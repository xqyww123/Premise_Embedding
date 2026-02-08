signature VECTOR_ARITH = sig
type vector = Word8Vector.vector
type dot_ret
val dot : vector * vector -> dot_ret
val top_k : ('a * Word8Vector.vector) array * Word8Vector.vector * int
         -> ('a * real) list
end

functor Vector_Arith (
      CONFIG : sig val DIM : int
                   val encoding : string (*e.g., Q.15*)
                   val encoding_size : int (*e.g., 2 bytes*)
                   type dot_ret
                   val dot_ret_conv : dot_ret Foreign.conversion
               end
      ) : VECTOR_ARITH = struct

type vector = Word8Vector.vector
type dot_ret = CONFIG.dot_ret

fun malloc_align_64bytes size =
  let open Foreign Foreign.Memory
      val ptr  = malloc (size + 0w64)
      val ptr' = SysWord.andb(voidStar2Sysword ptr + 0w63, SysWord.notb 0w63)
              |> sysWord2VoidStar
   in (ptr, ptr')
  end

val vec_size' = CONFIG.DIM * CONFIG.encoding_size
val vec_size = Word.fromInt (CONFIG.DIM * CONFIG.encoding_size)
val vec_size_32align = Word.andb (vec_size + 0w31, Word.notb 0w31)

(* The Funniest code I have ever written. Blame on Poly/ML's insufficient API support.
   A memcpy from Word8Vector to Foreign.Memory could have saved the world! *)
fun copy_vector_w8 (src, dest) =
  let val i = Unsynchronized.ref 0
      val j = Unsynchronized.ref 0w0
   in while !i < vec_size' do
       let val w = Word8Vector.sub (src, !i)
        in Foreign.Memory.set8 (dest, !j, w)
         ; i := !i + 1
         ; j := !j + 0w1
       end
  end

(*
fun copy_vector_w64 (src, dest) =
  let val j = Unsynchronized.ref 0w0
   in while !j < vec_size do
       let val w = RunCall.loadNativeWord (src, RunCall.bytesPerWord + !j)
        in @{print} (!j)
         ; Foreign.Memory.set64 (dest, !j, w)
         ; @{print} (!j)
         ; j := !j + 0w8
       end
  end

val _ = if CONFIG.DIM * CONFIG.encoding_size mod Word.toInt RunCall.bytesPerWord = 0 then ()
        else error "unimplemented silly"
*)

fun chk_vec_shape v =
      if Word8Vector.length v = CONFIG.DIM * CONFIG.encoding_size
      then ()
      else error (String.concat [
            "Expecting ", string_of_int CONFIG.DIM, " x ", CONFIG.encoding,
            ", but given ", string_of_int (Word8Vector.length v), " bytes. ",
            "(Should be ", string_of_int (CONFIG.DIM * CONFIG.encoding_size), " bytes)."])

local (*always little endian*)
    open Foreign
    open Foreign.Memory

    
    (* val wordSize: SysWord.word = SysWord.fromLarge(Word.toLarge RunCall.bytesPerWord) *)

    fun load _ = raise Foreign "Cannot load Word8Vector from C"
    fun store(m: voidStar, v: Word8Vector.vector) =
      let val _ = chk_vec_shape v
          val (ptr_origin, ptr_aligned) = malloc_align_64bytes vec_size
          
          (* We need a smart way to copy data!
              val _ = RunCall.moveBytes (v, RunCall.unsafeCast (voidStar2Sysword ptr_aligned), RunCall.bytesPerWord, 0w0, vec_size) *)
       in copy_vector_w8 (v, ptr_aligned)
        ; setAddress(m, 0w0, ptr_aligned)
        ; (fn () => free ptr_origin)
      end
in
    val cWord8Vector: Word8Vector.vector conversion =
        makeConversion {
            load = load,
            store = store,
            ctype = LowLevel.cTypePointer
        }
end

(*
fun ntimes n F =
  let val i = Unsynchronized.ref 0
   in while !i < n do (
        F (!i)
      ; i := !i + 1)
  end

fun ntimes_w n F =
  let val i = Unsynchronized.ref 0w0
   in while !i < n do (
        F (!i)
      ; i := !i + 0w1)
  end

fun ntimes_wi n F =
  let val i = Unsynchronized.ref 0
      val w = Unsynchronized.ref 0w0
   in while !i < n do (
        F (!i, !w)
      ; i := !i + 1
      ; w := !w + 0w1)
  end
*)

(*
local (*always little endian*)
    open Foreign
    open Foreign.Memory

    
    (* val wordSize: SysWord.word = SysWord.fromLarge(Word.toLarge RunCall.bytesPerWord) *)

    fun load len alist ptr =
      List.tabulate (len, fn i =>
        let val j = Word32.toInt (get32 (ptr, Word.fromInt i))
            val (v, _) = Array.sub (alist, j)
         in v
        end )
    fun store (m: voidStar, _) =
      let val (ptr_origin, ptr_aligned) =
                  malloc_align_64bytes (vec_size_32align * Word.fromInt (Array.length vs))
       in Array.appi (fn (i, (_,v)) =>
            copy_vector_w8 (v, ++ (ptr_aligned, vec_size_32align * Word.fromInt i))) vs
        ; setAddress(m, 0w0, ptr_aligned)
        ; (fn () => free ptr_origin)
      end
in
    fun cI32_AList k alist : int array conversion =
        makeConversion {
            load = load k alist,
            store = store,
            ctype = LowLevel.cTypePointer
        }
end
*)


(* TODO!
fun soft_dot (v1, v2) = 0
*)

val (dot_opr, tok_k_c) =
  let open Foreign
      val lib = loadLibrary (
            Resources.master_directory @{theory} + Path.make ["Tools", "Vector_Arith", "build", "libisabelle_vector.so"]
              |> Path.expand
              |> Path.implode)
      val suffix = CONFIG.encoding
      val dot_sym = getSymbol lib ("dot_" ^ suffix)
      val dot_fun = buildCall3 (dot_sym, (cWord8Vector, cWord8Vector, cSsize), CONFIG.dot_ret_conv)
      val top_k_sym = getSymbol lib ("top_k_" ^ suffix)
      val tok_k_c = buildCall7 (
            top_k_sym,
            (cPointer, cWord8Vector, cInt32, cInt32, cInt32, cPointer, cPointer),
            cVoid)
   in (dot_fun, tok_k_c)
  end handle Foreign.Foreign msg => (
        error ("Fail to load the Isabelle Vector library.\n" ^ msg))

fun dot (x,y) = dot_opr (x,y,CONFIG.DIM)
fun top_k (assoc_array, query, k) =
  let open Foreign Foreign.Memory
      val wk = Word.fromInt k
      val () = Array.app (chk_vec_shape o snd) assoc_array
      val (ptr_origin, ptr_aligned) =
              malloc_align_64bytes (vec_size_32align * Word.fromInt (Array.length assoc_array))
      val result_indexes_ptr = malloc (wk * 0w4)
      val result_scores_ptr  = malloc (wk * 0w2)
   in \<^try>\<open>
  let val () = Array.appi (fn (i, (_,v)) =>
                  copy_vector_w8 (v, ++ (ptr_aligned, vec_size_32align * Word.fromInt i))) assoc_array
      val () = tok_k_c (ptr_aligned, query, CONFIG.DIM, Array.length assoc_array, k,
                        result_indexes_ptr, result_scores_ptr)
   in List.tabulate (k, fn i =>
        let val wi = Word.fromInt i
            val j = Word32.toInt (get32 (result_indexes_ptr, wi))
            val raw = Word.toInt (get16 (result_scores_ptr, wi))
            val score = Real.fromInt (if raw >= 32768 then raw - 65536 else raw) / 32768.0
            val (v, _) = Array.sub (assoc_array, j)
         in (v, score)
        end )
  end finally (
      free ptr_origin
    ; free result_indexes_ptr
    ; free result_scores_ptr )\<close>
  end


end

structure Vector_Arith_Q15_D384 = Vector_Arith(struct
    val DIM = 384
    val encoding = "q15"
    val encoding_size = 2
    type dot_ret = int
    val dot_ret_conv = Foreign.cInt16
  end)

structure Vector_Arith_Q15_D24 = Vector_Arith(struct
    val DIM = 24
    val encoding = "q15"
    val encoding_size = 2
    type dot_ret = int
    val dot_ret_conv = Foreign.cInt16
  end)

structure Vector_Arith_Q15_D2048 = Vector_Arith(struct
    val DIM = 2048
    val encoding = "q15"
    val encoding_size = 2
    type dot_ret = int
    val dot_ret_conv = Foreign.cInt16
  end)
