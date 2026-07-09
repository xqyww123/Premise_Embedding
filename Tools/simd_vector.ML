signature VECTOR_ARITH = sig
type vector = Word8Vector.vector
type dot_ret
val dot : vector * vector -> dot_ret
val top_k : ('a * Word8Vector.vector) array * Word8Vector.vector * int
         -> ('a * real) list
end

(* Where libisabelle_vector.so lives is no longer fixed. A source checkout builds it
   under Tools/Vector_Arith/build, but a pip-installed wheel ships it as package data
   beside the Python module, under a site-packages path that depends on the
   interpreter. So ask the Python package over RPC: its library_path() resolves and
   validates the very file ctypes loads, which keeps ML and Python on one build.
   ISABELLE_VECTOR_SO overrides, and a checkout with no usable Python falls back to
   the in-tree path.

   Resolution is deferred until the library is first used, never done at theory-load
   time: Remote_Procedure_Calling auto-launches the Python host, and an `isabelle
   build` has no business spawning one just to load a theory. Lazy also means the
   three functor instantiations below share one lookup and one dlopen. *)
local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in
val vector_library_path_cmd : (unit, string) Remote_Procedure_Calling.command = {
  name = "Vector_Arith.library_path",
  arg_schema = packUnit,
  ret_schema = unpackString,
  callback = [],
  timeout = SOME (Time.fromSeconds 60)
}
end

val vector_library_path =
  let
    (* Isabelle_Semantic_Embedding._vecarith picks the same name from sys.platform. *)
    val lib_name =
      if ML_System.platform_is_windows then "isabelle_vector.dll"
      else if ML_System.platform_is_macos then "libisabelle_vector.dylib"
      else "libisabelle_vector.so"
    fun in_tree () =
      Resources.master_directory \<^theory>
        + Path.make ["Tools", "Vector_Arith", "build", lib_name]
        |> Path.expand |> Path.implode
    fun ask_python () =
      (Remote_Procedure_Calling.load ["Isabelle_Semantic_Embedding"];
       Remote_Procedure_Calling.call_command vector_library_path_cmd ())
    (* Any failure to reach the host -- not launched, module missing, library not
       built there -- means fall back to the checkout. Interrupts are not failures. *)
    val resolved = Lazy.lazy (fn () =>
      case getenv "ISABELLE_VECTOR_SO" of
        "" =>
          (case Exn.capture_body ask_python of
             Exn.Res path => path
           | Exn.Exn exn => if Exn.is_interrupt exn then Exn.reraise exn else in_tree ())
      | path => path)
    (* Unconditionally, because platform_path is right in every case and identity in
       most: on Unix it is `fn path => path`. On Windows the ML process is a native
       PE launched from a Cygwin shell, so Foreign.loadLibrary wants C:\..., while
       Python -- found by `command -v python3` under that same Cygwin bash -- may
       report either shape. platform_path maps C:\x and C:/x and /cygdrive/c/x all to
       C:\x, and /usr/lib/... through CYGWIN_ROOT (which getsettings does export: its
       `set -o allexport` spans the assignment). *)
  in fn () => ML_System.platform_path (Lazy.force resolved) end

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

(* Deferred: resolving the path talks to the Python host, so nothing here may run
   before the first dot/top_k. See vector_library_path above. *)
val foreign_calls = Lazy.lazy (fn () =>
  let open Foreign
      val lib = loadLibrary (vector_library_path ())
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
        error ("Fail to load the Isabelle Vector library.\n" ^ msg)))

fun dot_opr args = #1 (Lazy.force foreign_calls) args
fun tok_k_c args = #2 (Lazy.force foreign_calls) args

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
            (* The kernel accumulates round((a*b)/32768) per lane, so raw is Q1.15:
               raw = 32768 * |a| * |b| * cos. Vectors reach C already scaled to
               TARGET_NORM (see Isabelle_Semantic_Embedding._vecarith.encode_q15),
               which both keeps |raw| under int16 -- the accumulator wraps, it does
               not saturate -- and makes this a cosine up to that constant factor.
               Sledgehammer_MaSh.normalize_scores divides each list by its own mean,
               so the constant is irrelevant downstream. *)
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
