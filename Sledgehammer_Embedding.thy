theory Sledgehammer_Embedding
  imports Isabelle_RPC.Remote_Procedure_Calling
begin
 


ML_file \<open>Tools/simd_vector.ML\<close>

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>


(*TODO: move this to Pure*)
ML \<open>
val v1 = Word8Vector.tabulate (4096, (fn i => Word8.fromLargeInt (if i mod 2 = 1 then i div 2 else 0)))
val v2 = Word8Vector.tabulate (4096, (fn i => Word8.fromLargeInt (if i mod 2 = 1 then i div 2 else 0)))
\<close>

ML \<open>getenv "LD_LIBRARY_PATH"\<close>

ML \<open>
Vector_Arith_Q15_D2048.dot(v1,v2)
\<close>

ML \<open>
let val vec_size = 4096
    fun rand_vec () = Word8Vector.tabulate (vec_size, fn _ =>
          Word8.fromInt (Random.random_range 0 255))

    fun run_test (trial, (n, k)) =
      let val vecs = List.tabulate (n, fn _ => rand_vec ())
          val query = rand_vec ()

          (* Ground truth: dot 函数计算所有点积 *)
          val dots = map_index (fn (i, v) =>
              (i, Vector_Arith_Q15_D2048.dot (v, query))) vecs
          val sorted = sort (fn ((_,a), (_,b)) => Int.compare (b, a)) dots
          val expected_scores = map snd (take k sorted)

          (* top_k 结果 *)
          val arr = Array.tabulate (n, fn i => (i, nth vecs i))
          val result = Vector_Arith_Q15_D2048.top_k (arr, query, k)
          val actual_scores = map (fn (_, sc) => Real.round (sc * 32768.0)) result

          val pass = (expected_scores = actual_scores)
       in if pass then ()
          else ( writeln (String.concat [
                   "FAIL trial=", string_of_int trial,
                   " n=", string_of_int n, " k=", string_of_int k])
               ; writeln ("  expected: " ^ String.concatWith ","
                   (map string_of_int expected_scores))
               ; writeln ("  actual:   " ^ String.concatWith ","
                   (map string_of_int actual_scores)))
        ; pass
      end

    val configs = [(5,2), (10,5), (50,10), (100,20), (500,50), (1000,100)]
    val num_trials = 10
    val tests = maps (fn cfg => List.tabulate (num_trials, fn t => (t, cfg))) configs
    val results = map run_test tests
    val total = length results
    val passed = length (filter I results)
 in writeln (String.concat [
      "top_k randomized test: ", string_of_int passed,
      " / ", string_of_int total, " passed"])
end
\<close>




declare [[ML_debugger]]
 
ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding.ML\<close>

ML \<open>Path.expand (Resources.master_directory @{theory} + Path.make ["Tools", "Vector_Arith", "build", "libisabelle_vector.so"])\<close>
 
ML \<open>Premise_Embedding_Context_Info.string_of_term @{context}
      @{prop \<open>\<lbrakk> A ; B ; C \<rbrakk> \<Longrightarrow> D\<close>}\<close>
 
term "\<lbrakk> A ; B ; C \<rbrakk> \<Longrightarrow> D"

typ list 


thm finite_mono_strict_prefix_implies_finite_fixpoint
 
sledgehammer_params [fact_filter = "embd"]
  
lemma
  assumes A: P
  assumes B: Q 
  shows "P \<Longrightarrow> P \<and> Q" 
  sledgehammer [fact_filter = "embd"]

ML \<open>
fun trim_makrup msg =
  let val segs = Unsynchronized.ref []
      val s = size msg
      val i = Unsynchronized.ref (s - 1)
      val j = Unsynchronized.ref s
      val mode = Unsynchronized.ref true
   in while !i >= 0 do (
       (if String.sub (msg, !i) = #"\005"
        then let val m = !mode
          in mode := not m
           ; if m
             then let val s' = !j - !i - 1
               in if s' > 0
                then segs := String.substring (msg, !i + 1, s') :: !segs
                else ()
              end
             else j := !i
          end
        else ())
      ; i := !i - 1)
    ; String.concat (!segs)
  end

fun string_of_term ctxt =
    let val ctxt' = ctxt
              |> Config.put Printer.show_types true
              |> Config.put Printer.show_sorts true
              |> Config.put Printer.show_markup false
              |> Config.put Printer.show_structs false
              |> Config.put Printer.show_question_marks true
              |> Config.put Printer.show_brackets false
     in Syntax.string_of_term ctxt'
     #> trim_makrup
    end
\<close>

ML \<open>String.substring ("01234567", 2, 3)\<close>

ML \<open>
string_of_term \<^context> (Thm.prop_of @{thm allI})
\<close>


ML \<open>Remote_Procedure_Calling.load ["Isabelle_Premise_Embedding"]\<close> 

ML \<open>
val test_cmd : (unit,unit) Remote_Procedure_Calling.command = {
  name = "test",
  arg_schema = MessagePackBinIO.Pack.packUnit,
  ret_schema = MessagePackBinIO.Unpack.unpackUnit,
  timeout = SOME (Time.fromSeconds 10)
} 
\<close>
  
ML \<open>Remote_Procedure_Calling.call_command test_cmd ()\<close> 






end