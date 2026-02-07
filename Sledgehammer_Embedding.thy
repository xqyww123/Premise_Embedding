theory Sledgehammer_Embedding
  imports Isabelle_RPC.Remote_Procedure_Calling
begin
 
declare [[ML_debugger]]

ML_file \<open>Tools/simd_vector.ML\<close>

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
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
  sledgeha mmer [fact_filter = "embd"]

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




(*TODO: move this to Pure*)
ML \<open>
val v1 = Word8Vector.tabulate (48, (fn i => Word8.fromLargeInt (if i mod 2 = 1 then i div 2 else 0)))
val v2 = Word8Vector.tabulate (48, (fn i => Word8.fromLargeInt (if i mod 2 = 1 then i div 2 else 0)))
\<close>

ML \<open>getenv "LD_LIBRARY_PATH"\<close>

ML \<open>
Vector_Arith_Q15_D24.dot(v1,v2)
\<close>


end