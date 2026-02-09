theory Sledgehammer_Embedding
  imports HOL.Nunchaku
          Isabelle_RPC.Remote_Procedure_Calling
begin


ML_file \<open>Tools/simd_vector.ML\<close>
ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
(* declare [[ML_debugger]] *)
ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding.ML\<close>

(*
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

Test Vector Lib*)

(*TODO: move this to Pure*)



end