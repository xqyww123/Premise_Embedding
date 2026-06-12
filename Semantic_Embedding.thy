theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling
begin

(*declare [[ML_debugger, ML_exception_debugger]]*)

ML_file \<open>Tools/pide_state.ML\<close>

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
ML_file \<open>Tools/infra_filter.ML\<close>
ML_file \<open>Tools/explain_term.ML\<close>
ML_file \<open>Tools/theory_structure.ML\<close>
ML_file \<open>Tools/locale_instance.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>

(*
declare [[ML_print_depth = 1000]]
ML \<open>Semantic_Store.query_knn (Context.Proof \<^context>)
    {query_text="'strong/complete induction on natural numbers: to prove P(n) for all n, assume P(m) for all m<n'",
     k=200, kinds=[Universal_Key.InductionRuleK], domain=Semantic_Store.ContextAll,
     term_patterns=[], type_patterns=[],
     theories_include=[], name_contains=[], target_type="nat"}\<close>
*)

end
