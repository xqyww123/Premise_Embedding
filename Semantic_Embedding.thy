theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling
begin

(* datatype test = A | B | C *)

(*declare [[ML_debugger, ML_exception_debugger]]*)

ML_file \<open>Tools/pide_state.ML\<close>

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
(* ML_file \<open>Tools/term_serial_index.ML\<close> *)
ML_file \<open>Tools/infra_filter.ML\<close>
ML_file \<open>Tools/theory_structure.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>

end
