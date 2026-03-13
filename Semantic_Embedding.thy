theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling Performant_Isabelle_ML.Performant_Isabelle_ML
          Isa_REPL.Isa_REPL
begin

declare [[ML_debugger, ML_exception_debugger, ML_exception_trace, ML_print_depth=1000]]

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
ML_file \<open>Tools/term_serial_index.ML\<close>
ML_file \<open>Tools/theory_structure.ML\<close>
ML_file \<open>Tools/infra_filter.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>

ML \<open>Semantic_Stor e.interpret_file @{theory List}\<close>

thm List.List.list.map_cong

end
