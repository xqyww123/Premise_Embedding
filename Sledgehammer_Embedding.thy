theory Sledgehammer_Embedding
  imports Isabelle_RPC.Remote_Procedure_Calling
begin 

ML_file \<open>Tools/simd_vector.ML\<close>

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding.ML\<close>

end