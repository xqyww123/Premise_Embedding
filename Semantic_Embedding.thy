theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling
begin 

declare [[ML_debugger]]

ML_file \<open>Tools/semantic_store.ML\<close>

ML \<open>Resources.master_directory @{theory}\<close>

ML \<open>
fun get_theory_file_path thy =
  let
    val thy_name = Context.theory_long_name thy
    val master_dir = Resources.master_directory thy
    val thy_base = Long_Name.base_name thy_name
    val thy_file = Path.ext "thy" (master_dir + Path.basic thy_base)
    val expanded_path = File.full_path master_dir thy_file
  in
    File.platform_path expanded_path
  end
\<close>
ML \<open>get_theory_file_path @{theory HOL}\<close>
ML \<open>type t = Proof.context\<close>

end