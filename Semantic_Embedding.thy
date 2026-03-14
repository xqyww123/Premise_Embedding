theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling Performant_Isabelle_ML.Performant_Isabelle_ML
          Isa_REPL.Isa_REPL
begin

declare [[ML_debugger, ML_exception_debugger, ML_exception_trace, ML_print_depth=1000]]

ML_file \<open>Tools/pide_state.ML\<close>
ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
ML_file \<open>Tools/term_serial_index.ML\<close>
ML_file \<open>Tools/theory_structure.ML\<close>
ML_file \<open>Tools/infra_filter.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>

text \<open>Tes t
1\<close>

definition \<open>AAA = True\<close>

term AAA
ML \<open>val pos =
    map (Name_Space.the_entry (Proof_Context.consts_of \<^context> |> Consts.space_of) #> #pos)
        [@{const_name AAA}, @{const_name Nil}]\<close>

ML \<open>PIDE_State.goto_definition (Position.make {line=20, offset=526, end_offset=529, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>

ML \<open>PIDE_State.absolutize_id_based_pos {write_to_temp_file=true} pos\<close>




ML \<open>Semantic_Store.inter pret (Context.Theory @{theory Groups})\<close>

ML \<open>@{term \<open>OFCLASS('a, zero_class)\<close>}\<close>

typ List.list.list_IITN_list

thm List.List.list.map_cong

end
