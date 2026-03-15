theory Semantic_Embedding
  imports Isa_REPL.Isa_REPL Performant_Isabelle_ML.Performant_Isabelle_ML
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

ML \<open>Symbol_Pos.explode ("\005\006asdsa\<alpha>", Position.none)\<close>




ML "PIDE_State.get_session_databases ()"

definition \<alpha> where \<open>\<alpha> = True\<close>

term \<open>\<lambda>x'. x' \<and> True\<close>
term \<open>Nil\<close>
ML \<open>val pos =
    map (Name_Space.the_entry (Proof_Context.consts_of \<^context> |> Consts.space_of) #> #pos)
        [@{const_name \<alpha>}, @{const_name Nil}]\<close>

  
ML \<open>PIDE_State.goto_definition (Position.make {line=19, offset=485, end_offset=530, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>
ML \<open>PIDE_State.goto_definit ion (Position.make {line=17, offset=461, end_offset=530, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close> e (Position.make {line=17, offset=461, end_offset=530, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>

ML \<open>expand_path "~~/aaa.a"\<close>
 

ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/HOL.thy" 3230\<close>
ML \<open>PIDE_State.line_column_to_offset "~~/src/HOL/HOL.thy" 100 12\<close>
ML \<open>PIDE_State.line_column_to_offset "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 19 26\<close>
ML \<open>PIDE_State.offset_to_line_column "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 485\<close>
ML \<open>PIDE_State.offset_to_line_column "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 481\<close>

ML \<open>PIDE_State.line_column_to_offset "~~/src/HOL/List.thy" 74 6\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 1694\<close>
 
 
ML \<open>PIDE_State.goto_definition (Position.make {line=74, offset=1694, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 208\<close>
ML \<open>PIDE_State.line_column_to_offset "~~/src/HOL/List.thy" 394 29\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 13983\<close>
ML \<open>PIDE_State.goto_definition (Position.make {line=74, offset=13983, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 13702\<close>

ML \<open>PIDE_State.absolutize_id_based_pos {write_to_temp_file=true} pos\<close>

ML \<open>Resources.theory_qualifier "HOL.List"\<close>
ML \<open>Resources.global_session\<close>
ML \<open>Sessions.background\<close>


ML \<open>Semantic_Store.inter pret (Context.Theory @{theory Groups})\<close>

ML \<open>@{term \<open>OFCLASS('a, zero_class)\<close>}\<close>

typ List.list.list_IITN_list

thm List.List.list.map_cong

end
