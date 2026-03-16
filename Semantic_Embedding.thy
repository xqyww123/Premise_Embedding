theory Semantic_Embedding
  imports Isabelle_RPC.Remote_Procedure_Calling Performant_Isabelle_ML.Performant_Isabelle_ML
begin

declare [[ML_debugger, ML_exception_debugger, ML_exception_trace, ML_print_depth=1000]]

ML \<open>Context.theory_id @{theory}\<close>

ML_file \<open>Tools/theory_structure.ML\<close>
ML_file \<open>Tools/theory_hash.ML\<close>
ML_file \<open>Tools/Hasher.ML\<close>
ML_file \<open>Tools/pide_state.ML\<close>
ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
(* ML_file \<open>Tools/term_serial_index.ML\<close> *)
ML_file \<open>Tools/infra_filter.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>
  

ML \<open>Semantic_Store.interpret (Context.Theory @{theory List})\<close>


ML \<open>Thm.derivation_id\<close>
ML \<open>Word.fromLargeWord\<close>
ML \<open>PackWord.subVec\<close>
ML \<open>Word31.wordSize\<close>
ML \<open>type T = word\<close>
ML \<open>Context.theory_identifier @{theory}\<close>




locale A =
  fixes XX :: bool
begin

term XX

end

ML \<open>\<close>



(*
ML \<open>Context.theory_id @{theory}\<close>

term \<open>\<lambda>x'. x' \<and> True\<close>

ML \<open>PIDE_State.line_column_to_offset "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 14 36\<close>
ML \<open>PIDE_State.offset_to_line_column "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 443\<close>
ML \<open>PIDE_State.goto_definition (Position.make {line=14, offset=443, end_offset=40, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>
ML \<open>PIDE_State.entity_at_position (Position.make {line=14, offset=443, end_offset=530, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>

ML \<open>PIDE_State.line_column_to_offset "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 14 26\<close>
ML \<open>PIDE_State.offset_to_line_column "/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy" 438\<close>
ML \<open>PIDE_State.goto_definition (Position.make {line=14, offset=438, end_offset=40, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>
ML \<open>PIDE_State.entity_at_position (Position.make {line=14, offset=438, end_offset=530, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>

ML \<open>PIDE_State.goto_definition (Position.make {line=74, offset=1694, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>
ML \<open>PIDE_State.hover_message (Position.make {line=74, offset=1694, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>

ML \<open>PIDE_State.entity_at_position (Position.make {line=74, offset=1694, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>


ML \<open>Semantic_Stor e.interpret (Context.Theory @{theory String})\<close>

ML \<open>
val hol_simpset =
  Raw_Simplifier.simpset_of (Proof_Context.init_global \<^theory>\<open>HOL\<close>)
val hol_ss_ctxt = put_simpset hol_simpset \<^context>
fun is_trivial_thm thm =
      (let
        val thm' = Conv.fconv_rule (Object_Logic.atomize \<^context>) thm
        val thm'' = Simplifier.asm_full_simplify hol_ss_ctxt thm'
      in @{print} thm''
        ; Envir.beta_eta_contract (Thm.prop_of thm'') aconv \<^prop>\<open>True\<close> end)
      handle THM _ => false | CTERM _ => false | TERM _ => false | TYPE _ => false
val x= is_trivial_thm @{thm String.char.case_cong}
\<close>


   

ML \<open>Conv.fconv_rule (Object_Logic.atomize ctxt)\<close>
ML \<open>Simplifier.simpset_of (Proof_Context.init_global @{theory HOL})\<close>
ML \<open>
let val 
Simplifier.asm_full_simplify (Simplifier. )\<close>

datatype char =
  Char (digit0: bool) (digit1: bool) (digit2: bool) (digit3: bool)
       (digit4: bool) (digit5: bool) (digit6: bool) (digit7: bool)

typ char.char_IITN_char
thm String.char.exhaust_disc[]

lemma A: \<open>(char = char \<longrightarrow> P) \<longrightarrow> P\<close> by auto
thm A[simplified]

(*
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

   
ML \<open>PIDE_State.hover_message (Position.make {line=19, offset=485, end_offset=530, props={label="", file="/home/qiyuan/Current/MLML/contrib/Semantic_Embedding/Semantic_Embedding.thy", id=""}})\<close>
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
ML \<open>PIDE_State.hover_message (Position.make {line=74, offset=1694, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 208\<close>
ML \<open>PIDE_State.line_column_to_offset "~~/src/HOL/List.thy" 394 29\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 13983\<close>
ML \<open>PIDE_State.goto_definition (Position.make {line=74, offset=13983, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>
ML \<open>PIDE_State.offset_to_line_column "~~/src/HOL/List.thy" 13702\<close>
ML \<open>PIDE_State.hover_message (Position.make {line=74, offset=13983, end_offset=530, props={label="", file="~~/src/HOL/List.thy", id=""}})\<close>

ML \<open>PIDE_State.absolutize_id_based_pos {write_to_temp_file=true} pos\<close>

ML \<open>Resources.theory_qualifier "HOL.List"\<close>
ML \<open>Resources.global_session\<close>
ML \<open>Sessions.background\<close>


ML \<open>@{term \<open>OFCLASS('a, zero_class)\<close>}\<close>

typ List.list.list_IITN_list

thm List.List.list.map_cong
*) *)

end
