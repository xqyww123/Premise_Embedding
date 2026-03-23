theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling
begin

ML_file \<open>Tools/pide_state.ML\<close>

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
(* ML_file \<open>Tools/term_serial_index.ML\<close> *)
ML_file \<open>Tools/theory_structure.ML\<close>
ML_file \<open>Tools/infra_filter.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>


declare [[auto_interpret_for_embedding = false]]

ML \<open>Semantic_Store.is_thy_embedded (Context.Proof \<^context>) ("HOL.HOL", "oai.text-embedding-3-small")\<close>
ML \<open>Semantic_Store.embed_semantics (Context.Proof \<^context>) ([@{theory HOL}], "oai.text-embedding-3-small")\<close>

(*
ML \<open>Semantic_Store.query_semantics   (Context.Proof @{context})
      (Universal_Key.Constant "HOL.implies") false\<close>
*)
 
ML \<open>Semantic_Store.query_knn (Context.Proof @{context})
      "logical or" 10 [Universal_Key.ConstantK] NONE\<close>

term HOL.simp_implies

term Orderings.partial_preordering

ML \<open>@{theory Lazy_Sequence}\<close>
ML \<open>@{theory Predicate}\<close>



(*

ML \<open>Semantic_Store.int erpret (Context.Theory @{theory List})\<close> 
ML \<open>Semantic_Store.query  (Context.Proof @{context})
      (Universal_Key.Class "Orderings.partial_preordering") false\<close>

ML \<open>Context.theory_long_name @{theory}\<close>
ML \<open>Context.theory_base_name @{theory}\<close>

thm HOL.conj_comms

ML \<open>@{term "0::nat"}\<close> 
ML \<open>Universal_Key.key_of_co nstant (Context.Proof \<^context>) "Groups.zero_class.zero"\<close>

ML \<open>
val facts = Proof_Context.facts_of \<^context>
val fact_space = Facts.space_of facts
val ns_entry = Name_Space.the_entry fact_space "AAA"
\<close>

ML \<open>Thm.derivation_id @{thm allI}\<close>

ML \<open>type x = Proofterm.thm_id\<close>

ML \<open>\<^try>\<open>xxxx catch e => xxx\<close>\<close>





ML \<open>Thm.derivation_id\<close>

ML \<open>  l et
    val n = 400 * 1024 * 1024
    val src = Word8Array.array (n, 0w42)
    val dst = Word8Array.array (n, 0w0)
    val (t, _) = Timing.timing (fn () =>
      let val i = Unsynchronized.ref 0
      in while !i < n do
        (Word8Array.update (dst, !i, Word8Array.sub (src, !i)); i := !i + 1)
      end) ()
  in
    tracing ("Copied " ^ Int.toString n ^ " bytes. Time: " ^ Timing.message t)
  end\<close>

(* 
ML \<open>Semantic_Store.load_store (Path.explode "/tmp/xxx.mpl")\<close>
ML \<open>Semantic_Store.interpret  (Context.Theory @{theory List})\<close>
ML \<open>Semantic_Store.save_store (Path.explode "/tmp/xxx2.mpl")\<close>
 *)

(* ML \<open>Semantic_Store.interpret  (Context.Theory @{theory List})\<close> *)
(*
ML_file \<open>Tools/test_word64_to_bytes.ML\<close>

ML \<open>Semantic_Store.save_store (Path.explode "/tmp/xxx2.mpl")\<close>


ML \<open>Byte.stringToBytes\<close>
ML \<open>Word.PackWord32Little\<close>


ML \<open>map (Context.theory_name {long=true})
  [@{theory Pure}, @{theory Code_Generator}, @{theory Code_Evaluation}]\<close>
       
ML \<open>Theory_Hash.hash_of @{theory String}\<close>
ML \<open>Semantic_Store.interpret  (Context.Theory @{theory List})\<close>

ML \<open> @{thm append1_eq_conv}
  |> Thm.derivation_id\<close>

ML \<open>
#query_class (Semantic_Store.make_query_functions (Context.Theory @{theory}) [] true)
  "Groups.monoid_add" |> the |> tracing
\<close>  
ML \<open>
#query_constant (Semantic_Store.make_query_functions (Context.Theory @{theory}) [] true)
  "Cons" |> the |> tracing
\<close>
ML \<open> 
#query_theorem (Semantic_Store.make_query_functions (Context.Theory @{theory}) [] true)
  "Set.UNIV_I" |> the |> tracing
\<close>
term Nil
thm append1_eq_conv
typ "'a :: monoid_add"
term Bit_Operations.semiring_modulo_trivial
term Bit_Operations.semiring_modulo_trivial

thm List.List.list.pred_True

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


 
ML \<open>val old_tracing = !Private_Output.tracing_fn\<close>
ML \<open>Private_Output.tracing_fn := (fn x => old_tracing ("aaa" :: x))\<close>

ML \<open>tracing "asd"\<close>


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
*) *) *)

end
