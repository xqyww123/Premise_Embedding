theory Probe_Command_Header_Test
  imports Semantic_Embedding
begin

section \<open>Test targets\<close>

definition probe_const :: "nat \<Rightarrow> nat" where \<open>probe_const x = x + 1\<close>

fun probe_fun :: "nat \<Rightarrow> nat" where \<open>probe_fun 0 = 0\<close> | \<open>probe_fun (Suc n) = Suc (probe_fun n)\<close>

datatype probe_dt = A | B nat

type_synonym probe_tsyn = "nat list"

lemma probe_lemma: \<open>probe_const 0 = 1\<close>
  by (simp add: probe_const_def)

locale probe_locale =
  fixes x :: nat
  assumes \<open>x > 0\<close>

section \<open>Probe\<close>

ML \<open>
let
  fun probe_entity space name =
    let
      val pos = Name_Space.the_entry space name |> #pos
      val [abs_pos] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos]
      val file = the (Position.file_of abs_pos)
      val offset = the (Position.offset_of abs_pos)
      val input = XML.Encode.pair XML.Encode.string XML.Encode.int (file, offset)
      val result = Scala.function1 "pide_state.probe_command_header"
        (YXML.string_of_body input)
    in writeln ("=== " ^ name ^ " ===\n" ^ result) end

  val ctxt = \<^context>
  val const_space = Proof_Context.consts_of ctxt |> Consts.space_of
  val type_space = Sign.type_space (Proof_Context.theory_of ctxt)
  val fact_space = Proof_Context.facts_of ctxt |> Facts.space_of
  val locale_space = Locale.locale_space (Proof_Context.theory_of ctxt)
in
  probe_entity const_space @{const_name probe_const};
  probe_entity const_space @{const_name probe_fun};
  probe_entity type_space @{type_name probe_dt};
  (* type_synonym doesn't register in type_space the same way, skip *)
  probe_entity fact_space "Probe_Command_Header_Test.probe_lemma";
  probe_entity locale_space "Probe_Command_Header_Test.probe_locale"
end
\<close>

end
