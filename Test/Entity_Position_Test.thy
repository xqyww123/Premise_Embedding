theory Entity_Position_Test
  imports Semantic_Embedding
begin

section \<open>Test targets\<close>

definition Pos_Test_A :: bool where \<open>Pos_Test_A = True\<close>

definition Pos_Test_B :: bool where \<open>Pos_Test_B = False\<close>

lemma Pos_Test_Lemma: \<open>Pos_Test_A \<or> \<not> Pos_Test_A\<close>
  by (simp add: Pos_Test_A_def)

section \<open>Tests\<close>

ML \<open>File.standard_path (Path.explode "~~")\<close>
 
ML \<open>Run_Python.run "return str(1 + 2)"\<close>

ML \<open>
let
  val ctxt = \<^context>
  val thy = Proof_Context.theory_of ctxt

  (* ---- helpers ---- *)
  fun pos_info pos =
    (the_default "" (Position.file_of pos),
     the_default 0 (Position.line_of pos),
     the_default 0 (Position.offset_of pos))

  val const_space = Proof_Context.consts_of ctxt |> Consts.space_of
  val fact_space = Proof_Context.facts_of ctxt |> Facts.space_of

  (* ==== Part A: Compiled-session entities (from HOL heap) ==== *)

  (* ---- Test 1: compiled constant has file + line + offset ---- *)
  val true_pos = #pos (Name_Space.the_entry const_space @{const_name True})
  val (true_file, true_line, true_off) = pos_info true_pos
  val _ = writeln ("Test 1 - HOL.True position:")
  val _ = writeln ("  file: " ^ true_file)
  val _ = writeln ("  line: " ^ string_of_int true_line)
  val _ = writeln ("  offset: " ^ string_of_int true_off)
  val _ = @{assert} (true_file <> "")
  val _ = @{assert} (true_line > 0)
  val _ = @{assert} (true_off > 0)
  val _ = @{assert} (String.isSuffix ".thy" true_file)
  val _ = writeln "Test 1: OK"

  (* ---- Test 2: compiled theorem has file + line + offset ---- *)
  val refl_pos = #pos (Name_Space.the_entry fact_space "HOL.refl")
  val (refl_file, refl_line, refl_off) = pos_info refl_pos
  val _ = writeln ("Test 2 - HOL.refl position:")
  val _ = writeln ("  file: " ^ refl_file)
  val _ = writeln ("  line: " ^ string_of_int refl_line)
  val _ = writeln ("  offset: " ^ string_of_int refl_off)
  val _ = @{assert} (refl_file <> "")
  val _ = @{assert} (refl_line > 0)
  val _ = @{assert} (refl_off > 0)
  val _ = writeln "Test 2: OK"

  (* ---- Test 3: two different constants have different offsets ---- *)
  val conj_pos = #pos (Name_Space.the_entry const_space @{const_name conj})
  val disj_pos = #pos (Name_Space.the_entry const_space @{const_name disj})
  val (_, _, conj_off) = pos_info conj_pos
  val (_, _, disj_off) = pos_info disj_pos
  val _ = @{assert} (conj_off <> disj_off)
  val _ = writeln "Test 3 - conj and disj have different offsets: OK"

  (* ==== Part B: Live PIDE entities ==== *)

  (* ---- Test 4: live constant position via absolutize ---- *)
  val test_pos = #pos (Name_Space.the_entry const_space @{const_name Pos_Test_A})
  val [abs_pos] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [test_pos]
  val (abs_file, abs_line, abs_off) = pos_info abs_pos
  val _ = writeln ("Test 4 - Pos_Test_A (absolutized):")
  val _ = writeln ("  file: " ^ abs_file)
  val _ = writeln ("  line: " ^ string_of_int abs_line)
  val _ = writeln ("  offset: " ^ string_of_int abs_off)
  val _ = @{assert} (abs_file <> "")
  val _ = @{assert} (abs_off > 0)
  val _ = @{assert} (String.isSuffix "Entity_Position_Test.thy" abs_file)
  val _ = writeln "Test 4: OK"

  (* ---- Test 5: live lemma position via absolutize ---- *)
  val lemma_pos = #pos (Name_Space.the_entry fact_space
                          "Entity_Position_Test.Pos_Test_Lemma")
  val [abs_lemma] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [lemma_pos]
  val (lf, ll, lo) = pos_info abs_lemma
  val _ = @{assert} (lf <> "")
  val _ = @{assert} (lo > 0)
  val _ = @{assert} (String.isSuffix "Entity_Position_Test.thy" lf)
  val _ = writeln "Test 5 - Pos_Test_Lemma absolutized position: OK"

  (* ---- Test 6: absolutized position → command_at_position works ---- *)
  val cmd_A = PIDE_State.command_at_position abs_pos
  val _ = @{assert} (is_some cmd_A)
  val (src_A, _, _) = the cmd_A
  val _ = @{assert} (String.isSubstring "definition" src_A)
  val _ = @{assert} (String.isSubstring "Pos_Test_A" src_A)
  val _ = writeln ("Test 6 - command_at_position for Pos_Test_A: OK")
  val _ = writeln ("  source: " ^ src_A)

  (* ---- Test 7: Pos_Test_B is at a different position than Pos_Test_A ---- *)
  val test_B_pos = #pos (Name_Space.the_entry const_space @{const_name Pos_Test_B})
  val [abs_B] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [test_B_pos]
  val (_, _, abs_B_off) = pos_info abs_B
  val _ = @{assert} (abs_B_off > abs_off)
  val cmd_B = PIDE_State.command_at_position abs_B
  val _ = @{assert} (is_some cmd_B)
  val (src_B, _, _) = the cmd_B
  val _ = @{assert} (String.isSubstring "Pos_Test_B" src_B)
  val _ = writeln "Test 7 - Pos_Test_B at different position: OK"

  (* ---- Test 8: lemma command source ---- *)
  val cmd_lemma = PIDE_State.command_at_position abs_lemma
  val _ = @{assert} (is_some cmd_lemma)
  val (src_lemma, _, _) = the cmd_lemma
  val _ = @{assert} (String.isSubstring "lemma" src_lemma)
  val _ = @{assert} (String.isSubstring "Pos_Test_Lemma" src_lemma)
  val _ = writeln ("Test 8 - lemma command source: OK")
  val _ = writeln ("  source: " ^ src_lemma)

in
  writeln "\n=== All entity position tests passed ==="
end
\<close>

end
