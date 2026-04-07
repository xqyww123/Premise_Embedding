theory Command_At_Position_Test
  imports Semantic_Embedding
begin

section \<open>Test targets\<close>

definition Cmd_Test_A :: bool where \<open>Cmd_Test_A = True\<close>

definition Cmd_Test_B :: bool where \<open>Cmd_Test_B = False\<close>

lemma Cmd_Test_Lemma: \<open>Cmd_Test_A \<or> \<not> Cmd_Test_A\<close>
  by (simp add: Cmd_Test_A_def)

section \<open>Tests\<close>

ML \<open>
let
  (* ---- helpers ---- *)
  val const_space = Proof_Context.consts_of \<^context> |> Consts.space_of
  val pos_A = Name_Space.the_entry const_space @{const_name Cmd_Test_A} |> #pos
  val pos_B = Name_Space.the_entry const_space @{const_name Cmd_Test_B} |> #pos

  (* ---- Test 1: basic lookup returns SOME ---- *)
  val result_A = PIDE_State.command_at_position pos_A
  val _ = @{assert} (is_some result_A)
  val (src_A, start_A, end_A) = the result_A
  val _ = writeln ("Test 1 - command_at_position returns SOME for Cmd_Test_A: OK")
  val _ = writeln ("  source: " ^ src_A)
  val _ = writeln ("  span: [" ^ string_of_int start_A ^ ", " ^ string_of_int end_A ^ ")")

  (* ---- Test 2: source contains the definition keyword and name ---- *)
  val _ = @{assert} (String.isSubstring "definition" src_A)
  val _ = @{assert} (String.isSubstring "Cmd_Test_A" src_A)
  val _ = writeln ("Test 2 - source contains 'definition' and 'Cmd_Test_A': OK")

  (* ---- Test 3: second definition is at a different span ---- *)
  val result_B = PIDE_State.command_at_position pos_B
  val _ = @{assert} (is_some result_B)
  val (src_B, start_B, end_B) = the result_B
  val _ = @{assert} (String.isSubstring "Cmd_Test_B" src_B)
  val _ = @{assert} (start_B > start_A)
  val _ = writeln ("Test 3 - Cmd_Test_B command is after Cmd_Test_A: OK")
  val _ = writeln ("  span A: [" ^ string_of_int start_A ^ ", " ^ string_of_int end_A ^ ")")
  val _ = writeln ("  span B: [" ^ string_of_int start_B ^ ", " ^ string_of_int end_B ^ ")")

  (* ---- Test 4: commands do not overlap ---- *)
  val _ = @{assert} (end_A <= start_B)
  val _ = writeln ("Test 4 - commands do not overlap (end_A <= start_B): OK")

  (* ---- Test 5: span is non-empty ---- *)
  val _ = @{assert} (end_A > start_A)
  val _ = @{assert} (end_B > start_B)
  val _ = writeln ("Test 5 - spans are non-empty: OK")

  (* ---- Test 6: Position.none returns NONE ---- *)
  val _ = @{assert} (is_none (PIDE_State.command_at_position Position.none))
  val _ = writeln ("Test 6 - Position.none returns NONE: OK")

  (* ---- Test 7: lemma command ---- *)
  val fact_space = Proof_Context.facts_of \<^context> |> Facts.space_of
  val pos_lemma = Name_Space.the_entry fact_space "Command_At_Position_Test.Cmd_Test_Lemma" |> #pos
  val result_lemma = PIDE_State.command_at_position pos_lemma
  val _ = @{assert} (is_some result_lemma)
  val (src_lemma, _, _) = the result_lemma
  val _ = @{assert} (String.isSubstring "lemma" src_lemma)
  val _ = @{assert} (String.isSubstring "Cmd_Test_Lemma" src_lemma)
  val _ = writeln ("Test 7 - lemma command source: OK")
  val _ = writeln ("  source: " ^ src_lemma)

  (* ---- Test 8: query via file + offset (the RPC path) ---- *)
  (* Resolve pos_A to get an absolute file + offset, then query again *)
  val [abs_A] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos_A]
  val file_A = the (Position.file_of abs_A)
  val off_A = the (Position.offset_of abs_A)
  val direct_result = PIDE_State.command_at_position abs_A
  val _ = @{assert} (is_some direct_result)
  val (direct_src, _, _) = the direct_result
  val _ = @{assert} (String.isSubstring "Cmd_Test_A" direct_src)
  val _ = writeln ("Test 8 - query via absolute file+offset: OK")

in
  writeln "\n=== All PIDE_State.command_at_position tests passed ==="
end
\<close>

end
