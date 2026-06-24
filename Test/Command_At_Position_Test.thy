theory Command_At_Position_Test
  imports Semantic_Embedding.Semantic_Embedding
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

  (* ---- Test 9: WIP re-cut returns a SINGLE command, never the whole file ----
     The whole-theory-dump fix (command_at_position, s <= 1 branch) relies on
     command_spans_of_text / command_at_position_wip splitting a multi-command
     source into individual commands.  A position inside Cmd_Test_B must re-cut
     to ONLY that command's source — not the whole file (which would also carry
     Cmd_Test_A and Cmd_Test_Lemma). *)
  val [abs_B] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos_B]
  val file_B = the (Position.file_of abs_B)
  val off_B = the (Position.offset_of abs_B)
  val wip_B = PIDE_State.command_at_position_wip file_B off_B
  val _ = @{assert} (is_some wip_B)
  val (wip_src_B, _, _) = the wip_B
  val _ = @{assert} (String.isSubstring "Cmd_Test_B" wip_src_B)
  val _ = @{assert} (not (String.isSubstring "Cmd_Test_A" wip_src_B))
  val _ = @{assert} (not (String.isSubstring "Cmd_Test_Lemma" wip_src_B))
  val _ = writeln ("Test 9 - WIP re-cut returns only Cmd_Test_B, not the whole file: OK")

  (* ---- Test 10: WIP re-cut MISS returns NONE (no whole-file leak) ----
     When command_at_position's re-cut finds no span at the offset, the fix
     degrades to command_at_position_wip, which on a genuine miss returns NONE
     (caller then does a single-line search) rather than leaking the whole file.
     Verify an offset far past EOF yields NONE. *)
  val file_size = size (File.read (Path.explode file_B))
  val wip_miss = PIDE_State.command_at_position_wip file_B (file_size + 1000)
  val _ = @{assert} (is_none wip_miss)
  val _ = writeln ("Test 10 - WIP re-cut past EOF returns NONE (no whole-file leak): OK")

  (* ---- Test 11: recut_dump_command — the branch the whole-theory-leak fix
     ACTUALLY changed (command_at_position's s<=1 re-cut).  Tests 1-10 only
     touch unchanged code, so a regression to `| NONE => SOME (source, s, e)`
     (the original leak) would pass them.  Drive the helper with a SYNTHETIC
     whole-file dump and assert it NEVER returns the whole `source`:
       (a) an offset inside the 2nd command returns ONLY that command;
       (b) an offset past EOF (re-cut miss) returns NONE — not the whole source. *)
  val dump_src =
    "theory Synthetic_Dump\n" ^
    "  imports Main\n" ^
    "begin\n\n" ^
    "definition synth_alpha :: bool where \"synth_alpha = True\"\n\n" ^
    "definition synth_beta :: bool where \"synth_beta = False\"\n\n" ^
    "end\n"
  (* A path that does not exist: the in-memory re-cut uses `dump_src` directly
     and resolves keywords from its own header (imports Main); the disk fallback
     on a miss reads this absent file -> [] -> NONE. *)
  val dump_file = "/nonexistent/Synthetic_Dump.thy"
  fun off_of needle =  (* 1-based symbol offset of needle in dump_src (ASCII) *)
    Substring.size (#1 (Substring.position needle (Substring.full dump_src))) + 1
  (* (a) hit inside the 2nd command *)
  val hit = PIDE_State.recut_dump_command dump_file dump_src 1 (off_of "synth_beta = False")
  val _ = @{assert} (is_some hit)
  val (hit_src, _, _) = the hit
  val _ = @{assert} (String.isSubstring "synth_beta" hit_src)
  val _ = @{assert} (not (String.isSubstring "synth_alpha" hit_src))
  val _ = @{assert} (not (String.isSubstring "theory Synthetic_Dump" hit_src))
  val _ = @{assert} (hit_src <> dump_src)
  val _ = writeln ("Test 11a - recut_dump_command hit returns only the single command: OK")
  (* (b) re-cut miss must NOT return the whole source *)
  val miss = PIDE_State.recut_dump_command dump_file dump_src 1 (size dump_src + 50)
  val _ = @{assert} (is_none miss)
  val _ = writeln ("Test 11b - recut_dump_command miss returns NONE, not the whole theory: OK")

in
  writeln "\n=== All PIDE_State.command_at_position tests passed ==="
end
\<close>

end
