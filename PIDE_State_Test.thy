theory PIDE_State_Test
  imports Semantic_Embedding
begin

section \<open>Test: absolutize_id_based_pos resolves ID-based positions\<close>

definition AAA_test :: bool where \<open>AAA_test = True\<close>

definition BBB_test :: bool where \<open>BBB_test = False\<close>

ML \<open>
let
  (* ---- helpers ---- *)
  fun pos_str pos =
    String.concat [
      "file=", the_default "<none>" (Position.file_of pos),
      " line=", the_default "<none>" (Option.map string_of_int (Position.line_of pos)),
      " offset=", the_default "<none>" (Option.map string_of_int (Position.offset_of pos)),
      " end_offset=", the_default "<none>" (Option.map string_of_int (Position.end_offset_of pos)),
      " id=", the_default "<none>" (Position.id_of pos)]

  fun assert_resolved tag pos =
    ( @{assert} (is_some (Position.file_of pos))
    ; @{assert} (is_some (Position.line_of pos))
    ; writeln (tag ^ ": " ^ pos_str pos) )

  (* Get positions of known constants from the name space.
     These positions carry a command ID and command-relative offsets. *)
  val const_space = Proof_Context.consts_of \<^context> |> Consts.space_of
  val pos_AAA = Name_Space.the_entry const_space @{const_name AAA_test} |> #pos
  val pos_BBB = Name_Space.the_entry const_space @{const_name BBB_test} |> #pos

  val _ = writeln ("--- Before resolution ---")
  val _ = writeln ("AAA_test: " ^ pos_str pos_AAA)
  val _ = writeln ("BBB_test: " ^ pos_str pos_BBB)

  (* ---- Test 1: basic resolution ---- *)
  val [res_AAA, res_BBB] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos_AAA, pos_BBB]

  val _ = writeln ("\n--- After resolution ---")
  val _ = assert_resolved "AAA_test" res_AAA
  val _ = assert_resolved "BBB_test" res_BBB

  (* ---- Test 2: resolved positions have file-relative offsets ---- *)
  (* BBB_test is defined after AAA_test, so its offset must be larger *)
  val off_AAA = the (Position.offset_of res_AAA)
  val off_BBB = the (Position.offset_of res_BBB)
  val _ = @{assert} (off_BBB > off_AAA)
  val _ = writeln ("\noffset ordering: AAA=" ^ string_of_int off_AAA
                   ^ " < BBB=" ^ string_of_int off_BBB ^ "  OK")

  (* ---- Test 3: offsets are file-relative, not command-relative ---- *)
  (* A command-relative offset for a definition name is typically small (< 50).
     A file-relative offset for BBB_test must account for all preceding text,
     so it should be well above 1. More importantly, it should be larger than
     the offset of AAA_test. *)
  val _ = @{assert} (off_AAA > 1)
  val _ = @{assert} (off_BBB > 1)
  val _ = writeln ("file-relative offsets are positive: AAA=" ^ string_of_int off_AAA
                   ^ " BBB=" ^ string_of_int off_BBB ^ "  OK")

  (* ---- Test 4: pass-through for already-resolved positions ---- *)
  val already_resolved = Position.line_file 42 "/some/file.thy"
  val [unchanged] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [already_resolved]
  val _ = @{assert} (Position.line_of unchanged = SOME 42)
  val _ = @{assert} (Position.file_of unchanged = SOME "/some/file.thy")
  val _ = writeln ("\npass-through for already-resolved position: OK")

  (* ---- Test 5: pass-through for Position.none ---- *)
  val [unchanged_none] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [Position.none]
  val _ = @{assert} (Position.line_of unchanged_none = NONE)
  val _ = @{assert} (Position.file_of unchanged_none = NONE)
  val _ = writeln ("pass-through for Position.none: OK")

  (* ---- Test 6: mixed list preserves order ---- *)
  val mixed_input = [Position.none, pos_AAA, already_resolved, pos_BBB, Position.none]
  val mixed_output = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} mixed_input
  val _ = @{assert} (length mixed_output = 5)
  (* Position.none stays as-is *)
  val _ = @{assert} (Position.line_of (nth mixed_output 0) = NONE)
  (* pos_AAA gets resolved *)
  val _ = @{assert} (is_some (Position.file_of (nth mixed_output 1)))
  (* already_resolved stays as-is *)
  val _ = @{assert} (Position.line_of (nth mixed_output 2) = SOME 42)
  (* pos_BBB gets resolved *)
  val _ = @{assert} (is_some (Position.file_of (nth mixed_output 3)))
  (* Position.none stays as-is *)
  val _ = @{assert} (Position.line_of (nth mixed_output 4) = NONE)
  val _ = writeln ("mixed list order preserved: OK")

  (* ---- Test 7: line numbers are sensible ---- *)
  (* AAA_test is defined before BBB_test, so its line number should be <=  *)
  val line_AAA = the (Position.line_of res_AAA)
  val line_BBB = the (Position.line_of res_BBB)
  val _ = @{assert} (line_AAA > 0)
  val _ = @{assert} (line_BBB >= line_AAA)
  val _ = writeln ("line ordering: AAA=" ^ string_of_int line_AAA
                   ^ " <= BBB=" ^ string_of_int line_BBB ^ "  OK")

  (* ---- Test 8: end_offset is also file-relative when present ---- *)
  val end_off_AAA = Position.end_offset_of res_AAA
  val end_off_BBB = Position.end_offset_of res_BBB
  val _ = case (end_off_AAA, end_off_BBB) of
      (SOME ea, SOME eb) =>
        ( @{assert} (ea > off_AAA)
        ; @{assert} (eb > off_BBB)
        ; writeln ("end_offsets are beyond start offsets: OK") )
    | _ => writeln ("end_offsets not present (skipped)")

  (* ---- Test 9: duplicate IDs in input ---- *)
  val [dup1, dup2] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos_AAA, pos_AAA]
  val _ = @{assert} (Position.offset_of dup1 = Position.offset_of dup2)
  val _ = @{assert} (Position.file_of dup1 = Position.file_of dup2)
  val _ = writeln ("duplicate IDs produce consistent results: OK")

in
  writeln "\n=== All PIDE_State.absolutize_id_based_pos tests passed ==="
end
\<close>

end
