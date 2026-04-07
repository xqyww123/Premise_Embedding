theory Query_Tool_Test
  imports Semantic_Embedding
begin

section \<open>Test targets\<close>

definition Query_Test_Const :: nat where \<open>Query_Test_Const = 42\<close>

definition Query_Test_Fun :: \<open>nat \<Rightarrow> nat\<close> where \<open>Query_Test_Fun x = x + 1\<close>

lemma Query_Test_Lemma: \<open>Query_Test_Const = 42\<close>
  by (simp add: Query_Test_Const_def)

section \<open>Tests via Run_Python\<close>

ML \<open>
(* Context callbacks needed by Python-side entity enumeration *)
val query_test_cbs =
  let val ctx = Context.Proof \<^context>
  in [
    Universal_Key.make_universal_key_callback ctx,
    Context_Callbacks.make_constants_callback NONE ctx,
    Context_Callbacks.make_theorems_callback NONE ctx,
    Context_Callbacks.make_types_callback NONE ctx,
    Context_Callbacks.make_classes_callback NONE ctx,
    Context_Callbacks.make_locales_callback NONE ctx,
    Context_Callbacks.make_theory_long_name_callback ctx
  ] end

fun run_py source = Run_Python.run_with query_test_cbs source
\<close>

ML \<open>
(* ---- Test 1: entities_of returns entries, compiled entities have positions ---- *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind",
    "entries, _ = await entities_of(connection, [EntityKind.CONSTANT])",
    "with_pos = sum(1 for _, p in entries if p is not None)",
    "return f'total={len(entries)} with_pos={with_pos}'"
  ])
  val _ = writeln ("Test 1 - " ^ r)
  val _ = @{assert} (not (String.isSuffix "with_pos=0" r))
in writeln "Test 1: OK" end
\<close>

ML \<open>
(* ---- Test 2: compiled constant (HOL.True) has a valid position ---- *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.CONSTANT, 'True')",
    "entries, _ = await entities_of(connection, [EntityKind.CONSTANT])",
    "for k, p in entries:",
    "    if k == uk:",
    "        return f'file={p.file} line={p.line} offset={p.raw_offset}'",
    "return 'NOT_FOUND'"
  ])
  val _ = writeln ("Test 2 - HOL.True: " ^ r)
  val _ = @{assert} (String.isSubstring "HOL.thy" r)
  val _ = @{assert} (String.isSubstring "line=" r)
in writeln "Test 2: OK" end
\<close>

ML \<open>
(* ---- Test 3: live PIDE entity has no position (expected, not absolutized) ---- *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.CONSTANT, 'Query_Test_Const')",
    "entries, _ = await entities_of(connection, [EntityKind.CONSTANT])",
    "for k, p in entries:",
    "    if k == uk:",
    "        return f'pos={p}'",
    "return 'NOT_FOUND'"
  ])
  val _ = writeln ("Test 3 - Query_Test_Const: " ^ r)
  val _ = @{assert} (r = "pos=None")
in writeln "Test 3: OK (live PIDE \<rightarrow> None, as expected)" end
\<close>

ML \<open>
(* ---- Test 4: command_at_position via Python for live PIDE entity ---- *)
(* Absolutize the position on ML side, then pass file+offset to Python *)
let
  val const_space = Proof_Context.consts_of \<^context> |> Consts.space_of
  val pos = #pos (Name_Space.the_entry const_space @{const_name Query_Test_Const})
  val [abs_pos] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos]
  val file = the (Position.file_of abs_pos)
  val offset = the (Position.offset_of abs_pos)

  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_Semantic_Embedding.hover import command_at_position",
    "from Isabelle_RPC_Host.position import IsabellePosition",
    "pos = IsabellePosition(0, " ^ string_of_int offset ^ ", \"" ^ file ^ "\")",
    "result = await command_at_position(pos, connection)",
    "if result is None:",
    "    return 'NONE'",
    "src, start, end_off = result",
    "return src"
  ])
  val _ = writeln ("Test 4 - command source: " ^ r)
  val _ = @{assert} (String.isSubstring "definition" r)
  val _ = @{assert} (String.isSubstring "Query_Test_Const" r)
in writeln "Test 4: OK" end
\<close>

ML \<open>
(* ---- Test 5: theorems enumeration includes positions ---- *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind",
    "entries, _ = await entities_of(connection, [EntityKind.THEOREM])",
    "with_pos = sum(1 for _, p in entries if p is not None)",
    "return f'total={len(entries)} with_pos={with_pos}'"
  ])
  val _ = writeln ("Test 5 - theorems: " ^ r)
  val _ = @{assert} (not (String.isSuffix "with_pos=0" r))
in writeln "Test 5: OK" end
\<close>

ML \<open>
(* ---- Test 6: compiled theorem (refl) position points to HOL.thy ---- *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.THEOREM, 'refl')",
    "entries, _ = await entities_of(connection, [EntityKind.THEOREM])",
    "for k, p in entries:",
    "    if k == uk:",
    "        return f'{p.file}:{p.line}'",
    "return 'NOT_FOUND'"
  ])
  val _ = writeln ("Test 6 - refl: " ^ r)
  val _ = @{assert} (String.isSubstring "HOL.thy" r)
in writeln "Test 6: OK" end
\<close>

ML \<open>
(* ---- Test 7: _get_definition_source for live PIDE entity ---- *)
(* Currently returns None because entry_def_pos returns ("", 0, 0) for
   ID-based positions. The show_defs pipeline would need absolutization. *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_Semantic_Embedding.semantics import _get_definition_source",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.CONSTANT, 'Query_Test_Const')",
    "src = await _get_definition_source(connection, EntityKind.CONSTANT, uk)",
    "return repr(src)"
  ])
  val _ = writeln ("Test 7 - _get_definition_source (live PIDE): " ^ r)
  val _ = @{assert} (r = "None")
in writeln "Test 7: OK (None expected for live PIDE)" end
\<close>

ML \<open>
(* ---- Test 8: position file paths are expanded (no ~~ prefix) ---- *)
let
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.CONSTANT, 'True')",
    "entries, _ = await entities_of(connection, [EntityKind.CONSTANT])",
    "for k, p in entries:",
    "    if k == uk:",
    "        return 'abs' if p.file.startswith('/') else 'unexpanded'",
    "return 'NOT_FOUND'"
  ])
  val _ = writeln ("Test 8 - path expansion: " ^ r)
  val _ = @{assert} (r = "abs")
in writeln "Test 8: OK (paths are absolute)" end
\<close>

ML \<open>
writeln "\n=== All query tool RPC tests passed ==="
\<close>

section \<open>End-to-end query demos\<close>

ML \<open>
(* Demo 1: query_by_name output for a compiled constant *)
let val SOME r = run_py (space_implode "\n" [
    "from Isabelle_Semantic_Embedding.semantics import Semantic_DB, _get_definition_source",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.CONSTANT, 'conj')",
    "sem = Semantic_DB.query(uk, with_pretty=True)",
    "if sem is None:",
    "    sem = '(not interpreted)'",
    "src = await _get_definition_source(connection, EntityKind.CONSTANT, uk)",
    "if src is not None:",
    "    sem += '\\n\\nDefinition:\\n' + src",
    "return sem"
  ])
in writeln ("--- Demo 1: query conj ---\n" ^ r ^ "\n") end
\<close>

ML \<open>
(* Demo 2: query_by_name output for a compiled theorem *)
let val SOME r = run_py (space_implode "\n" [
    "from Isabelle_Semantic_Embedding.semantics import Semantic_DB, _get_definition_source",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "uk = await universal_key_of(connection, EntityKind.THEOREM, 'conjI')",
    "sem = Semantic_DB.query(uk, with_pretty=True)",
    "if sem is None:",
    "    sem = '(not interpreted)'",
    "src = await _get_definition_source(connection, EntityKind.THEOREM, uk)",
    "if src is not None:",
    "    sem += '\\n\\nDefinition:\\n' + src",
    "return sem"
  ])
in writeln ("--- Demo 2: query conjI ---\n" ^ r ^ "\n") end
\<close>

ML \<open>
(* Demo 3: live PIDE entity — manual absolutization to show what show_defs
   would produce once the pipeline handles live PIDE positions *)
let
  val const_space = Proof_Context.consts_of \<^context> |> Consts.space_of
  val pos = #pos (Name_Space.the_entry const_space @{const_name Query_Test_Const})
  val [abs_pos] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos]
  val file = the (Position.file_of abs_pos)
  val offset = the (Position.offset_of abs_pos)
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_Semantic_Embedding.hover import command_at_position",
    "from Isabelle_RPC_Host.position import IsabellePosition",
    "pos = IsabellePosition(0, " ^ string_of_int offset ^ ", \"" ^ file ^ "\")",
    "cmd = await command_at_position(pos, connection)",
    "if cmd is None:",
    "    return '(command_at_position returned None)'",
    "src, _, _ = cmd",
    "return 'constant Query_Test_Const: nat\\n(simulated interpretation)\\n\\nDefinition:\\n' + src"
  ])
in writeln ("--- Demo 3: Query_Test_Const (with absolutized pos) ---\n" ^ r ^ "\n") end
\<close>

ML \<open>
(* Demo 4: live PIDE lemma *)
let
  val fact_space = Proof_Context.facts_of \<^context> |> Facts.space_of
  val pos = #pos (Name_Space.the_entry fact_space "Query_Tool_Test.Query_Test_Lemma")
  val [abs_pos] = PIDE_State.absolutize_id_based_pos {write_to_temp_file = false} [pos]
  val file = the (Position.file_of abs_pos)
  val offset = the (Position.offset_of abs_pos)
  val SOME r = run_py (space_implode "\n" [
    "from Isabelle_Semantic_Embedding.hover import command_at_position",
    "from Isabelle_RPC_Host.position import IsabellePosition",
    "pos = IsabellePosition(0, " ^ string_of_int offset ^ ", \"" ^ file ^ "\")",
    "cmd = await command_at_position(pos, connection)",
    "if cmd is None:",
    "    return '(command_at_position returned None)'",
    "src, _, _ = cmd",
    "return 'lemma Query_Test_Lemma: Query_Test_Const = 42\\n(simulated interpretation)\\n\\nDefinition:\\n' + src"
  ])
in writeln ("--- Demo 4: Query_Test_Lemma (with absolutized pos) ---\n" ^ r ^ "\n") end
\<close>

ML \<open>
(* Demo 5: position info for a compiled constant *)
let val SOME r = run_py (space_implode "\n" [
    "from Isabelle_RPC_Host.context import entities_of",
    "from Isabelle_RPC_Host.universal_key import EntityKind, universal_key_of",
    "for name in ['conj', 'disj', 'implies', 'Not']:",
    "    uk = await universal_key_of(connection, EntityKind.CONSTANT, name)",
    "    entries, _ = await entities_of(connection, [EntityKind.CONSTANT])",
    "    for k, p in entries:",
    "        if k == uk:",
    "            break",
    "    else:",
    "        p = None",
    "    if p:",
    "        print(f'  {name}: {p.file}:{p.line}')",
    "    else:",
    "        print(f'  {name}: no position')",
    "return 'done'"
  ])
in writeln ("--- Demo 5: position lookup for HOL constants ---\n" ^ r ^ "\n") end
\<close>

end
