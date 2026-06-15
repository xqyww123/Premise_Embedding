(*  Title:      Tools/pide_state.ML
    Author:     Qiyuan Xu

Access PIDE document state via Scala functions:
  - save_thy_files: collect theory sources from PIDE state, write to backup files (path~)
  - absolutize_id_based_pos: resolve ID-based positions to file/line/offset positions
  - goto_definition: resolve entity reference at a position to its definition position
  - hover_message: extract tooltip information at a position
  - entity_at_position: return (kind, fully_qualified_name) of entity at a position
  - command_at_position: return source code of the command at a position
*)

signature PIDE_STATE =
sig
  val save_thy_files: unit -> string list

  (* Convert ID-based positions (containing only a command id and a relative offset)
     to absolute file-based positions (with file path, line number, and absolute offset).
     Positions that already have file/line information are returned unchanged.
     When write_to_temp_file is true, the PIDE source is saved to a backup file (path~)
     and the returned positions use the backup file path. *)
  val absolutize_id_based_pos: {write_to_temp_file: bool} -> Position.T list -> Position.T list

  (* Given a position (file + offset), return the definition position of the entity
     at that location. Tries live PIDE state first, falls back to session export DB. *)
  val goto_definition: Position.T -> Position.T option

  (* Given a position (file + offset), return the hover tooltip text.
     Tries live PIDE state first, falls back to session export DB. *)
  val hover_message: Position.T -> string option

  (* Given a position (file + offset), return the (kind, name) of the entity
     at that location, where name is the fully qualified name.
     Tries live PIDE state first, falls back to session export DB. *)
  val entity_at_position: Position.T -> (string * string) option

  (* Given a position (file + offset), return the source code of the command
     at that location together with its span (start_offset, end_offset) in the file.
     Tries live PIDE state first, falls back to session export DB. *)
  val command_at_position: Position.T -> (string * int * int) option

  (* Cut the command spans of a (possibly WIP) theory file locally with
     Outer_Syntax, returning (source, start_sym_off, end_sym_off) for each real
     command span.  Offsets are 1-based file-global symbol offsets.  This backs
     the WIP fallback of command_at_position for files not covered by live PIDE
     state or a session export DB. *)
  val command_spans_of_file: string -> (string * int * int) list
  val command_at_position_wip: string -> int -> (string * int * int) option

  (* Convert between symbol offset and line:column via Python RPC.
     Requires the Isabelle_RPC Python host to be running. *)
  val offset_to_line_column: string -> int -> int * int
  val line_column_to_offset: string -> int -> int -> int

  (* Return all ancestor sessions (whose heaps are loaded) with their export database paths.
     Each entry is (session_name, db_path). *)
  val get_session_databases: unit -> (string * string) list

  (* Given a file path and symbol offset, return the PIDE command ID of the
     command at that position.  Returns NONE if not in a live PIDE node.
     Only works for live theories, not heap-image theories. *)
  val command_id_at_position: string -> int -> int option

  (* Given a file path and symbol offset, attempt to extract the Toplevel.state
     of the command at that position via Document.command_exec.
     Returns NONE for heap-image theories or if the command hasn't been evaluated. *)
  val toplevel_state_at_position: string -> int -> Toplevel.state option

  type context_at_position_cfg = {re_eval_cache: bool}

  (* Given a file path and symbol offset, attempt to extract the Context.generic
     at that position.
     Fallback chain: live PIDE -> (if re_eval_cache) re-evaluation cache -> NONE *)
  val context_at_position: context_at_position_cfg -> string -> int -> Context.generic option

  (* Find the loaded theory whose master .thy file matches the given path. *)
  val theory_of_file: string -> string option

  (* Msgpack unpacker that reads an optional (file, offset) pair from the wire.
     NONE -> returns the fallback context.
     SOME (file, offset) -> calls context_at_position, falls back on failure. *)
  val position_context_unpacker: Context.generic ->
    Context.generic MessagePackBinIO.Unpack.unpacker
end;

structure PIDE_State: PIDE_STATE =
struct

fun save_thy_files () =
  let
    val yxml = Scala.function1 "pide_state.save_thy_files" "";
  in
    YXML.parse_body yxml |> XML.Decode.list XML.Decode.string
  end;

fun absolutize_id_based_pos {write_to_temp_file} positions =
  let
    val _ = if write_to_temp_file then (save_thy_files (); ()) else ();

    fun needs_resolution pos =
      is_some (Position.parse_id pos) andalso
      (is_none (Position.file_of pos) orelse is_none (Position.line_of pos));

    val to_resolve =
      positions |> map_filter (fn pos =>
        if needs_resolution pos then
          SOME (the (Position.parse_id pos), the_default 1 (Position.offset_of pos))
        else NONE);

    val resolved =
      if null to_resolve then []
      else
        let
          val input_body =
            XML.Encode.list (XML.Encode.pair XML.Encode.int XML.Encode.int) to_resolve;
          val yxml = Scala.function1 "pide_state.resolve_positions"
            (YXML.string_of_body input_body);
        in
          YXML.parse_body yxml
          |> XML.Decode.list (XML.Decode.pair XML.Decode.string
               (XML.Decode.pair XML.Decode.int XML.Decode.int))
        end;

    val results = Unsynchronized.ref resolved;
    fun pop () = case ! results of
        [] => ("", (0, 0))
      | x :: xs => (results := xs; x);
  in
    positions |> map (fn pos =>
      if needs_resolution pos then
        let
          val (file, (line, preceding_symbols)) = pop ()
          val file' = if write_to_temp_file then file ^ "~" else file
        in if file = "" then pos
           else Position.make0 line
             (case Position.offset_of pos of SOME off => preceding_symbols + off | NONE => 0)
             (case Position.end_offset_of pos of SOME off => preceding_symbols + off | NONE => 0)
             (the_default "" (Position.label_of pos))
             file'
             (the_default "" (Position.id_of pos)) end
      else pos)
  end;

(* Expand Isabelle path variables and make absolute.
   Fast path: skip if already an absolute path without variables. *)
fun expand_path s =
  if s = "" then ""
  else if String.isPrefix "/" s andalso not (Char.contains s #"$")
    andalso not (String.isSubstring "./" s)
  then s
  else File.absolute_path (Path.explode s) |> Path.implode;

(* Extract file path and offset from a Position.T, absolutizing if needed *)
fun pos_file_offset pos =
  let
    val pos' = hd (absolutize_id_based_pos {write_to_temp_file = false} [pos]);
    val file = the_default "" (Position.file_of pos') |> expand_path;
    val offset = the_default 0 (Position.offset_of pos');
  in (file, offset) end;

(* PIDE live queries via Scala *)

fun goto_definition_pide file_path symbol_offset =
  let
    val input_body =
      XML.Encode.pair XML.Encode.string XML.Encode.int
        (file_path, symbol_offset);
    val yxml = Scala.function1 "pide_state.goto_definition"
      (YXML.string_of_body input_body);
    val (def_file, (def_line, (def_offset, def_end_offset))) =
      YXML.parse_body yxml
      |> XML.Decode.pair XML.Decode.string
           (XML.Decode.pair XML.Decode.int
             (XML.Decode.pair XML.Decode.int XML.Decode.int));
  in
    if def_file = "" then NONE
    else SOME (Position.make0 def_line def_offset def_end_offset "" (expand_path def_file) "")
  end;

fun hover_message_pide file_path symbol_offset =
  let
    val input_body =
      XML.Encode.pair XML.Encode.string XML.Encode.int
        (file_path, symbol_offset);
    val yxml = Scala.function1 "pide_state.hover_message"
      (YXML.string_of_body input_body);
    val msg = if YXML.detect yxml
      then (warning "hover_message: unexpected YXML markup in result";
            XML.content_of (YXML.parse_body yxml))
      else yxml;
  in
    if msg = "" then NONE
    else SOME msg
  end;

(* Position conversion via Python RPC *)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val offset_to_line_column_cmd :
    (string * int, (int * int)) Remote_Procedure_Calling.command = {
  name = "position.offset_to_line_column",
  arg_schema = packPair (packString, packInt),
  ret_schema = unpackPair (unpackInt, unpackInt),
  callback = [],
  timeout = NONE
}

val line_column_to_offset_cmd :
    (string * int * int, int) Remote_Procedure_Calling.command = {
  name = "position.line_column_to_offset",
  arg_schema = packTuple3 (packString, packInt, packInt),
  ret_schema = unpackInt,
  callback = [],
  timeout = NONE
}

end (*local MessagePack*)

fun offset_to_line_column file_path offset =
  Remote_Procedure_Calling.call_command offset_to_line_column_cmd (expand_path file_path, offset);

fun line_column_to_offset file_path line column =
  Remote_Procedure_Calling.call_command line_column_to_offset_cmd (expand_path file_path, line, column);

(* Goto definition and hover message: Scala handles both live PIDE + DB fallback *)

fun goto_definition pos =
  let
    val (file, offset) = pos_file_offset pos
    val _ = warning ("goto_definition: file=" ^ file ^ " offset=" ^ string_of_int offset)
  in
    if file = "" orelse offset = 0 then (warning "goto_definition: empty file or zero offset"; NONE)
    else
      let val _ = warning "goto_definition: calling Scala..."
          val result = goto_definition_pide file offset
          val _ = warning ("goto_definition: Scala returned " ^
            (case result of NONE => "NONE" | SOME _ => "SOME(...)"))
      in result end
  end;

fun hover_message pos =
  let val (file, offset) = pos_file_offset pos
  in
    if file = "" orelse offset = 0 then NONE
    else hover_message_pide file offset
  end;

fun entity_at_position_pide file_path symbol_offset =
  let
    val input_body =
      XML.Encode.pair XML.Encode.string XML.Encode.int
        (file_path, symbol_offset);
    val yxml = Scala.function1 "pide_state.entity_at_position"
      (YXML.string_of_body input_body);
    val (kind, name) =
      YXML.parse_body yxml
      |> XML.Decode.pair XML.Decode.string XML.Decode.string;
  in
    if kind = "" andalso name = "" then NONE
    else SOME (kind, name)
  end;

fun entity_at_position pos =
  let val (file, offset) = pos_file_offset pos
  in
    if file = "" orelse offset = 0 then NONE
    else entity_at_position_pide file offset
  end;

fun command_at_position_pide file_path symbol_offset =
  let
    val input_body =
      XML.Encode.pair XML.Encode.string XML.Encode.int
        (file_path, symbol_offset);
    val yxml = Scala.function1 "pide_state.command_at_position"
      (YXML.string_of_body input_body);
    val (source, (start_offset, end_offset)) =
      YXML.parse_body yxml
      |> XML.Decode.pair XML.Decode.string
           (XML.Decode.pair XML.Decode.int XML.Decode.int);
  in
    if source = "" then NONE
    else SOME (source, start_offset, end_offset)
  end;

(* command_at_position is defined further below, after theory_of_file, so that
   it can fall back to the WIP (local Outer_Syntax) command-span path. *)

val get_session_databases : unit -> (string * string) list =
  let
    val cache = Synchronized.cache (fn () =>
      Scala.function1 "pide_state.get_session_databases" ""
      |> YXML.parse_body
      |> XML.Decode.list (XML.Decode.pair XML.Decode.string XML.Decode.string));
  in fn () => Synchronized.cache_eval {persistent = false} cache end;

fun command_id_at_position file_path symbol_offset =
  let
    val input_body =
      XML.Encode.pair XML.Encode.string XML.Encode.int
        (file_path, symbol_offset);
    val yxml = Scala.function1 "pide_state.command_id_at_position"
      (YXML.string_of_body input_body);
    val cmd_id = YXML.parse_body yxml |> XML.Decode.int;
  in if cmd_id = 0 then NONE else SOME cmd_id end;

fun toplevel_state_at_position file_path symbol_offset =
  case command_id_at_position file_path symbol_offset of
    NONE => NONE
  | SOME cmd_id =>
      \<^try>\<open>
        case Document.command_exec (Document.state ()) file_path cmd_id of
          NONE => NONE
        | SOME (eval, _) => SOME (Command.eval_result_state eval)
        catch _ => NONE
      \<close>;


(* --- Re-evaluation cache for finished theories --- *)

(* Cache: file_path -> sorted list of (symbol_offset, Toplevel.state).
   Populated on first access by re-evaluating the theory from its parents. *)
val reeval_cache : (int * Toplevel.state) list Symtab.table Synchronized.var =
  Synchronized.var "PIDE_State.reeval_cache" Symtab.empty;

fun lookup_reeval_cache file_path symbol_offset =
  case Symtab.lookup (Synchronized.value reeval_cache) file_path of
    NONE => NONE
  | SOME entries =>
      let fun find [] = NONE
            | find [(off, st)] = if symbol_offset >= off then SOME st else NONE
            | find ((off1, st1) :: (rest as (off2, _) :: _)) =
                if symbol_offset >= off1 andalso symbol_offset < off2 then SOME st1
                else find rest
      in find entries end;

(* Find the loaded theory whose master file matches the given path *)
fun theory_of_file file_path =
  let
    val target = Path.explode file_path
    fun matches name =
      \<^try>\<open>
        let val thy = Thy_Info.get_theory name
            val dir = Resources.master_directory thy
            val thy_file = Path.append dir (Path.basic (Long_Name.base_name name ^ ".thy"))
        in thy_file = target end
        catch _ => false
      \<close>
  in List.find matches (Thy_Info.get_names ()) end;


(* --- WIP command-span path (B1 cross-line completion) ---

   For a "work in progress" .thy file that is NOT covered by a live PIDE node
   nor by a session export DB, Scala's command_at_position returns NONE and the
   Python side degrades to single-line spans.  Here we re-cut the command spans
   locally with Outer_Syntax, so command_at_position can return a full,
   possibly multi-line, command span.

   Coordinate contract: the returned (start, end) are 1-based, file-global
   *symbol* offsets, matching the live branch (src/scala/pide_state.scala:104
   uses var pos = 1).  We obtain this by lexing with Position.file file_path,
   whose offset starts at 1 (Pure/General/position.ML:163). *)

(* Keywords for lexing the WIP file:
     1. preferred: the real theory matched by theory_of_file, via
        Thy_Header.get_keywords (Thy_Info.get_theory name);
     2. fallback: read the file header, merge the keywords of its (loaded)
        import theories with the header's own keyword declarations
        (cf. Pure/Thy/thy_info.ML:297-299). *)
fun keywords_of_wip_file file_path text =
  case theory_of_file file_path of
    SOME thy_name => Thy_Header.get_keywords (Thy_Info.get_theory thy_name)
  | NONE =>
      let
        val text_pos = Position.file file_path
        val header = Thy_Header.read text_pos text
        (* resolve each import to a loaded theory by base-name match *)
        val loaded = Thy_Info.get_names ()
        fun resolve_import (imp, _) =
          let val base = Long_Name.base_name imp in
            case Thy_Info.lookup_theory imp of
              SOME thy => SOME thy
            | NONE =>
                (* base-name fallback: only when it is unambiguous.  If several
                   loaded theories share the base name we cannot tell which one
                   is meant, so we skip (warning) rather than silently pick one
                   and risk merging the wrong keywords. *)
                (case filter (fn n => Long_Name.base_name n = base) loaded of
                   [n] => Thy_Info.lookup_theory n
                 | [] => NONE
                 | ns =>
                     (warning ("PIDE_State.keywords_of_wip_file: ambiguous import " ^
                        quote imp ^ " matches multiple loaded theories " ^
                        commas_quote ns ^ "; skipping"); NONE))
          end
        val parents = map_filter resolve_import (#imports header)
        val base_kws =
          fold (curry Keyword.merge_keywords o Thy_Header.get_keywords) parents
            Keyword.empty_keywords
      in
        Keyword.add_keywords (#keywords header) base_kws
      end;

(* Cut the command spans of a WIP file.
   Returns (source, start_sym_off, end_sym_off) for each real Command_Span,
   where start/end are 1-based file-global symbol offsets.
   Malformed_Span and Ignored_Span are dropped: a half-typed WIP file produces
   Malformed_Span frequently, and matching by kind avoids mis-hitting them.
   NOTE: this differs from the live PIDE path, which builds a Command for every
   span (Ignored ones too).  An offset landing in inter-command whitespace thus
   resolves to that Ignored command under live, but yields NONE here (the caller
   then degrades to a single-line search).  Harmless in practice: query offsets
   land inside commands and hover/definition never probe whitespace. *)
fun command_spans_of_file file_path : (string * int * int) list =
  \<^try>\<open>
    let
      val text = File.read (Path.explode file_path)
      val tpos = Position.file file_path
      val kws = keywords_of_wip_file file_path text
      val toks = Token.explode kws tpos text
      val spans = Outer_Syntax.parse_spans toks
      fun is_command span =
        case Command_Span.kind span of Command_Span.Command_Span _ => true | _ => false
      fun span_entry span =
        let
          val span_toks = Command_Span.content span
          val (p1, p2) = Token.range_of span_toks
          val start_off = the_default 0 (Position.offset_of p1)
          (* end uses the span's own token range (Command_Span.range); the
             Python consumer currently keys off start only, so trailing
             whitespace from Command_Span.command_ranges is not needed here. *)
          val end_off = the_default 0 (Position.offset_of p2)
          (* reconstruct the verbatim source: source_of yields each token's
             original text (symbols intact), and whitespace/comments are
             themselves tokens in the span content, so concatenation restores
             the original slice faithfully *)
          val source =
            implode (map Token.source_of span_toks)
        in
          if start_off > 0 then SOME (source, start_off, end_off) else NONE
        end
    in
      map_filter (fn span => if is_command span then span_entry span else NONE) spans
    end
    catch exn =>
      (warning ("PIDE_State.command_spans_of_file: " ^ Runtime.exn_message exn); [])
  \<close>;

fun command_at_position_wip file off =
  List.find (fn (_, s, e) => off >= s andalso off < e) (command_spans_of_file file);

fun command_at_position pos =
  let val (file, offset) = pos_file_offset pos
  in
    if file = "" orelse offset = 0 then NONE
    else
      case command_at_position_pide file offset of
        SOME r => SOME r
      | NONE => command_at_position_wip file offset
  end;


fun build_reeval_cache file_path =
  case theory_of_file file_path of
    NONE =>
      (warning ("PIDE_State.build_reeval_cache: no loaded theory for " ^ file_path); [])
  | SOME thy_name =>
      \<^try>\<open>
        let
          val thy = Thy_Info.get_theory thy_name
          val path = Path.explode file_path
          val text = File.read path
          val text_pos = Position.file file_path
          val header = Thy_Header.read text_pos text
          val parents = Theory.parents_of thy
          val master_dir = Resources.master_directory thy
          val fresh_thy = Resources.begin_theory master_dir header parents
          val trs = Outer_Syntax.parse_text fresh_thy (K fresh_thy) text_pos text

          val init_state = Toplevel.make_state NONE
          val (_, entries_rev) = fold (fn tr => fn (st, acc) =>
            let val st' = Toplevel.command_exception true tr st
                val offset = the_default 0 (Position.offset_of (Toplevel.pos_of tr))
            in if offset > 0 andalso not (Toplevel.is_ignored tr)
               then (st', (offset, st') :: acc)
               else (st', acc)
            end) trs (init_state, [])
          val entries = rev entries_rev
          val _ = writeln ("PIDE_State: cached " ^ string_of_int (length entries) ^
            " states for " ^ thy_name)
        in
          Synchronized.change reeval_cache (Symtab.update (file_path, entries));
          entries
        end
        catch exn => (warning ("PIDE_State.build_reeval_cache: " ^ Runtime.exn_message exn); [])
      \<close>;

fun reeval_context_at_position file_path symbol_offset =
  let
    val entries = case Symtab.lookup (Synchronized.value reeval_cache) file_path of
        SOME e => e
      | NONE => build_reeval_cache file_path
  in
    case entries of [] => NONE | _ =>
      let fun find [] = NONE
            | find [(off, st)] = if symbol_offset >= off then SOME st else NONE
            | find ((off1, st1) :: (rest as (off2, _) :: _)) =
                if symbol_offset >= off1 andalso symbol_offset < off2 then SOME st1
                else find rest
      in
        case find entries of
          NONE => NONE
        | SOME st =>
            \<^try>\<open>SOME (Context.Proof (Toplevel.context_of st)) catch _ => NONE\<close>
      end
  end;


type context_at_position_cfg = {re_eval_cache: bool};

fun context_at_position (cfg : context_at_position_cfg) file_path symbol_offset =
  case toplevel_state_at_position file_path symbol_offset of
    SOME st =>
      \<^try>\<open>SOME (Context.Proof (Toplevel.context_of st)) catch _ => NONE\<close>
  | NONE =>
      if #re_eval_cache cfg
      then reeval_context_at_position file_path symbol_offset
      else NONE;

fun position_context_unpacker (fallback : Context.generic)
    : Context.generic MessagePackBinIO.Unpack.unpacker =
  let open MessagePackBinIO.Unpack
  in fn src =>
    let val (pos_opt, src') =
      unpackOption (unpackPair (unpackString, unpackInt)) src
    in case pos_opt of
        NONE => (fallback, src')
      | SOME (file, offset) =>
          (case context_at_position {re_eval_cache = true} file offset of
            SOME ctx => (ctx, src')
          | NONE => (fallback, src'))
    end
  end;

end;

(* Register RPC callbacks so Python can call the auto functions *)
local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val _ = Theory.setup (Context.theory_map (
  Remote_Procedure_Calling.register_global_callback {
      name = "pide_state.goto_definition",
      arg_schema = unpackPair (unpackString, unpackInt),
      ret_schema = packOption (packTuple4 (packString, packInt, packInt, packInt)),
      function = fn (file, offset) =>
        let val pos = Position.make0 0 offset 0 "" file ""
        in PIDE_State.goto_definition pos
           |> Option.map (fn p =>
                (the_default "" (Position.file_of p),
                 the_default 0 (Position.line_of p),
                 the_default 0 (Position.offset_of p),
                 the_default 0 (Position.end_offset_of p)))
        end,
      timeout = SOME (Time.fromSeconds 10)
    }
  #> Remote_Procedure_Calling.register_global_callback {
      name = "pide_state.hover_message",
      arg_schema = unpackPair (unpackString, unpackInt),
      ret_schema = packOption packString,
      function = fn (file, offset) =>
        PIDE_State.hover_message (Position.make0 0 offset 0 "" file ""),
      timeout = SOME (Time.fromSeconds 10)
    }
  #> Remote_Procedure_Calling.register_global_callback {
      name = "pide_state.entity_at_position",
      arg_schema = unpackPair (unpackString, unpackInt),
      ret_schema = packOption (packPair (packString, packString)),
      function = fn (file, offset) =>
        PIDE_State.entity_at_position (Position.make0 0 offset 0 "" file ""),
      timeout = SOME (Time.fromSeconds 10)
    }
  #> Remote_Procedure_Calling.register_global_callback {
      name = "pide_state.command_at_position",
      arg_schema = unpackPair (unpackString, unpackInt),
      ret_schema = packOption (packTuple3 (packString, packInt, packInt)),
      function = fn (file, offset) =>
        PIDE_State.command_at_position (Position.make0 0 offset 0 "" file ""),
      timeout = SOME (Time.fromSeconds 10)
    }
  #> Remote_Procedure_Calling.register_global_callback {
      name = "pide_state.get_session_databases",
      arg_schema = unpackUnit,
      ret_schema = packList (packPair (packString, packString)),
      function = fn () => PIDE_State.get_session_databases (),
      timeout = NONE
    }
))

end
