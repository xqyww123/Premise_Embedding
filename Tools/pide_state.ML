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

  (* Re-cut a degenerate whole-file command_at_position result (source, start s)
     into the single command containing the file-global symbol `offset`; on a
     miss degrades to command_at_position_wip / NONE, never the whole `source`.
     Exposed for testing the whole-theory-leak fix. *)
  val recut_dump_command: string -> string -> int -> int -> (string * int * int) option

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
(* NORMALISE FIRST.  `s` reaches here from outside ML -- Position.file_of, or a path
   Python sent back after receiving it from Theory_Structure.get_theory_file_path, which
   is deliberately a PLATFORM path because Python does native file I/O.  On Windows that
   is `C:\...`, which fails the `isPrefix "/"` test, falls through to Path.explode, and
   raises: Path.check_elem rejects `:` and `\`.  Four SCALA callbacks reach this --
   hover_message, goto_definition, entity_at_position, command_at_position -- so on
   Windows they threw rather than degraded.  Those four want the result in Isabelle's
   INTERNAL form, which is what this returns: pide_state.scala matches it against
   `version.nodes … _.node` and feeds it to `Path.explode`.

   The two RPC consumers are NOT these: offset_to_line_column and line_column_to_offset
   send the string over MessagePack to position.py, which does `os.path.realpath` and
   `open` in NATIVE Python.  Internal form is exactly wrong there, so those two go through
   rpc_path, which composes this function with a conversion back to platform form.

   ML_System.standard_path is the sanctioned normalisation for a string from outside
   Isabelle's path world (Isabelle uses it that way at Pure/General/file.ML:76 for the OS
   cwd, and Pure/ML/exn_properties.ML:27 for PolyML's messages).  It is shape-agnostic --
   C:\x, C:/x and /cygdrive/c/x all normalise, `~~/...` passes through untouched -- and
   on non-Windows it is `fn path => path`, so every line below behaves exactly as before
   on Linux and macOS.

   Note the fast path is applied to the NORMALISED string, which is the point: a Windows
   platform path becomes /cygdrive/c/... and then takes the fast path, as it should. *)
fun expand_path s =
  if s = "" then ""
  else
    let val s = ML_System.standard_path s in
      if String.isPrefix "/" s andalso not (Char.contains s #"$")
        andalso not (String.isSubstring "./" s)
      then s
      else File.absolute_path (Path.explode s) |> Path.implode
    end;

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

(* PLATFORM path, not expand_path's internal form.  The receiver is position.py, which
   does os.path.realpath + open in native Python, so on Windows expand_path's
   /cygdrive/c/... would be a guaranteed FileNotFoundError.  Composing over expand_path
   keeps the variable-expansion and absolutisation it does, then converts at the boundary.

   ML_System.platform_path, NOT File.platform_path.  The latter takes a Path.T, which
   would mean exploding expand_path's result -- but expand_path's fast path deliberately
   returns the string WITHOUT exploding it, and Path.check_elem rejects ' ? * < > | " ,
   every one of which is a legal POSIX filename character.  Routing through Path.T would
   therefore turn working Linux calls into hard errors (neither caller has a \<^try>).
   ML_System.platform_path is string -> string and is `fn path => path` off Windows, so
   this really is a no-op there. *)
fun rpc_path file_path = ML_System.platform_path (expand_path file_path);

fun offset_to_line_column file_path offset =
  Remote_Procedure_Calling.call_command offset_to_line_column_cmd (rpc_path file_path, offset);

fun line_column_to_offset file_path line column =
  Remote_Procedure_Calling.call_command line_column_to_offset_cmd (rpc_path file_path, line, column);

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

(* The FILE PROPERTY Position.file sets (position.ML:148 -- `label` is a separate field,
   read elsewhere in this file via Position.label_of).

   Isabelle has TWO conventions for it, and this picks one deliberately.  The loader path
   uses the symbolic form (`~~/src/...`): Build/resources.ML:315,334, ML/ml_context.ML:216
   and Thy/thy_info.ML:368 all spell it `Position.file (File.symbolic_path path)`.  The
   PIDE path instead labels with the document node name: PIDE/document.ML:773,
   PIDE/command.ML:64, Build/resources.ML:340.  Since these positions are produced by
   re-lexing a file we just read from disk -- a loader-shaped act, not a document-node one
   -- the symbolic form is the closer fit, and it is what makes messages readable
   (File.symbolic_path folds prefixes listed in ISABELLE_DIRECTORIES, file.ML:51).
   Either way this is cosmetic: the offsets are independent of the name, verified.

   FALLS BACK to the raw string, deliberately.  This value is display text -- it reaches
   error messages and PIDE markup, and callers here only ever read Position.offset_of
   back out -- while Path.explode rejects ' ? * < > | " (path.ML:71), every one of them
   legal in a POSIX filename.  A strict version would turn a cosmetic improvement into a
   hard failure on Linux for such a file.  So: nicer name when it is derivable, the raw
   string when it is not.  (Not *exactly* the previous behaviour in every case -- where
   explode succeeds on a degenerate input the string is rewritten, e.g. "" becomes "."
   and "./Foo.thy" becomes "Foo.thy".  None of these reach here: File.read rejects them
   first.) *)
fun file_label file_path =
  the_default file_path
    (try (File.symbolic_path o Path.explode o ML_System.standard_path) file_path);

(* Cache: file_path -> sorted list of (symbol_offset, Toplevel.state).
   Populated on first access by re-evaluating the theory from its parents.

   KEYED ON THE NORMALISED PATH.  The same file arrives from the wire spelled either way
   on Windows (`C:\...` as Python sends it, `/cygdrive/c/...` after normalisation), and a
   raw key would file those as two distinct theories -- not a correctness bug, but
   build_reeval_cache replays an entire theory, so a spurious miss costs minutes.
   Identity on Linux and macOS, where the two spellings coincide. *)
fun cache_key file_path = ML_System.standard_path file_path;
val reeval_cache : (int * Toplevel.state) list Symtab.table Synchronized.var =
  Synchronized.var "PIDE_State.reeval_cache" Symtab.empty;

fun lookup_reeval_cache file_path symbol_offset =
  case Symtab.lookup (Synchronized.value reeval_cache) (cache_key file_path) of
    NONE => NONE
  | SOME entries =>
      let fun find [] = NONE
            | find [(off, st)] = if symbol_offset >= off then SOME st else NONE
            | find ((off1, st1) :: (rest as (off2, _) :: _)) =
                if symbol_offset >= off1 andalso symbol_offset < off2 then SOME st1
                else find rest
      in find entries end;

(* Find the loaded theory whose master file matches the given path.

   `file_path` arrives from OUTSIDE ML: Python sends it back as the `context_at` hint,
   having received it from us as Theory_Structure.get_theory_file_path, which is
   deliberately a PLATFORM path because Python does native file I/O.  On Windows that is
   `C:\...`, and Path.explode rejects `:` and `\` (Path.check_elem).  On Linux and macOS
   ML_System.standard_path is the identity, so the round trip was invisible.

   SCOPE, precisely: this fixes the callers that reach theory_of_file -- the
   position_context_unpacker / context_at_position route, and build_reeval_cache, which
   calls theory_of_file OUTSIDE its own \<^try> and so genuinely crashed.  It does NOT
   cover hover_message, goto_definition, entity_at_position or command_at_position:
   those reach the same platform string through `expand_path`, which was fixed separately
   (see its own comment above).  Do not read this comment as covering them.

   Why ML_System.standard_path is the right tool for a string from outside Isabelle's
   path world, and why it is a no-op on Linux and macOS: see expand_path's comment above.
   The reasoning is identical and is not repeated here. *)
(* NONE, not an exception, when the path will not explode.  Path.explode rejects
   ' * ? < > | " and a mid-element $ (path.ML:71) -- all legal in a POSIX filename, and
   Isabelle itself opens such files happily (verified: a theory under a directory named
   Don't_Touch loads clean).  standard_path only removes the WINDOWS-specific offenders
   `:` and `\`; it does nothing about these, on any platform.

   This used to raise, and build_reeval_cache calls it OUTSIDE its own \<^try> (below), so
   the error escaped to the RPC boundary as an unexplained "Illegal character" for anyone
   whose project sat under such a directory.  The return type was already string option
   and every caller already handles NONE, so NONE is both the honest and the cheap
   answer. *)
fun theory_of_file file_path =
  case try (Path.explode o ML_System.standard_path) file_path of
    NONE => NONE
  | SOME target =>
  let
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
   uses var pos = 1).  We obtain this by lexing with Position.file, whose offset starts
   at 1 (Pure/General/position.ML:163) -- independently of the name it is given, so
   file_label's fallback cannot affect the coordinate contract. *)

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
        val text_pos = Position.file (file_label file_path)
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
(* Cut a theory source TEXT into command spans (source, start_sym_off,
   end_sym_off) via local Outer_Syntax.  Pure function of (file_path, text):
   file_path only seeds the lexing Position and the keyword resolution. *)
fun command_spans_of_text file_path text : (string * int * int) list =
  \<^try>\<open>
    let
      val tpos = Position.file (file_label file_path)
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
      (warning ("PIDE_State.command_spans_of_text: " ^ Runtime.exn_message exn); [])
  \<close>;

fun command_spans_of_file file_path : (string * int * int) list =
  (* standard_path: file_path arrives from the wire and is a PLATFORM path on Windows.
     Without it Path.explode raises and the \<^try> below turns that into [] plus a
     warning naming this function -- so command spans degrade to nothing there, with the
     cause visible only to whoever reads the warnings. *)
  \<^try>\<open>command_spans_of_text file_path
      (File.read (Path.explode (ML_System.standard_path file_path)))
    catch exn =>
      (warning ("PIDE_State.command_spans_of_file: " ^ Runtime.exn_message exn); [])\<close>;

fun command_at_position_wip file off =
  List.find (fn (_, s, e) => off >= s andalso off < e) (command_spans_of_file file);

(* Re-cut a degenerate whole-file `command_at_position` result.

   A degenerate DB snapshot dumps the WHOLE theory file as one command's source
   (Build.read_theory rebuilds the theory as a single Command.unparsed; a
   whole-file COMMAND_SPAN then makes try_substring return everything).  Such a
   dump's span starts at the file head (s <= 1).  Given that dump as `source`
   with its start `s`, re-cut the ALREADY-RETURNED source — NOT the file on
   disk, so no extra read and no risk of a stale/edited buffer mismatch — and
   return the single command containing `offset`.

   If the re-cut locates NO span at the offset (the offset fell in inter-command
   whitespace, or `command_spans_of_text` failed to parse and returned []), we
   must NOT fall back to the raw whole-file `source`: returning it leaks the
   entire theory to the caller.  Degrade to the WIP path (re-cut from the clean
   on-disk source); if that also finds nothing it returns NONE, and the caller
   degrades to a single-line search.

   Exposed for testing: it is the only branch the whole-theory-leak fix changed
   (the leak was `... | NONE => SOME (source, s, e)` returning the whole file),
   yet that branch cannot be driven from a live theory (it needs an s <= 1
   degenerate snapshot), so it is unit-tested directly with a synthetic dump. *)
fun recut_dump_command file source s offset =
  let
    val spans = command_spans_of_text file source
    val local_off = offset - s + 1  (* source's symbol 1 = file symbol s *)
  in
    case List.find (fn (_, ss, ee) => local_off >= ss andalso local_off < ee) spans of
      SOME (src, ss, ee) => SOME (src, s + ss - 1, s + ee - 1)
    | NONE => command_at_position_wip file offset
  end;

fun command_at_position pos =
  let val (file, offset) = pos_file_offset pos
  in
    if file = "" orelse offset = 0 then NONE
    else
      case command_at_position_pide file offset of
        (* Every real entity command sits after `theory .. begin`, so s > 1 and
           is trusted directly with zero work.  An s <= 1 result is (or may be) a
           whole-file degenerate dump; re-cut it to the single command. *)
        SOME (r as (source, s, _)) =>
          if s > 1 then SOME r
          else recut_dump_command file source s offset
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
          (* standard_path for the same reason as command_spans_of_file above: a raw
             platform path raises here, and this function's \<^try> turns that into []
             plus a warning -- the re-eval cache just stops working on Windows. *)
          val path = Path.explode (ML_System.standard_path file_path)
          val text = File.read path
          (* No try around symbolic_path here -- but NOT because holding a Path.T makes it
             total.  It is not: symbolic_path runs Path.expand, which calls getenv_strict
             and errors on an unset variable (path.ML:228-236), so a filename containing
             `$` would raise.  It is safe only because File.read on the line above runs
             the same Path.expand and raises first.  Reordering these two lines would make
             that live; file_label exists for the sites that have no such predecessor. *)
          val text_pos = Position.file (File.symbolic_path path)
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
          Synchronized.change reeval_cache (Symtab.update (cache_key file_path, entries));
          entries
        end
        catch exn => (warning ("PIDE_State.build_reeval_cache: " ^ Runtime.exn_message exn); [])
      \<close>;

fun reeval_context_at_position file_path symbol_offset =
  let
    val entries = case Symtab.lookup (Synchronized.value reeval_cache) (cache_key file_path) of
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
