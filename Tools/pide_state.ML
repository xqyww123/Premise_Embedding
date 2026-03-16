(*  Title:      Tools/pide_state.ML
    Author:     Qiyuan Xu

Access PIDE document state via Scala functions:
  - save_thy_files: collect theory sources from PIDE state, write to backup files (path~)
  - absolutize_id_based_pos: resolve ID-based positions to file/line/offset positions
  - goto_definition: resolve entity reference at a position to its definition position
  - hover_message: extract tooltip information at a position
  - entity_at_position: return (kind, fully_qualified_name) of entity at a position
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

  (* Convert between symbol offset and line:column via Python RPC.
     Requires the Isabelle_RPC Python host to be running. *)
  val offset_to_line_column: string -> int -> int * int
  val line_column_to_offset: string -> int -> int -> int

  (* Return all ancestor sessions (whose heaps are loaded) with their export database paths.
     Each entry is (session_name, db_path). *)
  val get_session_databases: unit -> (string * string) list
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
      then (warning "hover_message: unexpected YXML markup in result"; YXML.content_of yxml)
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

val get_session_databases : unit -> (string * string) list =
  let
    val cache = Synchronized.cache (fn () =>
      Scala.function1 "pide_state.get_session_databases" ""
      |> YXML.parse_body
      |> XML.Decode.list (XML.Decode.pair XML.Decode.string XML.Decode.string));
  in fn () => Synchronized.cache_eval {persistent = false} cache end;

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
      name = "pide_state.get_session_databases",
      arg_schema = unpackUnit,
      ret_schema = packList (packPair (packString, packString)),
      function = fn () => PIDE_State.get_session_databases (),
      timeout = NONE
    }
))

end
