(*  Title:      Tools/pide_state.ML
    Author:     Qiyuan Xu

Access PIDE document state via Scala functions:
  - save_thy_files: collect theory sources from PIDE state, write to backup files (path~)
  - absolutize_id_based_pos: resolve ID-based positions to file/line/offset positions
  - goto_definition: resolve entity reference at a position to its definition position
  - hover_message: extract tooltip information at a position
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

(* Extract file path and offset from a Position.T, absolutizing if needed *)
fun pos_file_offset pos =
  let
    val pos' = hd (absolutize_id_based_pos {write_to_temp_file = false} [pos]);
    val file = the_default "" (Position.file_of pos');
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
    else SOME (Position.make0 def_line def_offset def_end_offset "" def_file "")
  end;

fun hover_message_pide file_path symbol_offset =
  let
    val input_body =
      XML.Encode.pair XML.Encode.string XML.Encode.int
        (file_path, symbol_offset);
    val yxml = Scala.function1 "pide_state.hover_message"
      (YXML.string_of_body input_body);
    val msg = YXML.parse_body yxml |> XML.Decode.string;
  in
    if msg = "" then NONE
    else SOME msg
  end;

(* DB fallback via Python RPC *)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val goto_definition_from_db_cmd :
    (string * int, (string * int * int * int) option) Remote_Procedure_Calling.command = {
  name = "pide_state.goto_definition_from_db",
  arg_schema = packPair (packString, packInt),
  ret_schema = unpackOption (unpackTuple4 (unpackString, unpackInt, unpackInt, unpackInt)),
  callback = [],
  timeout = NONE
}

val hover_message_from_db_cmd :
    (string * int, string option) Remote_Procedure_Calling.command = {
  name = "pide_state.hover_message_from_db",
  arg_schema = packPair (packString, packInt),
  ret_schema = unpackOption unpackString,
  callback = [],
  timeout = NONE
}

end (*local MessagePack*)

fun goto_definition_from_db file_path offset =
  Remote_Procedure_Calling.call_command goto_definition_from_db_cmd (file_path, offset)
  |> Option.map (fn (def_file, def_line, def_offset, def_end_offset) =>
       Position.make0 def_line def_offset def_end_offset "" def_file "");

fun hover_message_from_db file_path offset =
  Remote_Procedure_Calling.call_command hover_message_from_db_cmd (file_path, offset);

(* Auto: try live PIDE first, then DB fallback *)

fun goto_definition pos =
  let val (file, offset) = pos_file_offset pos
  in
    if file = "" orelse offset = 0 then NONE
    else
      (case goto_definition_pide file offset of
        SOME result => SOME result
      | NONE => goto_definition_from_db file offset)
  end;

fun hover_message pos =
  let val (file, offset) = pos_file_offset pos
  in
    if file = "" orelse offset = 0 then NONE
    else
      (case hover_message_pide file offset of
        SOME result => SOME result
      | NONE => hover_message_from_db file offset)
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
))

end
