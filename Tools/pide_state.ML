(*  Title:      Tools/pide_state.ML
    Author:     Qiyuan Xu

Access PIDE document state via Scala functions:
  - save_thy_files: collect theory sources from PIDE state, write to backup files (path~)
  - command_positions: resolve ID-based positions to file/line/column positions
*)

signature PIDE_STATE =
sig
  val save_thy_files: unit -> string list
  val command_positions: Position.T list -> Position.T list
end;

structure PIDE_State: PIDE_STATE =
struct

fun save_thy_files () =
  let
    val yxml = Scala.function1 "pide_state.save_thy_files" "";
  in
    YXML.parse_body yxml |> XML.Decode.list XML.Decode.string
  end;

fun command_positions positions =
  let
    val _ = save_thy_files ();

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
          |> XML.Decode.list (XML.Decode.pair XML.Decode.string XML.Decode.int)
        end;

    val results = Unsynchronized.ref resolved;
    fun pop () = case ! results of
        [] => ("", 0)
      | x :: xs => (results := xs; x);
  in
    positions |> map (fn pos =>
      if needs_resolution pos then
        let val (file, line) = pop ()
        in if file = "" then pos
           else Position.make0 line
             (the_default 0 (Position.offset_of pos))
             (the_default 0 (Position.end_offset_of pos))
             (the_default "" (Position.label_of pos))
             file
             (the_default "" (Position.id_of pos)) end
      else pos)
  end;

end;
