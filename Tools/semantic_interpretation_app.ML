(*
  Isa-REPL App for collecting semantic interpretations.

  Reads a theory name from the socket, calls Semantic_Store.interpret,
  and streams tracing messages back to the client in real-time.

  Protocol:
    Client sends: theory_name (msgpack string)
    Server streams: [msg, None] pairs (REPL output format) during interpretation
    Server sends:   packUnit when done successfully
    On error:       sends [None, error_msg] then packUnit
*)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val _ = REPL_Server.register_app "Semantic_Store.collect" (fn (cin, cout, state) =>
  let
    fun read unpacker =
      let val (ret, cin') = doUnpack unpacker (!cin)
      in cin := cin'; ret end
    fun done () = (REPL_Server.output cout packUnit ();
                   BinIO.StreamIO.flushOut cout)
  in
    \<^try>\<open>
      let
        val thy_name = read unpackString
        val thy = Thy_Info.get_theory thy_name
        val _ = Semantic_Store.interpret (Context.Theory thy)
        val _ = done ()
      in state end
    catch exn =>
      (REPL_Server.output cout packString ("ERROR: " ^ Runtime.exn_message exn);
       BinIO.StreamIO.flushOut cout;
       done ();
       state)\<close>
  end)

end
