(*
  Isa-REPL App for collecting semantic interpretations.

  Reads a theory name from the socket, calls Semantic_Store.interpret,
  and streams tracing messages back to the client in real-time.

  Protocol:
    Client sends: theory_name (msgpack string)
    Server streams: packString messages during interpretation (from tracing)
    Server sends:   packUnit when done successfully
    On error:       sends packString "ERROR: ..." then packUnit
*)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val _ = REPL_Server.register_app "Semantic_Store.collect" (fn (cin, cout, state) =>
  let
    fun read unpacker =
      let val (ret, cin') = doUnpack unpacker (!cin)
      in cin := cin'; ret end
    fun send msg = (REPL_Server.output cout packString msg;
                    BinIO.StreamIO.flushOut cout)
    fun done () = (REPL_Server.output cout packUnit ();
                   BinIO.StreamIO.flushOut cout)
  in
    \<^try>\<open>
      let
        val thy_name = read unpackString
        val thy = Thy_Info.get_theory thy_name

        (* Hijack tracing to stream messages over the socket *)
        val old_tracing = !Private_Output.tracing_fn
        val _ = Private_Output.tracing_fn :=
          (fn msgs => List.app send msgs)

        val _ = \<^try>\<open>Semantic_Store.interpret (Context.Theory thy)
          catch exn => (Private_Output.tracing_fn := old_tracing; Exn.reraise exn)\<close>
        val _ = Private_Output.tracing_fn := old_tracing
        val _ = done ()
      in state end
    catch exn =>
      (send ("ERROR: " ^ Runtime.exn_message exn);
       done ();
       state)\<close>
  end)

end
