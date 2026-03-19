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
        val _ = Theory_Hash.store_theory_hash thy

        (* Hijack output channels to stream messages over the socket *)
        val old_writeln = !Private_Output.writeln_fn
        val old_state = !Private_Output.state_fn
        val old_information = !Private_Output.information_fn
        val old_tracing = !Private_Output.tracing_fn
        val old_warning = !Private_Output.warning_fn
        val old_legacy = !Private_Output.legacy_fn
        val old_error_message = !Private_Output.error_message_fn
        fun set_channels out_fn =
          (Private_Output.writeln_fn := out_fn;
           Private_Output.state_fn := out_fn;
           Private_Output.information_fn := out_fn;
           Private_Output.tracing_fn := out_fn;
           Private_Output.warning_fn := (fn msgs => List.app (fn m => send ("### " ^ m)) msgs);
           Private_Output.legacy_fn := (fn msgs => List.app (fn m => send ("legacy: " ^ m)) msgs);
           Private_Output.error_message_fn := (fn (_, msgs) => List.app (fn m => send ("*** " ^ m)) msgs))
        fun restore_channels () =
          (Private_Output.writeln_fn := old_writeln;
           Private_Output.state_fn := old_state;
           Private_Output.information_fn := old_information;
           Private_Output.tracing_fn := old_tracing;
           Private_Output.warning_fn := old_warning;
           Private_Output.legacy_fn := old_legacy;
           Private_Output.error_message_fn := old_error_message)
        val _ = set_channels (fn msgs => List.app send msgs)

        val _ = \<^try>\<open>Semantic_Store.interpret (Context.Theory thy)
          catch exn => (restore_channels (); Exn.reraise exn)\<close>
        val _ = restore_channels ()
        val _ = done ()
      in state end
    catch exn =>
      (send ("ERROR: " ^ Runtime.exn_message exn);
       done ();
       state)\<close>
  end)

end
