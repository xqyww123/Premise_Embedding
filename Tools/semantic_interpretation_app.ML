(*
  Isa-REPL App for collecting semantic interpretations.

  Interpret-only: embedding is done Python-side by semantics_manage.py (offline
  `embed`, or the `collect --embed-models` chain), never here.

  Protocol:
    Client sends: theory_names (msgpack string list — the roots to interpret)
                  force (msgpack bool – when true, re-interpret already-finished theories)
    Server streams: packString messages during interpretation (from tracing)

  Concurrency width is the session's own -o threads=N, i.e. Multithreading.max_threads (),
  which is what Isabelle's future worker pool bounds itself by. There is no per-call override.
    Server sends:   packUnit when done successfully
    On error:       sends packString "ERROR: ..." then packUnit
*)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val _ = REPL_Server.register_app "Semantic_Store.collect" (fn (cin, cout, state) =>
  let
    fun read unpacker =
      let val (ret, cin') = doUnpack unpacker (!cin)
      in cin := cin'; ret end
    (* Serialize output: under -o threads>1 several interpret' worker threads
       stream tracing concurrently through these channels; without the lock their
       msgpack frames would interleave and corrupt the socket stream. *)
    val send_lock = Synchronized.var "Semantic_Store.collect.send" ()
    fun send msg =
      Synchronized.change send_lock (fn () =>
        (REPL_Server.output cout packString msg;
         BinIO.StreamIO.flushOut cout))
    fun done () =
      Synchronized.change send_lock (fn () =>
        (REPL_Server.output cout packUnit ();
         BinIO.StreamIO.flushOut cout))
  in
    \<^try>\<open>
      let
        val thy_names = read (unpackList unpackString)
        val force = read unpackBool
        val thys = map Thy_Info.get_theory thy_names
        val _ = List.app Theory_Hash.store_theory_hash thys
        val _ = Remote_Procedure_Calling.load ["Isabelle_Semantic_Embedding"]

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

        val _ = \<^try>\<open>
                  Semantic_Store.interpret_with_parallel force (map Context.Theory thys)
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
