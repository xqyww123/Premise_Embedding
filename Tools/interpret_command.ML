(*  Title:      Tools/interpret_command.ML

Outer-syntax command driving Semantic_Store interpretation, with an interactive
confirmation before any LLM work starts.  After interpretation (and on the
nothing-to-interpret path) it also completes the vector store:
Semantic_Store.embed_all_missing embeds every interpreted record in the whole DB
still missing a vector for the configured embedding model.

INTERACTIVE USE ONLY.  The confirmation blocks on Active.dialog, whose only responder in
the whole Isabelle tree is jEdit (Tools/jEdit/src/active.scala:100).  Under `isabelle
build`, Isa-REPL, VSCode, `isabelle server` or headless PIDE the join never returns.  Do
not put this command in a theory that gets built; batch collection goes through the
`Semantic_Store.collect` REPL app instead (semantic_interpretation_app.ML).
*)

structure Semantic_Interpret_Command =
struct

(* Resolve a theory name against the ancestors of the current theory.

   Mirrors Theory_Hash.resolve_theory (theory_hash.ML:33) rather than calling it, so the
   real source position can be threaded through: Theory.check then highlights the offending
   name in jEdit and lists the candidates.  resolve_theory hard-codes Position.none.

   The self case is NOT about the candidate set: Theory.check (theory.ML:163) resolves via
   Context.get_theory, which does handle name = thy itself (context.ML:353-359) -- but only
   by falling through to `theory_id_final (theory_id thy)`, and erroring "Unfinished theory"
   otherwise.  This command is interactive, so the theory under the cursor is still being
   constructed and hence not final; without the bypass, naming the theory you are sitting in
   would be rejected.  (`ancestors_of thy` feeds only `all ()`, the completion list used to
   build the error report -- not the lookup.)  Same bypass as resolve_theory. *)
fun resolve_root ctxt (name, pos) =
  let
    val thy = Proof_Context.theory_of ctxt
    val long = String.isSubstring "." name
    val self = if long then Context.theory_long_name thy else Context.theory_base_name thy
  in
    if name = self then thy
    else Theory.check {long = long} ctxt (name, pos)
  end

fun run (force : bool) names state =
  let
    val _ =
      if Toplevel.is_theory state then ()
      else error "run_semantic_interpretation: only available in a theory context"

    val ctxt = Toplevel.context_of state

    (* Resolve BEFORE touching the RPC: a typo should fail instantly, with a position,
       rather than after a cold RPC-host launch. *)
    val roots =
      if null names then [Toplevel.theory_of state]
      else map (resolve_root ctxt) names

    (* Connecting may cold-start the Python RPC host (RPC.ML:421 -> launch_and_retry ->
       launch_RPC_host), which takes 1-40 s.  Say so, or the command looks hung. *)
    val _ = writeln "Connecting to the semantic DB (this may launch the RPC host)..."
    val _ = Remote_Procedure_Calling.load ["Isabelle_Semantic_Embedding"]

    (* As in semantic_interpretation_app.ML:42 -- without this the Python side cannot
       resolve these theory hashes. *)
    val _ = List.app Theory_Hash.store_theory_hash roots

    val work = Semantic_Store.plan_interpretation force roots
    val n = length work

    (* Phase 2: whole-DB vector completion (missing-only) for the configured embedding
       model.  Runs after a successful interpretation run AND on the nothing-to-
       interpret path, so the command always leaves the vector store complete.
       Interrupting is safe: vectors are committed per 256-batch and the scan is
       presence-based, so a re-run resumes.  A failure fails the command — the
       interpretation results are already durable in LMDB; a recognisable failure
       (extract_user_error's marker) renders as one actionable line, anything else
       keeps the traceback. *)
    fun embed_phase () =
      (writeln "[Semantic_Embedding] Completing missing vector embeddings...";
       Semantic_Store.embed_all_missing (Context.Proof ctxt))
      handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
        (case Semantic_Store.extract_user_error message of
          SOME human => error human
        | NONE => error ("Vector embedding failed: " ^ message))
  in
    if n = 0 then (writeln "[Semantic_Store] Nothing to interpret."; embed_phase ())
    else
      let
        (* Name the theories, not just the count: "11 theories" gives the user nothing to
           judge by, while the names show at a glance whether this is the intended scope
           (say a couple of local theories) or an accidental sweep of half of HOL.  `work`
           comes from collect_cone in topological order, so these are also the ones that
           run first.  Capped at 20 -- past that the list stops informing and starts
           burying the question. *)
        val limit = 20
        val names = map Context.theory_long_name work
        val shown = take limit names
        val listing =
          Pretty.string_of (Pretty.block (Pretty.commas (map Pretty.str shown))) ^
          (if n > limit then ", ... (and " ^ string_of_int (n - limit) ^ " more)" else "")

        val (mk, promise) = Active.dialog_text ()
        val _ = writeln (String.concat [
          "[Semantic_Store] ", string_of_int n,
          (if n = 1 then " theory needs" else " theories need"), " interpretation:\n",
          listing, "\n",
          "This calls the LLM: it may take a long time and cost money.\n",
          "Afterwards, missing vector embeddings are computed as well (embedding API; far cheaper).\n",
          "  ", mk "Yes", "   ", mk "No"])
        (* Blocks until the user clicks.  Not caught: if the buffer is edited, jEdit
           cancels this exec and the interrupt propagates -- Active.dialog's promise carries
           an abort hook that removes it from the dialog table (active.ML:51), so that is
           clean.  Interrupting a run in progress is safe too: answers are written to LMDB
           as they arrive and each theory is marked as it completes, so a re-run resumes. *)
      in
        (case Future.join promise of
          "Yes" =>
            (* roots, NOT work: interpret_with_parallel re-derives the cone itself and needs
               the interpreted ancestors as relay nodes. *)
            (Semantic_Store.interpret_with_parallel force (map Context.Theory roots);
             embed_phase ())
        | _ => writeln "[Semantic_Store] Aborted.")
      end
  end

val _ =
  Outer_Syntax.command \<^command_keyword>\<open>run_semantic_interpretation\<close>
    "interpret the ancestor cone of the given theories into the semantic DB"
    (Parse.opt_bang -- Scan.repeat Parse.name_position >>
      (fn (force, names) => Toplevel.keep (run force names)))

end
