signature EXPLAIN_TERM = sig
  val raw_lambda_printer : Proof.context -> term -> string
  val make_desugar_callback : Context.generic -> Remote_Procedure_Calling.callback'
end

structure Explain_Term : EXPLAIN_TERM = struct

(* --- Compact term printer --- *)

fun try_dest_numeral t =
  try HOLogic.dest_number t |> Option.map snd

fun strip_app (f $ x) = let val (head, args) = strip_app f in (head, args @ [x]) end
  | strip_app t = (t, [])

fun resolve_bound env i =
  if i < length env then nth env i else "?b" ^ string_of_int i

fun is_atom (Const _) = true | is_atom (Free _) = true
  | is_atom (Var _) = true | is_atom (Bound _) = true
  | is_atom t = is_some (try_dest_numeral t)

fun is_binder ctxt t =
  (can Term.type_of t andalso is_some (Case_Translation.strip_case ctxt false t))
  orelse (case t of Const(\<^const_name>\<open>Let\<close>, _) $ _ $ Abs _ => true
                  | Abs _ => true | _ => false)

fun pt ctxt env parens t : string list =
  case try_dest_numeral t of
    SOME n => [string_of_int n]
  | NONE =>
  case (if can Term.type_of t then Case_Translation.strip_case ctxt false t else NONE) of
    SOME (scrutinee, clauses) =>
      let
        fun pt_rhs rhs = pt ctxt env (is_binder ctxt rhs) rhs
        fun print_clause (pat, rhs) =
          pt ctxt env false pat @ [" \<Rightarrow> "] @ pt_rhs rhs
        val inner = ["case "] @ pt ctxt env (is_binder ctxt scrutinee) scrutinee
          @ [" of "]
          @ flat (Library.separate [" | "] (map print_clause clauses))
      in if parens then ["("] @ inner @ [")"] else inner end
  | NONE =>
  case t of
    Const(\<^const_name>\<open>Let\<close>, _) $ value $ Abs(name, T, body) =>
      let
        val body' = Term.subst_bound (Free(name, T), body)
        val inner = ["let ", name, " = "] @ pt ctxt env (is_binder ctxt value) value
          @ [" in "] @ pt ctxt (name :: env) false body'
      in if parens then ["("] @ inner @ [")"] else inner end
  |
    Const(name, _) => [Long_Name.base_name name]
  | Free(name, _) => [name]
  | Var((name, i), _) =>
      [if i = 0 then "?" ^ name else "?" ^ name ^ "." ^ string_of_int i]
  | Bound i => [resolve_bound env i]
  | Abs(name, T, body) =>
      let val body' = Term.subst_bound (Free(name, T), body)
          val inner = ["\<lambda>", name, ". "] @ pt ctxt (name :: env) false body'
      in if parens then ["("] @ inner @ [")"] else inner end
  | _ $ _ =>
      let val (head, args) = strip_app t
          fun needs_parens a = not (is_atom a)
          val parts = pt ctxt env (needs_parens head) head
            @ maps (fn a => [" "] @ pt ctxt env (needs_parens a) a) args
      in if parens then ["("] @ parts @ [")"] else parts end

fun raw_lambda_printer ctxt t =
  let val ctxt' = Config.put Case_Translation.show_cases false ctxt
      val [t'] = Syntax.uncheck_terms ctxt' [t]
  in String.concat (pt ctxt' [] false t') end


(* --- Constant extraction and filtering --- *)

fun extract_constants (ctxt : Proof.context) (checked_term : term)
    : (string * Word8Vector.vector) list =
  let
    val thy = Proof_Context.theory_of ctxt
    val context = Context.Proof ctxt
    val const_space = Consts.space_of (Proof_Context.consts_of ctxt)
    val {is_infra_const, ...} = Infra_Filter.gen_infra_filters context

    val main_thy_ancestors =
      let val main_thy = Thy_Info.get_theory "Main"
      in Strhashtab.make_set
           (Context.theory_long_name main_thy ::
            map Context.theory_long_name (Theory.ancestors_of main_thy))
      end

    fun is_base_const name =
      case try (Name_Space.the_entry const_space) name of
        NONE => false
      | SOME entry => Strhashtab.defined main_thy_ancestors (#theory_long_name entry)

    val seen = Strhashtab.empty 64
    val consts = Term.fold_aterms (fn Const(name, _) =>
        (fn acc =>
          if Strhashtab.defined seen name then acc
          else if is_infra_const name then acc
          else if is_base_const name then acc
          else
            let val _ = Strhashtab.update seen (name, ())
                val uk = Universal_Key.key_of NONE context (Universal_Key.Constant name)
            in (name, uk) :: acc end)
      | _ => I) checked_term []
  in rev consts end


(* --- RPC callback factory --- *)

fun make_desugar_callback (fallback : Context.generic)
    : Remote_Procedure_Calling.callback' =
  let open MessagePackBinIO.Pack MessagePackBinIO.Unpack in
  Remote_Procedure_Calling.mk_callback {
    name = "explain_term.desugar_and_explain",
    arg_schema = unpackPair (
      unpackOption (unpackPair (unpackString, unpackInt)),
      unpackString),
    ret_schema = packPair (
      packString,
      packList (packPair (packString, packBytes))),
    function = fn (pos_opt, term_string) =>
      let
        val context = case pos_opt of
              NONE => fallback
            | SOME (file, offset) =>
                (case PIDE_State.context_at_position
                        {re_eval_cache = true} file offset of
                   SOME ctx => ctx
                 | NONE => fallback)
        val ctxt = Context.proof_of context
        (* mode_abbrev: type-check WITHOUT expanding abbreviations, so an
           abbreviation resolves to its own constant, not its expansion *)
        val checked = Proof_Context.read_term_abbrev ctxt term_string
        val compact = raw_lambda_printer ctxt checked
        val constants = extract_constants ctxt checked
      in (compact, constants) end,
    timeout = SOME (Time.fromSeconds 30)
  }
  end

end
