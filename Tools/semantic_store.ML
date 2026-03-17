signature SEMANTIC_STORE = sig
  type query_functions = {
    query_constant : string -> string option,
    query_theorem  : string -> string option,
    query_type     : string -> string option,
    query_class    : string -> string option,
    query_locale   : string -> string option
  }
  val make_query_functions : Context.generic -> string list -> bool -> query_functions
  val interpret : Context.generic -> unit
  val clean_wip : unit -> unit
  val save_store : Path.T -> unit
  val load_store : Path.T -> unit
end

structure Semantic_Store : SEMANTIC_STORE = struct

(* --- Store data structures --- *)

type theory_long_name = string
datatype key' = Constant_Key of string (*fully quantified name*)
              | Theorem_Key  of Term_Digest.digest128
              | Type_Key     of string
              | Class_Key    of string
              | Locale_Key   of string
type key = key' * theory_long_name
type semantics = string (*in English*)
type entry = {
        semantics: semantics,
        pretty_printed: string
     }

structure Thm_Dedup = Hash_Table(struct
  type key = Term_Digest.digest128
  val hash = Term_Digest.digest128_hasher
  val eq = op = : Term_Digest.digest128 * Term_Digest.digest128 -> bool
end)

structure EntryTab = Hash_Table(struct
  type key = key
  fun hash (Constant_Key name, thy) = Word.xorb (0w1, Hashers.string_key name + Hashers.string_key thy)
    | hash (Theorem_Key d, thy)     = Word.xorb (0w2, Term_Digest.digest128_hasher d + Hashers.string_key thy)
    | hash (Type_Key name, thy)     = Word.xorb (0w3, Hashers.string_key name + Hashers.string_key thy)
    | hash (Class_Key name, thy)    = Word.xorb (0w4, Hashers.string_key name + Hashers.string_key thy)
    | hash (Locale_Key name, thy)   = Word.xorb (0w5, Hashers.string_key name + Hashers.string_key thy)
  fun eq ((k1, v1), (k2, v2)) = k1 = k2 andalso v1 = v2
end)

val STORE : entry EntryTab.table = EntryTab.empty 4096
val STORE_lock = Synchronized.var "SEMANTIC_STORE" ()

structure Local_Store = Proof_Data(
  type T = entry EntryTab.table option Unsynchronized.ref
  val init = fn _ => Unsynchronized.ref NONE
)

fun get_local_store ctxt =
  let val r = Local_Store.get ctxt
  in case !r of
       SOME tab => tab
     | NONE =>
         let val tab = EntryTab.empty 256
         in r := SOME tab; tab end
  end

val INTERPRETED : unit Strhashtab.table = Strhashtab.empty 256
val _ = List.app (fn name => Strhashtab.update INTERPRETED (name, ()))
  ["Pure", "Tools.Code_Generator", "HOL.Code_Evaluation"]

fun is_interpreted thy =
  Synchronized.change_result STORE_lock (fn () =>
    (Strhashtab.defined INTERPRETED (Context.theory_long_name thy), ()))

fun mark_interpreted thy =
  Synchronized.change STORE_lock (fn () =>
    (Strhashtab.update INTERPRETED (Context.theory_long_name thy, ()); ()))

(* --- RPC protocol --- *)

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack

  fun pack_entry_kind (Constant_Key _, _) = packInt 0
    | pack_entry_kind (Theorem_Key _, _) = packInt 1
    | pack_entry_kind (Type_Key _, _) = packInt 2
    | pack_entry_kind (Class_Key _, _) = packInt 3
    | pack_entry_kind (Locale_Key _, _) = packInt 4

  val pack_arg = fn {file_path, theory_longname, thy_hash, deps_longname, entries, context_info} =>
    packTuple6 (packString, packString, packBytes, packList packString,
                packList (packTuple4 (pack_entry_kind, packString, packString, packInt)),
                packString)
               (file_path, theory_longname, thy_hash, deps_longname, entries, context_info)
in

type query_functions = {
  query_constant : string -> semantics option,
  query_theorem  : string -> semantics option,
  query_type     : string -> semantics option,
  query_class    : string -> semantics option,
  query_locale   : string -> semantics option
}

fun make_query_functions (context : Context.generic) (current_names : string list)
    (with_pretty : bool) : query_functions =
  let
    val thy = Context.theory_of context
    val consts = Context.cases Sign.consts_of Proof_Context.consts_of context
    val consts_space = Consts.space_of consts
    val ctxt = Context.cases Proof_Context.init_global I context
    val current_set = Symtab.make_set current_names

    fun check_not_current name =
      if Symtab.defined current_set name
      then error (String.concat
        ["Cannot query \"", name, "\": it is one of the entries you are asked to interpret. ",
         "Interpret it yourself from the source file."])
      else ()

    fun format_entry (e : entry) =
      if with_pretty andalso #pretty_printed e <> "" then
        #pretty_printed e ^ "\n" ^ #semantics e
      else #semantics e

    fun lookup_store key =
      Synchronized.change_result STORE_lock
        (fn () =>
          (case context of
             Context.Proof ctxt =>
               (case EntryTab.lookup (get_local_store ctxt) key of
                  SOME e => SOME (format_entry e)
                | NONE => Option.map format_entry (EntryTab.lookup STORE key))
           | _ => Option.map format_entry (EntryTab.lookup STORE key),
           ()))

    fun query_constant name =
      (let
        val full_name = Name_Space.intern consts_space name
        val _ = check_not_current full_name
        val ns_entry = Name_Space.the_entry consts_space full_name
      in
        lookup_store (Constant_Key full_name, #theory_long_name ns_entry)
      end
      handle ERROR msg =>
        if String.isPrefix (String.concat ["Undefined constant: ", name]) msg
        then error (String.concat ["Constant ", name, " is not defined in this context"])
        else raise ERROR msg)

    fun query_theorem name =
      (let
        val (base_name, idx_opt) = Theory_Structure.parse_thm_name name
        val facts = Proof_Context.facts_of ctxt
        val fact_space = Facts.space_of facts
        val full_name = Name_Space.intern fact_space base_name
        val _ = check_not_current full_name
        val ns_entry = Name_Space.the_entry fact_space full_name
        val {thms, ...} = case Facts.lookup context facts full_name of
            SOME result => result
          | NONE => error (String.concat ["Lemma ", name, " is not defined in this context"])
        val thm = case idx_opt of
            NONE => hd thms
          | SOME n => nth thms (n - 1)
        val s = Term_Digest.thm128 thm
      in
        lookup_store (Theorem_Key s, #theory_long_name ns_entry)
      end
      handle ERROR msg =>
        if String.isPrefix "Undefined " msg
        then error (String.concat ["Lemma ", name, " is not defined in this context"])
        else raise ERROR msg)

    fun query_ns_key make_key' space kind_label name =
      (let
        val full_name = Name_Space.intern space name
        val _ = check_not_current full_name
        val ns_entry = Name_Space.the_entry space full_name
      in
        lookup_store (make_key' full_name, #theory_long_name ns_entry)
      end
      handle ERROR msg =>
        if String.isPrefix "Undefined " msg
        then error (String.concat [kind_label, " ", name, " is not defined in this context"])
        else raise ERROR msg)
  in
    { query_constant = query_constant,
      query_theorem  = query_theorem,
      query_type     = query_ns_key Type_Key (Sign.type_space thy) "type",
      query_class    = query_ns_key Class_Key (Sign.class_space thy) "class",
      query_locale   = query_ns_key Locale_Key (Locale.locale_space thy) "locale" }
  end

fun make_query_callback (context : Context.generic) (current_names : string list)
    : Remote_Procedure_Calling.callback' =
  let
    val {query_constant, query_theorem, query_type, query_class, query_locale} =
      make_query_functions context current_names true
  in
    Remote_Procedure_Calling.mk_callback {
      name = "Semantic_Store.query",
      arg_schema = unpackPair (unpackInt, unpackString),
      ret_schema = packOption packString,
      function = fn (tag, name) =>
        case tag of
          0 => query_constant name
        | 1 => query_theorem name
        | 2 => query_type name
        | 3 => query_class name
        | 4 => query_locale name
        | _ => NONE,
      timeout = NONE
    }
  end

val check_theorem_name_in_file_cmd :
    (string * string list, (int * int * bool) list) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.check_theorem_name_in_file",
  arg_schema = packPair (packString, packList packString),
  ret_schema = unpackList (unpackTuple3 (unpackInt, unpackInt, unpackBool)),
  callback = [],
  timeout = NONE
}

fun make_interpret_file_cmd callbacks :
    ({file_path: string, theory_longname: string, thy_hash: Word8Vector.vector,
      deps_longname: string list, entries: (key * string * string * int) list,
      context_info: string},
     semantics option list * string list * (int * int * real) * (int * int * real))
    Remote_Procedure_Calling.command = {
  name = "Semantic_Store.interpret_file",
  arg_schema = pack_arg,
  ret_schema = unpackTuple4 (unpackList (unpackOption unpackString),
                             unpackList unpackString,
                             unpackTuple3 (unpackInt, unpackInt, unpackReal),
                             unpackTuple3 (unpackInt, unpackInt, unpackReal)),
  callback = callbacks,
  timeout = NONE
}

end (*local MessagePack*)

(* --- Utilities --- *)

fun trim_markup msg =
  let
    val len = size msg
    val i = Unsynchronized.ref 0
    val visible = Unsynchronized.ref true
    val start = Unsynchronized.ref 0
    val parts = Unsynchronized.ref ([] : string list)
  in
    while !i < len do
      let val c = String.sub (msg, !i) in
        if c = #"\005" then
          (if !visible then
             (if !start < !i then
                parts := String.substring (msg, !start, !i - !start) :: !parts
              else ();
              visible := false)
           else
             (visible := true;
              start := !i + 1);
           i := !i + 1)
        else
          (if not (!visible) then () else ();
           i := !i + 1)
      end;
    (if !visible andalso !start < len then
       parts := String.substring (msg, !start, len - !start) :: !parts
     else ());
    String.concat (rev (!parts))
  end

fun string_of_term ctxt =
    let val ctxt' = ctxt
              |> Config.put Printer.show_types false
              |> Config.put Printer.show_sorts false
              |> Config.put Printer.show_markup false
              |> Config.put Printer.show_structs false
              |> Config.put Printer.show_question_marks true
              |> Config.put Printer.show_brackets false
     in Syntax.string_of_term ctxt'
     #> trim_markup
    end

fun string_of_typ ctxt =
    let val ctxt' = ctxt
              |> Config.put Printer.show_sorts true
              |> Config.put Printer.show_markup false
     in Syntax.string_of_typ ctxt'
     #> trim_markup
    end

(* --- Main orchestration --- *)

local
  fun key'_kind_rank (Class_Key _) = 0
    | key'_kind_rank (Locale_Key _) = 1
    | key'_kind_rank (Type_Key _) = 2
    | key'_kind_rank (Constant_Key _) = 3
    | key'_kind_rank (Theorem_Key _) = 4
in
fun key_kind_ord ((k1, _), (k2, _)) = int_ord (key'_kind_rank k1, key'_kind_rank k2)
end
fun entry_ord (e1, e2) =
  (case int_ord (the_default 0 (Position.line_of (#3 e1)),
                 the_default 0 (Position.line_of (#3 e2))) of
    EQUAL => (case key_kind_ord (#1 e1, #1 e2) of
      EQUAL => string_ord (#2 e1, #2 e2)
    | ord => ord)
  | ord => ord)

(* Returns ((current input, output, cost), (historical input, output, cost)) *)
fun interpret' (context : Context.generic) : (int * int * real) * (int * int * real) =
  let
    (* 1. Get theory information *)
    val thy = Context.theory_of context
    val thy_name = Context.theory_long_name thy
    val thy_file_path = Theory_Structure.get_theory_file_path thy
    val deps_longname = Theory.parents_of thy |> map Context.theory_long_name
    val thy_hash = if Resources.loaded_theory thy_name
                   then Theory_Hash.hash_of thy
                   else Byte.stringToBytes thy_name

    (* 2. Extract constants and theorems, filtering infrastructure *)
    val {is_infra_const, is_infra_thm, is_infra_type, is_infra_class, is_infra_locale} =
      Infra_Filter.gen_infra_filters context

    val const_entries = Theory_Structure.get_constants_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_const name then NONE
          else SOME ((Constant_Key name, thy_name), name, pos))

    val raw_thm_entries = Theory_Structure.get_theorems_with_positions context
      |> filter (fn (name, _, _, thm, _) => not (is_infra_thm (name, thm)))

    (* 2c. Build named entries and check which appear in the source file. *)
    val raw_thm_entries_named = raw_thm_entries
      |> map (fn (name, idx, pos, thm, is_global) =>
        (Thm_Name.print (name, idx), pos, thm, is_global))
    val thy_prefix = thy_name ^ "."
    fun strip_thy_prefix name =
      if String.isPrefix thy_prefix name
      then String.extract (name, size thy_prefix, NONE)
      else name
    val file_match_names = map (strip_thy_prefix o #1) raw_thm_entries_named
    val name_check_results = Context.cases
      (fn _ => Remote_Procedure_Calling.call_command
          check_theorem_name_in_file_cmd (thy_file_path, file_match_names)
        handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
          (warning ("check_theorem_name_in_file RPC failed: " ^ message);
           map (K (0, ~1, false)) file_match_names))
      (fn _ => map (K (0, ~1, false)) file_match_names)
      context
    val raw_thm_entries_with_offsets =
          (raw_thm_entries_named ~~ name_check_results)

    (* 2d. Filter theorems with [code] attribute *)
    val raw_thm_entries_with_offsets = filter (fn (_, (_, _, has_code)) =>
        not has_code) raw_thm_entries_with_offsets

    (* 3. Register theorems and assign digest-based keys *)
    val thm_entries_with_keys = raw_thm_entries_with_offsets
      |> map (fn ((name, pos, thm, is_global), (word_off, goal_off, _)) =>
        let val s = Term_Digest.thm128 thm
        in (s, name, pos, thm, word_off, goal_off, is_global) end)

    (* 4. Deduplicate theorems by digest *)
    val best_thms : (string * Position.T * thm * int * int * bool) Thm_Dedup.table = Thm_Dedup.empty 1024
    val _ = thm_entries_with_keys
      |> List.app (fn (s, name, pos, thm, word_off, goal_off, is_global) =>
        case Thm_Dedup.lookup best_thms s of
          NONE => Thm_Dedup.update best_thms (s, (name, pos, thm, word_off, goal_off, is_global))
        | SOME (_, _, _, prev_word_off, prev_goal_off, _) =>
            let val dominated =
              if goal_off >= 0 andalso prev_goal_off < 0 then true
              else if prev_goal_off >= 0 andalso goal_off < 0 then false
              else if goal_off >= 0 then goal_off < prev_goal_off
              else word_off < prev_word_off
            in if dominated
               then Thm_Dedup.update best_thms (s, (name, pos, thm, word_off, goal_off, is_global))
               else ()
            end)
    val deduped_thm_entries = Thm_Dedup.fold
      (fn (s, (name, pos, thm, _, goal_off, is_global)) => fn acc =>
        ((Theorem_Key s, thy_name), name, pos, goal_off, SOME thm, is_global) :: acc)
      best_thms []

    (* 5. Extract types, classes, and locales *)
    val type_entries = Theory_Structure.get_types_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_type name then NONE
          else SOME ((Type_Key name, thy_name), name, pos))

    val class_entries = Theory_Structure.get_classes_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_class name then NONE
          else SOME ((Class_Key name, thy_name), name, pos))

    val locale_entries = Theory_Structure.get_locales_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_locale name then NONE
          else SOME ((Locale_Key name, thy_name), name, pos))

    (* 6. Build entries list, sorted by source position. *)
    val ctxt = Context.cases Proof_Context.init_global I context
    val pp = string_of_term ctxt
    val pp_typ = string_of_typ ctxt

    fun mk_prop_str (key, thm_opt) =
      case (key, thm_opt) of
        ((Theorem_Key _, _), SOME thm) => pp (Thm.prop_of thm)
      | ((Constant_Key cname, _), _) =>
          (let val T = Consts.the_constraint (Sign.consts_of thy) cname
           in pp_typ T end
           handle TYPE _ => "" | ERROR _ => "")
      | _ => ""

    fun mk_entry (key, name, pos, goal_off, thm_opt, is_global) =
      (key, name, pos, goal_off, is_global, mk_prop_str (key, thm_opt))

    val all_entries_with_pos =
      (map (fn (key, name, pos) => mk_entry (key, name, pos, ~1, NONE, true)) const_entries
       @ map (fn (key, name, pos, goal_off, thm_opt, is_global) =>
           mk_entry (key, name, pos, goal_off, thm_opt, is_global)) deduped_thm_entries
       @ map (fn (key, name, pos) => mk_entry (key, name, pos, ~1, NONE, true)) type_entries
       @ map (fn (key, name, pos) => mk_entry (key, name, pos, ~1, NONE, true)) class_entries
       @ map (fn (key, name, pos) => mk_entry (key, name, pos, ~1, NONE, true)) locale_entries)
      |> sort entry_ord

    val entries = all_entries_with_pos |> map (fn (key, name, pos, _, _, prop_str) =>
      (key, name, prop_str, the_default ~1 (Position.line_of pos)))

    val names = map #2 entries

    (* 6b. Build context_info *)
    val context_info = Context.cases (K "")
      (fn ctxt' =>
        let val pp' = string_of_term ctxt'
            val assumptions = Assumption.all_prems_of ctxt'
              |> map (fn thm => pp' (Thm.prop_of thm))
        in if null assumptions then ""
           else String.concatWith "\n" (map (fn a => "  - " ^ a) assumptions)
        end)
      context

    (* 7. Build callbacks capturing context *)
    val query_cb = make_query_callback context names
    val theory_info_cb = Theory_Structure.make_theory_info_callback context
    val cmd = make_interpret_file_cmd [query_cb, theory_info_cb]

    (* 8. Call RPC command *)
    val rpc_arg = {
      file_path = thy_file_path,
      theory_longname = thy_name,
      thy_hash = thy_hash,
      deps_longname = deps_longname,
      entries = entries,
      context_info = context_info
    }
    val (results, pretty_prints, (input_tokens, output_tokens, cost_usd),
                                  (cum_input_tokens, cum_output_tokens, cum_cost_usd)) =
      Remote_Procedure_Calling.call_command cmd rpc_arg
        handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
          error (String.concat ["Semantic interpretation RPC failed: ", message])
    val fmt_cost = Real.fmt (StringCvt.FIX (SOME 4))
    val _ = tracing (String.concat [
      "[Semantic_Store] ", thy_name,
      ": input=", Int.toString input_tokens,
      " output=", Int.toString output_tokens,
      " cost=$", fmt_cost cost_usd,
      " (historical: input=", Int.toString cum_input_tokens,
      " output=", Int.toString cum_output_tokens,
      " cost=$", fmt_cost cum_cost_usd, ")"])

    (* 9. Store results *)
    val _ = Synchronized.change STORE_lock (fn () =>
      (map2 (fn ((key, _, _, _, is_global, _), pretty) => fn result_opt =>
        case result_opt of
          SOME sem =>
            let val e = {semantics = sem, pretty_printed = pretty}
            in if is_global then EntryTab.update STORE (key, e)
               else (case context of
                       Context.Proof ctxt => EntryTab.update (get_local_store ctxt) (key, e)
                     | _ => EntryTab.update STORE (key, e))
            end
        | NONE => ())
      (all_entries_with_pos ~~ pretty_prints) results; ()))
  in
    ((input_tokens, output_tokens, cost_usd),
     (cum_input_tokens, cum_output_tokens, cum_cost_usd))
  end

fun clean_wip () =
  Synchronized.change STORE_lock (fn () =>
    let val cache = Strhashtab.empty 32
        fun is_loaded name =
          case Strhashtab.lookup cache name of
            SOME b => b
          | NONE => let val b = Resources.loaded_theory name
                    in Strhashtab.update cache (name, b); b end
    in
      EntryTab.fold (fn (key as (_, thy_name), _) => fn keys =>
          if is_loaded thy_name then keys else key :: keys) STORE []
      |> List.app (fn key => EntryTab.delete_safe STORE key);
      ()
    end)

fun collect_uninterpreted thy =
  let
    val seen = Inthashtab.empty 32
    fun collect thy acc =
      let val id = Context.theory_identifier thy
      in if is_interpreted thy orelse Inthashtab.defined seen id then acc
         else
           let val _ = Inthashtab.update seen (id, ())
               val acc' = fold collect (Theory.parents_of thy) acc
           in acc' @ [thy] end
      end
  in collect thy [] end

fun interpret (context : Context.generic) : unit =
  let
    val thy = Context.theory_of context
    val theories = collect_uninterpreted thy
    val contexts = map Context.Theory theories
      @ (case context of Context.Proof _ => [context] | _ => [])
    val _ = if null contexts then () else
      tracing ("[Semantic_Store] interpreting " ^ Int.toString (length contexts) ^ " theories")

    val ((total_in, total_out, total_cost), (hist_in, hist_out, hist_cost)) =
      fold (fn ctx => fn ((acc_in, acc_out, acc_cost), (hacc_in, hacc_out, hacc_cost)) =>
        let
          val ((i, o_, c), (hi, ho, hc)) = interpret' ctx
          val _ = Context.cases mark_interpreted (K ()) ctx
        in ((acc_in + i, acc_out + o_, acc_cost + c),
            (hacc_in + hi, hacc_out + ho, hacc_cost + hc)) end)
      contexts ((0, 0, 0.0), (0, 0, 0.0))

    val fmt_cost = Real.fmt (StringCvt.FIX (SOME 4))
    val _ = tracing (String.concat [
        "[Semantic_Store] total: input=", Int.toString total_in,
        " output=", Int.toString total_out,
        " cost=$", fmt_cost total_cost,
        " (historical: input=", Int.toString hist_in,
        " output=", Int.toString hist_out,
        " cost=$", fmt_cost hist_cost, ")"])
  in
    ()
  end

(* --- Save / Load --- *)

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack
  infix 3 >>

  fun pack_store_entry (((key', thy_name), {semantics, pretty_printed}) : key * entry) =
    let
      fun pack_named tag name =
        packPair (packInt, packTuple4 (packString, packString, packString, packString))
          (tag, (name, thy_name, semantics, pretty_printed))
    in
      case key' of
        Constant_Key name => pack_named 0 name
      | Theorem_Key (lo, hi) =>
          packPair (packInt, packTuple5 (packWord64, packWord64, packString, packString, packString))
            (1, (lo, hi, thy_name, semantics, pretty_printed))
      | Type_Key name => pack_named 2 name
      | Class_Key name => pack_named 3 name
      | Locale_Key name => pack_named 4 name
    end

  val unpack_store_entry : (key * entry) unpacker =
    unpackPair_dep' unpackInt (fn tag =>
      let fun mk mk_key' (name, thy_name, s, p) =
            ((mk_key' name, thy_name), {semantics = s, pretty_printed = p})
      in case tag of
        1 => unpackTuple5 (unpackWord64, unpackWord64, unpackString, unpackString, unpackString)
             >> (fn (lo, hi, thy_name, s, p) =>
                   ((Theorem_Key (lo, hi), thy_name), {semantics = s, pretty_printed = p}))
      | _ => unpackTuple4 (unpackString, unpackString, unpackString, unpackString)
             >> (fn args => mk (case tag of
                       0 => Constant_Key | 2 => Type_Key
                     | 3 => Class_Key | 4 => Locale_Key
                     | _ => raise Unpack) args)
      end)
in

fun save_store path =
  File_Stream.open_output (fn stream =>
    let val cout = BinIO.getOutstream stream
    in Synchronized.change STORE_lock (fn () =>
      let
        val entry_list = EntryTab.fold (fn (key, entry) => fn acc =>
          (key, entry) :: acc) STORE []
        val interpreted_names = Strhashtab.fold (fn (name, ()) => fn acc =>
          name :: acc) INTERPRETED []
      in
        doPack (packPair (packList packString, packList pack_store_entry))
          (interpreted_names, entry_list) cout;
        tracing ("[Semantic_Store] saved " ^ Int.toString (length entry_list) ^
                 " entries, " ^ Int.toString (length interpreted_names) ^
                 " interpreted theories to " ^ Path.print path);
        ()
      end)
    end) path

fun load_store path =
  let
    val (interpreted_names, entries) = File_Stream.open_input (fn stream =>
      let val cin = BinIO.getInstream stream
          val (result, _) = doUnpack
            (unpackPair (unpackList unpackString,
                         unpackList unpack_store_entry)) cin
      in result end) path
  in
    Synchronized.change STORE_lock (fn () =>
      (List.app (fn (key, e) => EntryTab.update STORE (key, e)) entries;
       List.app (fn name => Strhashtab.update INTERPRETED (name, ())) interpreted_names;
       ()));
    tracing ("[Semantic_Store] loaded " ^ Int.toString (length entries) ^
             " entries, " ^ Int.toString (length interpreted_names) ^
             " interpreted theories from " ^ Path.print path)
  end

end (*local save/load*)

end
