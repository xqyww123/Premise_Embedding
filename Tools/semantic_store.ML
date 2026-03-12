signature SEMANTIC_STORE = sig
  val interpret_file : theory -> unit
  val preserved_datatype_const_names : theory -> string list
end

structure Semantic_Store : SEMANTIC_STORE = struct

datatype key = Constant_Key of string * Position.T
             | Theorem_Key  of serial (*serial from Term_Serial_Index*)
type semantics = string (*in English*)
type entry = {
        semantics: semantics,
        position: Position.T
     }

structure EntryTab = Hash_Table(struct
  type key = key
  fun hash (Constant_Key (name, _)) = hash_string name
    | hash (Theorem_Key s) = hash_int s
  fun eq (Constant_Key (a1, p1), Constant_Key (a2, p2)) = a1 = a2 andalso p1 = p2
    | eq (Theorem_Key a, Theorem_Key b) = (a = b)
    | eq _ = false
end)

val STORE : entry EntryTab.table = EntryTab.empty 256
val STORE_lock = Synchronized.var "SEMANTIC_STORE" ()

(* Parse "name(N)" into (name, SOME N) or (name, NONE) *)
fun parse_thm_name s =
  let
    val len = String.size s
  in
    if len > 2 andalso String.sub (s, len - 1) = #")"
    then
      let
        fun find_paren i =
          if i < 0 then NONE
          else if String.sub (s, i) = #"(" then SOME i
          else find_paren (i - 1)
      in
        case find_paren (len - 2) of
          SOME i =>
            let val num_str = String.substring (s, i + 1, len - i - 2)
            in case Int.fromString num_str of
                SOME n => (String.substring (s, 0, i), SOME n)
              | NONE => (s, NONE)
            end
        | NONE => (s, NONE)
      end
    else (s, NONE)
  end

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack

  fun pack_entry_kind (Constant_Key _) = packInt 0
    | pack_entry_kind (Theorem_Key _) = packInt 1

  val pack_arg = fn {file_path, theory_longname, deps_longname, entries} =>
    packTuple4 (packString, packString, packList packString,
                packList (packTuple4 (pack_entry_kind, packString, packString, packInt)))
               (file_path, theory_longname, deps_longname, entries)

in

fun make_query_callback (context : Context.generic) (current_names : string list)
    : Remote_Procedure_Calling.callback' =
  let
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

    fun lookup_store key =
      Synchronized.change_result STORE_lock
        (fn () => (Option.map #semantics (EntryTab.lookup STORE key), ()))

    fun query_constant name =
      (let
        val full_name = Name_Space.intern consts_space name
        val _ = check_not_current full_name
        val ns_entry = Name_Space.the_entry consts_space full_name
      in
        lookup_store (Constant_Key (full_name, #pos ns_entry))
      end
      handle ERROR msg =>
        if String.isPrefix (String.concat ["Undefined constant: ", name]) msg
        then error (String.concat ["Constant ", name, " is not defined in this context"])
        else raise ERROR msg)

    fun query_theorem name =
      (let
        val _ = check_not_current name
        val (base_name, idx_opt) = parse_thm_name name
        val thms = Proof_Context.get_thms ctxt base_name
        val thm = case idx_opt of
            NONE => hd thms
          | SOME n => nth thms (n - 1)
        val s = Term_Serial_Index.serial_of_thm thm
      in
        lookup_store (Theorem_Key s)
      end
      handle ERROR msg =>
        if String.isPrefix (String.concat ["Undefined fact: ", name]) msg
        then error (String.concat ["Theorem ", name, " is not defined in this context"])
        else raise ERROR msg)
  in
    Remote_Procedure_Calling.mk_callback {
      name = "Semantic_Store.query",
      arg_schema = unpackPair (unpackInt, unpackString),
      ret_schema = packOption packString,
      function = fn (tag, name) =>
        case tag of
          0 => query_constant name
        | 1 => query_theorem name
        | _ => NONE,
      timeout = NONE
    }
  end

fun make_interpret_file_cmd (query_cb : Remote_Procedure_Calling.callback') :
    ({file_path: string, theory_longname: string,
      deps_longname: string list, entries: (key * string * string * int) list},
     semantics option list) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.interpret_file",
  arg_schema = pack_arg,
  ret_schema = unpackList (unpackOption unpackString),
  callback = [query_cb],
  timeout = NONE
}

val check_theorem_name_in_file_cmd :
    (string * string list, (int * int * bool) list) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.check_theorem_name_in_file",
  arg_schema = packPair (packString, packList packString),
  ret_schema = unpackList (unpackTuple3 (unpackInt, unpackInt, unpackBool)),
  callback = [],
  timeout = NONE
}

end

(* Helper: Get theory file path with expansion *)
fun get_theory_file_path thy =
  let
    val thy_name = Context.theory_long_name thy
    val master_dir = Resources.master_directory thy
    val thy_base = Long_Name.base_name thy_name
    val thy_file = Path.ext "thy" (master_dir + Path.basic thy_base)
    val expanded_path = File.full_path master_dir thy_file
  in
    File.platform_path expanded_path
  end

(* Helper: Get constants with positions from theory (local only) *)
fun get_constants_with_positions thy =
  let
    val thy_name = Context.theory_long_name thy
    val space = Consts.space_of (Sign.consts_of thy)
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        let val entry = Name_Space.the_entry space name
        in if #theory_long_name entry = thy_name
           then SOME (Constant_Key (name, #pos entry), name, #pos entry)
           else NONE
        end)
  end

(* Helper: Get theorems with positions from theory.
   Returns (display_name, position, thm) triples — no key yet,
   because keys are assigned after serial registration. *)
fun get_theorems_with_positions thy =
  let
    val parent_facts = map Global_Theory.facts_of (Theory.parents_of thy)
    val facts = Global_Theory.facts_of thy
    val space = Facts.space_of facts
    val facts_list = Facts.dest_static false parent_facts facts
  in
    facts_list |> maps (fn (name, thms) =>
      let val pos = #pos (Name_Space.the_entry space name)
      in thms |> map_index (fn (idx, thm) =>
        let
          val display_name = Thm_Name.print (name, if length thms = 1 then 0 else idx + 1)
        in
          (display_name, pos, thm)
        end)
      end)
  end

(* Get the names of datatype-related constants that should be preserved
   (constructors, selectors, case, map, pred, rel, sets) *)
fun preserved_datatype_const_names (thy : theory) : string list =
  let
    val thy_name = Context.theory_long_name thy
    val type_space = Sign.type_space thy
    val local_type_names = Name_Space.get_names type_space
      |> filter (fn name =>
          #theory_long_name (Name_Space.the_entry type_space name) = thy_name)
    val const_name = fst o Term.dest_Const
    fun ctr_sugar_names T =
      case Ctr_Sugar.ctr_sugar_of_global thy T of
        NONE => []
      | SOME cs =>
          map const_name (#ctrs cs)
          @ maps (map const_name) (#selss cs)
          @ [const_name (#casex cs)]
    fun bnf_names T =
      case BNF_Def.bnf_of_global thy T of
        NONE => []
      | SOME bnf =>
          [const_name (BNF_Def.map_of_bnf bnf),
           const_name (BNF_Def.pred_of_bnf bnf),
           const_name (BNF_Def.rel_of_bnf bnf)]
          @ map const_name (BNF_Def.sets_of_bnf bnf)
  in
    maps (fn T => ctr_sugar_names T @ bnf_names T) local_type_names
  end

(* Filter datatype infrastructure constants and return the infra predicate.
   Pre-BNF intermediate constants (e.g. pre_list.* ) are infrastructure.
   Class instance constants (e.g. equal_list_inst.* ) are infrastructure.
   Other type-prefixed constants are infrastructure unless in the preserved set. *)
fun gen_is_ADT_infra (thy : theory)
    : {is_infra_const: string -> bool, is_infra_thm: string -> bool} =
  let
    val thy_name = Context.theory_long_name thy
    val type_space = Sign.type_space thy
    val local_type_names = Name_Space.get_names type_space
      |> filter (fn name =>
          #theory_long_name (Name_Space.the_entry type_space name) = thy_name)
    val preserved_set = Strhashtab.make_set (preserved_datatype_const_names thy)
    val pre_prefixes = map_filter (fn T =>
      case BNF_FP_Def_Sugar.fp_sugar_of_global thy T of
        NONE => NONE
      | SOME fp =>
          let val map_name = fst (Term.dest_Const (BNF_Def.map_of_bnf (#pre_bnf fp)))
          in SOME (Long_Name.qualifier map_name ^ ".") end)
      local_type_names
    val inst_infixes = map (fn T => "_" ^ Long_Name.base_name T ^ "_inst.") local_type_names
    fun is_infra_const name =
      exists (fn pfx => String.isPrefix pfx name) pre_prefixes
      orelse exists (fn infix_ => String.isSubstring infix_ name) inst_infixes
      orelse (exists (fn T => String.isPrefix (T ^ ".") name) local_type_names
              andalso not (Strhashtab.defined preserved_set name))
    fun is_infra_thm display_name =
      exists (fn T =>
        String.isPrefix (T ^ ".") display_name
        andalso let val suffix = fst (parse_thm_name
                      (String.extract (display_name, size T + 1, NONE)))
                in String.isSuffix "_def" suffix
                   orelse suffix = "splits"
                   orelse String.isPrefix "split_" suffix end)
        local_type_names
  in
    {is_infra_const = is_infra_const, is_infra_thm = is_infra_thm}
  end

(* Filter internal theorems that don't need interpretation:
   - theorems with [code] attribute
   - theorems whose propositions reference Lifting/BNF_Def/Transfer or
     datatype infrastructure constants *)
fun filter_internal_thms (is_infra : string -> bool)
    (entries : ((string * Position.T * thm) * (int * int * bool)) list)
    : ((string * Position.T * thm) * (int * int * bool)) list =
  let
    val internal_prefixes = ["Lifting.", "BNF_Def.", "Transfer.", "BNF_Cardinal_Order_Relation.", "HOL.equal"]
    fun is_internal_constant (name, _) =
      exists (fn pfx => String.isPrefix pfx name) internal_prefixes
      orelse is_infra name
  in
    filter (fn ((_, _, thm), (_, _, has_code)) =>
      not has_code andalso not (Term.exists_Const is_internal_constant (Thm.prop_of thm)))
      entries
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
     #> REPL.trim_makrup
    end

(* Main function: interpret_file *)
fun interpret_file (thy : theory) : unit =
  let
    (* 1. Get theory information *)
    val thy_name = Context.theory_long_name thy
    val thy_file_path = get_theory_file_path thy
    val deps_longname = Theory.parents_of thy |> map Context.theory_long_name

    (* 2. Extract constants and theorems *)
    val const_entries = get_constants_with_positions thy
      (* (Constant_Key (name, pos), display_name, pos) *)
    val {is_infra_const, is_infra_thm} = gen_is_ADT_infra thy
    val const_entries = filter (fn (_, name, _) => not (is_infra_const name)) const_entries

    val raw_thm_entries = get_theorems_with_positions thy
      (* (display_name, pos, thm) *)

    (* 2b. Filter theorems whose names do not appear in the source file.
       The RPC returns (word_offset, goal_keyword_offset, has_code_attr) per name:
         word_offset >= 0        : name appears as a word at that byte offset
         goal_keyword_offset >= 0 : name follows a goal keyword (lemma/theorem/etc.)
         has_code_attr            : name follows a goal keyword with [code] attribute *)
    val thm_display_names = map #1 raw_thm_entries
    val name_check_results = Remote_Procedure_Calling.call_command
          check_theorem_name_in_file_cmd (thy_file_path, thm_display_names)
      handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
        (warning ("check_theorem_name_in_file RPC failed: " ^ message);
         map (K (0, ~1, false)) thm_display_names)
    val raw_thm_entries_with_offsets =
          (raw_thm_entries ~~ name_check_results)

    (* 2c. Remove internal theorems that don't need interpretation *)
    val raw_thm_entries_with_offsets = filter_internal_thms is_infra_const raw_thm_entries_with_offsets
    val raw_thm_entries_with_offsets = filter (fn ((name, _, _), _) =>
        not (is_infra_thm name)) raw_thm_entries_with_offsets

    (* 3. Register theorems and assign serial-based keys *)
    val transfer = Global_Theory.transfer_theories thy
    val thm_entries_with_keys = raw_thm_entries_with_offsets
      |> map (fn ((display_name, pos, thm), (word_off, goal_off, _)) =>
        let val s = Term_Serial_Index.serial_of_thm (transfer thm)
        in (Theorem_Key s, display_name, pos, word_off, goal_off) end)

    (* 4. Deduplicate theorems by serial key — keep earliest occurrence per serial.
       Prefer entries that follow a goal keyword; among those, pick the earliest.
       Fall back to earliest word occurrence otherwise. *)
    val best_serials : (string * Position.T * int * int) Inthashtab.table = Inthashtab.empty 64
    val _ = thm_entries_with_keys
      |> List.app (fn (Theorem_Key s, name, pos, word_off, goal_off) =>
        case Inthashtab.lookup best_serials s of
          NONE => Inthashtab.update best_serials (s, (name, pos, word_off, goal_off))
        | SOME (_, _, prev_word_off, prev_goal_off) =>
            let val dominated =
              if goal_off >= 0 andalso prev_goal_off < 0 then true
              else if prev_goal_off >= 0 andalso goal_off < 0 then false
              else if goal_off >= 0 then goal_off < prev_goal_off
              else word_off < prev_word_off
            in if dominated
               then Inthashtab.update best_serials (s, (name, pos, word_off, goal_off))
               else ()
            end)
    val deduped_thm_entries = Inthashtab.fold
      (fn (s, (name, pos, _, goal_off)) => fn acc =>
        (Theorem_Key s, name, pos, goal_off) :: acc)
      best_serials []

    (* 5. Build entries list, sorted by source position.
       For theorems not introduced by a goal keyword, include their pretty-printed
       proposition so the interpreter can see what they state. *)
    val ctxt = Proof_Context.init_global thy
    val pp = string_of_term ctxt

    fun key_kind_ord (Constant_Key _, Theorem_Key _) = LESS
      | key_kind_ord (Theorem_Key _, Constant_Key _) = GREATER
      | key_kind_ord _ = EQUAL
    fun entry_ord (e1, e2) =
      (case int_ord (the_default 0 (Position.line_of (#3 e1)),
                     the_default 0 (Position.line_of (#3 e2))) of
        EQUAL => (case key_kind_ord (#1 e1, #1 e2) of
          EQUAL => string_ord (#2 e1, #2 e2)
        | ord => ord)
      | ord => ord)
    val all_entries_with_pos =
      (map (fn (key, name, pos) => (key, name, pos, ~1)) const_entries
       @ deduped_thm_entries)
      |> sort entry_ord

    val entries = all_entries_with_pos |> map (fn (key, name, pos, goal_off) =>
      let val prop_str =
        case key of
          Theorem_Key s =>
            if goal_off < 0
            then (case Term_Serial_Index.thm_of_serial thy s of
                    SOME thm => pp (Thm.prop_of thm)
                  | NONE => "")
            else ""
        | _ => ""
        val line = the_default ~1 (Position.line_of pos)
      in (key, name, prop_str, line) end)

    val display_names = map #2 entries

    (* 6. Build query callback capturing theory context *)
    val query_cb = make_query_callback (Context.Theory thy) display_names
    val cmd = make_interpret_file_cmd query_cb

    (* 7. Call RPC command *)
    val rpc_arg = {
      file_path = thy_file_path,
      theory_longname = thy_name,
      deps_longname = deps_longname,
      entries = entries
    }
    val results = Remote_Procedure_Calling.call_command cmd rpc_arg
      handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
        error (String.concat ["Semantic interpretation RPC failed: ", message])

    (* 8. Store results — each result corresponds to an entry in all_entries_with_pos *)
    val _ = Synchronized.change STORE_lock (fn () =>
      (map2 (fn (key, _, pos, _) => fn result_opt =>
        case result_opt of
          SOME sem => EntryTab.update STORE (key, {semantics = sem, position = pos})
        | NONE => ())
      all_entries_with_pos results; ()))
  in
    ()
  end


end
