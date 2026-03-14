signature SEMANTIC_STORE = sig
  val interpret : Context.generic -> unit
end

structure Semantic_Store : SEMANTIC_STORE = struct

(* --- Store data structures --- *)

datatype key = Constant_Key of string (*fully quantified name*) * Position.T
             | Theorem_Key  of serial (*serial from Term_Serial_Index*)
             | Type_Key     of string * Position.T
             | Class_Key    of string * Position.T
             | Locale_Key   of string * Position.T
type semantics = string (*in English*)
type entry = {
        semantics: semantics,
        position: Position.T
     }

structure EntryTab = Hash_Table(struct
  type key = key
  fun hash (Constant_Key (name, _)) = hash_string name
    | hash (Theorem_Key s) = hash_int s
    | hash (Type_Key (name, _)) = hash_string name
    | hash (Class_Key (name, _)) = hash_string name
    | hash (Locale_Key (name, _)) = hash_string name
  fun eq (Constant_Key (a1, p1), Constant_Key (a2, p2)) = a1 = a2 andalso p1 = p2
    | eq (Theorem_Key a, Theorem_Key b) = (a = b)
    | eq (Type_Key (a1, p1), Type_Key (a2, p2)) = a1 = a2 andalso p1 = p2
    | eq (Class_Key (a1, p1), Class_Key (a2, p2)) = a1 = a2 andalso p1 = p2
    | eq (Locale_Key (a1, p1), Locale_Key (a2, p2)) = a1 = a2 andalso p1 = p2
    | eq _ = false
end)

val STORE : entry EntryTab.table = EntryTab.empty 256
val STORE_lock = Synchronized.var "SEMANTIC_STORE" ()

(* --- RPC protocol --- *)

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack

  fun pack_entry_kind (Constant_Key _) = packInt 0
    | pack_entry_kind (Theorem_Key _) = packInt 1
    | pack_entry_kind (Type_Key _) = packInt 2
    | pack_entry_kind (Class_Key _) = packInt 3
    | pack_entry_kind (Locale_Key _) = packInt 4

  val pack_arg = fn {file_path, theory_longname, deps_longname, entries, context_info} =>
    packTuple5 (packString, packString, packList packString,
                packList (packTuple4 (pack_entry_kind, packString, packString, packInt)),
                packString)
               (file_path, theory_longname, deps_longname, entries, context_info)
in

fun make_query_callback (context : Context.generic) (current_names : string list)
    : Remote_Procedure_Calling.callback' =
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
        val (base_name, idx_opt) = Theory_Structure.parse_thm_name name
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

    fun query_ns_key make_key space kind_label name =
      (let
        val full_name = Name_Space.intern space name
        val _ = check_not_current full_name
        val ns_entry = Name_Space.the_entry space full_name
      in
        lookup_store (make_key (full_name, #pos ns_entry))
      end
      handle ERROR msg =>
        if String.isPrefix (String.concat ["Undefined ", kind_label, ": ", name]) msg
        then error (String.concat [kind_label, " ", name, " is not defined in this context"])
        else raise ERROR msg)

    val query_type   = query_ns_key Type_Key (Sign.type_space thy) "Type"
    val query_class  = query_ns_key Class_Key (Sign.class_space thy) "Class"
    val query_locale = query_ns_key Locale_Key (Locale.locale_space thy) "Locale"
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
    ({file_path: string, theory_longname: string,
      deps_longname: string list, entries: (key * string * string * int) list,
      context_info: string},
     semantics option list) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.interpret_file",
  arg_schema = pack_arg,
  ret_schema = unpackList (unpackOption unpackString),
  callback = callbacks,
  timeout = NONE
}

end (*local MessagePack*)

(* --- Utilities --- *)

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

(* --- Main orchestration --- *)

local
  fun key_kind_rank (Class_Key _) = 0
    | key_kind_rank (Locale_Key _) = 1
    | key_kind_rank (Type_Key _) = 2
    | key_kind_rank (Constant_Key _) = 3
    | key_kind_rank (Theorem_Key _) = 4
in
fun key_kind_ord (k1, k2) = int_ord (key_kind_rank k1, key_kind_rank k2)
end
fun entry_ord (e1, e2) =
  (case int_ord (the_default 0 (Position.line_of (#3 e1)),
                 the_default 0 (Position.line_of (#3 e2))) of
    EQUAL => (case key_kind_ord (#1 e1, #1 e2) of
      EQUAL => string_ord (#2 e1, #2 e2)
    | ord => ord)
  | ord => ord)

fun interpret (context : Context.generic) : unit =
  let
    (* 1. Get theory information *)
    val thy = Context.theory_of context
    val thy_name = Context.theory_long_name thy
    val thy_file_path = Theory_Structure.get_theory_file_path thy
    val deps_longname = Theory.parents_of thy |> map Context.theory_long_name

    (* 2. Extract constants and theorems, filtering infrastructure *)
    val {is_infra_const, is_infra_thm, is_infra_type, is_infra_class, is_infra_locale} =
      Infra_Filter.gen_infra_filters context

    val const_entries = Theory_Structure.get_constants_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_const name then NONE
          else SOME (Constant_Key (name, pos), name, pos))

    val raw_thm_entries = Theory_Structure.get_theorems_with_positions context
      |> filter (fn (name, _, _, thm) => not (is_infra_thm (name, thm)))

    (* 2c. Build named entries and check which appear in the source file.
       The RPC returns (word_offset, goal_keyword_offset, has_code_attr) per name:
         word_offset >= 0        : name appears as a word at that byte offset
         goal_keyword_offset >= 0 : name follows a goal keyword (lemma/theorem/etc.)
         has_code_attr            : name follows a goal keyword with [code] attribute
       For proof contexts, skip the RPC and return dummy results. *)
    val raw_thm_entries_named = raw_thm_entries
      |> map (fn (name, idx, pos, thm) =>
        (Thm_Name.print (name, idx), pos, thm))
    (* Strip theory prefix for source file matching — source uses short names *)
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

    (* 3. Register theorems and assign serial-based keys *)
    val transfer = Global_Theory.transfer_theories thy
    val thm_entries_with_keys = raw_thm_entries_with_offsets
      |> map (fn ((name, pos, thm), (word_off, goal_off, _)) =>
        let val s = Term_Serial_Index.serial_of_thm (transfer thm)
        in (Theorem_Key s, name, pos, word_off, goal_off) end)

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

    (* 5. Extract types, classes, and locales *)
    val type_entries = Theory_Structure.get_types_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_type name then NONE
          else SOME (Type_Key (name, pos), name, pos))

    val class_entries = Theory_Structure.get_classes_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_class name then NONE
          else SOME (Class_Key (name, pos), name, pos))

    val locale_entries = Theory_Structure.get_locales_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_locale name then NONE
          else SOME (Locale_Key (name, pos), name, pos))

    (* 6. Build entries list, sorted by source position.
       For theorems not introduced by a goal keyword, include their pretty-printed
       proposition so the interpreter can see what they state. *)
    val ctxt = Context.cases Proof_Context.init_global I context
    val pp = string_of_term ctxt

    val all_entries_with_pos =
      (map (fn (key, name, pos) => (key, name, pos, ~1)) const_entries
       @ deduped_thm_entries
       @ map (fn (key, name, pos) => (key, name, pos, ~1)) type_entries
       @ map (fn (key, name, pos) => (key, name, pos, ~1)) class_entries
       @ map (fn (key, name, pos) => (key, name, pos, ~1)) locale_entries)
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

    val names = map #2 entries

    (* 6b. Build context_info — empty for theory, assumptions for proof context *)
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
      deps_longname = deps_longname,
      entries = entries,
      context_info = context_info
    }
    val results = Remote_Procedure_Calling.call_command cmd rpc_arg
      handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
        error (String.concat ["Semantic interpretation RPC failed: ", message])

    (* 9. Store results — each result corresponds to an entry in all_entries_with_pos *)
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
