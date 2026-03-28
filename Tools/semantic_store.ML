signature SEMANTIC_STORE = sig
  type semantics = string (*in English*)

  (* Look up the English semantic interpretation of an entity.
     If with_pretty is true, prepends the pretty-printed formal signature. *)
  val query_semantics : Context.generic -> Universal_Key.entity -> bool -> semantics option

  (* Interpret all uninterpreted ancestor theories of the current context,
     plus the current proof context if applicable. Calls Python via RPC
     to generate plain-English translations using a Claude agent. *)
  val interpret : Context.generic -> unit

  (* Remove all LMDB entries whose theory hashes are non-persistent
     (i.e., from non-loaded theories). Returns the number of entries deleted. *)
  val clean_wip : unit -> int

  (* Create a callback that Python can invoke to interpret a list of theories
     by name (short or long). Resolves names, skips already-interpreted ones. *)
  val make_interpret_theory_callback : Context.generic -> Remote_Procedure_Calling.callback'

  (* Config option: when true, Semantic_Vector_Store._auto_embed will
     automatically trigger semantic interpretation for missing theories. *)
  val auto_interpret_for_embedding : bool Config.T

  (* Config option: embedding model name used by Semantic_Vector_Store.
     Empty string means use the environment variable EMBEDDING_MODEL. *)
  val embedding_model : string Config.T

  (* Search k nearest entities by semantic similarity to a query string.
     kinds filters entity types; NONE domain means all entities of those kinds.
     Returns (score, entity) pairs sorted by similarity. *)
  val query_knn : Context.generic ->
    semantics * int * Universal_Key.entity_kind list * Universal_Key.universal_key list option ->
    (real * Universal_Key.entity) list

  (* Embed semantic interpretations into vectors for the given theories
     and all their unembedded ancestors, under the specified embedding model
     (empty string = use default). *)
  val embed_all_entities_in_theories : Context.generic -> theory list * string -> unit

  (* Embed a list of entities, skipping those already embedded. *)
  val embed_entities : Context.generic -> Universal_Key.entity list -> unit

  (* Check whether a theory's entities are all embedded under the given model. *)
  val is_thy_embedded : Context.generic -> string * string -> bool

  (* Interpret a list of entities, skipping those already interpreted. *)
  val interpret_entities : Context.generic -> Universal_Key.entity list -> unit
end

structure Semantic_Store : SEMANTIC_STORE = struct

type semantics = string (*in English*)

val auto_interpret_for_embedding = Attrib.setup_config_bool
  \<^binding>\<open>auto_interpret_for_embedding\<close> (K true)
val _ = Config.register_rpc_option "auto_interpret_for_embedding"
          (Config.bool_value auto_interpret_for_embedding)

val embedding_model = Attrib.setup_config_string
  \<^binding>\<open>embedding_model\<close> (K "")
val _ = Config.register_rpc_option "Semantic_Embedding.embedding_model"
          (Config.string_value embedding_model)

structure Thm_Dedup = Hash_Table(struct
  type key = Universal_Key.universal_key
  val hash = Universal_Key.hasher
  val eq = op = : Universal_Key.universal_key * Universal_Key.universal_key -> bool
end)

(* --- Python RPC wrappers --- *)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

fun query_semantics (context : Context.generic) (entity : Universal_Key.entity)
    (with_pretty : bool) : string option =
  let
    val key = Universal_Key.key_of NONE context entity
  in
    Remote_Procedure_Calling.call_command
      {name = "Semantic_Store.query",
       arg_schema = packPair (packBytes, packBool),
       ret_schema = unpackOption unpackString,
       callback = [], timeout = NONE}
      (key, with_pretty)
  end

fun is_interpreted' (key : Universal_Key.universal_key) : bool =
  Remote_Procedure_Calling.call_command
    {name = "Semantic_Store.is_interpreted",
     arg_schema = packBytes, ret_schema = unpackBool,
     callback = [], timeout = NONE}
    key

fun mark_interpreted' (key : Universal_Key.universal_key) : unit =
  Remote_Procedure_Calling.call_command
    {name = "Semantic_Store.mark_interpreted",
     arg_schema = packBytes, ret_schema = unpackUnit,
     callback = [], timeout = NONE}
    key

fun is_interpreted (thy : theory) : bool =
  is_interpreted' (Universal_Key.key_of_theory' thy)

fun mark_interpreted (thy : theory) : unit =
  mark_interpreted' (Universal_Key.key_of_theory' thy)

fun clean_wip () : int =
  Remote_Procedure_Calling.call_command
    {name = "Semantic_Store.clean_wip",
     arg_schema = packUnit, ret_schema = unpackInt,
     callback = [], timeout = NONE}
    ()

end (*local RPC wrappers*)

(* val base_theory_keys = map (Universal_Key.key_of_theory (Context.Theory @{theory}))
       ["Pure", "Tools.Code_Generator", "HOL.Code_Evaluation"] *)
val base_theory_ids = map Context.theory_identifier
      [@{theory Code_Generator}, @{theory Pure}, @{theory Code_Evaluation}]

(* --- RPC protocol --- *)

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack

  val pack_entry_kind = Universal_Key.pack_entity_kind

  val pack_arg =
    fn {file_path, theory_longname, theory_key, entries} =>
    packTuple4
      (packString, packString, packBytes,
       packList (packTuple5 (pack_entry_kind, packString, packString, packInt, packBytes)))
      (file_path, theory_longname, theory_key, entries)
in

val check_theorem_name_in_file_cmd :
    (string * string list, (int * int) list) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.check_theorem_name_in_file",
  arg_schema = packPair (packString, packList packString),
  ret_schema = unpackList (unpackPair (unpackInt, unpackInt)),
  callback = [],
  timeout = NONE
}

fun make_interpret_file_cmd callbacks :
    ({file_path: string, theory_longname: string, theory_key: Universal_Key.universal_key,
      entries: (Universal_Key.entity * string * string * int * Word8Vector.vector) list},
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
  open Universal_Key
  fun entity_kind_rank (Class _) = 0
    | entity_kind_rank (Locale _) = 1
    | entity_kind_rank (Type _) = 2
    | entity_kind_rank (Constant _) = 3
    | entity_kind_rank (Introduction_Rule _) = 5
    | entity_kind_rank (Named_Introduction_Rule _) = 5
    | entity_kind_rank (Elimination_Rule _) = 6
    | entity_kind_rank (Named_Elimination_Rule _) = 6
    | entity_kind_rank _ = 10
in
fun key_kind_ord (k1, k2) = int_ord (entity_kind_rank k1, entity_kind_rank k2)
end
fun entry_ord (e1, e2) =
  (case int_ord (the_default 999999999 (Position.line_of (#3 e1)),
                 the_default 999999999 (Position.line_of (#3 e2))) of
    EQUAL => (case key_kind_ord (#1 e1, #1 e2) of
      EQUAL => string_ord (#2 e1, #2 e2)
    | ord => ord)
  | ord => ord)

val contains_cmd
    : (Universal_Key.universal_key list, bool list) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.contains",
  arg_schema = MessagePackBinIO.Pack.packList Universal_Key.pack_universal_key,
  ret_schema = MessagePackBinIO.Unpack.unpackList MessagePackBinIO.Unpack.unpackBool,
  callback = [],
  timeout = NONE
}

(* Build (entity, name, prop_str, line, uk) entries from (entity, name, pos, uk) tuples.
   Shared by interpret' and interpret_entities. *)
fun build_entries (context : Context.generic)
    (raw : (Universal_Key.entity * string * Position.T * Universal_Key.universal_key) list)
    : (Universal_Key.entity * string * string * int * Universal_Key.universal_key) list =
  let
    val thy = Context.theory_of context
    val ctxt = Context.cases Proof_Context.init_global I context
    val pp = string_of_term ctxt
    val pp_typ = string_of_typ ctxt
    fun mk_prop_str entity =
      case entity of
        Universal_Key.Theorem thm =>
          (pp (Thm.prop_of thm)
           handle ERROR msg =>
             (warning ("mk_prop_str: Theorem pp failed: " ^ msg); raise ERROR msg))
      | Universal_Key.Introduction_Rule thm =>
          (pp (Thm.prop_of thm)
           handle ERROR msg =>
             (warning ("mk_prop_str: Introduction_Rule pp failed: " ^ msg); raise ERROR msg))
      | Universal_Key.Elimination_Rule thm =>
          (pp (Thm.prop_of thm)
           handle ERROR msg =>
             (warning ("mk_prop_str: Elimination_Rule pp failed: " ^ msg); raise ERROR msg))
      | Universal_Key.Constant cname =>
          (let val T = Consts.the_constraint (Sign.consts_of thy) cname
           in pp_typ T end
           handle TYPE _ => "" | ERROR _ => "")
      | _ => ""
  in
    raw |> map (fn (entity, name, pos, uk) =>
      (entity, name, mk_prop_str entity, the_default 999999999 (Position.line_of pos), uk))
  end

(* Returns ((current input, output, cost), (historical input, output, cost)) *)
fun interpret' (context : Context.generic) : (int * int * real) * (int * int * real) =
  let
    (* 1. Get theory information *)
    val thy = Context.theory_of context
    val thy_name = Context.theory_long_name thy
    val thy_file_path = Theory_Structure.get_theory_file_path thy
    (* 2. Extract constants and theorems, filtering infrastructure *)
    val {is_infra_const, is_infra_thm, is_infra_type, is_infra_class, is_infra_locale} =
      Infra_Filter.gen_infra_filters context

    val const_entries = Theory_Structure.get_constants_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_const name then NONE
          else let val entity = Universal_Key.Constant name
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    val raw_thm_entries = Theory_Structure.get_theorems_with_positions context
      |> filter (fn (name, _, _, thm, _) => not (is_infra_thm (name, thm)))

    (* 2c. Build named entries and check which appear in the source file. *)
    val raw_thm_entries_named = raw_thm_entries
      |> map (fn (name, idx, pos, thm, _) =>
        (Thm_Name.print (name, idx), pos, thm))
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
           map (K (0, ~1)) file_match_names))
      (fn _ => map (K (0, ~1)) file_match_names)
      context
    val raw_thm_entries_with_offsets =
          (raw_thm_entries_named ~~ name_check_results)

    (* 3. Compute universal keys for theorems *)
    val thm_entries_with_uks = raw_thm_entries_with_offsets
      |> map (fn ((name, pos, thm), (word_off, goal_off)) =>
        let val uk = Universal_Key.key_of NONE context (Universal_Key.Theorem thm)
        in (uk, name, pos, thm, word_off, goal_off) end)

    (* 4. Deduplicate theorems by universal key *)
    val best_thms : (string * Position.T * thm * int * int) Thm_Dedup.table =
      Thm_Dedup.empty 1024
    val _ = thm_entries_with_uks
      |> List.app (fn (uk, name, pos, thm, word_off, goal_off) =>
        case Thm_Dedup.lookup best_thms uk of
          NONE => Thm_Dedup.update best_thms
            (uk, (name, pos, thm, word_off, goal_off))
        | SOME (_, _, _, prev_word_off, prev_goal_off) =>
            let val dominated =
              if goal_off >= 0 andalso prev_goal_off < 0 then true
              else if prev_goal_off >= 0 andalso goal_off < 0 then false
              else if goal_off >= 0 then goal_off < prev_goal_off
              else word_off < prev_word_off
            in if dominated
               then Thm_Dedup.update best_thms
                 (uk, (name, pos, thm, word_off, goal_off))
               else ()
            end)
    val deduped_thm_entries = Thm_Dedup.fold
      (fn (uk, (name, pos, thm, _, _)) => fn acc =>
        (Universal_Key.Theorem thm, name, pos, uk) :: acc)
      best_thms []

    (* 5. Extract types, classes, and locales *)
    val type_entries = Theory_Structure.get_types_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_type name then NONE
          else let val entity = Universal_Key.Type name
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    val class_entries = Theory_Structure.get_classes_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_class name then NONE
          else let val entity = Universal_Key.Class name
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    val locale_entries = Theory_Structure.get_locales_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_locale name then NONE
          else let val entity = Universal_Key.Locale name
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    (* 5b. Extract introduction and elimination rules *)
    val intro_entries = Theory_Structure.get_introduction_rules_with_positions context
      |> map_filter (fn (name, pos, thm) =>
          if is_infra_thm (name, thm) then NONE
          else let val entity = Universal_Key.Introduction_Rule thm
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    val elim_entries = Theory_Structure.get_elimination_rules_with_positions context
      |> map_filter (fn (name, pos, thm) =>
          if is_infra_thm (name, thm) then NONE
          else let val entity = Universal_Key.Elimination_Rule thm
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    (* 6. Build entries list, sorted by source position. *)
    val all_entries_with_pos =
      (const_entries @ deduped_thm_entries
       @ type_entries @ class_entries @ locale_entries
       @ intro_entries @ elim_entries)
      |> sort entry_ord
    val entries = build_entries context all_entries_with_pos

    (* 7. Build callbacks capturing context *)
    val uk_cb = Universal_Key.make_universal_key_callback context
    val theory_info_cb = Context_Callbacks.make_theory_long_name_and_path_callback context
    val cmd = make_interpret_file_cmd [uk_cb, theory_info_cb]

    (* 8. Call RPC command *)
    val thy_key = Universal_Key.key_of_theory' thy
    val rpc_arg = {
      file_path = thy_file_path,
      theory_longname = thy_name,
      theory_key = thy_key,
      entries = entries
    }
    val (_, _, (input_tokens, output_tokens, cost_usd),
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
  in
    ((input_tokens, output_tokens, cost_usd),
     (cum_input_tokens, cum_output_tokens, cum_cost_usd))
  end

fun collect_uninterpreted thy =
  let
    val seen = Inthashtab.empty 64
    val _ = List.app (Inthashtab.update seen o rpair ()) base_theory_ids
    fun collect thy acc =
      let val id = Context.theory_identifier thy
      in
        if is_interpreted thy orelse Inthashtab.defined seen id
        then acc
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
      tracing ("[Semantic_Store] interpreting " ^
               Int.toString (length contexts) ^ " theories")

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

fun mk_interpret_theory (context : Context.generic) (names : string list) : unit =
  let
    val theories = map (Theory_Hash.resolve_theory context) names
    val uninterpreted = filter (not o is_interpreted) theories
    val contexts = map Context.Theory uninterpreted
    val _ = if null contexts then () else
      tracing ("[Semantic_Store] mk_interpret_theory: interpreting " ^
               Int.toString (length contexts) ^ " theories")
  in
    List.app (fn ctx =>
      let val _ = interpret' ctx
          val _ = Context.cases mark_interpreted (K ()) ctx
      in () end)
    contexts
  end

fun make_interpret_theory_callback (context : Context.generic)
    : Remote_Procedure_Calling.callback' =
  Remote_Procedure_Calling.mk_callback {
    name = "Semantic_Store.interpret_theories",
    arg_schema = MessagePackBinIO.Unpack.unpackList MessagePackBinIO.Unpack.unpackString,
    ret_schema = MessagePackBinIO.Pack.packUnit,
    function = mk_interpret_theory context,
    timeout = NONE
  }

fun interpret_entities (context : Context.generic) (entities : Universal_Key.entity list) : unit =
  let
    (* 1. Compute universal keys *)
    val keyed = map (fn e => (e, Universal_Key.key_of NONE context e)) entities
    val keys = map #2 keyed

    (* 2. Batch check which are already interpreted *)
    val exists = Remote_Procedure_Calling.call_command contains_cmd keys
    val missing = map_filter (fn ((e, uk), ex) =>
      if ex then NONE else SOME (e, uk)) (keyed ~~ exists)
  in
    if null missing then ()
    else
      let
        val _ = tracing ("[Semantic_Store] interpret_entities: " ^
                  Int.toString (length missing) ^ " of " ^
                  Int.toString (length entities) ^ " entities need interpretation")
        (* 3. Get names and positions *)
        val names_pos = Theory_Structure.get_entities_with_positions context (map #1 missing)
        val raw_entries = map2 (fn (entity, uk) => fn (name, pos) =>
          (entity, name, pos, uk)) missing names_pos

        (* 4. Build entries and call interpret RPC *)
        val entries = build_entries context raw_entries
        val thy = Context.theory_of context
        val thy_name = Context.theory_long_name thy
        val thy_file_path = Theory_Structure.get_theory_file_path thy
        val uk_cb = Universal_Key.make_universal_key_callback context
        val theory_info_cb = Context_Callbacks.make_theory_long_name_and_path_callback context
        val cmd = make_interpret_file_cmd [uk_cb, theory_info_cb]
        val thy_key = Universal_Key.key_of_theory' thy
        val rpc_arg = {
          file_path = thy_file_path,
          theory_longname = thy_name,
          theory_key = thy_key,
          entries = entries
        }
        val _ = Remote_Procedure_Calling.call_command cmd rpc_arg
      in () end
  end

fun extract_intro_rules ctx =
  let val cs = Classical.rep_cs (Classical.get_cs ctx)
  in map #1 (Item_Net.content (#safeIs cs) @ Item_Net.content (#unsafeIs cs)) end

fun extract_elim_rules ctx =
  let val cs = Classical.rep_cs (Classical.get_cs ctx)
  in map #1 (Item_Net.content (#safeEs cs) @ Item_Net.content (#unsafeEs cs)) end

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

fun query_knn (context : Context.generic) =
  let
    val {is_infra_const, is_infra_thm, is_infra_type, is_infra_class, is_infra_locale} =
      Infra_Filter.gen_infra_filters context
    val callbacks = [
      Context_Callbacks.make_constants_callback (SOME (not o is_infra_const)) context,
      Context_Callbacks.make_theorems_callback (SOME (not o is_infra_thm)) context,
      Context_Callbacks.make_types_callback (SOME (not o is_infra_type)) context,
      Context_Callbacks.make_classes_callback (SOME (not o is_infra_class)) context,
      Context_Callbacks.make_locales_callback (SOME (not o is_infra_locale)) context,
      Context_Callbacks.make_introduction_rules_callback extract_intro_rules (SOME (not o is_infra_thm)) context,
      Context_Callbacks.make_elimination_rules_callback extract_elim_rules (SOME (not o is_infra_thm)) context,
      make_interpret_theory_callback context,
      Context_Callbacks.make_theory_long_name_callback context,
      Config.make_config_lookup_callback context
    ]
    val cmd : (string * int * Universal_Key.entity_kind list * Universal_Key.universal_key list option,
               (real * Universal_Key.entity) list)
              Remote_Procedure_Calling.command = {
      name = "Semantic_Embedding.query_knn",
      arg_schema = packTuple4 (packString, packInt,
                     packList Universal_Key.pack_entity_kind',
                     packOption (packList Universal_Key.pack_universal_key)),
      ret_schema = unpackList (unpackPair (unpackReal, Universal_Key.unpack_entity)),
      callback = callbacks,
      timeout = NONE
    }
  in
    fn (query_text, k, kinds, domain) =>
      Remote_Procedure_Calling.call_command cmd (query_text, k, kinds, domain)
  end

fun is_thy_embedded (context : Context.generic) (theory_name : string, model : string) : bool =
  let
    val cmd : (string * string, bool) Remote_Procedure_Calling.command = {
      name = "Semantic_Embedding.is_thy_embedded",
      arg_schema = packPair (packString, packString),
      ret_schema = unpackBool,
      callback = [Universal_Key.make_universal_key_callback context,
                  Config.make_config_lookup_callback context],
      timeout = NONE
    }
  in Remote_Procedure_Calling.call_command cmd (theory_name, model) end

fun embed_all_entities_in_theories (context : Context.generic) (theories : theory list, model : string) : unit =
  let
    val seen = Inthashtab.empty 64
    val _ = List.app (Inthashtab.update seen o rpair ()) base_theory_ids
    fun collect thy acc =
      let val id = Context.theory_identifier thy
      in
        if Inthashtab.defined seen id then acc
        else
          let val _ = Inthashtab.update seen (id, ())
              val name = Context.theory_long_name thy
              val acc' = fold collect (Theory.parents_of thy) acc
          in if is_thy_embedded context (name, model) then acc'
             else acc' @ [name]
          end
      end
    val unembedded = fold collect theories []
  in
    if null unembedded then ()
    else
      let
        val _ = tracing ("[Semantic_Store] embed_all_entities_in_theories: " ^
                   Int.toString (length unembedded) ^ " theories to embed")
        val {is_infra_const, is_infra_thm, is_infra_type, is_infra_class, is_infra_locale} =
          Infra_Filter.gen_infra_filters context
        val callbacks = [
          Context_Callbacks.make_constants_callback (SOME (not o is_infra_const)) context,
          Context_Callbacks.make_theorems_callback (SOME (not o is_infra_thm)) context,
          Context_Callbacks.make_types_callback (SOME (not o is_infra_type)) context,
          Context_Callbacks.make_classes_callback (SOME (not o is_infra_class)) context,
          Context_Callbacks.make_locales_callback (SOME (not o is_infra_locale)) context,
          Context_Callbacks.make_introduction_rules_callback extract_intro_rules (SOME (not o is_infra_thm)) context,
          Context_Callbacks.make_elimination_rules_callback extract_elim_rules (SOME (not o is_infra_thm)) context,
          Universal_Key.make_universal_key_callback context,
          Config.make_config_lookup_callback context
        ]
        val cmd : (string list * string, unit) Remote_Procedure_Calling.command = {
          name = "Semantic_Embedding.embed_all_entities_in_theories",
          arg_schema = packPair (packList packString, packString),
          ret_schema = unpackUnit,
          callback = callbacks,
          timeout = NONE
        }
      in Remote_Procedure_Calling.call_command cmd (unembedded, model) end
  end

fun embed_entities (context : Context.generic) (entities : Universal_Key.entity list) : unit =
  let
    val keys = map (Universal_Key.key_of NONE context) entities
    val cmd : (Universal_Key.universal_key list, unit) Remote_Procedure_Calling.command = {
      name = "Semantic_Embedding.embed_entities",
      arg_schema = packList Universal_Key.pack_universal_key,
      ret_schema = unpackUnit,
      callback = [Config.make_config_lookup_callback context],
      timeout = NONE
    }
  in Remote_Procedure_Calling.call_command cmd keys end

end (*local MessagePack*)

end
