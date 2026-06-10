signature SEMANTIC_STORE = sig
  type semantics = string (*in English*)
  type usage = {input: int, cache_write: int, cache_read: int, output: int, cost: real}
  val zero_usage : usage
  val add_usage : usage * usage -> usage

  (* Domain for semantic search: which entities to search over. *)
  datatype search_domain =
      ContextAll
        (* All entities of the requested kinds in the current context.
           Pattern filtering is applied by Python via ML callbacks. *)
    | ContextExtended of (Universal_Key.universal_key * string) list
        (* Like ContextAll, plus additional (key, name) pairs
           (e.g. local contextual theorems from a proof state). *)
    | Restricted of Universal_Key.universal_key list
        (* Search only within the given keys; no callback-based filtering. *)

  (* Look up the English semantic interpretation of an entity.
     If with_pretty is true, prepends the pretty-printed formal signature. *)
  val query_semantics : Context.generic -> Universal_Key.entity -> bool -> semantics option

  (* Interpret all uninterpreted ancestor theories of the current context,
     plus the current proof context if applicable. Calls Python via RPC
     to generate plain-English translations using a Claude agent. *)
  val interpret : bool -> Context.generic -> unit

  (* Remove all LMDB entries whose theory hashes are non-persistent
     (i.e., from non-loaded theories). Returns the number of entries deleted. *)
  val clean_wip : unit -> int

  (* Create a callback that Python can invoke to interpret a list of theories
     by name (short or long). Resolves names, skips already-interpreted ones. *)
  val make_interpret_theory_callback : Context_Callbacks.context_unpacker -> Remote_Procedure_Calling.callback'

  (* Config option: when true, Semantic_Vector_Store._auto_embed will
     automatically trigger semantic interpretation for missing theories. *)
  val auto_interpret_for_embedding : bool Config.T

  (* Config option: embedding model name used by Semantic_Vector_Store.
     Empty string means use the environment variable EMBEDDING_MODEL. *)
  val embedding_model : string Config.T

  (* Config option: reranker model name for re-ranking search results.
     Empty string means disabled (use env var RERANKER_MODEL or no reranking). *)
  val reranker_model : string Config.T

  (* Search k nearest entities by semantic similarity to a query string.
     Returns ((score, entity) list, warnings). Warnings include notices about
     undeclared free variables in term patterns. *)
  val query_knn : Context.generic ->
    {query_text: semantics, k: int, kinds: Universal_Key.entity_kind list,
     domain: search_domain,
     term_patterns: string list, type_patterns: string list,
     theories_include: string list, name_contains: string list,
     target_type: string} ->
    (real * Universal_Key.entity) list * string list

  (* Embed semantic interpretations into vectors for the given theories
     and all their unembedded ancestors, under the specified embedding model
     (empty string = use default). *)
  val embed_all_entities_in_theories : bool -> Context.generic -> theory list * string -> unit

  (* Embed a list of entities, skipping those already embedded. *)
  val embed_entities : Context.generic -> Universal_Key.entity list -> unit

  (* Check whether a theory's entities are all embedded under the given model. *)
  val is_thy_embedded : Context.generic -> string * string -> bool

  (* Interpret a list of entities, skipping those already interpreted. *)
  val interpret_entities : Context.generic -> Universal_Key.entity list -> unit

  (* Context callbacks with infra filters and theorem cache baked in.
     Computes infra filters once and returns all entity callbacks. *)
  val make_entity_callbacks : Context.generic -> Context_Callbacks.context_unpacker ->
    {constants: Remote_Procedure_Calling.callback',
     theorems: Remote_Procedure_Calling.callback',
     types: Remote_Procedure_Calling.callback',
     classes: Remote_Procedure_Calling.callback',
     locales: Remote_Procedure_Calling.callback',
     named_theorems: Remote_Procedure_Calling.callback',
     methods: Remote_Procedure_Calling.callback',
     introduction_rules: Remote_Procedure_Calling.callback',
     elimination_rules: Remote_Procedure_Calling.callback',
     induction_rules: Remote_Procedure_Calling.callback',
     case_split_rules: Remote_Procedure_Calling.callback'}

  val strip_trailing_begin : string -> string
end

structure Semantic_Store : SEMANTIC_STORE = struct

type semantics = string (*in English*)

datatype search_domain =
    ContextAll
  | ContextExtended of (Universal_Key.universal_key * string) list
  | Restricted of Universal_Key.universal_key list

type usage = {input: int, cache_write: int, cache_read: int, output: int, cost: real}
val zero_usage : usage = {input=0, cache_write=0, cache_read=0, output=0, cost=0.0}
fun add_usage (a : usage, b : usage) : usage =
  {input = #input a + #input b, cache_write = #cache_write a + #cache_write b,
   cache_read = #cache_read a + #cache_read b, output = #output a + #output b,
   cost = #cost a + #cost b}
fun fmt_usage ({input, cache_write, cache_read, output, cost} : usage) =
  String.concat [
    "input=", Int.toString input,
    " cache_write=", Int.toString cache_write,
    " cache_read=", Int.toString cache_read,
    " output=", Int.toString output,
    " cost=$", Real.fmt (StringCvt.FIX (SOME 4)) cost]

val auto_interpret_for_embedding = Attrib.setup_config_bool
  \<^binding>\<open>auto_interpret_for_embedding\<close> (K true)
val _ = Config.register_rpc_option "auto_interpret_for_embedding"
          (Config.bool_value auto_interpret_for_embedding)

val embedding_model = Attrib.setup_config_string
  \<^binding>\<open>embedding_model\<close> (K "")
val _ = Config.register_rpc_option "Semantic_Embedding.embedding_model"
          (Config.string_value embedding_model)

val reranker_model = Attrib.setup_config_string
  \<^binding>\<open>reranker_model\<close> (K "")
val _ = Config.register_rpc_option "Semantic_Embedding.reranker_model"
          (Config.string_value reranker_model)

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

val is_infra_session = Infra_Filter.is_infra_session

(* --- RPC protocol --- *)

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack

  val pack_entry_kind = Universal_Key.pack_entity_kind

  val pack_arg =
    fn {file_path, theory_longname, theory_key, entries} =>
    packTuple4
      (packString, packString, packBytes,
       packList (packTuple6 (pack_entry_kind, packString, packString, packInt, packBytes, packString)))
      (file_path, theory_longname, theory_key, entries)

  fun unpackUsage src =
    let val ((i, cw, cr, o_, c), src') = unpackTuple5 (unpackInt, unpackInt, unpackInt, unpackInt, unpackReal) src
    in ({input = i, cache_write = cw, cache_read = cr, output = o_, cost = c} : usage, src') end
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
      entries: (Universal_Key.entity * string * string * int * Word8Vector.vector * string) list},
     semantics option list * string list * usage * usage)
    Remote_Procedure_Calling.command = {
  name = "Semantic_Store.interpret_file",
  arg_schema = pack_arg,
  ret_schema = unpackTuple4 (unpackList (unpackOption unpackString),
                             unpackList unpackString,
                             unpackUsage, unpackUsage),
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

fun strip_trailing_begin src =
  let
    val len = size src
    fun rskip i = if i < 0 then i
      else if Char.isSpace (String.sub (src, i)) then rskip (i - 1) else i
    val last = rskip (len - 1)
  in
    if last >= 4 andalso String.substring (src, last - 4, 5) = "begin"
       andalso (last < 5 orelse String.sub (src, last - 5) = #"\n")
    then String.substring (src, 0, if last >= 5 then last - 5 else 0)
    else src
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
    | entity_kind_rank (Named_Theorems _) = 4
    | entity_kind_rank (Method _) = 4
    | entity_kind_rank (Introduction_Rule _) = 5
    | entity_kind_rank (Named_Introduction_Rule _) = 5
    | entity_kind_rank (Elimination_Rule _) = 6
    | entity_kind_rank (Named_Elimination_Rule _) = 6
    | entity_kind_rank (Induction_Rule _) = 7
    | entity_kind_rank (Named_Induction_Rule _) = 7
    | entity_kind_rank (Case_Split_Rule _) = 8
    | entity_kind_rank (Named_Case_Split_Rule _) = 8
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

(* Build (entity, name, prop_str, line, uk, prompt_hint) entries from
   (entity, name, pos, uk) tuples.  prompt_hint is extra context shown to the
   interpretation agent only (NOT stored as expr): currently the first members
   of a named_theorems collection.  Shared by interpret' and interpret_entities. *)
fun build_entries (context : Context.generic)
    (raw : (Universal_Key.entity * string * Position.T * Universal_Key.universal_key) list)
    : (Universal_Key.entity * string * string * int * Universal_Key.universal_key * string) list =
  let
    val thy = Context.theory_of context
    val ctxt = Context.cases Proof_Context.init_global I context
    val pp = string_of_term ctxt
    val pp_typ = string_of_typ ctxt
    fun source_at_pos pos =
      case PIDE_State.command_at_position pos of
        SOME (src, _, _) => strip_trailing_begin src
      | NONE => ""
    fun mk_prop_str (entity, pos) =
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
      | Universal_Key.Induction_Rule thm =>
          (pp (Thm.prop_of thm)
           handle ERROR msg =>
             (warning ("mk_prop_str: Induction_Rule pp failed: " ^ msg); raise ERROR msg))
      | Universal_Key.Case_Split_Rule thm =>
          (pp (Thm.prop_of thm)
           handle ERROR msg =>
             (warning ("mk_prop_str: Case_Split_Rule pp failed: " ^ msg); raise ERROR msg))
      | Universal_Key.Constant cname =>
          (let val T = Consts.the_constraint (Sign.consts_of thy) cname
           in pp_typ T end
           handle TYPE _ => "" | ERROR _ => "")
      | Universal_Key.Type _ => source_at_pos pos
      | Universal_Key.Class _ => source_at_pos pos
      | Universal_Key.Locale _ => source_at_pos pos
      | Universal_Key.Named_Theorems _ => source_at_pos pos
      | Universal_Key.Method _ => source_at_pos pos
      | _ => ""
    (* Extra prompt-only context (never stored as expr).  For a named_theorems,
       show up to 10 current member propositions so the agent can describe the
       collection's purpose; members are deliberately kept out of expr/key. *)
    fun mk_prompt_hint (Universal_Key.Named_Theorems nt_name, _) =
          (case try (Named_Theorems.get ctxt) nt_name of
             NONE => ""
           | SOME [] => ""
           | SOME thms =>
               let
                 val shown = Library.take 10 thms
                 fun line thm = "  - " ^ (pp (Thm.prop_of thm) handle ERROR _ => "?")
                 val more = if length thms > 10
                            then ["  \226\128\166 (" ^ string_of_int (length thms - 10) ^ " more)"]
                            else []
               in
                 "Current members (context only \226\128\148 do NOT list them in your answer):\n"
                 ^ cat_lines (map line shown @ more)
               end)
      | mk_prompt_hint _ = ""
  in
    raw |> map (fn (entity, name, pos, uk) =>
      (entity, name, mk_prop_str (entity, pos), the_default 999999999 (Position.line_of pos),
       uk, mk_prompt_hint (entity, pos)))
  end

fun extract_intro_rules ctx =
  let val cs = Classical.rep_cs (Classical.get_cs ctx)
  in map #1 (Item_Net.content (#safeIs cs) @ Item_Net.content (#unsafeIs cs)) end

fun extract_elim_rules ctx =
  let val cs = Classical.rep_cs (Classical.get_cs ctx)
  in map #1 (Item_Net.content (#safeEs cs) @ Item_Net.content (#unsafeEs cs)) end

(* Query-time counterpart of Theory_Structure.get_induct_rules_with_positions:
   both sites surface the same rule set (Induct.dest_rules + suffix-matched
   fact scan with has_rule_shape). Specific to induction / case-split rules. *)
local
fun name_ends_in_any suffixes name =
  let val base = Long_Name.base_name name
  in exists (fn s => String.isSuffix s base) suffixes end

fun extract_induct_or_cases_rules selector suffixes ctx =
  let
    val thy = Context.theory_of ctx
    val ctxt = Context.cases Proof_Context.init_global I ctx
    val seen : unit Strhashtab.table = Strhashtab.empty 64

    val src1 = selector (Induct.dest_rules ctxt)
      |> map_filter (fn (_, thm) =>
          let val thm = Thm.transfer thy thm in
            if not (Thm.has_name_hint thm) then SOME thm
            else let val n = Thm.get_name_hint thm in
              if Strhashtab.defined seen n then NONE
              else (Strhashtab.update seen (n, ()); SOME thm)
            end
          end)

    (* Scan ALL visible facts (including those inherited from parent theories),
       not only facts new to the current theory. Many useful induction rules
       (e.g. wellorder_class.less_induct, nat_less_induct) are NOT registered
       in Induct.dest_rules and live in parent theories — restricting to
       current-theory-only facts via `Facts.dest_static false parent_facts`
       silently drops them. The `seen` table further down already prevents
       double-counting against src1's names. *)
    val facts = Context.cases Global_Theory.facts_of Proof_Context.facts_of ctx
    val src2 = Facts.dest_static false [] facts
      |> maps (fn (name, thms) =>
          if not (name_ends_in_any suffixes name) then []
          else map_filter (fn thm =>
            let val thm = Thm.transfer thy thm
                val k = if Thm.has_name_hint thm then Thm.get_name_hint thm else name
            in if Strhashtab.defined seen k
                  orelse not (Theory_Structure.has_rule_shape thm) then NONE
               else (Strhashtab.update seen (k, ()); SOME thm)
            end) thms)
  in src1 @ src2 end
in

val extract_induct_rules =
  extract_induct_or_cases_rules
    (fn r => #type_induct r @ #pred_induct r) ["_induct", "_ind"]
val extract_case_split_rules =
  extract_induct_or_cases_rules
    (fn r => #type_cases r @ #pred_cases r) ["_cases", "_case"]

end

(* --- Theorem cache: pre-computed non-infra theorems with universal keys --- *)

structure Thm_Cache = Theory_Data(struct
  type T = {
    entries: Context_Callbacks.cached_thm_entry list,
    facts: Facts.T
  }
  val empty = {entries = [], facts = Facts.empty}
  fun merge ({entries = e1, facts = f1}, {entries = e2, facts = f2}) =
    let
      val seen = Bytehashtab.empty (length e1 + length e2)
      fun add entry acc =
        if Bytehashtab.defined seen (#uk entry) then acc
        else (Bytehashtab.update seen (#uk entry, ()); entry :: acc)
    in {entries = fold add e1 (fold add e2 []),
        facts = Facts.merge (f1, f2)} end
end)

(* Process a list of (name, thm list) facts into cached_thm_entry list.
   Filters out concealed, hidden, and infra theorems; computes universal keys. *)
fun process_facts_into_cache (context : Context.generic)
    (is_infra_thm : string * thm -> bool)
    (facts_to_process : (string * thm list) list)
    : Context_Callbacks.cached_thm_entry list =
  let
    val thy = Context.theory_of context
    val all_facts = Context.cases Global_Theory.facts_of Proof_Context.facts_of context
    val fact_space = Facts.space_of all_facts
  in
    maps (fn (name, thms) =>
      let val entry_opt = try (Name_Space.the_entry fact_space) name
      in case entry_opt of
           NONE => []
         | SOME e =>
             if #concealed e orelse
                Long_Name.is_hidden (Name_Space.intern fact_space name)
             then []
             else let
               val ethy = #theory_long_name e
               val epos = Context_Callbacks.entry_def_pos e
             in map_filter (fn thm =>
               let val thm' = Thm.transfer thy thm
               in if is_infra_thm (name, thm') then NONE
                  else SOME {
                    uk = Universal_Key.key_of_theorem' NONE context thm',
                    name = name, thy_name = ethy,
                    pos = epos,
                    thm = thm'
                  } : Context_Callbacks.cached_thm_entry option
               end) thms
             end
      end) facts_to_process
  end

(* Dedup new_entries against existing entries by universal key. *)
fun dedup_against existing new_entries =
  let
    val seen = Bytehashtab.empty (length existing)
    val _ = List.app (fn e => Bytehashtab.update seen (#uk e, ())) existing
  in filter (fn e => not (Bytehashtab.defined seen (#uk e))) new_entries end

(* Convenience callbacks with infra filters and theorem cache baked in.
   Processes any uncached facts at creation time so callbacks see 0 uncached. *)
fun make_entity_callbacks (init_context: Context.generic) (ctxt_unpack: Context_Callbacks.context_unpacker) =
  let (* Populate coercion-transparent matching ignore set (idempotent). *)
      val _ = List.app (fn n => Strhashtab.update Context_Callbacks.coercion_ignores (n, ())) [
        "Nat.semiring_1_class.of_nat",
        "Int.ring_1_class.of_int",
        "Rat.field_char_0_class.of_rat",
        "Real_Vector_Spaces.of_real",
        "Rings.zero_neq_one_class.of_bool",
        "Extended_Nat.enat",
        "Extended_Real.ereal.ereal",
        "Extended_Nonnegative_Real.ennreal"]
      val {is_infra_const, is_infra_thm, is_infra_induct_thm,
           is_infra_type, is_infra_class, is_infra_locale,
           is_infra_named_theorems, is_infra_method} =
        Infra_Filter.gen_infra_filters init_context
      val {entries = cached_entries, facts = cached_facts} =
        Thm_Cache.get (Context.theory_of init_context)
      (* Process uncached facts now, so callbacks see 0 uncached *)
      val current_facts = Context.cases Global_Theory.facts_of Proof_Context.facts_of init_context
      val uncached_facts = Facts.dest_static false [cached_facts] current_facts
      val combined_entries =
        if null uncached_facts then cached_entries
        else let
          val new_entries = process_facts_into_cache init_context is_infra_thm uncached_facts
          val unique_new = dedup_against cached_entries new_entries
        in cached_entries @ unique_new end
      val thm_cache = SOME (combined_entries, current_facts)
  in {
    constants = Context_Callbacks.make_constants_callback (SOME (not o is_infra_const)) ctxt_unpack,
    theorems = Context_Callbacks.make_theorems_callback (SOME (not o is_infra_thm)) thm_cache ctxt_unpack,
    types = Context_Callbacks.make_types_callback (SOME (not o is_infra_type)) ctxt_unpack,
    classes = Context_Callbacks.make_classes_callback (SOME (not o is_infra_class)) ctxt_unpack,
    locales = Context_Callbacks.make_locales_callback (SOME (not o is_infra_locale)) ctxt_unpack,
    named_theorems = Context_Callbacks.make_named_theorems_callback (SOME (not o is_infra_named_theorems)) ctxt_unpack,
    methods = Context_Callbacks.make_methods_callback (SOME (not o is_infra_method)) ctxt_unpack,
    introduction_rules = Context_Callbacks.make_introduction_rules_callback
      extract_intro_rules (SOME (not o is_infra_thm)) ctxt_unpack,
    elimination_rules = Context_Callbacks.make_elimination_rules_callback
      extract_elim_rules (SOME (not o is_infra_thm)) ctxt_unpack,
    induction_rules = Context_Callbacks.make_induction_rules_callback
      extract_induct_rules (SOME (not o is_infra_induct_thm)) ctxt_unpack,
    case_split_rules = Context_Callbacks.make_case_split_rules_callback
      extract_case_split_rules (SOME (not o is_infra_thm)) ctxt_unpack
  } end

fun interpret' (context : Context.generic) : usage * usage =
  let
    (* 1. Get theory information *)
    val thy = Context.theory_of context
    val thy_name = Context.theory_long_name thy
    val thy_file_path = Theory_Structure.get_theory_file_path thy
    (* 2. Extract constants and theorems, filtering infrastructure *)
    val {is_infra_const, is_infra_thm, is_infra_type, is_infra_class, is_infra_locale,
         is_infra_named_theorems, is_infra_method, ...} =
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

    (* 5a'. Extract named_theorems collections and proof methods *)
    val named_theorems_entries = Theory_Structure.get_named_theorems_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_named_theorems name then NONE
          else let val entity = Universal_Key.Named_Theorems name
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    val method_entries = Theory_Structure.get_methods_with_positions context
      |> map_filter (fn (name, pos) =>
          if is_infra_method name then NONE
          else let val entity = Universal_Key.Method name
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

    (* 5c. Extract induction and case-split rules *)
    val induct_entries = Theory_Structure.get_induction_rules_with_positions context
      |> map_filter (fn (name, pos, thm) =>
          if is_infra_thm (name, thm) then NONE
          else let val entity = Universal_Key.Induction_Rule thm
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    val case_split_entries = Theory_Structure.get_case_split_rules_with_positions context
      |> map_filter (fn (name, pos, thm) =>
          if is_infra_thm (name, thm) then NONE
          else let val entity = Universal_Key.Case_Split_Rule thm
               in SOME (entity, name, pos,
                    Universal_Key.key_of NONE context entity) end)

    (* 6. Build entries list, sorted by source position. *)
    val all_entries_with_pos =
      (const_entries @ deduped_thm_entries
       @ type_entries @ class_entries @ locale_entries
       @ named_theorems_entries @ method_entries
       @ intro_entries @ elim_entries
       @ induct_entries @ case_split_entries)
      |> sort entry_ord
    val entries = build_entries context all_entries_with_pos
  in
    if null entries then (zero_usage, zero_usage)
    else let
      (* 7. Build callbacks capturing context *)
      val su = Context_Callbacks.static_context_unpacker context
      val pu = PIDE_State.position_context_unpacker context
      val uk_cb = Universal_Key.make_universal_key_callback pu
      val theory_info_cb = Context_Callbacks.make_theory_long_name_and_path_callback su
      val desugar_cb = Explain_Term.make_desugar_callback context
      val {constants, theorems, types, classes, locales, named_theorems, methods,
           introduction_rules, elimination_rules,
           induction_rules, case_split_rules} = make_entity_callbacks context su
      val cmd = make_interpret_file_cmd [uk_cb, theory_info_cb, desugar_cb,
            constants, theorems, types, classes, locales, named_theorems, methods,
            introduction_rules, elimination_rules,
            induction_rules, case_split_rules]

      (* 8. Call RPC command *)
      val thy_key = Universal_Key.key_of_theory' thy
      val rpc_arg = {
        file_path = thy_file_path,
        theory_longname = thy_name,
        theory_key = thy_key,
        entries = entries
      }
      val (_, _, current, cumulative) =
        Remote_Procedure_Calling.call_command cmd rpc_arg
          handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
            error (String.concat ["Semantic interpretation RPC failed: ", message])
      val _ = tracing (String.concat [
        "[Semantic_Store] ", thy_name, ": ", fmt_usage current,
        " (historical: ", fmt_usage cumulative, ")"])
    in
      (current, cumulative)
    end
  end

fun collect_uninterpreted (force : bool) thy =
  let
    val seen = Inthashtab.empty 64
    val _ = List.app (Inthashtab.update seen o rpair ()) base_theory_ids
    fun collect thy acc =
      let val id = Context.theory_identifier thy
      in
        if (not force andalso is_interpreted thy) orelse Inthashtab.defined seen id
           orelse is_infra_session thy
        then acc
        else
          let val _ = Inthashtab.update seen (id, ())
              val acc' = fold collect (Theory.parents_of thy) acc
          in acc' @ [thy] end
      end
  in collect thy [] end

(* Scheduler state for DAG-parallel interpretation (one Isabelle session).
   See contrib/Semantic_Embedding plan: width N = Multithreading.max_threads (),
   blocking interpret' runs on off-pool Isabelle_Thread workers, concurrency bounded
   by `running < n`, a theory is admitted only after all its in-set ancestors are
   `done` (so its agent can read their interpretations from LMDB). Failure policy is
   HARD CRASH: the first error sets `failed`, all siblings are interrupted, and the
   exception is reraised out of `interpret` (the pipeline is resume-by-design). *)
type sched_state =
  {running: int, completed: int, done: Intset.T,
   failed: exn option, total: usage, hist: usage}

fun interpret_parallel (n : int) (theories : theory list) : usage * usage =
  let
    val collected_ids = Intset.make (map Context.theory_identifier theories)
    (* deps = transitive ancestors restricted to the collected set, so a skipped /
       already-interpreted intermediate never drops a real prerequisite edge. *)
    val nodes = map (fn thy =>
      {id = Context.theory_identifier thy,
       thy = thy,
       deps =
         Theory.ancestors_of thy
         |> map Context.theory_identifier
         |> filter (Intset.member collected_ids)
         |> Intset.make}) theories
    val total_count = length nodes

    val st = Synchronized.var "Semantic_Store.interpret_sched"
      ({running = 0, completed = 0, done = Intset.empty, failed = NONE,
        total = zero_usage, hist = zero_usage} : sched_state)

    fun admit {id = _, thy = _, deps} =
      Synchronized.guarded_access st
        (fn (s as {running, completed, done, failed, total, hist}) =>
          if is_some failed then SOME (false, s)
          else if running < n andalso Intset.subset (deps, done)
          then SOME (true, {running = running + 1, completed = completed,
                            done = done, failed = failed, total = total, hist = hist})
          else NONE)

    fun finish id (cur, cum) =
      Synchronized.change st
        (fn {running, completed, done, failed, total, hist} =>
          {running = running - 1, completed = completed + 1,
           done = Intset.insert id done, failed = failed,
           total = add_usage (total, cur), hist = add_usage (hist, cum)})

    fun set_failed exn =
      Synchronized.change st
        (fn {running, completed, done, failed, total, hist} =>
          {running = running, completed = completed, done = done,
           failed = SOME (the_default exn failed), total = total, hist = hist})

    (* `\<^try> ... catch` re-raises interrupts and only routes genuine errors to
       set_failed; bare `handle exn =>` would be a compile error here. *)
    fun worker (node as {id, thy, deps = _}) () =
      \<^try>\<open>
        if admit node
        then
          let val (cur, cum) = interpret' (Context.Theory thy)
              val _ = mark_interpreted thy
          in finish id (cur, cum) end
        else ()  (* failure already recorded: skip cleanly, no deadlock *)
      catch exn => set_failed exn\<close>

    val params =
      Isabelle_Thread.interrupts (Isabelle_Thread.params "Semantic_Store.interpret")
    val handles = map (fn node => Isabelle_Thread.fork params (worker node)) nodes

    (* Wait until a failure occurs or all theories have finished. *)
    val _ = Synchronized.guarded_access st
      (fn (s as {failed, completed, running = _, done = _, total = _, hist = _}) =>
        if is_some failed orelse completed = total_count
        then SOME ((), s) else NONE)

    val {failed, total, hist, running = _, completed = _, done = _} =
      Synchronized.value st
    val _ = if is_some failed
            then List.app Isabelle_Thread.interrupt_thread handles else ()
    val _ = List.app Isabelle_Thread.join handles
    val _ = case failed of SOME e => Exn.reraise e | NONE => ()
  in (total, hist) end

fun interpret (force : bool) (context : Context.generic) : unit =
  let
    val thy = Context.theory_of context
    val theories = collect_uninterpreted force thy
    val contexts = map Context.Theory theories
      @ (case context of Context.Proof _ => [context] | _ => [])
    val _ = if null contexts then () else
      tracing ("[Semantic_Store] interpreting " ^
               Int.toString (length contexts) ^ " theories")

    val n = Multithreading.max_threads ()
    val (total, hist) =
      if n <= 1
      then
        fold (fn ctx => fn (acc, hacc) =>
          let val (cur, cum) = interpret' ctx
              val _ = Context.cases mark_interpreted (K ()) ctx
          in (add_usage (acc, cur), add_usage (hacc, cum)) end)
        contexts (zero_usage, zero_usage)
      else
        let
          val (ttotal, thist) = interpret_parallel n theories
          (* A trailing Proof context (if any) runs last, sequentially; like the
             sequential path it is interpreted but not marked. *)
          val (ptotal, phist) =
            case context of
              Context.Proof _ => interpret' context
            | _ => (zero_usage, zero_usage)
        in (add_usage (ttotal, ptotal), add_usage (thist, phist)) end

    val _ = tracing (String.concat [
        "[Semantic_Store] total: ", fmt_usage total,
        " (historical: ", fmt_usage hist, ")"])
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

fun make_interpret_theory_callback (ctxt_unpack : Context_Callbacks.context_unpacker)
    : Remote_Procedure_Calling.callback' =
  Remote_Procedure_Calling.mk_callback {
    name = "Semantic_Store.interpret_theories",
    arg_schema = MessagePackBinIO.Unpack.unpackPair (ctxt_unpack, MessagePackBinIO.Unpack.unpackList MessagePackBinIO.Unpack.unpackString),
    ret_schema = MessagePackBinIO.Pack.packUnit,
    function = fn (context, names) => mk_interpret_theory context names,
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
        val su = Context_Callbacks.static_context_unpacker context
        val pu = PIDE_State.position_context_unpacker context
        val uk_cb = Universal_Key.make_universal_key_callback pu
        val theory_info_cb = Context_Callbacks.make_theory_long_name_and_path_callback su
        val desugar_cb = Explain_Term.make_desugar_callback context
        val {constants, theorems, types, classes, locales, named_theorems, methods,
             introduction_rules, elimination_rules,
             induction_rules, case_split_rules} = make_entity_callbacks context su
        val cmd = make_interpret_file_cmd [uk_cb, theory_info_cb, desugar_cb,
              constants, theorems, types, classes, locales, named_theorems, methods,
              introduction_rules, elimination_rules,
              induction_rules, case_split_rules]
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

(* --- Theorem cache update hook --- *)

fun update_thm_cache thy =
  let
    val {entries = cached, facts = cached_facts} = Thm_Cache.get thy
    val current_facts = Global_Theory.facts_of thy
    val new_facts = Facts.dest_static false [cached_facts] current_facts
  in
    if null new_facts then NONE
    else let
      val t0 = Time.now ()
      val context = Context.Theory thy
      val {is_infra_thm, ...} = Infra_Filter.gen_infra_filters context
      val new_entries = process_facts_into_cache context is_infra_thm new_facts
      val unique_new = dedup_against cached new_entries
      val _ = tracing (String.concat [
        "[Semantic_Store] Thm_Cache updated: ", Int.toString (length unique_new), " new entries (",
        Int.toString (length cached + length unique_new), " total) in ",
        Int.toString (Time.toMilliseconds (Time.- (Time.now (), t0))), "ms"])
    in
      SOME (Thm_Cache.put {
        entries = cached @ unique_new,
        facts = current_facts
      } thy)
    end
  end

val _ = Theory.setup (Theory.at_begin update_thm_cache #> Theory.at_end update_thm_cache)

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

fun pack_search_domain ContextAll =
      packPair (packInt, packUnit) (0, ())
  | pack_search_domain (ContextExtended extras) =
      packPair (packInt, packList (packPair (packBytes, packString))) (1, extras)
  | pack_search_domain (Restricted keys) =
      packPair (packInt, packList packBytes) (2, keys)

fun query_knn (context : Context.generic) =
  let
    val su = Context_Callbacks.static_context_unpacker context
    val {constants, theorems, types, classes, locales, named_theorems, methods,
         introduction_rules, elimination_rules,
         induction_rules, case_split_rules} = make_entity_callbacks context su
    val callbacks = [
      constants, theorems, types, classes, locales, named_theorems, methods,
      introduction_rules, elimination_rules,
      induction_rules, case_split_rules,
      make_interpret_theory_callback su,
      Context_Callbacks.make_theory_long_name_callback su,
      Config.make_config_lookup_callback su
    ]
    val cmd : (string * int * Universal_Key.entity_kind list *
               search_domain * string list * string list * string list * string list * string,
               (real * Universal_Key.entity) list * string list)
              Remote_Procedure_Calling.command = {
      name = "Semantic_Embedding.query_knn",
      arg_schema = packTuple9 (packString, packInt,
                     packList Universal_Key.pack_entity_kind',
                     pack_search_domain,
                     packList packString, packList packString,
                     packList packString, packList packString,
                     packString (* target_type, "" = no filter; only honoured by
                                   induction/case-split rule kinds *)),
      ret_schema = unpackPair (
                     unpackList (unpackPair (unpackReal, Universal_Key.unpack_entity)),
                     unpackList unpackString),
      callback = callbacks,
      timeout = NONE
    }
  in
    fn {query_text, k, kinds, domain, term_patterns, type_patterns,
        theories_include, name_contains, target_type} =>
      Remote_Procedure_Calling.call_command cmd
        (query_text, k, kinds, domain, term_patterns, type_patterns,
         theories_include, name_contains, target_type)
  end

fun is_thy_embedded (context : Context.generic) (theory_name : string, model : string) : bool =
  let
    val su = Context_Callbacks.static_context_unpacker context
    val cmd : (string * string, bool) Remote_Procedure_Calling.command = {
      name = "Semantic_Embedding.is_thy_embedded",
      arg_schema = packPair (packString, packString),
      ret_schema = unpackBool,
      callback = [Universal_Key.make_universal_key_callback su,
                  Config.make_config_lookup_callback su],
      timeout = NONE
    }
  in Remote_Procedure_Calling.call_command cmd (theory_name, model) end

fun embed_all_entities_in_theories (force : bool) (context : Context.generic) (theories : theory list, model : string) : unit =
  let
    val seen = Inthashtab.empty 64
    val _ = List.app (Inthashtab.update seen o rpair ()) base_theory_ids
    fun collect thy acc =
      let val id = Context.theory_identifier thy
      in
        if Inthashtab.defined seen id orelse is_infra_session thy then acc
        else
          let val _ = Inthashtab.update seen (id, ())
              val name = Context.theory_long_name thy
              val acc' = fold collect (Theory.parents_of thy) acc
          in if not force andalso is_thy_embedded context (name, model) then acc'
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
        val su = Context_Callbacks.static_context_unpacker context
        val {constants, theorems, types, classes, locales, named_theorems, methods,
             introduction_rules, elimination_rules,
             induction_rules, case_split_rules} = make_entity_callbacks context su
        val callbacks = [
          constants, theorems, types, classes, locales, named_theorems, methods,
          introduction_rules, elimination_rules,
          induction_rules, case_split_rules,
          Universal_Key.make_universal_key_callback su,
          Config.make_config_lookup_callback su
        ]
        val cmd : (string list * string * bool, unit) Remote_Procedure_Calling.command = {
          name = "Semantic_Embedding.embed_all_entities_in_theories",
          arg_schema = packTuple3 (packList packString, packString, packBool),
          ret_schema = unpackUnit,
          callback = callbacks,
          timeout = NONE
        }
      in Remote_Procedure_Calling.call_command cmd (unembedded, model, force) end
  end

fun embed_entities (context : Context.generic) (entities : Universal_Key.entity list) : unit =
  let
    val su = Context_Callbacks.static_context_unpacker context
    val keys = map (Universal_Key.key_of NONE context) entities
    val cmd : (Universal_Key.universal_key list, unit) Remote_Procedure_Calling.command = {
      name = "Semantic_Embedding.embed_entities",
      arg_schema = packList Universal_Key.pack_universal_key,
      ret_schema = unpackUnit,
      callback = [Config.make_config_lookup_callback su],
      timeout = NONE
    }
  in Remote_Procedure_Calling.call_command cmd keys end

end (*local MessagePack*)

end
