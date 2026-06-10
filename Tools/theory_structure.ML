signature THEORY_STRUCTURE = sig
  val resolve_theory : Context.generic -> string -> theory
  val get_theory_file_path : theory -> string
  val get_constants_with_positions : Context.generic -> (string (* full_name *) * Position.T) list
  val get_theorems_with_positions : Context.generic ->
    (string (* full_name *) * int (* idx *) * Position.T * thm * bool (* is_global *)) list
  val get_types_with_positions : Context.generic -> (string * Position.T) list
  val get_classes_with_positions : Context.generic -> (string * Position.T) list
  val get_locales_with_positions : Context.generic -> (string * Position.T) list
  val get_named_theorems_with_positions : Context.generic -> (string * Position.T) list
  val get_methods_with_positions : Context.generic -> (string * Position.T) list
  val parse_thm_name : string -> string * int option
  val get_introduction_rules_with_positions : Context.generic -> (string * Position.T * thm) list
  val get_elimination_rules_with_positions  : Context.generic -> (string * Position.T * thm) list
  val get_induction_rules_with_positions    : Context.generic -> (string * Position.T * thm) list
  val get_case_split_rules_with_positions   : Context.generic -> (string * Position.T * thm) list
  (* Schema guard for name-based induction/case-split detection: conclusion
     and some premise tail both end in the same schematic variable (possibly
     0-ary, for case-split rules whose conclusion is a bare Boolean). *)
  val has_rule_shape : thm -> bool
  val get_entities_with_positions : Context.generic -> Universal_Key.entity list ->
    (string * Position.T) list
  val get_theorems_with_positions_of : Context.generic -> Universal_Key.entity list ->
    (string * int * Position.T * thm * bool) list
end

structure Theory_Structure : THEORY_STRUCTURE = struct

fun resolve_theory (context : Context.generic) (name : string) : theory =
  let
    val thy = Context.theory_of context
    val ctxt = Context.proof_of context
    val thy_long = Context.theory_long_name thy
    val thy_short = Context.theory_base_name thy
  in
    (* Special case: the current theory may be unfinished,
       so Theory.check would reject it *)
    if name = thy_long orelse name = thy_short
    then thy
    else
      (* Try short name first, then long name *)
      (Theory.check {long = false} ctxt (name, Position.none)
       handle ERROR _ =>
         Theory.check {long = true} ctxt (name, Position.none))
  end

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

(* Get types with positions (local definitions only) *)
fun get_types_with_positions context =
  let
    val thy_name = Context.theory_long_name (Context.theory_of context)
    val space = Context.cases Sign.type_space Proof_Context.type_space context
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        let val entry = Name_Space.the_entry space name
        in if #theory_long_name entry = thy_name
           then SOME (name, #pos entry)
           else NONE
        end)
  end

(* Get classes with positions (local definitions only) *)
fun get_classes_with_positions context =
  let
    val thy_name = Context.theory_long_name (Context.theory_of context)
    val space = Context.cases Sign.class_space Proof_Context.class_space context
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        let val entry = Name_Space.the_entry space name
        in if #theory_long_name entry = thy_name
           then SOME (name, #pos entry)
           else NONE
        end)
  end

(* Get locales with positions (theory-level only, uses Context.theory_of) *)
fun get_locales_with_positions context =
  let
    val thy = Context.theory_of context
    val parent_locale_names =
      Theory.parents_of thy
      |> maps (fn parent => Name_Space.get_names (Locale.locale_space parent))
      |> Symtab.make_set
    val space = Locale.locale_space thy
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        if Symtab.defined parent_locale_names name then NONE
        else SOME (name, #pos (Name_Space.the_entry space name)))
  end

(* Get named_theorems collections with positions (local definitions only).
   Named_Theorems are registered as dynamic facts, so Facts.is_dynamic (cheap,
   no content forcing) prefilters candidates; Named_Theorems.get then confirms
   (only run on the few dynamic facts new to this theory). *)
fun get_named_theorems_with_positions context =
  let
    val ctxt = Context.cases Proof_Context.init_global I context
    val facts = Proof_Context.facts_of ctxt
    val space = Facts.space_of facts
    val parent_names =
      Theory.parents_of (Context.theory_of context)
      |> maps (fn parent => Name_Space.get_names (Facts.space_of (Global_Theory.facts_of parent)))
      |> Symtab.make_set
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        if Symtab.defined parent_names name then NONE
        else if Facts.is_dynamic facts name andalso can (Named_Theorems.get ctxt) name
        then SOME (name, #pos (Name_Space.the_entry space name))
        else NONE)
  end

(* Get proof methods with positions (local definitions only). *)
fun get_methods_with_positions context =
  let
    val space = Method.method_space context
    val parent_names =
      Theory.parents_of (Context.theory_of context)
      |> maps (fn parent => Name_Space.get_names (Method.method_space (Context.Theory parent)))
      |> Symtab.make_set
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        if Symtab.defined parent_names name then NONE
        else SOME (name, #pos (Name_Space.the_entry space name)))
  end

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

(* Get constants with positions from theory (local definitions only) *)
fun get_constants_with_positions ctxt =
  let
    val thy_name = Context.theory_long_name (Context.theory_of ctxt)
    val space = Consts.space_of (Context.cases Sign.consts_of Proof_Context.consts_of ctxt)
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        let val entry = Name_Space.the_entry space name
        in if #theory_long_name entry = thy_name
           then SOME (name, #pos entry)
           else NONE
        end)
  end

(* Get theorems with positions.
   Returns (full_name, index, position, thm) tuples.
   full_name is the fully qualified fact name from Facts.dest_static. *)
fun get_theorems_with_positions context =
  let
    val thy = Context.theory_of context
    val (parent_facts, facts) = Context.cases
      (fn thy => (map Global_Theory.facts_of (Theory.parents_of thy), Global_Theory.facts_of thy))
      (fn ctxt => ([Global_Theory.facts_of (Proof_Context.theory_of ctxt)], Proof_Context.facts_of ctxt))
      context
    val global_facts = Global_Theory.facts_of thy
    val is_global = Facts.defined global_facts
    val space = Facts.space_of facts
    val facts_list = Facts.dest_static false parent_facts facts
  in
    facts_list |> maps (fn (name, thms) =>
      let val pos = (case try (Name_Space.the_entry space) name of
              SOME entry => #pos entry | NONE => Position.none)
          val n = length thms
          val global = is_global name
      in thms |> map_index (fn (idx, thm) =>
          (name, if n = 1 then 0 else idx + 1, pos,
           Thm.transfer thy thm,
           global))
      end)
  end

(* Get introduction/elimination rules.
   Combines two sources:
   1. Classical reasoner rules (intro or elim/dest declarations, excluding
      the "extra" intro?/elim?/dest? kinds)
   2. Name-based heuristic: facts whose base name ends with I (intro) or E (elim),
      where the preceding character is not an uppercase letter.
   Uses Thm.get_name_hint for names; rules without name hints are skipped. *)
local

fun has_rule_suffix suffix name =
  let val base = Long_Name.base_name name
      val len = size base
  in len >= 2
     andalso String.sub (base, len - 1) = suffix
     andalso not (Char.isUpper (String.sub (base, len - 2)))
  end

fun get_rules_with_positions is_elim suffix context =
  let
    val thy = Context.theory_of context
    val thy_base = Context.theory_base_name thy
    val thy_prefix = thy_base ^ "."
    val ctxt = Context.cases Proof_Context.init_global I context
    val fact_space = Facts.space_of (Proof_Context.facts_of ctxt)
    val seen : unit Strhashtab.table = Strhashtab.empty 128

    (* Source 1: classical reasoner *)
    val cs_rules = Classical.dest_decls ctxt (fn (_, {kind, ...}) =>
      Bires.kind_is_elim kind = is_elim andalso not (Bires.kind_extra kind))
    val cs_results = map_filter (fn (thm, _) =>
      let val thm = Thm.transfer thy thm
      in if not (Thm.has_name_hint thm) then NONE
         else let val thm_name = Thm.get_name_hint thm
                  val name = Thm_Name.print thm_name
      in if Strhashtab.defined seen name then NONE
         else if not (String.isPrefix thy_prefix name) then NONE
         else (Strhashtab.update seen (name, ());
               (* the_entry is keyed on the bare fact name; the Thm_Name index
                  selects a theorem within a fact and is not a name-space key. *)
               let val pos = case try (Name_Space.the_entry fact_space) (#1 thm_name) of
                     SOME entry => #pos entry | NONE => Position.none
               in SOME (name, pos, thm) end)
      end end) cs_rules

    (* Source 2: name-based heuristic from facts *)
    val (parent_facts, facts) = Context.cases
      (fn thy => (map Global_Theory.facts_of (Theory.parents_of thy), Global_Theory.facts_of thy))
      (fn ctxt => ([Global_Theory.facts_of (Proof_Context.theory_of ctxt)], Proof_Context.facts_of ctxt))
      context
    val facts_list = Facts.dest_static false parent_facts facts
    val name_results = maps (fn (name, thms) =>
      if not (has_rule_suffix suffix name) then []
      else map_filter (fn thm =>
        let val thm = Thm.transfer thy thm
            val key = if Thm.has_name_hint thm then Thm_Name.print (Thm.get_name_hint thm) else name
        in if Strhashtab.defined seen key then NONE
           else (Strhashtab.update seen (key, ());
                 let val pos = case try (Name_Space.the_entry fact_space) name of
                       SOME entry => #pos entry | NONE => Position.none
                 in SOME (name, pos, thm) end)
        end) thms) facts_list
  in cs_results @ name_results end

in
val get_introduction_rules_with_positions = get_rules_with_positions false #"I"
val get_elimination_rules_with_positions  = get_rules_with_positions true #"E"
end

(* Get induction / case-split rules by unioning:
   1. Induct.dest_rules — theorems tagged with [induct] / [cases].
   2. Fact-space scan by name suffix, guarded by has_rule_shape. Needed
      because class-axiom rules (e.g. Orderings.wellorder_class.less_induct)
      are not registered in the [induct] table. *)
local

fun strip_trueprop (Const (\<^const_name>\<open>Trueprop\<close>, _) $ x) = x
  | strip_trueprop t = t

fun schematic_head t =
  case Term.strip_comb (strip_trueprop t) of
    (Var (v, _), _) => SOME v
  | _ => NONE

fun tail_concl t =
  let val body = Term.strip_all_body t
      val c = Logic.strip_imp_concl body
  in if c aconv t then c else tail_concl c end

fun name_ends_in_any suffixes name =
  let val base = Long_Name.base_name name
  in exists (fn s => String.isSuffix s base) suffixes end

fun has_rule_shape_impl thm =
  let
    val body = Term.strip_all_body (Thm.prop_of thm)
    val prems = Logic.strip_imp_prems body
    val concl = Logic.strip_imp_concl body
  in
    case schematic_head concl of
      NONE => false
    | SOME p =>
        exists (fn prem =>
          case schematic_head (tail_concl prem) of
            SOME q => p = q
          | NONE => false) prems
  end

fun get_induct_rules_with_positions selector suffixes context =
  let
    val thy = Context.theory_of context
    val thy_prefix = Context.theory_base_name thy ^ "."
    val ctxt = Context.cases Proof_Context.init_global I context
    val fact_space = Facts.space_of (Proof_Context.facts_of ctxt)
    val seen : unit Strhashtab.table = Strhashtab.empty 64
    fun pos_of name =
      case try (Name_Space.the_entry fact_space) name of
        SOME entry => #pos entry | NONE => Position.none

    val src1 = selector (Induct.dest_rules ctxt)
      |> map_filter (fn (_, thm) =>
          let val thm = Thm.transfer thy thm in
            if not (Thm.has_name_hint thm) then NONE
            else let val thm_name = Thm.get_name_hint thm
                     val name = Thm_Name.print thm_name in
              if Strhashtab.defined seen name
                 orelse not (String.isPrefix thy_prefix name) then NONE
              else (Strhashtab.update seen (name, ());
                    (* the name space is keyed on the bare fact name *)
                    SOME (name, pos_of (#1 thm_name), thm))
            end
          end)

    val (parent_facts, facts) = Context.cases
      (fn thy => (map Global_Theory.facts_of (Theory.parents_of thy), Global_Theory.facts_of thy))
      (fn ctxt => ([Global_Theory.facts_of (Proof_Context.theory_of ctxt)], Proof_Context.facts_of ctxt))
      context
    val src2 = Facts.dest_static false parent_facts facts
      |> maps (fn (name, thms) =>
          if not (name_ends_in_any suffixes name) then []
          else map_filter (fn thm =>
            let val thm = Thm.transfer thy thm
                val key = if Thm.has_name_hint thm then Thm_Name.print (Thm.get_name_hint thm) else name
            in if Strhashtab.defined seen key
                  orelse not (String.isPrefix thy_prefix key)
                  orelse not (has_rule_shape_impl thm) then NONE
               else (Strhashtab.update seen (key, ());
                     SOME (name, pos_of name, thm))
            end) thms)
  in src1 @ src2 end

in

val has_rule_shape = has_rule_shape_impl

val get_induction_rules_with_positions =
  get_induct_rules_with_positions
    (fn r => #type_induct r @ #pred_induct r) ["_induct", "_ind"]
val get_case_split_rules_with_positions =
  get_induct_rules_with_positions
    (fn r => #type_cases r @ #pred_cases r) ["_cases", "_case"]

end

(* Look up name and position for a list of entities *)
fun get_entities_with_positions context entities =
  let
    val thy = Context.theory_of context
    val const_space = Consts.space_of (Context.cases Sign.consts_of Proof_Context.consts_of context)
    val type_space = Context.cases Sign.type_space Proof_Context.type_space context
    val class_space = Context.cases Sign.class_space Proof_Context.class_space context
    val locale_space = Locale.locale_space thy
    val method_space = Method.method_space context
    val fact_space = Facts.space_of
      (Context.cases Global_Theory.facts_of Proof_Context.facts_of context)
    fun pos_of space name =
      case try (Name_Space.the_entry space) name of
        SOME entry => #pos entry | NONE => Position.none
    fun fact_pos name =
      let val (base, _) = parse_thm_name name
      in pos_of fact_space base end
    fun thm_name_pos thm =
      if Thm.has_name_hint thm
      then let val name = Thm_Name.print (Thm.get_name_hint thm) in (name, fact_pos name) end
      else error "Anonymous theorem (no name hint)"
  in
    map (fn Universal_Key.Constant name => (name, pos_of const_space name)
          | Universal_Key.Type name => (name, pos_of type_space name)
          | Universal_Key.Class name => (name, pos_of class_space name)
          | Universal_Key.Locale name => (name, pos_of locale_space name)
          | Universal_Key.Named_Theorems name => (name, fact_pos name)
          | Universal_Key.Method name => (name, pos_of method_space name)
          | Universal_Key.Named_Theorem name => (name, fact_pos name)
          | Universal_Key.Named_Introduction_Rule name => (name, fact_pos name)
          | Universal_Key.Named_Elimination_Rule name => (name, fact_pos name)
          | Universal_Key.Named_Induction_Rule name => (name, fact_pos name)
          | Universal_Key.Named_Case_Split_Rule name => (name, fact_pos name)
          | Universal_Key.Theorem thm => thm_name_pos thm
          | Universal_Key.Introduction_Rule thm => thm_name_pos thm
          | Universal_Key.Elimination_Rule thm => thm_name_pos thm
          | Universal_Key.Induction_Rule thm => thm_name_pos thm
          | Universal_Key.Case_Split_Rule thm => thm_name_pos thm
        ) entities
  end

(* Look up full theorem details for a list of entities.
   Only processes theorem-like entities; others are skipped. *)
fun get_theorems_with_positions_of context entities =
  let
    val thy = Context.theory_of context
    val facts = Context.cases Global_Theory.facts_of Proof_Context.facts_of context
    val global_facts = Global_Theory.facts_of thy
    val is_global = Facts.defined global_facts
    val fact_space = Facts.space_of facts
    fun pos_of name =
      case try (Name_Space.the_entry fact_space) name of
        SOME entry => #pos entry | NONE => Position.none
    fun from_thm thm =
      let val thm = Thm.transfer thy thm
      in if not (Thm.has_name_hint thm)
         then error "Anonymous theorem (no name hint)"
         else let val thm_name = Thm.get_name_hint thm
                  (* the fact space and Facts.defined are keyed on the bare fact name *)
              in [(Thm_Name.print thm_name, 0, pos_of (#1 thm_name), thm, is_global (#1 thm_name))] end
      end
    fun from_name name =
      case Facts.lookup context facts name of
        NONE => error ("Unknown fact: " ^ quote name)
      | SOME {thms, ...} =>
          let val pos = pos_of name
              val n = length thms
              val global = is_global name
          in map_index (fn (idx, thm) =>
              (name, if n = 1 then 0 else idx + 1, pos,
               Thm.transfer thy thm, global)) thms
          end
  in
    maps (fn Universal_Key.Theorem thm => from_thm thm
           | Universal_Key.Named_Theorem name => from_name name
           | _ => []) entities
  end

end
