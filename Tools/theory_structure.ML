signature THEORY_STRUCTURE = sig
  val resolve_theory : Context.generic -> string -> theory
  val get_theory_file_path : theory -> string
  val get_constants_with_positions : Context.generic -> (string (* full_name *) * Position.T) list
  val get_theorems_with_positions : Context.generic ->
    (string (* full_name *) * int (* idx *) * Position.T * thm * bool (* is_global *)) list
  val get_types_with_positions : Context.generic -> (string * Position.T) list
  val get_classes_with_positions : Context.generic -> (string * Position.T) list
  val get_locales_with_positions : Context.generic -> (string * Position.T) list
  val parse_thm_name : string -> string * int option
  val get_introduction_rules_with_positions : Context.generic -> (string * Position.T * thm) list
  val get_elimination_rules_with_positions  : Context.generic -> (string * Position.T * thm) list
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

(* Get introduction/elimination rules from the classical reasoner (local only).
   Uses Thm.derivation_name for names; rules with empty names are skipped. *)
local
fun get_rules_with_positions selector context =
  let
    val thy = Context.theory_of context
    val thy_base = Context.theory_base_name thy
    val thy_prefix = thy_base ^ "."
    val cs = Classical.rep_cs (Classical.get_cs context)
    val rules = selector cs
    val ctxt = Context.cases Proof_Context.init_global I context
    val fact_space = Facts.space_of (Proof_Context.facts_of ctxt)
    val seen : unit Strhashtab.table = Strhashtab.empty 128
  in
    map_filter (fn (thm, _, _) =>
      let val thm = Thm.transfer thy thm
          val name = Thm.derivation_name thm
      in if name = "" orelse Strhashtab.defined seen name then NONE
         else if not (String.isPrefix thy_prefix name) then NONE
         else (Strhashtab.update seen (name, ());
               let val pos = case try (Name_Space.the_entry fact_space) name of
                     SOME entry => #pos entry | NONE => Position.none
               in SOME (name, pos, thm) end)
      end) rules
  end
in
val get_introduction_rules_with_positions =
  get_rules_with_positions (fn cs =>
    Item_Net.content (#safeIs cs) @ Item_Net.content (#unsafeIs cs))
val get_elimination_rules_with_positions =
  get_rules_with_positions (fn cs =>
    Item_Net.content (#safeEs cs) @ Item_Net.content (#unsafeEs cs))
end

end
