signature THEORY_STRUCTURE = sig
  val resolve_theory : Context.generic -> string -> theory
  val get_theory_file_path : theory -> string
  val get_constants_with_positions : theory -> (string (* full_name *) * Position.T) list
  val get_theorems_with_positions : theory ->
    (string (* full_name *) * int (* idx *) * Position.T * thm) list
  val parse_thm_name : string -> string * int option
  val make_theory_info_callback : Context.generic -> Remote_Procedure_Calling.callback'
end

structure Theory_Structure : THEORY_STRUCTURE = struct

fun resolve_theory (context : Context.generic) (name : string) : theory =
  let
    val thy = Context.cases I Proof_Context.theory_of context
    val ctxt = Context.cases Proof_Context.init_global I context
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
fun get_constants_with_positions thy =
  let
    val thy_name = Context.theory_long_name thy
    val space = Consts.space_of (Sign.consts_of thy)
  in
    Name_Space.get_names space
    |> map_filter (fn name =>
        let val entry = Name_Space.the_entry space name
        in if #theory_long_name entry = thy_name
           then SOME (name, #pos entry)
           else NONE
        end)
  end

(* Get theorems with positions from theory.
   Returns (full_name, index, position, thm) tuples.
   full_name is the fully qualified fact name from Facts.dest_static. *)
fun get_theorems_with_positions thy =
  let
    val parent_facts = map Global_Theory.facts_of (Theory.parents_of thy)
    val facts = Global_Theory.facts_of thy
    val space = Facts.space_of facts
    val facts_list = Facts.dest_static false parent_facts facts
  in
    facts_list |> maps (fn (name, thms) =>
      let val pos = #pos (Name_Space.the_entry space name)
          val n = length thms
      in thms |> map_index (fn (idx, thm) =>
          (name, if n = 1 then 0 else idx + 1, pos, thm))
      end)
  end

fun make_theory_info_callback (context : Context.generic)
    : Remote_Procedure_Calling.callback' =
  let
    fun theory_info name =
      let val t = resolve_theory context name
          val file_path = get_theory_file_path t
      in (Context.theory_long_name t, file_path) end
  in
    Remote_Procedure_Calling.mk_callback {
      name = "Theory_Structure.theory_info",
      arg_schema = MessagePackBinIO.Unpack.unpackString,
      ret_schema = MessagePackBinIO.Pack.packPair
                     (MessagePackBinIO.Pack.packString, MessagePackBinIO.Pack.packString),
      function = theory_info,
      timeout = NONE
    }
  end

end
