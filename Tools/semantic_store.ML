signature SEMANTIC_STORE = sig
  val interpret_file : theory -> unit
end

structure Semantic_Store : SEMANTIC_STORE = struct

datatype key = Constant_Key of string (*fully quantified name*)
             | Theorem_Key  of string (*fully quantified name*)
                             * int option (*the index if the name is associated with multiple
                                            theorems, staring from 0*)
type semantics = string (*in English*)
type entry = {
        semantics: semantics,
        position: Position.T
     }

fun key_ord (Constant_Key name1, Constant_Key name2) = string_ord (name1, name2)
  | key_ord (Constant_Key _, Theorem_Key _) = LESS
  | key_ord (Theorem_Key _, Constant_Key _) = GREATER
  | key_ord (Theorem_Key (name1, idx1), Theorem_Key (name2, idx2)) =
      (case string_ord (name1, name2) of
        EQUAL => option_ord int_ord (idx1, idx2)
      | ord => ord)

structure EntryTab = Table(type key = key val ord = key_ord);

val STORE = Synchronized.var "SEMANTIC_STORE" EntryTab.empty

local
  open MessagePackBinIO.Pack MessagePackBinIO.Unpack

  (* Pack key *)
  fun pack_key (Constant_Key name) =
        packPair (packInt, packString) (0, name)
    | pack_key (Theorem_Key (name, NONE)) =
        packPair (packInt, packString) (1, name)
    | pack_key (Theorem_Key (name, SOME idx)) =
        packTuple3 (packInt, packString, packInt) (1, name, idx)

  val pack_arg = fn {file_path, theory_longname, deps_longname, entries} =>
    packTuple4 (packString, packString, packList packString, packList (packPair (pack_key, packString)))
               (file_path, theory_longname, deps_longname, entries)
in

val interpret_file_cmd : (
        { file_path: string,
          theory_longname: string,
          deps_longname: string list,
          entries: (key * xstring (*displayed name*)) list
        },
        semantics option list
      ) Remote_Procedure_Calling.command = {
  name = "Semantic_Store.interpret_file",
  arg_schema = pack_arg,
  ret_schema = unpackList (unpackOption unpackString),
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

(* Helper: Get constants with positions from theory *)
fun get_constants_with_positions thy =
  let
    val {constants, ...} = Consts.dest (Sign.consts_of thy)
  in
    map (fn (name, _) =>
      (Constant_Key name, Long_Name.base_name name, Position.none)) constants
  end

(* Helper: Get theorems with positions from theory *)
fun get_theorems_with_positions thy =
  let
    val parent_facts = map Global_Theory.facts_of (Theory.parents_of thy)
    val facts = Global_Theory.facts_of thy
    val facts_list = Facts.dest_static false parent_facts facts
  in
    facts_list |> maps (fn (name, thms) =>
      thms |> map_index (fn (idx, _) =>
        let
          val key = Theorem_Key (name, if length thms = 1 then NONE else SOME idx)
          val display_name = Thm_Name.print (name, if length thms = 1 then 0 else idx + 1)
        in
          (key, display_name, Position.none)
        end))
  end

(* Main function: interpret_file *)
fun interpret_file (thy : theory) : unit =
  let
    (* 1. Get theory information *)
    val thy_name = Context.theory_long_name thy
    val thy_file_path = get_theory_file_path thy
    val deps_longname = Theory.parents_of thy |> map Context.theory_long_name

    (* 2. Extract constants and theorems with positions *)
    val const_entries = get_constants_with_positions thy
    val thm_entries = get_theorems_with_positions thy

    (* 3. Build entries list for RPC call *)
    val all_entries_with_pos = const_entries @ thm_entries
    val entries = map (fn (key, display_name, _) => (key, display_name)) all_entries_with_pos

    (* 4. Call RPC command *)
    val rpc_arg = {
      file_path = thy_file_path,
      theory_longname = thy_name,
      deps_longname = deps_longname,
      entries = entries
    }
    val results = Remote_Procedure_Calling.call_command interpret_file_cmd rpc_arg
      handle Remote_Procedure_Calling.Remote_Calling_Failure {message, ...} =>
        error ("Semantic interpretation RPC failed: " ^ message)

    (* 5. Combine results with keys and positions *)
    val store_updates =
      map2 (fn (key, _, pos) => fn result_opt =>
        Option.map (fn sem => (key, {semantics = sem, position = pos})) result_opt)
      all_entries_with_pos results
      |> List.mapPartial I

    (* 6. Update the store *)
    val _ = Synchronized.change STORE (fn store =>
      fold (fn (key, entry) => EntryTab.update (key, entry)) store_updates store)
  in
    ()
  end


end