signature INFRA_FILTER = sig
  val gen_infra_filters : theory ->
    {is_infra_const: string -> bool,
     is_infra_thm: string * thm -> bool,
     is_infra_type: string -> bool,
     is_infra_class: string -> bool,
     is_infra_locale: string -> bool}
end

structure Infra_Filter : INFRA_FILTER = struct

fun gen_infra_filters (thy : theory)
    : {is_infra_const: string -> bool, is_infra_thm: string * thm -> bool,
       is_infra_type: string -> bool, is_infra_class: string -> bool,
       is_infra_locale: string -> bool} =
  let
    val thy_name = Context.theory_long_name thy
    val type_space = Sign.type_space thy
    val local_type_names = Name_Space.get_names type_space
      |> filter (fn name =>
          #theory_long_name (Name_Space.the_entry type_space name) = thy_name)

    (* --- ADT infrastructure constants --- *)

    (* Preserved datatype constant names: constructors, selectors, case, map, pred, rel, sets *)
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
    val preserved_set = Strhashtab.make_set
      (maps (fn T => ctr_sugar_names T @ bnf_names T) local_type_names)

    (* Pre-BNF intermediate constant prefixes *)
    val pre_prefixes = map_filter (fn T =>
      case BNF_FP_Def_Sugar.fp_sugar_of_global thy T of
        NONE => NONE
      | SOME fp =>
          let val map_name = fst (Term.dest_Const (BNF_Def.map_of_bnf (#pre_bnf fp)))
          in SOME (Long_Name.qualifier map_name ^ ".") end)
      local_type_names

    (* Class instance constant infixes *)
    val inst_infixes = map (fn T => "_" ^ Long_Name.base_name T ^ "_inst.") local_type_names

    fun is_infra_const name =
      exists (fn pfx => String.isPrefix pfx name) pre_prefixes
      orelse exists (fn infix_ => String.isSubstring infix_ name) inst_infixes
      orelse (exists (fn T => String.isPrefix (T ^ ".") name) local_type_names
              andalso not (Strhashtab.defined preserved_set name))

    (* --- ADT infrastructure theorem names --- *)

    fun is_ADT_infra_thm full_name =
      exists (fn T =>
        String.isPrefix (T ^ ".") full_name
        andalso let val suffix = String.extract (full_name, size T + 1, NONE)
                in String.isSuffix "_def" suffix
                 (* orelse suffix = "splits" orelse suffix = "split"
                   orelse String.isPrefix "split_" suffix *) end)
        local_type_names

    (* --- Inductive infrastructure theorem names --- *)

    val ctxt = Proof_Context.init_global thy
    val local_const_names = Theory_Structure.get_constants_with_positions thy |> map #1
    val inductive_names =
      local_const_names
      |> map_filter (fn name =>
          let val t = Const (name, Sign.the_const_type thy name)
          in (Inductive.the_inductive ctxt t; SOME name)
          end
          handle ERROR _ => NONE | Match => NONE | Empty => NONE
               | List.Empty => NONE | TYPE _ => NONE)
      |> Symtab.make_set

    (* Check if name or its predicate form (name ^ "p") is inductive.
       inductive_set registers the inductive info under the predicate form
       (e.g. "listsp"), not the set form ("lists"). *)
    fun check_inductive name =
      Symtab.defined inductive_names name
      orelse Symtab.defined inductive_names (name ^ "p")

    fun is_inductive_infra_thm full_name =
      let val base = Long_Name.base_name full_name
      in
        if String.isSuffix "_def" base
        then check_inductive
               (String.substring (full_name, 0, size full_name - 4))
        else if base = "cong"
        then check_inductive (Long_Name.qualifier full_name)
        else false
      end

    (* --- Proposition content check --- *)

    val internal_prefixes = ["Lifting.", "BNF_Def.", "Transfer.", "BNF_Cardinal_Order_Relation.", "HOL.equal"]
    fun is_internal_constant (name, _) =
      exists (fn pfx => String.isPrefix pfx name) internal_prefixes
      orelse is_infra_const name

    (* --- Combined theorem predicate --- *)

    fun is_infra_thm (full_name, thm) =
      is_ADT_infra_thm full_name
      orelse is_inductive_infra_thm full_name
      orelse Term.exists_Const is_internal_constant (Thm.prop_of thm)
  in
    {is_infra_const = is_infra_const, is_infra_thm = is_infra_thm,
     is_infra_type = K false, is_infra_class = K false, is_infra_locale = K false}
  end

end
