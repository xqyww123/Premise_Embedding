signature INFRA_FILTER = sig
  val smart_size_of_term : term -> int
  val gen_infra_filters : Context.generic ->
    {is_infra_const: string -> bool,
     is_infra_thm: string * thm -> bool,
     is_infra_type: string -> bool,
     is_infra_class: string -> bool,
     is_infra_locale: string -> bool}
end

structure Infra_Filter : INFRA_FILTER = struct

val smart_size_of_term = smart_size_of_term (* from Performant_Isabelle_ML *)

fun gen_infra_filters (context : Context.generic)
    : {is_infra_const: string -> bool, is_infra_thm: string * thm -> bool,
       is_infra_type: string -> bool, is_infra_class: string -> bool,
       is_infra_locale: string -> bool} =
  let
    val type_space = Context.cases Sign.type_space Proof_Context.type_space context
    val consts = Context.cases Sign.consts_of Proof_Context.consts_of context
    val const_space = Consts.space_of consts
    val fact_space = Facts.space_of
      (Context.cases Global_Theory.facts_of Proof_Context.facts_of context)
    val class_space = Context.cases Sign.class_space Proof_Context.class_space context
    val locale_space = Locale.locale_space (Context.theory_of context)
    fun is_hidden space name =
      Long_Name.is_hidden (Name_Space.intern space name)
    val type_names = Name_Space.get_names type_space
      |> filter (fn name =>
          name <> "fun"
          andalso not (Name_Space.is_concealed type_space name)
          andalso not (Long_Name.is_hidden (Name_Space.intern type_space name)))
    val type_name_prefixes = map (fn T => T ^ ".") type_names

    (* --- ADT/record infrastructure constants --- *)

    (* Preserved constant names: datatype constructors/selectors/case, BNF map/pred/rel/sets,
       record selectors/updaters/make/fields/extend/truncate *)
    val const_name = fst o Term.dest_Const
    val thy = Context.theory_of context
    val (preserved_names, adt_record_prefixes) =
      let
        fun ctr_sugar_names T =
          case Context.cases Ctr_Sugar.ctr_sugar_of_global Ctr_Sugar.ctr_sugar_of context T of
            NONE => []
          | SOME cs =>
              map const_name (#ctrs cs)
              @ maps (map const_name) (#selss cs)
              @ [const_name (#casex cs)]
        fun bnf_names T =
          case Context.cases BNF_Def.bnf_of_global BNF_Def.bnf_of context T of
            NONE => []
          | SOME bnf =>
              [const_name (BNF_Def.map_of_bnf bnf),
               const_name (BNF_Def.pred_of_bnf bnf),
               const_name (BNF_Def.rel_of_bnf bnf)]
              @ map const_name (BNF_Def.sets_of_bnf bnf)
        fun record_names T =
          case Record.get_info thy T of
            NONE => []
          | SOME info =>
              let val fields = map fst (#fields info)
                  val prefix = T ^ "."
              in
                fields
                @ map (fn f => f ^ Record.updateN) fields
                @ [prefix ^ "more", prefix ^ "more" ^ Record.updateN,
                   prefix ^ "make", prefix ^ "fields", prefix ^ "extend", prefix ^ "truncate"]
              end
        fun co_rec_names T =
          case Context.cases BNF_FP_Def_Sugar.fp_sugar_of_global BNF_FP_Def_Sugar.fp_sugar_of context T of
            NONE => []
          | SOME fp =>
              (case #fp_co_induct_sugar fp of
                NONE => []
              | SOME sugar => [const_name (#co_rec sugar)])
        val (all_names, prefixes) = fold (fn T => fn (names, pfxs) =>
          let val ns = ctr_sugar_names T @ bnf_names T @ record_names T @ co_rec_names T
          in (ns @ names, if null ns then pfxs else (T ^ ".") :: pfxs) end)
          type_names ([], [])
      in (all_names, prefixes) end
    val preserved_set = Strhashtab.make_set preserved_names

    (* Quotient type infrastructure constants:
       - Abs_T/Rep_T (typedef-level; users work with quotient-level abs_T/rep_T instead)
       - pcr_T/cr_T (Lifting package correspondence relations) *)
    val quotient_typedef_infra = Strhashtab.make_set
      (maps (fn T =>
        case Quotient_Info.lookup_quotients_global thy T of
          NONE => []
        | SOME _ =>
            let val base = Long_Name.base_name T
                val qual = Long_Name.qualifier T
                fun mk name = if qual = "" then name else qual ^ "." ^ name
            in
              maps (fn ({Rep_name, Abs_name, ...}, _) => [Rep_name, Abs_name])
                (Typedef.get_info_global thy T)
              @ [mk ("pcr_" ^ base), mk ("cr_" ^ base)]
            end)
        type_names)

    (* Pre-BNF intermediate constant prefixes *)
    val pre_prefixes = map_filter (fn T =>
      case Context.cases BNF_FP_Def_Sugar.fp_sugar_of_global BNF_FP_Def_Sugar.fp_sugar_of context T of
        NONE => NONE
      | SOME fp =>
          let val map_name = fst (Term.dest_Const (BNF_Def.map_of_bnf (#pre_bnf fp)))
          in SOME (Long_Name.qualifier map_name ^ ".") end)
      type_names

    (* Class instance constant infixes *)
    val inst_infixes = map (fn T => "_" ^ Long_Name.base_name T ^ "_inst.") type_names

    val internal_prefixes = ["Lifting.", "BNF_Def.", "Transfer.", "BNF_Cardinal_Order_Relation.",
          "HOL.equal", "ATP.", "Num."]

    (* Pre-compute names that have a _class variant as a constant.
       For each constant "a.b_class.c", record "a.b.c" as having a class variant. *)
    val has_class_variant_set =
      let val all_const_names = #constants (Consts.dest consts) |> map fst
      in Strhashtab.make_set (map_filter (fn cname =>
           if String.isSubstring "_class." cname then
             let val qual = Long_Name.qualifier cname
                 val base = Long_Name.base_name cname
             in if String.isSuffix "_class" qual then
                  SOME (String.concat [
                    String.substring (qual, 0, size qual - 6), ".", base])
                else NONE
             end
           else NONE) all_const_names)
      end
    fun has_class_variant name = Strhashtab.defined has_class_variant_set name

    fun returns_prop (Type("fun", [_, T'])) = returns_prop T'
      | returns_prop (Type("prop", [])) = true
      | returns_prop _ = false

    val infra_const_cache : bool Strhashtab.table = Strhashtab.empty 4096
    fun is_infra_const' name typ_opt =
      case Strhashtab.lookup infra_const_cache name of
        SOME result => result
      | NONE =>
          let val result =
            Name_Space.is_concealed const_space name
            orelse is_hidden const_space name
            orelse exists (fn pfx => String.isPrefix pfx name) internal_prefixes
            orelse exists (fn pfx => String.isPrefix pfx name) pre_prefixes
            orelse exists (fn infix_ => String.isSubstring infix_ name) inst_infixes
            orelse has_class_variant name
            orelse Strhashtab.defined quotient_typedef_infra name
            orelse (exists (fn pfx => String.isPrefix pfx name) adt_record_prefixes
                    andalso not (Strhashtab.defined preserved_set name))
            orelse String.isSubstring ".class." name
            orelse (String.isSuffix "_class" (Long_Name.base_name name)
                    andalso returns_prop (case typ_opt of
                              SOME T => T
                            | NONE => (case try (Consts.the_constraint consts) name of
                                         SOME T => T | NONE => dummyT)))
              val _ = Strhashtab.update infra_const_cache (name, result)
          in result end
    fun is_infra_const name = is_infra_const' name NONE

    (* --- ADT infrastructure theorem names --- *)

    fun is_ADT_infra_thm full_name =
      String.isSuffix "_def" full_name
      andalso exists (fn pfx => String.isPrefix pfx full_name) type_name_prefixes

    (* --- Inductive infrastructure theorem names --- *)

    val ctxt = Context.proof_of context
    (* Check if name or its predicate form (name ^ "p") is inductive.
       inductive_set registers the inductive info under the predicate form
       (e.g. "listsp"), not the set form ("lists").
       the_inductive_global stores locale names (e.g. "ccpo.iteratesp") but
       fact names use class names (e.g. "ccpo_class.iteratesp").
       We strip "_class" from qualifier segments to recover the locale name. *)
    val is_inductive = can (Inductive.the_inductive_global ctxt)
    fun declass name =
      let val segs = Long_Name.explode name
      in case segs of
           [] => NONE
         | _ => let val segs' = map (fn s =>
                      if String.isSuffix "_class" s
                      then String.substring (s, 0, size s - 6)
                      else s) segs
                    val name' = Long_Name.implode segs'
                in if name' = name then NONE else SOME name' end
      end
    fun check_inductive name =
      is_inductive name orelse is_inductive (name ^ "p")
      orelse (String.isSubstring "_class." name
              andalso case declass name of
                SOME name' => is_inductive name' orelse is_inductive (name' ^ "p")
              | NONE => false)

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

    fun is_internal_class (Const(name1, _) $ Const(\<^const_name>\<open>Pure.type\<close>, _)) =
          String.isSuffix "_class" name1
      | is_internal_class (Const(\<^const_name>\<open>Pure.eq\<close>, _) $ A $ B) =
          is_internal_class A orelse is_internal_class B
      | is_internal_class (Const(\<^const_name>\<open>Pure.imp\<close>, _) $ _ $ X) = is_internal_class X
      | is_internal_class (Const(\<^const_name>\<open>Pure.conjunction\<close>, _) $ A $ B) =
          is_internal_class A orelse is_internal_class B
      | is_internal_class _ = false
    fun is_internal_constant (name, T) = is_infra_const' name (SOME T)

    fun is_infra_thm (full_name, thm) =
        Name_Space.is_concealed fact_space full_name
        orelse is_hidden fact_space full_name
        orelse is_ADT_infra_thm full_name
        orelse is_inductive_infra_thm full_name
        orelse Term.exists_Const is_internal_constant (Thm.prop_of thm)
        orelse is_internal_class (Thm.prop_of thm)
        orelse smart_size_of_term (Thm.prop_of thm) > 1000
  in
    {is_infra_const = is_infra_const, is_infra_thm = is_infra_thm,
     is_infra_type = fn name => Name_Space.is_concealed type_space name
                                orelse is_hidden type_space name,
     is_infra_class = fn name => Name_Space.is_concealed class_space name
                                 orelse is_hidden class_space name,
     is_infra_locale = fn name => Name_Space.is_concealed locale_space name
                                  orelse is_hidden locale_space name}
  end

end
