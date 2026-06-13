signature INFRA_FILTER = sig
  val smart_size_of_term : term -> int
  val is_infra_session : theory -> bool
  val gen_infra_filters : Context.generic ->
    {is_infra_const: string -> bool,
     is_infra_thm: string * thm -> bool,
     is_infra_induct_thm: string * thm -> bool,
     is_infra_type: string -> bool,
     is_infra_class: string -> bool,
     is_infra_locale: string -> bool,
     is_infra_dynamic_fact: string -> bool,
     is_infra_method: string -> bool}
end

structure Infra_Filter : INFRA_FILTER = struct

val smart_size_of_term = smart_size_of_term (* from Performant_Isabelle_ML *)

val infra_session_names = ["HOL-Decision_Procs"]
fun is_infra_session thy =
  member (op =) infra_session_names
    (Long_Name.qualifier (Context.theory_long_name thy))

fun gen_infra_filters (context : Context.generic)
    : {is_infra_const: string -> bool,
       is_infra_thm: string * thm -> bool,
       is_infra_induct_thm: string * thm -> bool,
       is_infra_type: string -> bool, is_infra_class: string -> bool,
       is_infra_locale: string -> bool,
       is_infra_dynamic_fact: string -> bool, is_infra_method: string -> bool} =
  let
    val type_space = Context.cases Sign.type_space Proof_Context.type_space context
    val consts = Context.cases Sign.consts_of Proof_Context.consts_of context
    val const_space = Consts.space_of consts
    val facts = Context.cases Global_Theory.facts_of Proof_Context.facts_of context
    val fact_space = Facts.space_of facts
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

    (* BNF origin position set: positions (file, offset) where BNF-generated constants live.
       Used to distinguish auto-generated ADT constants from lift_definition/user constants. *)
    val bnf_origin_pos_set =
      let
        fun safe_const_name t = try (fst o Term.dest_Const) t
        fun collect_bnf_names T =
          let
            val from_ctr = case Context.cases Ctr_Sugar.ctr_sugar_of_global Ctr_Sugar.ctr_sugar_of context T of
                NONE => []
              | SOME cs =>
                  map const_name (#ctrs cs)
                  @ maps (map const_name) (#selss cs)
                  @ [const_name (#casex cs)]
                  @ map_filter safe_const_name (#discs cs)
            val from_bnf = case Context.cases BNF_Def.bnf_of_global BNF_Def.bnf_of context T of
                NONE => []
              | SOME bnf =>
                  [const_name (BNF_Def.map_of_bnf bnf),
                   const_name (BNF_Def.pred_of_bnf bnf),
                   const_name (BNF_Def.rel_of_bnf bnf)]
                  @ map const_name (BNF_Def.sets_of_bnf bnf)
            val from_fp = case Context.cases BNF_FP_Def_Sugar.fp_sugar_of_global BNF_FP_Def_Sugar.fp_sugar_of context T of
                NONE => []
              | SOME fp =>
                  let val res = #fp_res fp
                      val co_rec_names = case #fp_co_induct_sugar fp of
                          NONE => []
                        | SOME sugar => map_filter safe_const_name [#co_rec sugar]
                  in
                    map_filter safe_const_name
                      (#ctors res @ #dtors res @ #xtor_un_folds res @ #xtor_co_recs res)
                    @ co_rec_names
                  end
            val from_pre_bnf = case Context.cases BNF_FP_Def_Sugar.fp_sugar_of_global BNF_FP_Def_Sugar.fp_sugar_of context T of
                NONE => []
              | SOME fp =>
                  let val pre = #pre_bnf fp
                  in map_filter safe_const_name
                      ([BNF_Def.map_of_bnf pre] @ BNF_Def.sets_of_bnf pre)
                  end
          in from_ctr @ from_bnf @ from_fp @ from_pre_bnf end
        val all_bnf_names = maps collect_bnf_names type_names
        fun pos_key_of cname =
          case try (Name_Space.the_entry const_space) cname of
            NONE => NONE
          | SOME entry =>
              (case (Position.file_of (#pos entry), Position.offset_of (#pos entry)) of
                (SOME file, SOME offset) => SOME (String.concat [file, "\t", Int.toString offset])
              | _ => NONE)
      in Strhashtab.make_set (map_filter pos_key_of all_bnf_names) end

    fun on_bnf_origin_pos name =
      case try (Name_Space.the_entry const_space) name of
        NONE => false
      | SOME entry =>
          (case (Position.file_of (#pos entry), Position.offset_of (#pos entry)) of
            (SOME file, SOME offset) =>
              Strhashtab.defined bnf_origin_pos_set (String.concat [file, "\t", Int.toString offset])
          | _ => false)

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

    val infra_session_thy_prefixes =
      map_filter (fn anc =>
        if is_infra_session anc
        then SOME (Context.theory_base_name anc ^ ".")
        else NONE) (Theory.ancestors_of thy)

    val internal_prefixes = ["Lifting.", "BNF_Def.", "Transfer.", "BNF_Cardinal_Order_Relation.",
          "HOL.equal", "ATP.", "Code_Evaluation."] @ infra_session_thy_prefixes

    val infra_theory_prefixes = ["Code_Evaluation.", "Num."] @ infra_session_thy_prefixes
    fun is_from_infra_theory name =
      exists (fn pfx => String.isPrefix pfx name) infra_theory_prefixes

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
            not (Strhashtab.defined preserved_set name)
            andalso (
            Name_Space.is_concealed const_space name
            orelse is_hidden const_space name
            orelse exists (fn pfx => String.isPrefix pfx name) internal_prefixes
            orelse exists (fn pfx => String.isPrefix pfx name) pre_prefixes
            orelse exists (fn infix_ => String.isSubstring infix_ name) inst_infixes
            orelse has_class_variant name
            orelse Strhashtab.defined quotient_typedef_infra name
            orelse let val base = Long_Name.base_name name
                   in String.isPrefix "Abs_" base orelse String.isPrefix "Rep_" base end
            orelse (exists (fn pfx => String.isPrefix pfx name) adt_record_prefixes
                    andalso on_bnf_origin_pos name)
            orelse String.isSubstring ".class." name
            orelse (String.isSuffix "_class" (Long_Name.base_name name)
                    andalso returns_prop (case typ_opt of
                              SOME T => T
                            | NONE => (case try (Consts.the_constraint consts) name of
                                         SOME T => T | NONE => dummyT))))
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
        (* No accessible short name in this context (e.g. shadowed, or a
           class-localized fact outside its locale): the agent cannot cite it by
           name, so it is useless to surface. Facts.extern marks such names with
           a "??." prefix (this is the exact string the agent would see). *)
        orelse String.isPrefix "??." (Facts.extern ctxt facts full_name)
        orelse is_from_infra_theory full_name
        orelse is_ADT_infra_thm full_name
        orelse is_inductive_infra_thm full_name
        orelse Term.exists_Const is_internal_constant (Thm.prop_of thm)
        orelse is_internal_class (Thm.prop_of thm)
        orelse smart_size_of_term (Thm.prop_of thm) > 1000

    (* Stricter than is_infra_thm: also rejects induction rules from theory
       prefixes that emit too-generic auto-generated rules (e.g. Basic_BNFs's
       fsts.inducts / snds.inducts). *)
    val infra_induct_thy_prefixes = ["Basic_BNFs."]
    fun is_infra_induct_thm (full_name, thm) =
        is_infra_thm (full_name, thm)
        orelse exists (fn p => String.isPrefix p full_name) infra_induct_thy_prefixes

    (* --- Methods: drop any method defined in a Main-ancestor theory --- *)
    val main_anc_set =
      let val thy = Context.theory_of context
          val ancs = Theory.ancestors_of thy
      in case find_first (fn t => Context.theory_base_name t = "Main") ancs of
           SOME m => Symtab.make_set (map Context.theory_long_name (m :: Theory.ancestors_of m))
         (* current theory is Main or below it: filter everything inherited *)
         | NONE => Symtab.make_set (map Context.theory_long_name (thy :: ancs))
      end
    val method_space = Method.method_space context
    (* Same concealed/hidden/shadowed guards as is_infra_dynamic_fact:
       a method the agent cannot cite by name is useless to surface, and an
       inaccessible name crashes Universal_Key.key_of_method (intern yields the
       "??."-hidden form, the_entry then fails with "Inaccessible method"). *)
    fun is_infra_method name =
      (case try (Name_Space.the_entry method_space) name of
         SOME e => Symtab.defined main_anc_set (#theory_long_name e)
       | NONE => true)
      orelse Name_Space.is_concealed method_space name
      orelse is_hidden method_space name
      orelse String.isPrefix "??." (Name_Space.extern ctxt method_space name)

    (* --- Theorem collections (dynamic facts): tool-internal blacklist --- *)
    (* A theorem collection is any dynamic fact (named_theorems or raw
       add_thms_dynamic).  The blacklist keys by fact name; it covers known
       tool-internal collections regardless of which kind they are, so it can be
       extended with non-named_theorems names as needed. *)
    val dynamic_fact_blacklist = Symtab.make_set [
        "HOL.no_atp", "HOL.atomize_elim",
        "HOL.nitpick_simp", "HOL.nitpick_psimp", "HOL.nitpick_unfold", "HOL.nitpick_choice_spec",
        "HOL.code_pred_def", "HOL.code_pred_inline", "HOL.code_pred_simp",
        "SMT.smt_arith_combine", "SMT.smt_arith_multiplication", "SMT.smt_arith_simplify",
        "Quotient.quot_equiv", "Quotient.quot_preserve", "Quotient.quot_respect",
        "Quotient.quot_thm", "Quotient.id_simps",
        "Lifting.relator_eq_onp",
        "Fun_Def.measure_function", "Fun_Def_Base.termination_simp",
        "Partial_Function.partial_function_mono"]
    (* Theory prefixes whose dynamic facts are all tool-internal (Transfer/Lifting
       relators, SMT proof rules, Pure method_facts).  None of these four packages
       emits a dynamic collection worth retrieving, so filter the whole prefix
       rather than maintaining a per-name list.  NB: this is local to the
       dynamic-fact filter, NOT the shared is_from_infra_theory (which also gates
       constants/types/classes/locales). *)
    val infra_dynamic_thy_prefixes = ["Transfer.", "Lifting.", "SMT.", "Pure."]
    (* Codatatype packages auto-generate, per codatatype, coinduction-upto
       congruence collections (<codatatype>.cong_* and <codatatype>.coinduct_upto)
       as raw dynamic facts.  These are machinery, not user-facing rule sets, and
       recur for every codatatype, so match them structurally on the base name. *)
    fun is_codatatype_cong name =
      let val base = Long_Name.base_name name
      in String.isPrefix "cong_" base orelse base = "coinduct_upto" end
    (* Beyond the blacklist/prefixes, drop collections the agent cannot cite by
       name, mirroring the is_infra_thm checks above: concealed/private-scope
       entries (e.g. Eisbach `method ... uses` parameter slots, declared concealed
       in a private scope) and shadowed names ("??." extern).  Such names also
       crash Universal_Key.key_of_theorem_collection: Name_Space.intern yields the
       "??."-hidden form and Name_Space.the_entry then fails with
       "Inaccessible fact". *)
    fun is_infra_dynamic_fact name =
      Symtab.defined dynamic_fact_blacklist name
      orelse exists (fn pfx => String.isPrefix pfx name) infra_dynamic_thy_prefixes
      orelse is_codatatype_cong name
      orelse Name_Space.is_concealed fact_space name
      orelse is_hidden fact_space name
      orelse String.isPrefix "??." (Facts.extern ctxt facts name)
  in
    {is_infra_const = is_infra_const, is_infra_thm = is_infra_thm,
     is_infra_induct_thm = is_infra_induct_thm,
     is_infra_type = fn name => Name_Space.is_concealed type_space name
                                orelse is_hidden type_space name
                                orelse is_from_infra_theory name,
     is_infra_class = fn name => Name_Space.is_concealed class_space name
                                 orelse is_hidden class_space name
                                 orelse is_from_infra_theory name,
     is_infra_locale = fn name => Name_Space.is_concealed locale_space name
                                  orelse is_hidden locale_space name
                                  orelse is_from_infra_theory name,
     is_infra_dynamic_fact = is_infra_dynamic_fact,
     is_infra_method = is_infra_method}
  end

end
