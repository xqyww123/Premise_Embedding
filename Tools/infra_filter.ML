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

(* Term size that counts number literals and string literals as small constants.
   - numeral <bits>: size 1
   - uminus (numeral <bits>): size 2
   - Suc (Suc ... 0): size 2
   - String.Literal b0..b6 (Literal ... zero): size 1
   - String.char.Char b0..b7: size 1
   - list of chars (String.char list): size 1 per char + 1 for the list *)
local
  fun is_numeral (Const ("Num.num.One", _)) = true
    | is_numeral (Const ("Num.num.Bit0", _) $ t) = is_numeral t
    | is_numeral (Const ("Num.num.Bit1", _) $ t) = is_numeral t
    | is_numeral _ = false

  fun is_nat_literal (Const ("Groups.zero_class.zero", _)) = true
    | is_nat_literal (Const ("Nat.Suc", _) $ t) = is_nat_literal t
    | is_nat_literal _ = false

  fun is_literal_chain (Const ("Groups.zero_class.zero", Type ("String.literal", _))) = true
    | is_literal_chain (Const ("String.Literal", _) $ _ $ _ $ _ $ _ $ _ $ _ $ _ $ t) =
        is_literal_chain t
    | is_literal_chain _ = false

  fun is_char (Const ("String.char.Char", _) $ _ $ _ $ _ $ _ $ _ $ _ $ _ $ _) = true
    | is_char _ = false

  fun is_char_list (Const ("List.list.Nil", Type ("List.list", [Type ("String.char", _)]))) = true
    | is_char_list (Const ("List.list.Cons", _) $ c $ cs) = is_char c andalso is_char_list cs
    | is_char_list _ = false

  fun add_size (Const ("Num.numeral_class.numeral", _) $ t) n =
        if is_numeral t then n + 1 else add_size t (n + 1)
    | add_size (Const ("Groups.uminus_class.uminus", _) $
                  (Const ("Num.numeral_class.numeral", _) $ t)) n =
        if is_numeral t then n + 2 else add_size t (n + 2)
    | add_size (t as (Const ("Nat.Suc", _) $ _)) n =
        if is_nat_literal t then n + 2 else add_size_default t n
    | add_size (t as (Const ("String.Literal", _) $ _ $ _ $ _ $ _ $ _ $ _ $ _ $ _)) n =
        if is_literal_chain t then n + 1 else add_size_default t n
    | add_size (t as (Const ("List.list.Cons", _) $ _ $ _)) n =
        if is_char_list t then n + 1 else add_size_default t n
    | add_size (t as (Const ("List.list.Nil", Type ("List.list", [Type ("String.char", _)])))) n =
        n + 1
    | add_size t n = add_size_default t n
  and add_size_default (t $ u) n = add_size t (add_size u n)
    | add_size_default (Abs (_, _, t)) n = add_size t (n + 1)
    | add_size_default _ n = n + 1
in
  val smart_size_of_term = fn t => add_size t 0
end

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

    (* --- ADT infrastructure constants --- *)

    (* Preserved datatype constant names: constructors, selectors, case, map, pred, rel, sets *)
    val const_name = fst o Term.dest_Const
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
    val preserved_set = Strhashtab.make_set
      (maps (fn T => ctr_sugar_names T @ bnf_names T) type_names)

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

    (* For a.b.c, check if a.b_class.c exists as a constant (class method shorthand) *)
    fun has_class_variant name =
      Long_Name.is_qualified name andalso
      let val class_name = String.concat [Long_Name.qualifier name, "_class.", Long_Name.base_name name]
      in can (Consts.the_constraint consts) class_name end

    fun returns_prop (Type("fun", [_, T'])) = returns_prop T'
      | returns_prop (Type("prop", [])) = true
      | returns_prop _ = false

    fun is_infra_const' name typ_opt =
      Name_Space.is_concealed const_space name
      orelse is_hidden const_space name
      orelse exists (fn pfx => String.isPrefix pfx name) internal_prefixes
      orelse exists (fn pfx => String.isPrefix pfx name) pre_prefixes
      orelse exists (fn infix_ => String.isSubstring infix_ name) inst_infixes
      orelse has_class_variant name
      orelse (exists (fn T => String.isPrefix (T ^ ".") name) type_names
              andalso not (Strhashtab.defined preserved_set name))
      orelse String.isSubstring ".class." name
      orelse (String.isSuffix "_class" (Long_Name.base_name name)
              andalso returns_prop (case typ_opt of
                        SOME T => T
                      | NONE => (case try (Consts.the_constraint consts) name of
                                   SOME T => T | NONE => dummyT)))
    fun is_infra_const name = is_infra_const' name NONE

    (* --- ADT infrastructure theorem names --- *)

    fun is_ADT_infra_thm full_name =
      exists (fn T =>
        String.isPrefix (T ^ ".") full_name
        andalso let val suffix = String.extract (full_name, size T + 1, NONE)
                in String.isSuffix "_def" suffix
                 (* orelse suffix = "splits" orelse suffix = "split"
                   orelse String.isPrefix "split_" suffix *) end)
        type_names

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
