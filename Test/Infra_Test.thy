theory Infra_Test
  imports "../Semantic_Embedding"
begin
(*
section \<open>Test: record type\<close>

record my_point =
  xcoord :: nat
  ycoord :: nat

section \<open>Test: polymorphic record with inheritance\<close>

record 'a tagged_pair =
  fst_val :: 'a
  snd_val :: 'a
  tag :: string

record 'a weighted_pair = "'a tagged_pair" +
  weight :: nat

section \<open>Test: deep inheritance (3 levels) with multiple type params\<close>

record 'a base_obj =
  label :: string
  payload :: 'a

record ('a, 'b) ext_obj = "'a base_obj" +
  meta :: 'b

record ('a, 'b) full_obj = "('a, 'b) ext_obj" +
  priority :: nat

ML \<open>
let
  val context = Context.Proof \<^context>
  val thy = \<^theory>
  val type_space = Proof_Context.type_space \<^context>
  val type_names = Name_Space.get_names type_space
    |> filter (fn name =>
        name <> "fun"
        andalso not (Name_Space.is_concealed type_space name)
        andalso not (Long_Name.is_hidden (Name_Space.intern type_space name)))
  val has_ctr_sugar = is_some o Ctr_Sugar.ctr_sugar_of \<^context>
  val has_bnf = is_some o BNF_Def.bnf_of \<^context>
  val has_record = is_some o Record.get_info thy

  (* Types that have type_name_prefix but NOT adt/record info *)
  val uncovered = filter (fn T =>
    not (has_ctr_sugar T) andalso not (has_bnf T) andalso not (has_record T)) type_names

  (* Check if any of these have constants under their prefix *)
  val consts = Proof_Context.consts_of \<^context>
  val all_const_names = #constants (Consts.dest consts) |> map fst
  val uncovered_with_consts = map_filter (fn T =>
    let val pfx = T ^ "."
        val cs = filter (String.isPrefix pfx) all_const_names
    in if null cs then NONE else SOME (T, cs) end) uncovered

  val {is_infra_const, ...} = Infra_Filter.gen_infra_filters context
  val const_space = Consts.space_of consts

  val _ = writeln ("=== Types without ctr_sugar/bnf/record but with constants ("
                   ^ Int.toString (length uncovered_with_consts) ^ ") ===")
  val _ = map (fn (T, cs) => (
    writeln ("  " ^ T);
    map (fn c =>
      let val concealed = Name_Space.is_concealed const_space c
          val hidden = Long_Name.is_hidden (Name_Space.intern const_space c)
          val infra = is_infra_const c
          val flags = String.concat [
            if concealed then "C" else " ",
            if hidden then "H" else " ",
            if infra then " " else "*"]
      in writeln ("    [" ^ flags ^ "] " ^ c) end) cs; ())) uncovered_with_consts
in () end
\<close>

section \<open>Test: typedef\<close>

typedef my_nat = "{n::nat. n > 0}"
  by (rule exI[of _ 1], simp)

typedef 'a my_wrapper = "{x::'a option. x \<noteq> None}"
  by (rule exI[of _ "Some undefined"], simp)

typedef ('a, 'b) my_tagged = "{(x::'a, y::'b). True}"
  morphisms unwrap wrap
  by auto

typedef my_unit = "{()}"
  by auto


ML \<open>
let
  val thy = \<^theory>
  fun print_typedef name =
    case Typedef.get_info_global thy name of
      [] => writeln (name ^ ": no typedef info")
    | infos => (map (fn ({Rep_name, Abs_name, ...}, _) =>
        writeln ("  " ^ name ^ ": Rep=" ^ Rep_name ^ ", Abs=" ^ Abs_name)) infos; ())
  val _ = writeln "=== Typedef info ==="
  val _ = print_typedef "Infra_Test.my_nat"
  val _ = print_typedef "Infra_Test.my_wrapper"
  val _ = print_typedef "Infra_Test.my_tagged"
  val _ = print_typedef "Infra_Test.my_unit"
in () end
\<close>

section \<open>Test: codatatype\<close>

codatatype 'a stream = SCons (shd: 'a) (stl: "'a stream")

codatatype ('a, 'b) cotree =
  CLeaf
  | CNode (cval: 'a) (cleft: "('a, 'b) cotree") (cright: "('a, 'b) cotree") (cinfo: 'b)

codatatype 'a llist = LNil | LCons (lhd: 'a) (ltl: "'a llist")

primcorec nats_from :: "nat \<Rightarrow> nat stream" where
  "shd (nats_from n) = n"
| "stl (nats_from n) = nats_from (Suc n)"

primcorec zeroes :: "nat stream" where
  "shd zeroes = 0"
| "stl zeroes = zeroes"

ML \<open>
let
  val context = Context.Proof \<^context>
  val {is_infra_const, is_infra_thm, ...} = Infra_Filter.gen_infra_filters context
  val facts = Proof_Context.facts_of \<^context>
  val all_facts = Facts.dest_static false [] facts
  val my_facts = filter (fn (name, _) => String.isPrefix "Infra_Test." name) all_facts
  val _ = writeln "=== Theorems under Infra_Test ==="
  val _ = map (fn (name, thms) =>
    let val infra = List.all (fn thm => is_infra_thm (name, thm)) thms
        val mark = if infra then "  " else "* "
        val props = map (fn thm =>
          Syntax.string_of_term \<^context> (Thm.prop_of thm)) thms
    in writeln (String.concat ["  ", mark, name]);
       map (fn p => writeln (String.concat ["  ", mark, "  ", p])) props; () end)
    (sort (string_ord o apply2 fst) my_facts)
in () end
\<close>
*)
section \<open>Test: quotient_type\<close>

definition intrel :: "int \<times> int \<Rightarrow> int \<times> int \<Rightarrow> bool" where
  "intrel p q \<equiv> fst p * snd q = snd p * fst q"

lemma intrel_equivp: "equivp intrel"
  sorry

\<comment> \<open>Simple: non-polymorphic, full equivalence, default morphisms\<close>
quotient_type my_frac = "int \<times> int" / intrel
  by (rule intrel_equivp)

\<comment> \<open>Polymorphic: unordered pairs\<close>
definition eq_upair :: "'a \<times> 'a \<Rightarrow> 'a \<times> 'a \<Rightarrow> bool" where
  "eq_upair p q \<equiv> p = q \<or> (fst p = snd q \<and> snd p = fst q)"

lemma eq_upair_equivp: "equivp eq_upair"
  sorry

quotient_type 'a my_uprod = "'a \<times> 'a" / eq_upair
  by (rule eq_upair_equivp)

\<comment> \<open>Partial equivalence with custom morphisms\<close>
definition my_ratrel :: "int \<times> int \<Rightarrow> int \<times> int \<Rightarrow> bool" where
  "my_ratrel p q \<equiv> snd p \<noteq> 0 \<and> snd q \<noteq> 0 \<and> fst p * snd q = fst q * snd p"

lemma my_ratrel_part_equivp: "part_equivp my_ratrel"
  sorry

quotient_type my_rat = "int \<times> int" / partial: my_ratrel
  morphisms rep_rat mk_rat
  by (rule my_ratrel_part_equivp)

ML \<open>
let
  val context = Context.Proof \<^context>
  val {is_infra_const, ...} = Infra_Filter.gen_infra_filters context
  val consts = Proof_Context.consts_of \<^context>
  val all_const_names = #constants (Consts.dest consts) |> map fst
  val my_consts = sort string_ord (filter (fn n =>
    String.isPrefix "Infra_Test.my_frac" n
    orelse String.isPrefix "Infra_Test.abs_my_frac" n
    orelse String.isPrefix "Infra_Test.rep_my_frac" n
    orelse String.isPrefix "Infra_Test.my_uprod" n
    orelse String.isPrefix "Infra_Test.abs_my_uprod" n
    orelse String.isPrefix "Infra_Test.rep_my_uprod" n
    orelse String.isPrefix "Infra_Test.my_rat" n
    orelse String.isPrefix "Infra_Test.mk_rat" n
    orelse String.isPrefix "Infra_Test.rep_rat" n) all_const_names)
  val _ = writeln "=== Constants (quotient_type) ==="
  val _ = map (fn name =>
    writeln (String.concat ["  ", if is_infra_const name then "  " else "* ", name])) my_consts
in () end
\<close>

ML \<open>
let
  val context = Context.Proof \<^context>
  val {is_infra_thm, ...} = Infra_Filter.gen_infra_filters context
  val facts = Proof_Context.facts_of \<^context>
  val all_facts = Facts.dest_static false [] facts
  val my_facts = filter (fn (name, _) =>
    String.isPrefix "Infra_Test.my_frac" name
    orelse String.isPrefix "Infra_Test.my_uprod" name
    orelse String.isPrefix "Infra_Test.my_rat" name) all_facts
  val _ = writeln "=== Theorems (quotient_type) ==="
  val _ = map (fn (name, thms) =>
    let val infra = List.all (fn thm => is_infra_thm (name, thm)) thms
        val mark = if infra then "  " else "* "
        val props = map (fn thm =>
          Syntax.string_of_term \<^context> (Thm.prop_of thm)) thms
    in writeln (String.concat ["  ", mark, name]);
       map (fn p => writeln (String.concat ["  ", mark, "  ", p])) props; () end)
    (sort (string_ord o apply2 fst) my_facts)
in () end
\<close>

thm my_rat
term Abs_my_rat_cases
term pcr_my_rat
thm Abs_my_rat_induct
thm Rep_my_rat_cases

end
