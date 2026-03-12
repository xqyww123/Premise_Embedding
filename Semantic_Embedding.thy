theory Semantic_Embedding
  imports Main Isabelle_RPC.Remote_Procedure_Calling Performant_Isabelle_ML.Performant_Isabelle_ML
          Isa_REPL.Isa_REPL
begin

declare [[ML_debugger, ML_exception_debugger, ML_exception_trace, ML_print_depth=1000]]

ML_file \<open>Tools/Sledgehammer/sledgehammer_embedding_ctxt.ML\<close>
ML_file \<open>Tools/term_serial_index.ML\<close>
ML_file \<open>Tools/semantic_store.ML\<close>


ML \<open>Ctr_Sugar.ctr_sugar_of \<^context> "List.list"\<close>
ML \<open>
val bnf = the (BNF_Def.bnf_of \<^context> "List.list")
val mappers = BNF_Def.map_of_bnf bnf
val predicaters = BNF_Def.pred_of_bnf bnf
val relators = BNF_Def.rel_of_bnf bnf
val sets = BNF_Def.sets_of_bnf bnf
\<close>

ML \<open>@{theory List}\<close>    
ML \<open>Semantic_Store.interpre t_file @{theory List}\<close>
 
ML \<open>Semantic_Store.preserved_datatype_const_names @{theory List}\<close>
ML \<open>BNF_FP_Def_Sugar.fp_sugar_of_global @{theory} "List.list"\<close>


term dtor_list
term List.pre_list.list.set2_pre_list
thm no_atp
thm List.List.list.set_map
thm List.list.Quotient
thm basic_trans_rules
ML \<open>@{thm basic_trans_rules(1)} |> Thm.derivation_id\<close>
ML \<open>@{thm order_trans_rules(1)} |> Thm.derivation_id\<close>
thm order_trans_rules
thm List.length_code[code]
thm code
thm List.folding_insort_key.axioms
term folding_insort_key

corollary

ML \<open>
  let
    val thy = @{theory List}
    val transfer = Global_Theory.transfer_theories thy
    val facts = Global_Theory.facts_of thy
    val parent_facts = map Global_Theory.facts_of (Theory.parents_of thy)
    val entries = Facts.dest_static false parent_facts facts

    val (name, thms) = hd entries
    val thm = hd thms
    val thm' = transfer thm

    fun show_did (did : Proofterm.thm_id option) =
      case did of SOME {serial, ...} => Int.toString serial | NONE => "NONE"

    val did_before = try Thm.derivation_id thm
    val did_after = SOME (Thm.derivation_id thm')

    val _ = writeln (String.concat ["Theorem: ", name])
    val _ = writeln (String.concat ["Before transfer: ",
      case did_before of SOME did => show_did did | NONE => "CONTEXT raised"])
    val _ = writeln (String.concat ["After transfer: ",
      case did_after of SOME did => show_did did | NONE => "impossible"])
en 
\<close>

ML \<open>
val consts_space = Consts.space_of (Sign.consts_of @{theory})
fun query_constant name =
      let
        val full_name = Name_Space.intern consts_space name
        val ns_entry = Name_Space.the_entry consts_space full_name
      in
        (full_name, #pos ns_entry)
      end
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
fun query_theorem name =
      let
        val (base_name, idx_opt) = parse_thm_name name
        val thms = Proof_Context.get_thms \<^context> base_name
        val thm = case idx_opt of
            NONE => hd thms
          | SOME n => nth thms (n - 1)
        val s = Term_Serial_Index.serial_of_thm thm
      in
        s
      end
\<close>
ML \<open>ERROR s\<close>
ML \<open>query_theorem "append_Nsil"\<close>

ML \<open>Resources.master_directory @{theory}\<close>

ML \<open>
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
\<close>
ML \<open>get_theory_file_path @{theory HOL}\<close>
ML \<open>type t = Proof.context\<close>

end