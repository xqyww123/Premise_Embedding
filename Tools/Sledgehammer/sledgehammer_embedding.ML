signature SLEDGEHAMMER_EMBEDDING = sig
end

structure Sledgehammer_Embedding : SLEDGEHAMMER_EMBEDDING = struct

val embedding_model_provider = Attrib.setup_config_string \<^binding>\<open>embedding_model_provider\<close> (K "")
val embedding_model = Attrib.setup_config_string \<^binding>\<open>embedding_model\<close> (K "reasonwang/embedding-qwen3-1.7b-embedding_unicode_shuf")
val embedding_API_key = Attrib.setup_config_string \<^binding>\<open>embedding_API_key\<close> (K "")
val embedding_dim = 2048
val embedding_token_limit = Attrib.setup_config_int \<^binding>\<open>embedding_token_limit\<close> (K 8000)

fun config_of ctxt =
  let fun chk s = if s = "" then NONE else SOME s
   in ( chk (Config.get ctxt embedding_model_provider)
      , Config.get ctxt embedding_model
      , chk (Config.get ctxt embedding_API_key)
      , embedding_dim)
  end

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack
  val pack_defctxt = packString
  val pack_term = packString
  val pack_goal = packTuple3 (packList pack_term, pack_term, pack_defctxt)
  val pack_premise = packPair (pack_term, pack_defctxt)

  val unpackVec = unpackBytes
in
(* val embed_cmd = {
  name = "embed_goal_premises",
  arg_schema = packPair (pack_goal, packList pack_premise),
  ret_schema = unpackPair (unpackVec, unpackList unpackVec),
  callback = NONE,
  timeout = SOME (Time.fromSeconds 120)
} *)
val packConfig = packTuple4 (
      packOption packString,
      packString,
      packOption packString,
      packInt)
val packGoal = packTuple3 (
      packList packString, (*premises*)
      packString,          (*conclusion*)
      packList packString  (*variables*))
val packPremise =
      packPair (packString (*statement*), packList packString (*variables*))
val packCtxt = packString

val embed_cmd = {
  name = "embed",
  arg_schema = packPair (packList (packBytes o Byte.stringToBytes), packConfig),
  ret_schema = unpackList unpackVec,
  callback = NONE,
  timeout = NONE (* SOME (Time.fromSeconds 120) *)
}
val embed_goal_cmd = {
  name = "embed_goal",
  arg_schema = packTuple4 (packGoal, packCtxt, packConfig, packInt),
  ret_schema = unpackVec,
  callback = NONE,
  timeout = NONE
}
val embed_premises_cmd = {
  name = "embed_premises",
  arg_schema = packTuple3 (packList (packPair (packPremise, packCtxt)), packConfig, packInt),
  ret_schema = unpackList unpackVec,
  callback = NONE,
  timeout = NONE
}
val embed_goal_and_premises_cmd = {
  name = "embed_goal_and_premises",
  arg_schema = packTuple5 (
                  packGoal, packCtxt,
                  packList (packPair (packPremise, packCtxt)),
                  packConfig, packInt),
  ret_schema = unpackPair (unpackVec, unpackList unpackVec),
  callback = NONE,
  timeout = NONE
}

end

structure Vector_Arith = Vector_Arith_Q15_D2048
type vector = Vector_Arith.vector

fun print_typ_full_sort ctxt =
    let val ctxt' = ctxt
              |> Config.put Printer.show_types true
              |> Config.put Printer.show_sorts true
              |> Config.put Printer.show_markup false
              |> Config.put Printer.show_structs false
              |> Config.put Printer.show_question_marks true
              |> Config.put Printer.show_brackets false
     in Syntax.string_of_typ ctxt'
     #> Premise_Embedding_Context_Info.trim_markup
    end

fun add_typs (Abs (name,typ,X)) rets = add_typs X (insert (op =) (name,typ) rets)
  | add_typs (A $ B) rets = add_typs B (add_typs A rets)
  | add_typs (Free (name,typ)) rets = insert (op =) (name,typ) rets
  | add_typs (Var ((name,0),typ)) rets = insert (op =) (name,typ) rets
  | add_typs (Var ((name,i),typ)) rets = insert (op =) (name^"."^string_of_int i,typ) rets
  | add_typs _ rets = rets

(*
fun vars_of_goal ctxt =
  let val print_typ_full_sort = print_typ_full_sort ctxt
   in fn (assms, concl) =>
  let val vs = fold add_typs (concl::assms) []
   in map (fn (v,T) => String.concat [v, "::", print_typ_full_sort T]) vs
  end
  end
*)

fun vars_of_term ctxt =
  let val print_typ_full_sort = print_typ_full_sort ctxt
   in fn term =>
  let val vs = add_typs term []
   in map (fn (v,T) => String.concat [v, "::", print_typ_full_sort T]) vs
  end
  end

open ATP_Util
open ATP_Problem_Generate
open Sledgehammer_Util
open Sledgehammer_Fact
open Sledgehammer_Prover
open Sledgehammer_Prover_Minimize
open Sledgehammer_MePo

fun steep_weight_of_fact rank = Math.pow (0.62, log2 (Real.fromInt (rank + 1))) (* FUDGE *)
fun weight_facts_steeply facts = map_index (swap o apfst steep_weight_of_fact) facts

structure Thm_Vector = Theory_Data (
  type T = vector Termtab.table Synchronized.var option
  val empty = NONE
  fun merge (NONE, NONE) = SOME (Synchronized.var "Thm_Vector" Termtab.empty)
    | merge (NONE, some) = some
    | merge (some, NONE) = some
    | merge (SOME va, SOME vb) = SOME (Synchronized.var "Thm_Vector" (
        Termtab.merge (K true) (Synchronized.value va, Synchronized.value vb)))
)

structure Thm_Vector_Local = Proof_Data (
  type T = vector Termtab.table Synchronized.var
  fun init thy = (case Thm_Vector.get thy
                    of NONE => Synchronized.var "Thm_Vector_Local" Termtab.empty
                     | SOME v => Synchronized.var "Thm_Vector_Local" (Synchronized.value v))
)

val mepo_weight = 0.5 (* FUDGE *)
val embd_weight = 0.5 (* FUDGE *)

fun gen_eq_thm ctxt = if Config.get ctxt Sledgehammer_MaSh.duplicates
                      then Thm.eq_thm_strict
                      else Thm.eq_thm_prop

fun cached_embed_premises ctxt vecs_global_var_opt vecs_var goal_opt facts =
  let val SOT = Premise_Embedding_Context_Info.string_of_term ctxt
      val defctxt_string_of = Premise_Embedding_Context_Info.defctxt_string_of ctxt
      val vars_of_term = vars_of_term ctxt

      val vec_store = Synchronized.value vecs_var
      val vecs'0 = map (fn origin as ((_, (scope, _)), thm) =>
                let val prop = Thm.prop_of thm
                 in (origin, Termtab.lookup vec_store prop)
                end) facts
      val query = Par_List.map (
              fn ((tag, thm), NONE) =>
                  let val prop = Thm.prop_of thm
                   in SOME ((SOT prop, vars_of_term prop), defctxt_string_of prop)
                  end
               | _ => NONE) vecs'0
                       |> map_filter I

      val config = config_of ctxt
      val embedding_token_limit = Config.get ctxt embedding_token_limit
      val (vgoal_opt, vpremises') =
            case (goal_opt, null query) of
              (NONE, true) => (NONE, [])
            | (NONE, false) =>
                (NONE, Remote_Procedure_Calling.call_command embed_premises_cmd
                    (query, config, embedding_token_limit))
            | (SOME (goal_enc, goal_ctxt), true) =>
                (SOME (Remote_Procedure_Calling.call_command embed_goal_cmd
                    (goal_enc, goal_ctxt, config, embedding_token_limit)), [])
            | (SOME (goal_enc, goal_ctxt), false) =>
                let val (vgoal, vps) =
                      Remote_Procedure_Calling.call_command embed_goal_and_premises_cmd
                          (goal_enc, goal_ctxt, query, config, embedding_token_limit)
                 in (SOME vgoal, vps)
                end

      fun is_global ((_,(Global,_)),_) = true
        | is_global _ = false
      fun merge_ret ret ([],[]) = ret
        | merge_ret _ (_,[]) = error "cached_embed_premises BUG"
        | merge_ret (vecs', new_add, global_add) (vprem::vpremises, (tag_thm, NONE) :: vecs) =
            let val ele = (Thm.prop_of (snd tag_thm), vprem)
             in merge_ret ( (tag_thm, vprem)::vecs'
                          , ele::new_add
                          , if is_global tag_thm then ele::global_add else global_add )
                          (vpremises, vecs)
            end
        | merge_ret _ ([], (_, NONE) :: _) = error "cached_embed_premises BUG"
        | merge_ret (vecs', new_add, global_add) (vpremises, (tag_thm, SOME some) :: vecs) =
            merge_ret ((tag_thm, some)::vecs', new_add, global_add) (vpremises, vecs)

      val (vpremises, new_add, global_add) = merge_ret ([],[],[]) (vpremises', vecs'0)

      val _ = if null new_add then ()
              else Synchronized.change vecs_var (fold Termtab.update_new new_add)
      val _ = case vecs_global_var_opt
                of NONE => ()
                 | SOME gvar =>
                    if null global_add then ()
                    else Synchronized.change gvar (fold Termtab.update_new global_add)
  in (vgoal_opt, vpremises)
  end

fun relevant_facts ctxt (params as {verbose, fact_filter, timeout, ...}) prover
      max_facts ({add, only, ...} : fact_override) hyp_ts concl_t facts =
  let val add_ths = Attrib.eval_thms ctxt add
      val unique_facts = drop_duplicate_facts facts

      fun embd () =
        let val vecs_global_var =
                  case Thm_Vector.get (Proof_Context.theory_of ctxt)
                    of SOME v => v
                     | NONE => error "Sledgehammer_Embedding BUG: variable Thm_Vector is not initialized"
            val vecs_var = Thm_Vector_Local.get ctxt

            val SOT = Premise_Embedding_Context_Info.string_of_term ctxt
            val defctxt_string_of = Premise_Embedding_Context_Info.defctxt_string_of ctxt
            val vars_of_term = vars_of_term ctxt

            val assms = map (SOT o Thm.term_of) (Assumption.all_assms_of ctxt)
            val goal = Logic.list_implies (hyp_ts, concl_t)
            val goal_ctxt = defctxt_string_of goal
            val goal_enc = (assms, SOT goal, vars_of_term goal)

            val (SOME vgoal, vpremises) =
                  cached_embed_premises ctxt (SOME vecs_global_var) vecs_var
                      (SOME (goal_enc, goal_ctxt)) unique_facts

            val vpremises : (Sledgehammer_Fact.lazy_fact * Word8Vector.vector) Array.array
                  = Array.fromList vpremises
            val top_k_facts = Vector_Arith.top_k (vpremises, vgoal, max_facts)
                          |> map (apfst (apfst (apfst (fn f => f ()))))
         in (top_k_facts, [])
        end

      fun mepo () =
        (mepo_suggested_facts ctxt params max_facts NONE hyp_ts concl_t unique_facts
         |> weight_facts_steeply, [])

      val mess = [(mepo_weight, mepo), (embd_weight, embd)]
        (* the order is important for the "case" expression below *)
           |> Par_List.map (apsnd (fn f => f ()))


      fun in_add (_, th) = member Thm.eq_thm_prop add_ths th

      fun add_and_take accepts =
        (case add_ths of
           [] => accepts
         | _ =>
           (unique_facts |> filter in_add |> map fact_of_lazy_fact)
           @ (accepts |> filter_out in_add))
        |> take max_facts

      val mesh =
        Sledgehammer_MaSh.mesh_facts (fact_distinct (op aconv)) (eq_snd (gen_eq_thm ctxt)) max_facts mess
          |> add_and_take
   in [("embd", mesh)]
(* [("mepo", mepo_suggested_facts ctxt params max_facts NONE hyp_ts concl_t unique_facts),
       (* ("embd", map fst top_k_facts) *)] *)
  end

val _ = Theory.setup (Sledgehammer_MaSh.add_custom_fact_filter "embd" relevant_facts)

(*
      val xtab = Termtab.build (fold (fn (_, prop) => Termtab.insert (op =) (Thm.prop_of prop, 1)) facts)
      val (time, _) = Timing.timing (List.app (fn (_, thm) => (
          Termtab.lookup xtab (Thm.prop_of thm); ()))) facts
      val _ = @{print} time
*)

val update_premises_lock = Synchronized.var "update_premises_lock" ()

fun update_premises thy =
  Synchronized.change update_premises_lock (fn () =>
  let val ctxt = Proof_Context.init_global thy
      val facts = Sledgehammer_Fact.all_facts ctxt false (Thy_Header.get_keywords thy) [] []
                      (Sledgehammer_Fact.clasimpset_rule_table_of ctxt)
                  |> drop_duplicate_facts
      val vecs_global_var =
            case Thm_Vector.get thy
              of SOME v => v
               | NONE => error "update_premises: Thm_Vector is not initialized"
      val _ = cached_embed_premises ctxt NONE vecs_global_var NONE facts
   in ()
  end)

val _ = Theory.setup (
            Thm_Vector.put (SOME (Synchronized.var "Thm_Vector" Termtab.empty))
         #> Theory.at_begin (fn thy =>
              let val env = getenv "ISABELLE_PREBUILD_PREMISE_EMBEDDING"
                  val enabled = (env = "yes" orelse env = "true")
               in if enabled then update_premises thy else ()
                ; NONE
              end))

end
