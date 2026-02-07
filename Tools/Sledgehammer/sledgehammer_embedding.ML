signature SLEDGEHAMMER_EMBEDDING = sig
end

structure Sledgehammer_Embedding : SLEDGEHAMMER_EMBEDDING = struct

local open MessagePackBinIO.Pack MessagePackBinIO.Unpack
  val pack_defctxt = packString
  val pack_term = packString
  val pack_goal = packTuple3 (packList pack_term, pack_term, pack_defctxt)
  val pack_premise = packPair (pack_term, pack_defctxt)

  val unpackVec = unpackBytes
in
val embed_cmd = {
  name = "embed_goal_premises",
  arg_schema = packPair (pack_goal, packList pack_premise),
  ret_schema = unpackPair (unpackVec, unpackList unpackVec),
  callback = NONE,
  timeout = SOME (Time.fromSeconds 120)
}
end

structure Vector_Arith = Vector_Arith_Q15_D384
type vector = Vector_Arith.vector

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

val _ = Theory.setup (Thm_Vector.put (SOME (Synchronized.var "Thm_Vector" Termtab.empty)))

val mepo_weight = 0.5 (* FUDGE *)
val embd_weight = 0.5 (* FUDGE *)

fun gen_eq_thm ctxt = if Config.get ctxt Sledgehammer_MaSh.duplicates
                      then Thm.eq_thm_strict
                      else Thm.eq_thm_prop

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
      
            val assms = map (SOT o Thm.term_of) (Assumption.all_assms_of ctxt)
            val goal = Logic.list_implies (hyp_ts, concl_t)
            val goal_ctxt = defctxt_string_of goal
      
            val (vecs'0, query) = Synchronized.change_result vecs_var (fn vec_store =>
              let val vecs'0 = map (fn origin as ((_, (scope, _)), thm) =>
                        let val prop = Thm.prop_of thm
                         in (origin, Termtab.lookup vec_store prop)
                        end) facts
                  val query = map_filter (
                      fn ((tag, thm), NONE) =>
                          let val prop = Thm.prop_of thm
                           in SOME (SOT prop, defctxt_string_of prop)
                          end
                       | _ => NONE) vecs'0
               in ((vecs'0, query), vec_store)
              end)
      
            val (vgoal, vpremises') =
                  Remote_Procedure_Calling.call_command embed_cmd ((assms, SOT goal, goal_ctxt), query)
      
            fun is_global ((_,(Global,_)),_) = true
              | is_global _ = false
            fun merge_ret ret ([],[]) = ret
              | merge_ret _ (_,[]) = error "relevant_facts BUG"
              | merge_ret (vecs', new_add, global_add) (vprem::vpremises, (tag_thm, NONE) :: vecs) =
                  let val ele = (Thm.prop_of (snd tag_thm), vprem)
                   in merge_ret ( (tag_thm, vprem)::vecs'
                                , ele::new_add
                                , if is_global tag_thm then ele::global_add else global_add )
                                (vpremises, vecs)
                  end
              | merge_ret _ ([], (_, NONE) :: _) = error "relevant_facts BUG"
              | merge_ret (vecs', new_add, global_add) (vpremises, (tag_thm, SOME some) :: vecs) =
                  merge_ret ((tag_thm, some)::vecs', new_add, global_add) (vpremises, vecs)
      
            val (vpremises, new_add, global_add) = merge_ret ([],[],[]) (vpremises', vecs'0)
            val vpremises : (Sledgehammer_Fact.lazy_fact * Word8Vector.vector) Array.array
                  = Array.fromList vpremises
      
            val _ = if null new_add then ()
                    else Synchronized.change vecs_var (fold Termtab.update_new new_add)
            val _ = if null global_add then ()
                    else Synchronized.change vecs_global_var (fold Termtab.update_new global_add)
            val top_k_facts = Vector_Arith.top_k (vpremises, vgoal, max_facts)
                          |> map (apfst (apfst (apfst (fn f => f ()))))
         in (top_k_facts, [])
        end
(*
      val vector_facts = map (fn (tag as (_, (scope, _)), thm) =>
          let 
           in x
          end) facts *)

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
end
