(*
  Term Serial Index: assigns serial numbers to terms.

  Two-level caching:
  - Global registry (Synchronized.var + global hash tables) for theory-level entries.
  - Local registry (Proof_Data, lazily initialized) for proof-context-level entries.

  Theorems with a derivation_id are always registered globally.
  The same term may receive different serials in different registries,
  but one serial always corresponds to exactly one term/thm.
*)

signature TERM_SERIAL_INDEX = sig
  (* val serial_of_term: Context.generic -> term -> serial *)
  val serial_of_thm: Context.generic -> thm -> serial
  (* val term_of_serial: Context.generic -> serial -> term option *)
  val thm_of_serial: Context.generic -> serial -> thm option
end

structure Term_Serial_Index : TERM_SERIAL_INDEX = struct

(* --- Global state --- *)

(* serial -> term, for aconv filtering after iNet match *)
val term_tab : term Inthashtab.table = Inthashtab.empty 256

(* derivation_id serial -> term serial *)
val deriv_cache : serial Inthashtab.table = Inthashtab.empty 256

(* serial -> thm, for reverse lookup from serial to theorem *)
val thm_tab : thm Inthashtab.table = Inthashtab.empty 256

(* iNet maps terms to serials; guarded by Synchronized.var *)
val state = Synchronized.var "TERM_SERIAL_INDEX" (iNet.empty : serial iNet.net)

(* --- Local state (per proof context, lazily initialized) --- *)

type local_state = {
  net_ref: serial iNet.net Unsynchronized.ref,
  term_tab: term Inthashtab.table,
  thm_tab: thm Inthashtab.table
}

structure Local_Registry = Proof_Data (
  type T = local_state option Unsynchronized.ref
  val init = fn _ => Unsynchronized.ref NONE
)

fun get_local ctxt =
  let val r = Local_Registry.get ctxt
  in case !r of
       SOME st => st
     | NONE =>
         let val st : local_state = {
               net_ref = Unsynchronized.ref iNet.empty,
               term_tab = Inthashtab.empty 16,
               thm_tab = Inthashtab.empty 16
             }
             val _ = r := SOME st
         in st end
  end

fun get_local_opt ctxt = !(Local_Registry.get ctxt)

(* --- Internal helpers (parameterized by term_tab) --- *)

fun lookup_aconv ttab net term =
  let
    val candidates = iNet.match_term net term
  in
    List.find (fn s =>
      case Inthashtab.lookup ttab s of
        SOME t => Term.aconv (t, term)
      | NONE => false) candidates
  end

fun serial_of_term_internal ttab net term =
  case lookup_aconv ttab net term of
    SOME s => (s, net)
  | NONE =>
      let
        val s = serial ()
        val _ = Inthashtab.update ttab (s, term)
        val net' = iNet.insert_term (op =) (term, s) net
      in (s, net') end

(* --- Public interface --- *)

(*
fun serial_of_term (Context.Theory _) term =
      Synchronized.change_result state (fn net =>
        serial_of_term_internal term_tab net term)
  | serial_of_term (Context.Proof ctxt) term =
      Synchronized.change_result state (fn net =>
        let val {net_ref, term_tab = local_tt, ...} = get_local ctxt
            val (s, net') = serial_of_term_internal local_tt (!net_ref) term
            val _ = net_ref := net'
        in (s, net) end)
*)

fun serial_of_thm context thm =
  case Thm.derivation_id thm of
    SOME {serial = did, ...} =>
      (* Always global for theorems with derivation_id *)
      Synchronized.change_result state (fn net =>
        let val (s, net') =
          case Inthashtab.lookup deriv_cache did of
            SOME s => (s, net)
          | NONE =>
              let val (s, net') = serial_of_term_internal term_tab net (Thm.prop_of thm)
                  val _ = Inthashtab.update deriv_cache (did, s)
              in (s, net') end
        in Inthashtab.update thm_tab (s, Thm.trim_context thm); (s, net') end)
  | NONE =>
      (case context of
        Context.Theory _ =>
          Synchronized.change_result state (fn net =>
            let val (s, net') = serial_of_term_internal term_tab net (Thm.prop_of thm)
            in Inthashtab.update thm_tab (s, Thm.trim_context thm); (s, net') end)
      | Context.Proof ctxt =>
          Synchronized.change_result state (fn net =>
            let val {net_ref, term_tab = local_tt, thm_tab = local_tht} =
                  get_local ctxt
                val (s, net') = serial_of_term_internal local_tt (!net_ref) (Thm.prop_of thm)
                val _ = net_ref := net'
                val _ = Inthashtab.update local_tht (s, Thm.trim_context thm)
            in (s, net) end))

(*
fun term_of_serial (Context.Theory _) s = Inthashtab.lookup term_tab s
  | term_of_serial (Context.Proof ctxt) s =
      (case get_local_opt ctxt of
         SOME {term_tab = local_tt, ...} =>
           (case Inthashtab.lookup local_tt s of
              SOME t => SOME t
            | NONE => Inthashtab.lookup term_tab s)
       | NONE => Inthashtab.lookup term_tab s)
*)

fun thm_of_serial (Context.Theory thy) s =
      Option.map (Global_Theory.transfer_theories thy) (Inthashtab.lookup thm_tab s)
  | thm_of_serial (Context.Proof ctxt) s =
      let val thy = Proof_Context.theory_of ctxt
      in case get_local_opt ctxt of
           SOME {thm_tab = local_tht, ...} =>
             (case Inthashtab.lookup local_tht s of
                SOME thm => SOME thm
              | NONE => Option.map (Global_Theory.transfer_theories thy)
                          (Inthashtab.lookup thm_tab s))
         | NONE => Option.map (Global_Theory.transfer_theories thy)
                     (Inthashtab.lookup thm_tab s)
      end

end
