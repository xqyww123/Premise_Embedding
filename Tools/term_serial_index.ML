(*
  Term Serial Index: assigns globally unique serial numbers to terms.
  Alpha-equivalent terms share one serial.

  Uses iNet for fast term lookup and Thm.derivation_id for caching.
  All mutable state is guarded by a single Synchronized.var lock.
*)

signature TERM_SERIAL_INDEX = sig
  (* Get or assign serial for a term (new or existing for aconv-equal). *)
  val serial_of_term: term -> serial

  (* Get or assign serial for a thm: uses Thm.derivation_id cache when available,
     else falls back to serial_of_term on the proposition. *)
  val serial_of_thm: thm -> serial

  (* Reverse lookups *)
  val term_of_serial: serial -> term option
  val thm_of_serial: theory -> serial -> thm option
end

structure Term_Serial_Index : TERM_SERIAL_INDEX = struct

(* serial -> term, for aconv filtering after iNet match *)
val term_tab : term Inthashtab.table = Inthashtab.empty 256

(* derivation_id serial -> term serial *)
val deriv_cache : serial Inthashtab.table = Inthashtab.empty 256

(* serial -> thm, for reverse lookup from serial to theorem *)
val thm_tab : thm Inthashtab.table = Inthashtab.empty 256

(* iNet maps terms to serials; guarded by Synchronized.var *)
val state = Synchronized.var "TERM_SERIAL_INDEX" (iNet.empty : serial iNet.net)

(* Internal: find existing serial for an aconv-equal term in the net.
   Must be called within the Synchronized.change critical section. *)
fun lookup_aconv net term =
  let
    val candidates = iNet.match_term net term
  in
    List.find (fn s =>
      case Inthashtab.lookup term_tab s of
        SOME t => Term.aconv (t, term)
      | NONE => false) candidates
  end

(* Internal: get or assign serial within the critical section, returning (serial, updated_net). *)
fun serial_of_term_internal net term =
  case lookup_aconv net term of
    SOME s => (s, net)
  | NONE =>
      let
        val s = serial ()
        val _ = Inthashtab.update term_tab (s, term)
        val net' = iNet.insert_term (op =) (term, s) net
      in (s, net') end

fun serial_of_term term =
  Synchronized.change_result state (fn net => serial_of_term_internal net term)

fun serial_of_thm thm =
  Synchronized.change_result state (fn net =>
    let val (s, net') =
      case Thm.derivation_id thm of
        SOME {serial = did, ...} =>
          (case Inthashtab.lookup deriv_cache did of
            SOME s => (s, net)
          | NONE =>
              let
                val (s, net') = serial_of_term_internal net (Thm.prop_of thm)
                val _ = Inthashtab.update deriv_cache (did, s)
              in (s, net') end)
      | NONE => serial_of_term_internal net (Thm.prop_of thm)
    in Inthashtab.update thm_tab (s, thm); (s, net') end)

fun term_of_serial s = Inthashtab.lookup term_tab s

fun thm_of_serial thy s =
  Option.map (Global_Theory.transfer_theories thy) (Inthashtab.lookup thm_tab s)

end
