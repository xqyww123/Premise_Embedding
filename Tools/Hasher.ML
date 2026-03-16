(*
  Fast Non-Cryptographic Hash Library using FNV-1a algorithm

  Provides both 64-bit and 128-bit digests for strings, types, terms, and goals.

  The 128-bit variant runs two independent FNV-1a instances with different
  primes/offsets and pairs their results.

  ALGORITHM: FNV-1a (Fowler-Noll-Vo) 64-bit variant
    - For each byte: hash = (hash XOR byte) * prime
    - Fast, simple, and good distribution for non-cryptographic use

  KEY OPTIMIZATION TECHNIQUES:
    1. Direct fold accumulation without intermediate string lists
    2. Byte-level string processing via Byte.stringToBytes (zero-overhead cast)
    3. Imperative while loop for inner byte iteration
    4. Direct integer hashing (no string_of_int)
    5. Constructor tagging for structural distinction
*)

functor Hasher_Fn(val fnv_offset_basis : Word64.word val fnv_prime : Word64.word) :
sig
  type digest = Word64.word
  val to_hex : digest -> string
  val string : string -> digest
  val class  : class -> digest
  val sort   : sort -> digest
  val typ    : typ -> digest
  val term   : term -> digest
  val goal   : context_state -> digest
end =
struct

type digest = Word64.word

(* Convert Word64 to 16-char hex string *)
fun to_hex w =
  let
    val hex_digits = "0123456789abcdef"
    fun digit i = String.sub (hex_digits, i)
    fun loop i acc =
      if i < 0 then acc
      else
        let
          val b = Word8.fromLargeWord (Word64.>> (w, Word.fromInt (i * 8)))
          val hi = Word8.toInt (Word8.>> (b, 0w4))
          val lo = Word8.toInt (Word8.andb (b, 0w15))
        in
          loop (i - 1) (digit hi :: digit lo :: acc)
        end
  in
    String.implode (loop 7 [])
  end

(* Hash a string using imperative while loop with direct byte access *)
fun hash_string (s: string) (h: Word64.word) : Word64.word =
  let
    val bytes = Byte.stringToBytes s
    val len = Word8Vector.length bytes
    val hash_ref = Unsynchronized.ref h
    val i = Unsynchronized.ref 0
  in
    while !i < len do
      let
        val byte = Word8Vector.sub (bytes, !i)
        val w64 = Word64.fromLargeWord (Word8.toLargeWord byte)
        val h_val = !hash_ref
      in
        hash_ref := Word64.* (Word64.xorb (h_val, w64), fnv_prime);
        i := !i + 1
      end;
    !hash_ref
  end

fun hash_tagged (tag: Word64.word) (s: string) (h: Word64.word) : Word64.word =
  hash_string s (Word64.xorb (h, tag))

fun hash_int (i: int) (h: Word64.word) : Word64.word =
  Word64.* (Word64.xorb (h, Word64.fromInt i), fnv_prime)

(* Constructor tags *)
val tag_Type  = 0wx1 : Word64.word
val tag_TFree = 0wx2 : Word64.word
val tag_TVar  = 0wx3 : Word64.word
val tag_Const = 0wx4 : Word64.word
val tag_Free  = 0wx5 : Word64.word
val tag_Var   = 0wx6 : Word64.word
val tag_Bound = 0wx7 : Word64.word
val tag_Abs   = 0wx8 : Word64.word
val tag_App   = 0wx9 : Word64.word

fun hash_typ (ty: typ) (h: Word64.word) : Word64.word =
  case ty of
    Type (name, args) =>
      fold hash_typ args (hash_tagged tag_Type name h)
  | TFree (name, sort) =>
      fold hash_string sort (hash_tagged tag_TFree name h)
  | TVar ((name, idx), sort) =>
      fold hash_string sort (hash_int idx (hash_tagged tag_TVar name h))

fun hash_term (t: term) (h: Word64.word) : Word64.word =
  case t of
    Const (name, ty) =>
      hash_typ ty (hash_tagged tag_Const name h)
  | Free (name, ty) =>
      hash_typ ty (hash_tagged tag_Free name h)
  | Var ((name, idx), ty) =>
      hash_typ ty (hash_int idx (hash_tagged tag_Var name h))
  | Bound idx =>
      hash_int idx (Word64.xorb (h, tag_Bound))
  | Abs (name, ty, body) =>
      hash_term body (hash_typ ty (hash_tagged tag_Abs name h))
  | t1 $ t2 =>
      hash_term t2 (hash_term t1 (Word64.xorb (h, tag_App)))

(* Public API *)
fun string s = hash_string s fnv_offset_basis

fun class c = string c

fun sort s = fold hash_string s fnv_offset_basis

fun typ ty = hash_typ ty fnv_offset_basis

fun term t = hash_term t fnv_offset_basis

fun goal (ctxt, stat) =
  let
    val thy_name = Context.theory_name {long=false} (Proof_Context.theory_of ctxt)
    val h0 = hash_string thy_name fnv_offset_basis
    val h1 = fold (hash_term o Thm.term_of) (Assumption.all_assms_of ctxt) h0
  in
    hash_term (Thm.prop_of stat) h1
  end

end

signature TERM_DIGEST = sig

type digest = Word64.word
type digest128 = Word64.word * Word64.word

val to_hex : digest -> string
val string : string -> digest
val class  : class -> digest
val sort   : sort -> digest
val typ    : typ -> digest
val term   : term -> digest
val goal   : context_state -> digest

val digest_hasher : digest -> word
val digest128_hasher : digest128 -> word

val to_hex128 : digest128 -> string
val string128 : string -> digest128
val class128  : class -> digest128
val sort128   : sort -> digest128
val typ128    : typ -> digest128
val term128   : term -> digest128
val thm128    : thm -> digest128
val goal128   : context_state -> digest128

end

structure Term_Digest : TERM_DIGEST = struct

type digest = Word64.word
type digest128 = Word64.word * Word64.word

structure Hasher_Lo = Hasher_Fn(
  val fnv_offset_basis : Word64.word = 0wxcbf29ce484222325
  val fnv_prime : Word64.word = 0wx100000001b3)

(* Second instance uses the high half of the official FNV-1a 128-bit
   offset basis (0x6c62272e07bb014262b821756295c58d) and a different
   FNV-form prime: 2^40 + 2^8 + 0xc3 = 0x100000001c3. *)
structure Hasher_Hi = Hasher_Fn(
  val fnv_offset_basis : Word64.word = 0wx6c62272e07bb0142
  val fnv_prime : Word64.word = 0wx100000001c3)

open Hasher_Lo

fun digest_hasher (w : digest) : word = Word.fromLargeWord (Word64.toLargeWord w)
fun digest128_hasher (lo, hi : digest) : word =
  Word.xorb (digest_hasher lo, Word.fromLargeWord (Word64.toLargeWord (Word64.>> (hi, 0w1))))

fun to_hex128 (lo, hi) = Hasher_Lo.to_hex lo ^ Hasher_Hi.to_hex hi
fun string128 s = (Hasher_Lo.string s, Hasher_Hi.string s)
fun class128 c = (Hasher_Lo.class c, Hasher_Hi.class c)
fun sort128 s = (Hasher_Lo.sort s, Hasher_Hi.sort s)
fun typ128 t = (Hasher_Lo.typ t, Hasher_Hi.typ t)
fun term128 t = (Hasher_Lo.term t, Hasher_Hi.term t)
fun goal128 g = (Hasher_Lo.goal g, Hasher_Hi.goal g)

val thm128_cache : digest128 Inthashtab.table Synchronized.var =
  Synchronized.var "Hasher.thm128_cache" (Inthashtab.empty 1024)

fun thm128 thm =
  case Thm.derivation_id thm of
    NONE => term128 (Thm.prop_of thm)
  | SOME {serial, ...} =>
      Synchronized.change_result thm128_cache (fn tab =>
        case Inthashtab.lookup tab serial of
          SOME h => (h, tab)
        | NONE =>
            let val h = term128 (Thm.prop_of thm)
            in (h, (Inthashtab.update tab (serial, h); tab)) end)

end
