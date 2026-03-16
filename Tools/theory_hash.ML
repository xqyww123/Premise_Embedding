(*
  Theory Hash: xxhash128-based content hashing for Isabelle theories.

  Computes a 16-byte hash for a theory that depends on:
    1. The theory's .thy file content
    2. The hashes of all parent theories (recursive)

  This means if any ancestor theory's file changes, the hash of all
  descendant theories also changes.

  Only theories from saved heap images can be hashed. Attempting to hash
  a theory that is not from a heap image (e.g., one being processed in
  jEdit) raises Cannot_Hash.

  The xxhash128 computation is delegated to Python via Isabelle RPC,
  since PolyML does not have a native xxhash implementation.
*)

signature THEORY_HASH = sig
  type hash = Word8Vector.vector  (* 16 bytes, xxhash128 *)
  exception Cannot_Hash of theory
  val hash_of : theory -> hash
  val theory_of : hash -> theory option
  val hasher : hash -> word
  val to_hex : hash -> string
end

structure Theory_Hash : THEORY_HASH = struct

type hash = Word8Vector.vector

exception Cannot_Hash of theory

(* RPC command: (file_path, parent_hashes) -> hash *)
local open MessagePackBinIO.Pack MessagePackBinIO.Unpack in

val xxhash128_cmd : (string * hash list, hash) Remote_Procedure_Calling.command = {
  name = "xxhash128_theory",
  arg_schema = packPair (packString, packList packBytes),
  ret_schema = unpackBytes,
  callback = [],
  timeout = SOME (Time.fromSeconds 60)
}

end

(* Convert hash to native word for use as Hash_Table key *)
val hasher : hash -> word =
  if Word.wordSize >= 64 then
    fn h =>
      let
        val lo = PackWord32Little.subVec (h, 0)
        val hi = PackWord32Little.subVec (h, 1)
      in
        Word.fromLargeWord (LargeWord.orb (lo, LargeWord.<< (hi, 0w32)))
      end
  else
    fn h => Word.fromLargeWord (PackWord32Little.subVec (h, 0))

(* Cache: theory identifier -> hash *)
val cache : hash Inttab.table Synchronized.var =
  Synchronized.var "Theory_Hash.cache" Inttab.empty

(* Reverse map: hash -> theory *)
structure Hash_Tab = Hash_Table(
  type key = hash
  val hash = hasher
  val eq = op = : hash * hash -> bool)

val hash_to_thy : theory Hash_Tab.table Synchronized.var =
  Synchronized.var "Theory_Hash.hash_to_thy" (Hash_Tab.empty 64)

fun theory_of h =
  Synchronized.change_result hash_to_thy (fn tab => (Hash_Tab.lookup tab h, tab))

(* Convert 16-byte hash to 32-char hex string *)
local
  val hex = "0123456789abcdef"
  fun byte_hex b =
    let val n = Word8.toInt b
    in String.implode [String.sub (hex, n div 16), String.sub (hex, n mod 16)]
    end
in
fun to_hex (h : hash) =
  String.concat (List.tabulate (Word8Vector.length h, fn i =>
    byte_hex (Word8Vector.sub (h, i))))
end

fun hash_of thy =
  let
    val id = Context.theory_identifier thy
  in
    case Synchronized.change_result cache (fn tab =>
      case Inttab.lookup tab id of
        SOME h => (SOME h, tab)
      | NONE => (NONE, tab)) of
      SOME h => h
    | NONE =>
        let
          val name = Context.theory_long_name thy

          (* Check: theory must be from a saved heap image *)
          val _ =
            if Resources.loaded_theory name then ()
            else raise Cannot_Hash thy

          (* Get file path; check existence *)
          val file_path = \<^try>\<open>Theory_Structure.get_theory_file_path thy
            catch _ => raise Cannot_Hash thy\<close>
          val _ =
            if File.is_file (Path.explode file_path) then ()
            else raise Cannot_Hash thy

          (* Recursively hash parent theories *)
          val parent_hashes = map hash_of (Theory.parents_of thy)

          (* RPC call to Python *)
          val _ = Remote_Procedure_Calling.load ["Isabelle_Semantic_Embedding"]
          val h = Remote_Procedure_Calling.call_command xxhash128_cmd
                    (file_path, parent_hashes)

          val _ =
            if Word8Vector.length h = 16 then ()
            else raise Cannot_Hash thy

          (* Cache the result *)
          val _ = Synchronized.change cache (Inttab.update (id, h))
          val _ = Synchronized.change hash_to_thy (fn tab => (Hash_Tab.update tab (h, thy); tab))
        in h end
  end

end
