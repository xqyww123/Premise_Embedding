#!/usr/bin/env python3
"""Manage the Isabelle semantic interpretation database.

Subcommands:
  collect   Collect semantic interpretations for a theory (requires Isa-REPL)
  list      List all theories in the semantic database (offline)
  remove    Remove specific theories from the database (offline)
  reindex   Rebuild experience_index.lmdb from semantics.lmdb (offline)
  fsck      Check semantics.lmdb invariants; --fix repairs the derived ones (offline)
"""
import argparse
import os
import sys
from collections import defaultdict

import lmdb
import msgpack
import platformdirs

from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key


CACHE_DIR = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
SEMANTICS_DB_PATH = os.path.join(CACHE_DIR, "semantics.lmdb")
THEORY_HASH_CACHE_DIR = platformdirs.user_cache_dir("Isabelle_Theory_Hash", "Qiyuan")
THEORY_HASH_DB_PATH = os.path.join(THEORY_HASH_CACHE_DIR, "theory_hash.lmdb")


def _record_constituent_hashes(raw: bytes) -> 'set[bytes] | None':
    """Constituent theory hashes of a theorem/rule record (the theory_constituents
    field — the 6th tuple element, index 5; layout mirrors _Semantic_DB._decode).
    Theorem/rule keys carry an XOR pseudo-theory prefix, so membership in a theory
    is decided by this list, never by the key prefix.  None for legacy records."""
    vals = msgpack.unpackb(raw)
    if len(vals) <= 5 or vals[5] is None:
        return None
    return {bytes(h) for _, h in vals[5]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer")
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _load_theory_names() -> dict[bytes, str]:
    """Load hash→name mapping from theory_hash.lmdb."""
    if not os.path.exists(THEORY_HASH_DB_PATH):
        return {}
    env = lmdb.open(THEORY_HASH_DB_PATH, readonly=True, lock=False)
    result: dict[bytes, str] = {}
    with env.begin() as txn:
        for key, val in txn.cursor():
            name, _ts = msgpack.unpackb(val)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            result[bytes(key)] = name
    env.close()
    return result


def _vector_store_paths() -> list[str]:
    """Return paths of all vector_*.lmdb stores on disk."""
    if not os.path.isdir(CACHE_DIR):
        return []
    paths = []
    for entry in os.listdir(CACHE_DIR):
        if entry.startswith("vector_") and entry.endswith(".lmdb"):
            path = os.path.join(CACHE_DIR, entry)
            if os.path.isdir(path):
                paths.append(path)
    return paths


def _resolve_identifiers(identifiers: list[str],
                         thy_hashes_in_db: set[bytes],
                         hash_to_name: dict[bytes, str],
                         ) -> list[bytes]:
    """Resolve theory names or hex prefixes to theory hashes.

    Returns list of resolved hashes. Prints errors and returns empty list
    on ambiguity or missing identifiers.
    """
    name_to_hashes: dict[str, list[bytes]] = defaultdict(list)
    for h, name in hash_to_name.items():
        if h in thy_hashes_in_db:
            name_to_hashes[name].append(h)

    resolved: list[bytes] = []
    had_error = False

    for ident in identifiers:
        # Try as theory name first
        if ident in name_to_hashes:
            matches = name_to_hashes[ident]
            if len(matches) == 1:
                resolved.append(matches[0])
                continue
            else:
                print(f"Error: '{ident}' matches multiple universal keys:", file=sys.stderr)
                for h in matches:
                    print(f"  {h.hex()}", file=sys.stderr)
                print(f"Use a universal key prefix to disambiguate.", file=sys.stderr)
                had_error = True
                continue

        # Try as hex prefix
        try:
            prefix_bytes = bytes.fromhex(ident) if len(ident) % 2 == 0 else None
        except ValueError:
            prefix_bytes = None

        if prefix_bytes is not None:
            matches = [h for h in thy_hashes_in_db if h.startswith(prefix_bytes)]
        else:
            matches = [h for h in thy_hashes_in_db if h.hex().startswith(ident.lower())]

        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) == 0:
            print(f"Error: '{ident}' does not match any theory in the database.", file=sys.stderr)
            had_error = True
        else:
            print(f"Error: '{ident}' is ambiguous, matches {len(matches)} theories:", file=sys.stderr)
            for h in matches:
                name = hash_to_name.get(h, "?")
                print(f"  {h.hex()}  {name}", file=sys.stderr)
            had_error = True

    if had_error:
        return []
    return resolved


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    if not os.path.exists(SEMANTICS_DB_PATH):
        print(f"No semantic database found at {SEMANTICS_DB_PATH}")
        return

    hash_to_name = _load_theory_names()

    # Scan semantics.lmdb
    env = lmdb.open(SEMANTICS_DB_PATH, readonly=True, lock=False)
    theory_meta: dict[bytes, dict] = {}  # theory_hash -> {finished, cost_usd}
    entity_counts: dict[bytes, int] = defaultdict(int)

    legacy_thm_count = 0
    with env.begin() as txn:
        for key, val in txn.cursor():
            key = bytes(key)
            if len(key) == 16:
                data = msgpack.unpackb(val)
                finished = data.get(b"finished", data.get("finished", False))
                cost = data.get(b"cost_usd", data.get("cost_usd", 0.0))
                model = data.get(b"model", data.get("model", b""))
                if isinstance(model, bytes):
                    model = model.decode("utf-8", errors="replace")
                theory_meta[key] = {"finished": finished, "cost_usd": cost, "model": model}
            elif is_xor_prefixed_key(key):
                # XOR pseudo-theory prefix: attribute to each constituent
                # theory (a record mentioning N theories is counted N times)
                consts = _record_constituent_hashes(bytes(val))
                if consts is None:
                    legacy_thm_count += 1
                else:
                    for h in consts:
                        entity_counts[h] += 1
            elif len(key) > 16:
                entity_counts[key[:16]] += 1
    env.close()

    all_hashes = sorted(
        set(theory_meta) | set(entity_counts),
        key=lambda h: hash_to_name.get(h, h.hex()),
    )

    if not all_hashes:
        print("Database is empty.")
        return

    # Print table
    name_w = max(max((len(hash_to_name.get(h, "?")) for h in all_hashes), default=6), 6)
    name_w = min(name_w, 60)

    print(f"{'Theory':<{name_w}}  {'Entities':>8}  {'Status':<8}  {'Cost':>9}  {'Model':<20}  Universal Key")
    print("─" * (name_w + 8 + 8 + 9 + 20 + 16 + 12))

    for h in all_hashes:
        name = hash_to_name.get(h, "?")
        count = entity_counts.get(h, 0)
        meta = theory_meta.get(h, {})
        finished = meta.get("finished", False)
        cost = meta.get("cost_usd", 0.0)
        model = meta.get("model", "")
        status = "done" if finished else "WIP"
        from Isabelle_RPC_Host.theory_hash import is_persistent
        if not is_persistent(h):
            status += " *"
        print(f"{name:<{name_w}}  {count:>8}  {status:<8}  ${cost:>8.4f}  {model:<20}  {h.hex()}")

    print()
    total_entities = sum(entity_counts.values())
    n_done = sum(1 for h in all_hashes if theory_meta.get(h, {}).get("finished", False))
    print(f"{len(all_hashes)} theories ({n_done} done, {len(all_hashes) - n_done} WIP), "
          f"{total_entities} entities total "
          f"(theorem/rule records are counted once per constituent theory)")
    if legacy_thm_count:
        print(f"WARNING: {legacy_thm_count} legacy theorem/rule records without "
              f"constituent lists (pre-XOR keys); run migrate_xor_thm_keys.py to purge them.")


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def cmd_remove(args: argparse.Namespace) -> None:
    if not os.path.exists(SEMANTICS_DB_PATH):
        print(f"No semantic database found at {SEMANTICS_DB_PATH}", file=sys.stderr)
        sys.exit(1)

    hash_to_name = _load_theory_names()

    # Discover all theory hashes present in semantics DB.  Theorem/rule keys
    # carry XOR pseudo-theory prefixes: their theories come from the record's
    # constituent list, never from the key prefix.
    env = lmdb.open(SEMANTICS_DB_PATH, readonly=True, lock=False)
    thy_hashes_in_db: set[bytes] = set()
    with env.begin() as txn:
        for key, val in txn.cursor():
            key = bytes(key)
            if is_xor_prefixed_key(key):
                thy_hashes_in_db |= _record_constituent_hashes(bytes(val)) or set()
            else:
                thy_hashes_in_db.add(key[:16])
    env.close()

    resolved = _resolve_identifiers(args.identifiers, thy_hashes_in_db, hash_to_name)
    if not resolved:
        sys.exit(1)
    resolved_set = set(resolved)

    # Count what will be deleted from semantics DB; collect the theorem/rule
    # keys to delete (membership = the constituent list mentions a target).
    env = lmdb.open(SEMANTICS_DB_PATH, readonly=True, lock=False)
    del_counts: dict[bytes, int] = defaultdict(int)
    thm_keys_to_delete: set[bytes] = set()
    with env.begin() as txn:
        for key, val in txn.cursor():
            key = bytes(key)
            if is_xor_prefixed_key(key):
                consts = _record_constituent_hashes(bytes(val))
                if consts:
                    matched = consts & resolved_set
                    if matched:
                        thm_keys_to_delete.add(key)
                        for thy in matched:
                            del_counts[thy] += 1
            elif key[:16] in resolved_set and len(key) > 16:
                del_counts[key[:16]] += 1
    env.close()

    # Print summary
    print("Will remove:")
    for h in resolved:
        name = hash_to_name.get(h, "?")
        count = del_counts.get(h, 0)
        print(f"  {name:<50}  {count:>5} entities  [{h.hex()}]")

    vec_paths = _vector_store_paths()
    if vec_paths:
        print(f"Also cleaning {len(vec_paths)} vector store(s).")

    if not args.force:
        try:
            answer = input("\nConfirm? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if answer != "y":
            print("Aborted.")
            return

    # Delete from semantics DB: prefix-addressed keys by prefix, theorem/rule
    # keys from the constituent-matched set collected above.
    total_deleted = 0
    env = lmdb.open(SEMANTICS_DB_PATH, map_size=1 << 33)
    with env.begin(write=True) as txn:
        to_delete: list[bytes] = []
        for key, _ in txn.cursor():
            key = bytes(key)
            if key in thm_keys_to_delete or \
               (not is_xor_prefixed_key(key) and key[:16] in resolved_set):
                to_delete.append(key)
        for key in to_delete:
            txn.delete(key)
        total_deleted += len(to_delete)
    env.close()

    # Delete from vector stores (keyed by the same universal keys; theorem/
    # rule vectors are matched via the semantic-DB key set since vector
    # stores hold no constituent lists)
    vec_deleted = 0
    for path in vec_paths:
        venv = lmdb.open(path, map_size=1 << 33)
        with venv.begin(write=True) as txn:
            to_delete = []
            for key, _ in txn.cursor():
                key = bytes(key)
                if key in thm_keys_to_delete or \
                   (not is_xor_prefixed_key(key) and key[:16] in resolved_set):
                    to_delete.append(key)
            for key in to_delete:
                txn.delete(key)
            vec_deleted += len(to_delete)
        venv.close()

    theories_word = "theory" if len(resolved) == 1 else "theories"
    print(f"Removed {len(resolved)} {theories_word} ({total_deleted} entries from semantics DB"
          f", {vec_deleted} from vector stores).")


# ---------------------------------------------------------------------------
# reindex / fsck
# ---------------------------------------------------------------------------

def _require_db() -> None:
    if not os.path.exists(SEMANTICS_DB_PATH):
        print(f"No semantic database found at {SEMANTICS_DB_PATH}", file=sys.stderr)
        sys.exit(1)


def cmd_reindex(args: argparse.Namespace) -> None:
    _require_db()
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    n = Semantic_DB.rebuild_experience_index()
    print(f"Rebuilt the experience index from {n} EXPERIENCE record(s) in semantics.lmdb.")


def cmd_fsck(args: argparse.Namespace) -> None:
    """Check the invariants of semantics.lmdb that can break silently.

    Deliberately NOT checked: whether a record has a vector.  Vectors are a
    lazily-filled derived cache — topk hands unknown keys to _auto_embed, which
    embeds anything whose interpretation is already stored.  A record with no
    vector is legitimate; reporting a cold cache as damage would be noise.

    Both repairable classes are repaired the same way, by recomputing a derived
    artefact from the primary data it is derived from:
      * the experience index is a derived view of the EXPERIENCE records -> rebuild
      * an XOR key prefix is derived from the record's constituent list -> re-key
    """
    _require_db()
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB, SEMANTICS_MAP_SIZE
    from Isabelle_Semantic_Embedding.experience_index import Experience_Index

    c = Semantic_DB.check_consistency()
    indexed = Experience_Index.all_keys()
    missing_from_index = c.experience_keys - indexed
    stale_in_index = indexed - c.experience_keys

    def row(label: str, n: int, note: str = "") -> None:
        print(f"  {label:<46}{n:>7}{'   ' + note if note else ''}")

    print(f"semantics.lmdb   : {c.n_records} records, of which {len(c.experience_keys)} experiences")
    print(f"experience_index : {len(indexed)} keys")
    print()

    print("[repairable by --fix]")
    row("EXPERIENCE record present, missing from index", len(missing_from_index))
    row("index key with no EXPERIENCE record", len(stale_in_index))
    row("XOR key prefix disagrees with constituents", len(c.xor_mismatches))
    for bad, good in c.xor_mismatches[:5]:
        rec = Semantic_DB[bad]
        print(f"      {rec.name if rec else '?'}")
        print(f"        prefix {bad[:16].hex()} -> should be {good[:16].hex()}")
        if rec is not None and rec.theory_constituents:
            print(f"        constituents: {', '.join(n for n, _ in rec.theory_constituents)}")
    if len(c.xor_mismatches) > 5:
        print(f"      ... and {len(c.xor_mismatches) - 5} more")
    if c.xor_mismatches:
        print("      !! This invariant cannot break on its own. Something wrote a key that")
        print("         disagrees with its own constituent list — check that ML's")
        print("         Universal_Key.compute_constituents still mirrors xor_theory_prefix.")

    print("\n[report only]")
    row("legacy XOR record (no constituent list)", c.legacy_xor, "(run migrate_xor_thm_keys.py)")

    db_bytes = os.path.getsize(os.path.join(SEMANTICS_DB_PATH, "data.mdb"))
    pct = 100.0 * db_bytes / SEMANTICS_MAP_SIZE
    print(f"\n  semantics.lmdb size {db_bytes / 1024 ** 2:.1f} MiB / "
          f"{SEMANTICS_MAP_SIZE / 1024 ** 3:.0f} GiB map_size ({pct:.1f}%)"
          f"{'   ** writes fail once this reaches 100% **' if pct > 80 else ''}")

    problems = (len(missing_from_index) + len(stale_in_index)
                + len(c.xor_mismatches) + c.legacy_xor)

    if args.fix:
        repaired = 0
        if c.xor_mismatches:
            print("\n--fix: re-keying records whose XOR prefix disagrees with their constituents...")
            moved, conflicts = Semantic_DB.repair_xor_prefixes(c.xor_mismatches)
            for bad, good in moved:
                print(f"  moved {bad.hex()[:24]}… -> {good.hex()[:24]}…")
            repaired += len(moved)
            for bad in conflicts:
                print(f"  CONFLICT, left alone: {bad.hex()} — the correct key already holds "
                      f"a different record", file=sys.stderr)
        if missing_from_index or stale_in_index or repaired:
            print("\n--fix: rebuilding the experience index...")
            # The index is derived, so one rebuild subsumes every index-level
            # discrepancy: the entries missing above, the stale ones, and the
            # re-keyed experiences just moved by repair_xor_prefixes.
            n = Semantic_DB.rebuild_experience_index()
            print(f"  Rebuilt from {n} EXPERIENCE record(s) in semantics.lmdb.")
            repaired += len(missing_from_index) + len(stale_in_index)
        problems -= repaired

    print()
    if problems:
        print(f"{problems} problem(s) remain.")
        sys.exit(1)
    print("All checks passed.")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> None:
    if args.reinterpret and args.migrate_on_hash_change:
        print("Error: --reinterpret and --migrate-on-hash-change are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)
    if args.re_embed and args.reinterpret:
        print("Error: --re-embed and --reinterpret are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)
    if args.re_embed and args.migrate_on_hash_change:
        print("Error: --re-embed and --migrate-on-hash-change are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)
    if args.re_embed and not args.embed_models:
        print("Error: --re-embed requires --embed-models.", file=sys.stderr)
        sys.exit(1)

    import threading
    import time
    import asyncio
    import Isabelle_RPC_Host
    import Isabelle_Semantic_Embedding
    import Isabelle_Semantic_Embedding.semantic_interpretation as si
    si.interpretation_model = args.model
    import Isabelle_Semantic_Embedding.semantics as sem
    sem.migrate_on_hash_change = args.migrate_on_hash_change
    from IsaREPL import Client

    import socket
    host, port = args.rpc_addr.split(":")
    port = int(port)
    rpc_already_running = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        rpc_already_running = s.connect_ex((host, port)) == 0

    if rpc_already_running:
        print(f"RPC server already running on {args.rpc_addr}, reusing.", flush=True)
    else:
        logger = Isabelle_RPC_Host.mk_logger_(args.rpc_addr, None)
        rpc_thread = threading.Thread(
            target=Isabelle_RPC_Host.launch_server_,
            args=(args.rpc_addr, logger), daemon=True)
        rpc_thread.start()
        time.sleep(1)

    async def main():
        async with Client(args.repl_addr, args.session, timeout=None) as c:
            await c.set_register_thy(False)
            print("Loading theories...", flush=True)
            fullnames = await c.load_theory([args.theory, "Semantic_Embedding.Semantic_Collection_App"])
            print(f"Loaded: {fullnames}", flush=True)

            print("Running app...", flush=True)
            await c.run_app("Semantic_Store.collect")
            models = [m.strip() for m in args.embed_models.split(",") if m.strip()] if args.embed_models else []
            await c._write(args.theory, models, args.reinterpret, args.re_embed, args.parallel)

            has_error = False
            try:
                while True:
                    raw = await c._feed_and_unpack()
                    if isinstance(raw, (list, tuple)) and len(raw) == 2:
                        msg, err = raw
                        if err is not None and err != ():
                            err_str = err.decode("utf-8") if isinstance(err, bytes) else str(err)
                            print(err_str, file=sys.stderr, flush=True)
                            has_error = True
                            continue
                        if msg is None or msg == ():
                            if (msg is None or msg == ()) and (err is None or err == ()):
                                break
                            continue
                        if isinstance(msg, bytes):
                            msg = msg.decode("utf-8", errors="replace")
                        if isinstance(msg, str):
                            if msg.startswith("ERROR:"):
                                print(msg, file=sys.stderr, flush=True)
                                has_error = True
                            else:
                                print(msg, flush=True)
                    elif raw is None:
                        break
                    else:
                        print(f"[unexpected: {raw!r}]", flush=True)
            except Exception as e:
                print(f"Connection error: {e}", file=sys.stderr, flush=True)
                has_error = True
                raise

            if has_error:
                print("Failed.", file=sys.stderr)
                sys.exit(1)
            else:
                print("Done.")

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Manage the Isabelle semantic interpretation database.")
sub = parser.add_subparsers(dest="command", required=True)

# collect
p_collect = sub.add_parser("collect", help="Collect semantic interpretations for a theory")
p_collect.add_argument("theory", help="Theory name to interpret (e.g., HOL.List)")
p_collect.add_argument("--repl-addr", default="127.0.0.1:6666", help="Isa-REPL server address")
p_collect.add_argument("--rpc-addr", default="127.0.0.1:27182", help="RPC host address")
p_collect.add_argument("--session", default="HOL", help="Session qualifier for theory name resolution")
p_collect.add_argument("--model", default="claude-opus-4-8[1m]",
    help="LLM model for semantic interpretation (default: claude-opus-4-8[1m])")
p_collect.add_argument("--embed-models", default="",
    help="Comma-separated canonical (HuggingFace) embedding model names "
         "(e.g., 'Qwen/Qwen3-Embedding-8B'). NOTE: all listed models are embedded "
         "via the one active driver+base_url, so they must be served by the same "
         "endpoint.")
p_collect.add_argument("--reinterpret", action="store_true",
    help="Re-interpret already-finished theories to pick up new entities")
p_collect.add_argument("--migrate-on-hash-change", action="store_true",
    help="Copy old data to new hash instead of re-interpreting when hash changes")
p_collect.add_argument("--re-embed", action="store_true",
    help="Re-embed vectors without re-interpreting (requires --embed-models)")
p_collect.add_argument("--parallel", type=_positive_int, metavar="N",
    help="Override semantic interpretation parallelism (default: Isabelle worker count)")

# list
p_list = sub.add_parser("list", help="List all theories in the semantic database")

# remove
p_remove = sub.add_parser("remove", help="Remove theories from the database")
p_remove.add_argument("identifiers", nargs="+",
    help="Theory names or universal key hex prefixes (from 'list' output)")
p_remove.add_argument("--force", action="store_true", help="Skip confirmation prompt")

# reindex
p_reindex = sub.add_parser("reindex",
    help="Rebuild experience_index.lmdb from the EXPERIENCE records in semantics.lmdb")

# fsck
p_fsck = sub.add_parser("fsck", help="Check semantics.lmdb invariants")
p_fsck.add_argument("--fix", action="store_true",
    help="Repair the derived artefacts: rebuild the experience index, and re-key "
         "records whose XOR prefix disagrees with their constituent list.")

args = parser.parse_args()
{"collect": cmd_collect, "list": cmd_list, "remove": cmd_remove,
 "reindex": cmd_reindex, "fsck": cmd_fsck}[args.command](args)
