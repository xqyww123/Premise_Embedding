#!/usr/bin/env python3
"""Migrate semantics.lmdb entity records from legacy 4-tuples to the unified
5-tuple format (kind, name, expr, interpretation, provenance).

Legacy records gain provenance = None.  Theory-status records (16-byte keys,
msgpack dicts) are untouched.  Idempotent: 5-tuple records are skipped.

A timestamped backup copy of the environment is written next to the original
before any modification (lmdb's live-safe Environment.copy)."""

import os
import sys
import time

import lmdb
import msgpack
import platformdirs

DB_PATH = os.path.join(
    platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan"),
    "semantics.lmdb")


def main() -> None:
    if not os.path.isdir(DB_PATH):
        sys.exit(f"semantic DB not found: {DB_PATH}")

    backup = f"{DB_PATH}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(backup)
    env = lmdb.open(DB_PATH, map_size=1 << 30)
    env.copy(backup, compact=True)
    print(f"backup written: {backup}")

    migrated = skipped5 = theory = other = 0
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            key = bytes(key)
            if len(key) == 16:
                theory += 1
                continue
            vals = msgpack.unpackb(val)
            if not isinstance(vals, (list, tuple)):
                other += 1
                continue
            if len(vals) == 5:
                skipped5 += 1
                continue
            if len(vals) != 4:
                other += 1
                print(f"  unexpected record arity {len(vals)} at key {key.hex()[:16]}…, left as is")
                continue
            kind, name, expr, sem = vals
            txn.put(key, msgpack.packb((kind, name, expr, sem, None)))  # type: ignore[arg-type]
            migrated += 1
    env.close()
    print(f"migrated {migrated} records to 5-tuple; "
          f"already-5-tuple {skipped5}, theory-status {theory}, other {other}")


if __name__ == "__main__":
    main()
