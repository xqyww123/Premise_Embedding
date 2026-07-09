"""Tests for the experience index and the semantics.lmdb consistency checks.

The experience index is a derived view of the EXPERIENCE records in
semantics.lmdb, kept in its own LMDB.  An experience is written to three stores
(semantics.lmdb, the vector store, this index) in three separate transactions
with no cross-store atomicity, so the index can drift; rebuilding it from the
records is the repair.  These tests pin down the rebuild's correctness and, more
importantly, the locking that stops the rebuild from *causing* the drift it
repairs.

Isolation: every test runs against a throwaway cache directory.  platformdirs
resolves ``user_cache_dir`` from ``XDG_CACHE_HOME`` on each call, but the package
caches its LMDB environments in class-level singletons, so the directory must be
set before anything opens one — hence the session-scoped autouse fixture and the
in-function imports.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import msgpack
import pytest

from Isabelle_RPC_Host.universal_key import EntityKind, xor_theory_prefix

H1 = bytes.fromhex("11" * 16)
H2 = bytes.fromhex("22" * 16)
EXP = int(EntityKind.EXPERIENCE)
THM = int(EntityKind.THEOREM)
CONST = int(EntityKind.CONSTANT)
GLOBAL_BUCKET = b"\x00" * 16


@pytest.fixture(scope="session", autouse=True)
def isolated_cache(tmp_path_factory):
    """Point the package's caches at a throwaway directory for the whole session."""
    cache = tmp_path_factory.mktemp("cache")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    import platformdirs
    resolved = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
    assert str(cache) in resolved, f"cache not isolated: {resolved}"
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _record(kind: int, name: str, constituents, interpretation: str = "when to use") -> bytes:
    """A stored record: (kind, name, expr, interpretation, prov, constituents, experience)."""
    return msgpack.packb((kind, name, "[]", interpretation, None, constituents, "how to prove"))


def _key(prefix: bytes, kind: int, suffix: bytes) -> bytes:
    return prefix + bytes([kind]) + suffix.ljust(15, b"\x00")


def _write_records(records: dict[bytes, bytes]) -> None:
    """Seed semantics.lmdb through the package's own environment.

    Not lmdb.open(): py-lmdb refuses to open one environment twice in a single
    process, even sequentially, and Semantic_DB holds a singleton handle.
    """
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    with Semantic_DB._ensure_env().begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k, v)


def _reset(isolated_cache: str) -> None:
    """Empty both stores between tests (the envs are singletons, so drop keys, not files)."""
    from Isabelle_Semantic_Embedding.experience_index import Experience_Index
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    for env in (Semantic_DB._ensure_env(), Experience_Index._ensure_env()):
        with env.begin(write=True) as txn:
            for k in [bytes(k) for k, _ in txn.cursor()]:
                txn.delete(k)


# ---------------------------------------------------------------------------
# rebuild correctness
# ---------------------------------------------------------------------------

def test_rebuild_derives_the_index_from_the_records(isolated_cache):
    from Isabelle_Semantic_Embedding.experience_index import Experience_Index
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    _reset(isolated_cache)

    one = _key(xor_theory_prefix([H1]), EXP, b"\x01")
    two = _key(xor_theory_prefix([H1, H2]), EXP, b"\x02")
    glob = _key(xor_theory_prefix([]), EXP, b"\x03")          # no constituents
    thm = _key(xor_theory_prefix([H1]), THM, b"\x04")         # not an experience
    const = _key(H1, CONST, b"\x05")                          # prefix-addressed entity

    _write_records({
        one: _record(EXP, "one", [("T1", H1)]),
        two: _record(EXP, "two", [("T1", H1), ("T2", H2)]),
        glob: _record(EXP, "glob", []),
        thm: _record(THM, "thm", [("T1", H1)]),
        const: _record(CONST, "const", None),
    })

    # A stale entry and two missing ones: the rebuild must fix both directions.
    Experience_Index.add(b"\xAA" * 32, [H1])

    assert Semantic_DB.rebuild_experience_index() == 3
    assert Experience_Index.all_keys() == {one, two, glob}

    # Bucket membership: an experience lands under each constituent, and a
    # constituent-less one under the _GLOBAL sentinel (always a candidate).
    assert Experience_Index.candidates({H1}) == {one, two, glob}
    assert Experience_Index.candidates({H2}) == {two, glob}
    assert Experience_Index.candidates(set()) == {glob}

    # Idempotent: a second rebuild is a no-op.
    assert Semantic_DB.rebuild_experience_index() == 3
    assert Experience_Index.all_keys() == {one, two, glob}


def test_rebuild_from_a_stale_snapshot_erases_a_concurrent_experience(isolated_cache):
    """Why rebuild_experience_index holds the semantics write transaction.

    Rebuilding from a snapshot another writer has moved past wipes the buckets of
    whatever that writer added.  The record and its vector survive, but no bucket
    mentions it: candidates() never returns it and AoA's all_keys()-based dedup
    re-learns it.  That is precisely the drift the rebuild exists to repair.
    """
    from Isabelle_Semantic_Embedding.experience_index import Experience_Index
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    _reset(isolated_cache)

    old = _key(xor_theory_prefix([H1]), EXP, b"\x01")
    _write_records({old: _record(EXP, "old", [("T1", H1)])})
    Semantic_DB.rebuild_experience_index()

    snapshot = Semantic_DB.experience_entries()               # what the old code did first

    new = _key(xor_theory_prefix([H1]), EXP, b"\x02")         # ... and then a writer lands
    Semantic_DB[new] = Semantic_DB.Record(
        EntityKind.EXPERIENCE, "new", "[]", "when", None, [("T1", H1)], "how")
    Experience_Index.add(new, [H1])
    assert new in Experience_Index.all_keys()

    Experience_Index.rebuild(snapshot)                        # ... and the stale rebuild lands

    assert new not in Experience_Index.all_keys(), "the stale rebuild should erase it"
    assert Semantic_DB[new] is not None, "the record itself survives — a silent loss, not a delete"


_WRITER = """
import os, sys, time
from Isabelle_RPC_Host.universal_key import EntityKind, xor_theory_prefix
from Isabelle_Semantic_Embedding.semantics import Semantic_DB
from Isabelle_Semantic_Embedding.experience_index import Experience_Index

go = sys.argv[1]
H1 = bytes.fromhex("11" * 16)
new = xor_theory_prefix([H1]) + bytes([int(EntityKind.EXPERIENCE)]) + b"\\xFF" * 15

Semantic_DB.experience_entries()                 # open the env before timing anything
open(go + ".loaded", "w").write("1")
while not os.path.exists(go):
    time.sleep(0.005)
time.sleep(0.05)                                 # let the parent take the write lock first

start = time.time()
Semantic_DB[new] = Semantic_DB.Record(
    EntityKind.EXPERIENCE, "concurrent", "[]", "when", None, [("T1", H1)], "how")
blocked = time.time() - start
Experience_Index.add(new, [H1])
open(go + ".done", "w").write(str(blocked))
"""


def test_rebuild_holds_the_semantics_write_lock_across_the_scan(isolated_cache, tmp_path):
    """A concurrent writer must block for the whole rebuild, scan included.

    This is the property that makes the stale snapshot above unreachable: the
    scan and the index wipe live in one semantics write transaction, and LMDB's
    single-writer lock is inter-process.  Enough experiences are written that the
    scan is measurably slow, so a writer starting mid-scan is observably blocked.
    """
    from Isabelle_Semantic_Embedding.experience_index import Experience_Index
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    _reset(isolated_cache)

    n = 50_000
    prefix = xor_theory_prefix([H1])
    _write_records({
        prefix + bytes([EXP]) + b"\x01" + i.to_bytes(14, "big"):
            _record(EXP, f"e{i}", [("T1", H1)], "when to use " * 8)
        for i in range(n)
    })

    go = str(tmp_path / "go")
    script = tmp_path / "writer.py"
    script.write_text(_WRITER)
    writer = subprocess.Popen([sys.executable, str(script), go], env=os.environ.copy())
    while not os.path.exists(go + ".loaded"):
        time.sleep(0.01)

    open(go, "w").write("go")
    started = time.time()
    indexed = Semantic_DB.rebuild_experience_index()
    rebuild_secs = time.time() - started
    assert writer.wait(timeout=60) == 0
    blocked = float(open(go + ".done").read())

    assert indexed == n
    assert rebuild_secs > 0.2, f"scan too fast ({rebuild_secs:.2f}s) to observe the lock"
    assert blocked > 0.2, f"the writer was not blocked ({blocked:.2f}s): the lock does not span the scan"

    new = prefix + bytes([EXP]) + b"\xFF" * 15
    assert new in Experience_Index.all_keys(), "the concurrent experience was erased"


# ---------------------------------------------------------------------------
# check_consistency / repair_xor_prefixes
# ---------------------------------------------------------------------------

def test_xor_prefix_repair_moves_conflicts_and_leaves_sound_records_alone(isolated_cache):
    """An XOR prefix is derived from the record's constituent list, so the list wins.

    A record already sitting at the correct key with *different* content is a real
    conflict and must be refused, not overwritten.
    """
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    _reset(isolated_cache)

    good = xor_theory_prefix([H1, H2])
    bad = bytes(b ^ 0xFF for b in good)
    consts = [("T1", H1), ("T2", H2)]

    moved_from, moved_to = _key(bad, THM, b"\x01"), _key(good, THM, b"\x01")
    conflict_bad, conflict_good = _key(bad, THM, b"\x02"), _key(good, THM, b"\x02")
    dup_bad, dup_good = _key(bad, THM, b"\x03"), _key(good, THM, b"\x03")
    sound = _key(good, THM, b"\x04")
    duplicate = _record(THM, "dup", consts)

    _write_records({
        moved_from: _record(THM, "moved", consts),
        conflict_bad: _record(THM, "conflict_bad", consts),
        conflict_good: _record(THM, "conflict_good_DIFFERENT", consts),
        dup_bad: duplicate,
        dup_good: duplicate,
        sound: _record(THM, "sound", consts),
    })

    c = Semantic_DB.check_consistency()
    assert {bad_key for bad_key, _ in c.xor_mismatches} == {moved_from, conflict_bad, dup_bad}
    assert c.legacy_xor == 0

    moved, conflicts = Semantic_DB.repair_xor_prefixes(c.xor_mismatches)

    assert conflicts == [conflict_bad]
    assert set(moved) == {(moved_from, moved_to), (dup_bad, dup_good)}

    assert Semantic_DB[moved_from] is None and Semantic_DB[moved_to].name == "moved"
    assert Semantic_DB[dup_bad] is None and Semantic_DB[dup_good].name == "dup"
    assert Semantic_DB[conflict_bad].name == "conflict_bad", "a conflict must not be dropped"
    assert Semantic_DB[conflict_good].name == "conflict_good_DIFFERENT", "nor overwritten"
    assert Semantic_DB[sound].name == "sound", "a sound record must not be touched"

    assert Semantic_DB.check_consistency().xor_mismatches == [(conflict_bad, conflict_good)]


def test_legacy_records_are_reported_not_repaired(isolated_cache):
    from Isabelle_Semantic_Embedding.semantics import Semantic_DB
    _reset(isolated_cache)

    legacy = _key(xor_theory_prefix([H1]), THM, b"\x01")
    _write_records({
        legacy: msgpack.packb((THM, "legacy", "e", "i", None)),   # only 5 fields
    })

    c = Semantic_DB.check_consistency()
    assert c.legacy_xor == 1
    assert c.xor_mismatches == [], "a record with no constituent list has nothing to compare"
