"""Semantic query tools for looking up interpretations from parent theories."""

import asyncio
import json
import os
import threading
from collections.abc import Iterator
from typing import Any, NamedTuple

import lmdb
import msgpack
import numpy as np
from ._paths import semantic_DB_dir
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from Isabelle_RPC_Host.rpc import IsabelleError
from Isabelle_RPC_Host.position import IsabellePosition
from Isabelle_RPC_Host.unicode import pretty_unicode, ascii_of_unicode
from Isabelle_RPC_Host.universal_key import EntityKind, UndefinedEntity, universal_key, universal_key_of, destruct_key, is_WIP, RULE_ONLY_TAG_BYTES, RULE_ONLY_KINDS
from claude_agent_sdk import SdkMcpTool, tool

from .semantic_embedding import (Vector_Store, Embedding_Provider, make_embedding_provider,
                                 sanitize_model, unsanitize_model,
                                 Reranker_Provider, reranker_provider, key)
from ._vecarith import library_path as _vector_library_path

from .base import ToolCall_ret, mk_ret as _mk_ret
from .hover import resolve_context_at
from .embedding_config import _DEFAULT_KINDS_PHRASE
from .document_text import document_text_of, entity_document_text


# Map an EntityKind to the plural noun phrase used in a query instruction's
# {kinds} slot (e.g. [CONSTANT, THEOREM] -> "constants and theorems"). Keyed on
# the real EntityKind members; do NOT pluralize EntityKind.label (those are
# singular display strings, e.g. "lemma"/"named theorem bundles"). The four rule
# kinds are intentionally absent -> they collapse to "inference rules" below.
_KIND_PHRASE = {
    EntityKind.CONSTANT:           "constants",
    EntityKind.THEOREM:            "theorems",
    EntityKind.TYPE:               "types",
    EntityKind.CLASS:              "type classes",
    EntityKind.LOCALE:             "locales",
    EntityKind.THEOREM_COLLECTION: "theorem collections",
    EntityKind.METHOD:             "proof methods",
}


def render_kinds(kinds: list) -> str:
    """Render an EntityKind filter into the noun phrase for a query instruction's
    {kinds} slot. One kind -> its phrase; several -> an Oxford-style join; the
    four rule kinds (INTRODUCTION/ELIMINATION/INDUCTION/CASE_SPLIT) collapse to a
    single "inference rules"; empty or unknown -> the shared default phrase.
    Total over every EntityKind value (uses .get, never raises KeyError)."""
    if not kinds:
        return _DEFAULT_KINDS_PHRASE
    phrases: list[str] = []
    for k in kinds:
        p = _KIND_PHRASE.get(k)
        if p is None:
            p = "inference rules" if k in RULE_ONLY_KINDS else _DEFAULT_KINDS_PHRASE
        if p not in phrases:
            phrases.append(p)
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def unpack_thy_status(raw: bytes) -> dict:
    """Unpack a 16-byte theory-status record, normalizing every key to ``bytes``.

    Theory-status records (cost / token counts / ``finished`` / ``model`` / embed
    ``total_tokens``) are keyed by ``bytes`` (``d[b"cost_usd"]``) throughout this
    module.  Historically some were written with str keys (``packb`` of
    str-literal dicts) and some read-modify-write paths injected bytes keys into
    str-keyed dicts, leaving str-only or mixed records on disk.  Decode through
    this helper and index by bytes everywhere: str keys are encoded to bytes so
    legacy str/mixed records still read correctly, and re-packing the normalized
    dict rewrites the record with clean bytes keys on its next write.

    NOTE: entity/interpretation records are positional msgpack tuples, NOT keyed
    dicts, so they are unaffected by the key type and never pass through here."""
    d = msgpack.unpackb(raw)
    return {(k.encode() if isinstance(k, str) else k): v for k, v in d.items()}


def record_constituent_hashes(raw: bytes) -> 'set[bytes] | None':
    """Constituent theory hashes of a stored theorem/rule/experience record.

    The theory_constituents field is the 6th tuple element (index 5); the layout
    mirrors _Semantic_DB._decode.  Such keys carry an XOR pseudo-theory prefix,
    so membership in a theory is decided by this list and never by the prefix.
    None for legacy records, which predate the field."""
    vals = msgpack.unpackb(raw)
    if len(vals) <= 5 or vals[5] is None:
        return None
    return {bytes(h) for _, h in vals[5]}


# Long theory names to exclude from interpretation and entity enumeration.
_SKIP_THEORY_LONG_NAMES = ["Pure", "Tools.Code_Generator", "HOL.Code_Evaluation", "HOL.Typerep"]

# Base names (after the last dot) for matching theories under different session qualifiers.
_SKIP_THEORY_BASES = {n.rsplit(".", 1)[-1] for n in _SKIP_THEORY_LONG_NAMES}

def is_thy_skipped(name: str) -> bool:
    """Check whether a theory should be skipped from interpretation."""
    base = name.rsplit(".", 1)[-1] if "." in name else name
    return base in _SKIP_THEORY_BASES


migrate_on_hash_change: bool = False
persist_wip: bool = os.getenv("SEMANTIC_PERSIST_WIP", "") != ""

# Write ceiling for semantics.lmdb.  LMDB does not preallocate: this only
# reserves virtual address space, and the value passed at open() is the hard
# limit for *writes* by that process (exceeding it raises MapFullError).
# Read-only openers are unaffected — lmdb adopts the file's actual size.
# Keep every writer of semantics.lmdb on this one constant.  Raised from 1<<30
# because `r2_sync` merges a remote snapshot into this store, and 1 GiB was the
# lowest ceiling anywhere in the tree (semantics_manage's `remove` used 1<<33).
SEMANTICS_MAP_SIZE: int = 1 << 32   # 4 GiB


def _iter_vector_store_envs() -> 'Iterator[lmdb.Environment]':
    """Every ``vector_*.lmdb`` store on disk, as an open (process-cached) environment.

    One store per embedding model; the model name is encoded in the directory name.
    """
    cache_dir = semantic_DB_dir()
    if not os.path.isdir(cache_dir):
        return
    from .semantic_embedding import _get_lmdb_env
    for entry in sorted(os.listdir(cache_dir)):
        if entry.startswith("vector_") and entry.endswith(".lmdb"):
            path = os.path.join(cache_dir, entry)
            if os.path.isdir(path):
                yield _get_lmdb_env(path)


class Provenance(NamedTuple):
    """Locale-interpretation provenance of an instance fact.

    Serialized in the semantic DB as a msgpack map (None fields omitted);
    None as a whole for ordinary entries."""
    template_uk: 'bytes | None' = None
    locale_uk: 'bytes | None' = None
    qualifier: 'str | None' = None

EXPR_DISPLAY_LIMIT = 500

def trunc_expr(s: str, limit: int = EXPR_DISPLAY_LIMIT) -> str:
    """Truncate an expression string to the given limit (default EXPR_DISPLAY_LIMIT)."""
    return s[:limit] + "…" if len(s) > limit else s


class _Semantic_DB:
    """Process-wide LMDB-backed store of semantic interpretations.

    CONCURRENCY INVARIANT — keep write methods synchronous (no ``await`` inside a
    ``with env.begin(write=True)`` block). LMDB allows only one open write
    transaction per environment. Under ``-o threads>1`` the Isabelle scheduler runs
    several ``interpret_file`` coroutines concurrently on this host's single event
    loop; they remain safe ONLY because every write-txn body here runs to completion
    without yielding, so the loop cannot interleave two write transactions. Inserting
    an ``await`` between ``begin(write=True)`` and the block exit would let a second
    coroutine open a concurrent write txn and raise ``lmdb.Error`` — which, under the
    scheduler's hard-crash policy, aborts the whole run. (``_lock`` below only guards
    env *creation*, not write transactions.) No async lock is needed while these
    methods stay synchronous; if a writer ever must become async, add an
    ``asyncio.Lock`` around the write path instead of relying on this property.
    """
    _env: lmdb.Environment | None = None
    _lock = threading.Lock()

    class Record(NamedTuple):
        # NB (EXPERIENCE kind): an AoA "experience memory" (a reusable proof
        # strategy for a general class of goals; see AoA/docs/EXPERIENCE_MEMORY.md)
        # is stored as a Record of kind EXPERIENCE, REUSING the entity fields:
        #   name                -> agent-provided experience name (short, stable id)
        #   expr                -> the goal_patterns, joined (the term patterns the
        #                          strategy applies to)
        #   interpretation      -> goal_description (the WHEN-to-use text; this is
        #                          what gets embedded for semantic retrieval)
        #   theory_constituents -> minimal antichain of constituent theories of the
        #                          patterns' constants (drives availability), same
        #                          XOR-prefix convention as thm/rule keys
        #   experience          -> the how-to-prove payload (NOT embedded)
        # locale_provenance is always None for experiences.
        kind: EntityKind
        name: str
        expr: str | None
        interpretation: str | None
        # locale-interpretation provenance; None for entries that are not
        # locale-generated facts
        locale_provenance: 'Provenance | None' = None
        # constituent theories of theorem/rule (and experience) entities, as a
        # sorted (theory long name, 16-byte theory hash) list: the key's theory
        # prefix is the XOR of these hashes.  Used to find the records
        # belonging to / affected by a theory (deletion, migration).
        # None for non-theorem kinds and for legacy records.
        theory_constituents: 'list[tuple[str, bytes]] | None' = None
        # EXPERIENCE-only: the how-to-prove payload (natural language).  Not
        # embedded (only interpretation is).  None for all other kinds.
        experience: 'str | None' = None

        @property
        def pretty_print(self) -> str:
            pp = f"{self.kind.label} {self.name}"
            if self.expr:
                pp += f": {trunc_expr(self.expr)}"
            return pp

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            with self._lock:
                if self._env is None:
                    import atexit
                    cache_dir = semantic_DB_dir()
                    os.makedirs(cache_dir, exist_ok=True)
                    _Semantic_DB._env = lmdb.open(os.path.join(cache_dir, "semantics.lmdb"),
                                                  map_size=SEMANTICS_MAP_SIZE)
                    atexit.register(_Semantic_DB._close)
        return self._env  # type: ignore

    @staticmethod
    def _close() -> None:
        with _Semantic_DB._lock:
            if _Semantic_DB._env is not None:
                _Semantic_DB._env.close()
                _Semantic_DB._env = None

    @staticmethod
    def _dec(v: Any) -> str:
        return v.decode() if isinstance(v, bytes) else v

    @staticmethod
    def _decode(raw: bytes) -> 'Record':
        """Decode a stored record.  Records with fewer than 7 fields read with
        the missing trailing fields (locale_provenance, theory_constituents,
        experience) = None."""
        vals = list(msgpack.unpackb(raw))
        vals += [None] * (7 - len(vals))
        kind, name, expr, sem, prov_raw, consts_raw, experience = vals[:7]
        d = _Semantic_DB._dec
        prov = None
        if isinstance(prov_raw, dict):
            def g(k: str):
                return prov_raw.get(k, prov_raw.get(k.encode()))
            tuk, luk, qual = g("template_uk"), g("locale_uk"), g("qualifier")
            prov = Provenance(
                bytes(tuk) if tuk is not None else None,
                bytes(luk) if luk is not None else None,
                d(qual) if qual is not None else None)
        consts = None
        if consts_raw is not None:
            consts = [(d(n), bytes(h)) for n, h in consts_raw]
        return _Semantic_DB.Record(EntityKind(kind), d(name), d(expr), d(sem), prov,
                                   consts, d(experience) if experience is not None else None)

    @staticmethod
    def _encode(record: 'Record') -> bytes:
        prov_map = None
        if record.locale_provenance is not None:
            prov_map = {}
            if record.locale_provenance.template_uk is not None:
                prov_map["template_uk"] = record.locale_provenance.template_uk
            if record.locale_provenance.locale_uk is not None:
                prov_map["locale_uk"] = record.locale_provenance.locale_uk
            if record.locale_provenance.qualifier is not None:
                prov_map["qualifier"] = record.locale_provenance.qualifier
        return msgpack.packb((int(record.kind), record.name, record.expr,
                              record.interpretation, prov_map,
                              record.theory_constituents,
                              record.experience))  # type: ignore[return-value]

    def __getitem__(self, key: universal_key) -> 'Record | None':
        with self._ensure_env().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return None
        return self._decode(raw)

    def __contains__(self, key: universal_key) -> bool:
        with self._ensure_env().begin() as txn:
            return txn.get(key) is not None

    def contains(self, keys: list[universal_key]) -> list[bool]:
        """Check existence for a batch of keys in a single transaction."""
        with self._ensure_env().begin() as txn:
            return [txn.get(k) is not None for k in keys]

    def get_many(self, keys: list[universal_key]) -> 'list[Record | None]':
        """Fetch a batch of records in a SINGLE read transaction -- the batch counterpart
        of ``__getitem__``, mirroring ``contains``.  Result is positionally aligned with
        ``keys``; a key with no record yields None.

        ``[self[k] for k in keys]`` would open (and commit) one txn PER key.  Callers
        like ``_auto_embed`` run on the event loop with a ``missing`` list that can be
        10^5 long when a library has not been embedded for the active model yet, so the
        per-key form turns one cheap scan into 10^5 synchronous begin/commit pairs."""
        with self._ensure_env().begin() as txn:
            out: 'list[_Semantic_DB.Record | None]' = []
            for k in keys:
                raw = txn.get(k)
                out.append(self._decode(raw) if raw is not None else None)
            return out

    def iter_entity_records(self) -> 'Iterator[tuple[universal_key, Record]]':
        """Yield ``(key, Record)`` for every non-status record, in ONE read txn.

        Decodes each value inline so the caller never has to re-open the env — the
        env's default per-thread read slot allows only one live read txn, so opening
        a second (e.g. via ``query``/``__getitem__``) while iterating would fail.
        Build any text you need from the yielded Record, not by re-querying.

        Skips the 16-byte theory-status keys and any value that does not decode as a
        Record (legacy / non-entity). This is the whole-DB enumeration the offline
        embed drives off — using the singleton env, NEVER a second ``lmdb.open`` of
        semantics.lmdb (which py-lmdb refuses in-process)."""
        with self._ensure_env().begin() as txn:
            for k, v in txn.cursor():
                k = bytes(k)
                if len(k) == 16:
                    continue
                try:
                    rec = self._decode(v)
                except Exception:
                    continue
                yield k, rec

    def __setitem__(self, key: universal_key, record: 'Record') -> None:
        with self._ensure_env().begin(write=True) as txn:
            txn.put(key, self._encode(record))

    def delete(self, key: universal_key) -> bool:
        """Delete a record by key. Returns True if a record existed and was removed.
        Used e.g. to overwrite an experience memory (see write_memory)."""
        with self._ensure_env().begin(write=True) as txn:
            return txn.delete(key)

    def update_expr(self, key: universal_key, new_expr: str) -> None:
        """Update the expr field of an existing record, leaving all other fields intact."""
        with self._ensure_env().begin(write=True) as txn:
            raw = txn.get(key)
            if raw is None:
                return
            vals = list(msgpack.unpackb(raw))
            vals[2] = new_expr
            txn.put(key, msgpack.packb(vals))  # type: ignore

    def query(self, key: universal_key, with_pretty: bool = False) -> str | None:
        """Look up a semantic interpretation by universal key."""
        rec = self[key]
        if rec is None:
            return None
        if rec.interpretation is None:
            return None
        if with_pretty:
            # Display/reranker path. entity_document_text is the single definition of
            # the entity convention (pretty_print + "\n" + interpretation); applied to
            # any kind here it reproduces today's kind-blind string byte-for-byte, so
            # this is a pure de-duplication (no behavior change). The embed path no
            # longer routes through query -- it uses document_text_of directly.
            return entity_document_text(rec)
        return rec.interpretation

    def is_thy_interpreted(self, key: universal_key) -> bool:
        """Check whether a theory has been fully interpreted."""
        with self._ensure_env().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return False
        return unpack_thy_status(raw).get(b"finished", False)

    def mark_interpreted(self, key: universal_key) -> None:
        """Mark a theory as interpreted (finished) in the semantic store.
        Skips WIP (non-persistent) theories unless persist_wip is enabled."""
        if is_WIP(key) and not persist_wip:
            return
        with self._ensure_env().begin(write=True) as txn:
            raw = txn.get(key)
            if raw is not None:
                data = unpack_thy_status(raw)
                data[b"finished"] = True
                txn.put(key, msgpack.packb(data))  # type: ignore
            else:
                txn.put(key, msgpack.packb({
                    b"input_tokens": 0, b"cache_creation_tokens": 0,
                    b"cache_read_tokens": 0, b"output_tokens": 0,
                    b"cost_usd": 0.0, b"finished": True,
                    b"model": "",
                }))  # type: ignore

    def clean_wip(self) -> int:
        """Remove all entries with non-persistent theory hashes."""
        from Isabelle_RPC_Host.theory_hash import is_persistent
        to_delete: list[bytes] = []
        with self._ensure_env().begin(write=True) as txn:
            for key, _ in txn.cursor():
                if not is_persistent(key):
                    to_delete.append(bytes(key))
            for key in to_delete:
                txn.delete(key)
        return len(to_delete)

    @staticmethod
    def _scan_experiences(txn: Any) -> 'list[tuple[universal_key, list[bytes]]]':
        """``(key, constituent theory hashes)`` for every EXPERIENCE record seen by txn.

        Experience keys are XOR-prefixed (32 bytes, kind tag at byte 16), so they
        are recognized from the key alone; only the matches are decoded."""
        from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key
        tag = int(EntityKind.EXPERIENCE)
        entries: list[tuple[bytes, list[bytes]]] = []
        for key, val in txn.cursor():
            key = bytes(key)
            if not is_xor_prefixed_key(key) or key[16] != tag:
                continue
            rec = _Semantic_DB._decode(bytes(val))
            entries.append((key, [h for _, h in (rec.theory_constituents or [])]))
        return entries

    def experience_entries(self) -> 'list[tuple[universal_key, list[bytes]]]':
        """``(key, constituent theory hashes)`` for every EXPERIENCE record."""
        with self._ensure_env().begin() as txn:
            return self._scan_experiences(txn)

    def rebuild_experience_index(self) -> int:
        """Rebuild experience_index.lmdb from the EXPERIENCE records stored here.

        The index is a derived view of these records (see Experience_Index.rebuild),
        and the three stores an experience lives in are written without cross-store
        atomicity, so it can drift.  Call this after any bulk mutation of
        semantics.lmdb that bypasses Experience_Index (e.g. merging in another
        machine's snapshot).  Returns the number of experiences indexed.

        The scan and the index rebuild both run inside ONE semantics write
        transaction.  That is load-bearing, not incidental: without it, an
        experience written between the scan and the wipe has its bucket erased and
        becomes silently unretrievable — exactly the drift this method repairs.
        Holding the write transaction excludes every writer of semantics.lmdb
        (``Semantic_DB[key] = ...``) for the whole span, so the scan cannot go
        stale.  Readers are untouched; LMDB is MVCC.

        Lock order is semantics -> experience_index, matching every other writer:
        mcp_http_server commits the record (closing that transaction) before
        calling Experience_Index.add, and _migrate_constituent_records likewise
        touches the index only after its write transaction exits.  Nothing ever
        holds the index lock while wanting the semantics one, so no deadlock."""
        from .experience_index import Experience_Index
        with self._ensure_env().begin(write=True) as txn:
            return Experience_Index.rebuild(self._scan_experiences(txn))

    class Consistency(NamedTuple):
        """What a whole-store consistency scan found.  See semantics_manage fsck."""
        n_records: int                       # entity records (theory status excluded)
        experience_keys: set[bytes]
        legacy_xor: int                      # XOR-prefixed records with no constituent list
        # (wrong_key, correct_key) for records whose XOR prefix disagrees with
        # xor_theory_prefix(their constituent list)
        xor_mismatches: 'list[tuple[bytes, bytes]]'

    def check_consistency(self) -> 'Consistency':
        """Scan the store once and report the invariants that can break silently.

        Only genuine invariants: a *missing vector* is NOT one of them.  Vectors are
        a lazily-filled derived cache — topk hands unknown keys to _auto_embed, which
        embeds anything whose interpretation is already stored.  Reporting a cold
        cache as damage would be noise."""
        from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key, xor_theory_prefix
        tag = int(EntityKind.EXPERIENCE)
        n_records = 0
        experience_keys: set[bytes] = set()
        legacy_xor = 0
        xor_mismatches: list[tuple[bytes, bytes]] = []
        with self._ensure_env().begin() as txn:
            for key, val in txn.cursor():
                key = bytes(key)
                if len(key) == 16:
                    continue                      # theory status record, not an entity
                n_records += 1
                if not is_xor_prefixed_key(key):
                    continue
                if key[16] == tag:
                    experience_keys.add(key)
                rec = self._decode(bytes(val))
                if rec.theory_constituents is None:
                    legacy_xor += 1
                    continue
                expect = xor_theory_prefix([h for _, h in rec.theory_constituents])
                if expect != key[:16]:
                    xor_mismatches.append((key, expect + key[16:]))
        return _Semantic_DB.Consistency(n_records, experience_keys, legacy_xor, xor_mismatches)

    def repair_xor_prefixes(self,
                            mismatches: 'list[tuple[bytes, bytes]]',
                            ) -> 'tuple[list[tuple[bytes, bytes]], list[bytes]]':
        """Re-key records whose XOR prefix disagrees with their constituent list.

        The prefix is *derived* (xor_theory_prefix of the constituents), the
        constituent list is stored primary data, so the list wins and the record
        moves to the recomputed key.  A record already sitting at the correct key
        with different content is a real conflict: refuse it rather than guess.

        The semantics moves run in one write transaction, so they are atomic.
        Vectors are moved on a best-effort basis afterwards — they are a cache, and
        _auto_embed refills whatever is missing.  Callers must rebuild the
        experience index afterwards: an EXPERIENCE record that moved is still
        indexed under its old key.

        Returns (moved, conflicts)."""
        moved: list[tuple[bytes, bytes]] = []
        conflicts: list[bytes] = []
        with self._ensure_env().begin(write=True) as txn:
            for bad, good in mismatches:
                val = txn.get(bad)
                if val is None:
                    continue                      # vanished since the scan
                val = bytes(val)
                existing = txn.get(good)
                if existing is None:
                    txn.put(good, val)
                elif bytes(existing) != val:
                    conflicts.append(bad)
                    continue
                txn.delete(bad)
                moved.append((bad, good))
        for env in _iter_vector_store_envs():
            with env.begin(write=True) as vtxn:
                for bad, good in moved:
                    v = vtxn.get(bad)
                    if v is None:
                        continue
                    if vtxn.get(good) is None:
                        vtxn.put(good, bytes(v))
                    vtxn.delete(bad)
        return moved, conflicts

    @staticmethod
    def _copy_prefix(env: lmdb.Environment, old_prefix: bytes, new_prefix: bytes) -> int:
        """Rekey prefix-addressed entries (theory records, namespace entities).

        XOR-prefixed keys (theorem/rule AND experience keys) are skipped even
        when their XOR pseudo-theory prefix coincides with old_prefix (a
        single-constituent such key equals its constituent's hash byte-for-byte):
        blindly rewriting the prefix would desynchronize the key from the
        record's constituent list — they are rekeyed by
        _migrate_constituent_records instead."""
        from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key
        assert len(old_prefix) == 16 and len(new_prefix) == 16
        count = 0
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(old_prefix):
                while True:
                    key = bytes(cursor.key())
                    if not key.startswith(old_prefix):
                        break
                    if not is_xor_prefixed_key(key):
                        txn.put(new_prefix + key[16:], bytes(cursor.value()))
                        count += 1
                    if not cursor.next():
                        break
        return count

    def keys_belonging_to(self, theory_hashes: 'set[bytes]') -> list[bytes]:
        """All semantic-DB keys belonging to the given theories.

        Keys whose 16-byte prefix is one of the hashes (theory records and
        namespace entities), plus XOR-prefixed keys (theorem/rule AND experience
        keys) — whose prefix is an XOR pseudo-theory — whose stored constituent
        list mentions one of them (mention-based membership).  Legacy thm/rule
        records without a constituent list are never matched."""
        from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key
        result: list[bytes] = []
        with self._ensure_env().begin() as txn:
            for key, val in txn.cursor():
                key = bytes(key)
                if is_xor_prefixed_key(key):
                    consts = self._decode(bytes(val)).theory_constituents
                    if consts is not None and any(h in theory_hashes for _, h in consts):
                        result.append(key)
                elif key[:16] in theory_hashes:
                    result.append(key)
        return result

    def _migrate_constituent_records(self, old_hash: bytes, new_hash: bytes) -> int:
        """Rekey XOR-prefixed records (theorem/rule AND experience) whose
        constituents mention old_hash.

        Replaces old_hash with new_hash in the constituent list, recomputes
        the XOR prefix, and copies the record (and any vector-store entries)
        under the new key.  Old entries are left in place, mirroring
        _copy_prefix's copy semantics.  Experience records always carry a
        constituent list, so they never trip the legacy-record guard below."""
        from Isabelle_RPC_Host.universal_key import (
            is_xor_prefixed_key, xor_theory_prefix, EntityKind)
        env = self._ensure_env()
        rekeys: list[tuple[bytes, bytes, bytes]] = []
        # Experience keys additionally live in the (separate) inverted index,
        # keyed by their constituent hashes — rekeying moves them there too.
        # (old_key, new_key, old_hashes, new_hashes)
        exp_rekeys: list[tuple[bytes, bytes, list[bytes], list[bytes]]] = []
        with env.begin() as txn:
            for key, val in txn.cursor():
                key = bytes(key)
                if not is_xor_prefixed_key(key):
                    continue
                rec = self._decode(bytes(val))
                if rec.theory_constituents is None:
                    raise ValueError(
                        f"XOR-prefixed record {key.hex()} has no constituent list "
                        "(pre-XOR legacy theorem/rule record); run migrate_xor_thm_keys.py first")
                if not any(h == old_hash for _, h in rec.theory_constituents):
                    continue
                new_consts = [(n, new_hash if h == old_hash else h)
                              for n, h in rec.theory_constituents]
                new_key = xor_theory_prefix([h for _, h in new_consts]) + key[16:]
                rekeys.append((key, new_key,
                               self._encode(rec._replace(theory_constituents=new_consts))))
                if key[16] == int(EntityKind.EXPERIENCE):
                    exp_rekeys.append((key, new_key,
                                       [h for _, h in rec.theory_constituents],
                                       [h for _, h in new_consts]))
        if not rekeys:
            return 0
        with env.begin(write=True) as txn:
            for _, new_key, new_val in rekeys:
                txn.put(new_key, new_val)
        if exp_rekeys:
            from .experience_index import Experience_Index
            for old_key, new_key, old_hashes, new_hashes in exp_rekeys:
                Experience_Index.remove(old_key, old_hashes)
                Experience_Index.add(new_key, new_hashes)
        for venv in _iter_vector_store_envs():
            with venv.begin(write=True) as vtxn:
                for old_key, new_key, _ in rekeys:
                    v = vtxn.get(old_key)
                    if v is not None:
                        vtxn.put(new_key, bytes(v))
        return len(rekeys)

    def _try_migrate(self, new_key: universal_key) -> bool:
        from Isabelle_RPC_Host.theory_hash import open_theory_hash_store

        new_hash = bytes(new_key[:16])
        th_env = open_theory_hash_store()
        with th_env.begin() as txn:
            raw = txn.get(new_hash)
            if raw is None:
                return False
            new_name, _ = msgpack.unpackb(raw)
            if isinstance(new_name, bytes):
                new_name = new_name.decode("utf-8")

        candidates: list[tuple[bytes, int]] = []
        with th_env.begin() as txn:
            for k, v in txn.cursor():
                k = bytes(k)
                if k == new_hash:
                    continue
                name, ts = msgpack.unpackb(v)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                if name == new_name:
                    candidates.append((k, ts))

        if not candidates:
            return False
        candidates.sort(key=lambda x: x[1], reverse=True)

        sem_env = self._ensure_env()
        old_hash: bytes | None = None
        for candidate_hash, _ in candidates:
            with sem_env.begin() as txn:
                raw = txn.get(candidate_hash)
            if raw is not None and unpack_thy_status(raw).get(b"finished", False):
                old_hash = candidate_hash
                break

        if old_hash is None:
            return False

        n = self._copy_prefix(sem_env, old_hash, new_hash)

        cache_dir = semantic_DB_dir()
        if os.path.isdir(cache_dir):
            from .semantic_embedding import _get_lmdb_env
            for entry in os.listdir(cache_dir):
                if entry.startswith("vector_") and entry.endswith(".lmdb"):
                    path = os.path.join(cache_dir, entry)
                    if os.path.isdir(path):
                        self._copy_prefix(_get_lmdb_env(path), old_hash, new_hash)

        # Theorem/rule AND experience records reference theories through their
        # constituent lists, not their key prefix — rekey them by XOR recomputation.
        n_thm = self._migrate_constituent_records(old_hash, new_hash)

        print(f"Migrated {n} entries and {n_thm} constituent records for {new_name} "
              f"from {old_hash.hex()[:12]}… to {new_hash.hex()[:12]}…")
        return True


Semantic_DB = _Semantic_DB()
SemanticRecord = _Semantic_DB.Record


def clean_wip() -> int:
    """Remove all WIP (non-persistent) entries from the semantic DB and all vector stores.

    Returns:
        Number of entries deleted from the semantic DB.
    """
    deleted = Semantic_DB.clean_wip()
    Semantic_Vector_Store.clean_all_wip_in_created_dbs()
    return deleted


# --- MCP tool factories ---

_NAME_DESCRIPTION_BASE = (
    "The short or full name of the entity to look up. "
    "For type='constant', a notation token (e.g. an operator, infix, or binder "
    "symbol such as '⊑') is also accepted and resolved to its underlying constant.")
_NAME_DESCRIPTION_INTERP = (
    _NAME_DESCRIPTION_BASE +
    " For multi-variant theorems, include a '(idx)' suffix (e.g. 'conjI(2)').")

def _mk_query_by_name_schema(working_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["constant", "lemma", "type", "typeclass", "locale",
                        "named theorem bundles", "proof method",
                        "introduction rule", "elimination rule",
                        "induction rule", "case-split rule"],
                "description": "The kind of entity to query.",
            },
            "name": {
                "type": "string",
                "description": _NAME_DESCRIPTION_INTERP if working_names else _NAME_DESCRIPTION_BASE,
            },
            "show_defs": {
                "type": "boolean",
                "description": "If true, include the Isabelle source code of the command defining the entity.",
                "default": False,
            },
            "context_at": {
                "type": "object",
                "description": "Resolve the name under the proof context at this source position. "
                               "Omit to use the theory's global context.",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Path to the theory file. Defaults to the current theory file.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number.",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "A token on that line to pin the context to "
                                       "(ASCII or Unicode). If omitted, uses the end of the line.",
                    },
                },
                "required": ["line"],
                "additionalProperties": False,
            },
        },
        "required": ["type", "name"],
    }


import re as _re
_TRAILING_BEGIN_RE = _re.compile(r'\nbegin\s*\Z')

def _strip_trailing_begin(source: str) -> str:
    """Remove a trailing ``begin`` line from command source."""
    return _TRAILING_BEGIN_RE.sub('', source)


async def _get_definition_with_pos(
    connection: Connection, kind: EntityKind, uk: universal_key,
    ctxt: Any = None,
) -> tuple[str, IsabellePosition] | None:
    """Look up the source code and position of the command defining an entity.

    Uses cached entity enumeration to find the definition position,
    then calls command_at_position to retrieve the source. A non-None
    *ctxt* resolves the enumeration under that (file, offset) context —
    note this bypasses the per-connection enumeration cache, costing a
    full uncached enumeration RPC per call.

    Returns ``(source, cmd_pos)`` where *cmd_pos* is an
    `IsabellePosition` for the command start (symbol offset), or ``None``
    if the position or source is unavailable.
    """
    from Isabelle_RPC_Host.context import entities_of
    from .hover import command_at_position
    entries, _, _ = await entities_of(connection, [kind], ctxt=ctxt)
    pos = None
    for key, _, p in entries:
        if key == uk:
            pos = p
            break
    if pos is None:
        return None
    cmd = await command_at_position(pos, connection)
    if cmd is None:
        return None
    source, start_offset, _ = cmd
    # Strip trailing "begin" keyword (part of class/locale/context command spans)
    source = _strip_trailing_begin(source)
    source = pretty_unicode(source)
    cmd_pos = IsabellePosition(0, start_offset, pos.file)
    return (source, cmd_pos)


async def _get_definition_source(
    connection: Connection, kind: EntityKind, uk: universal_key,
    ctxt: Any = None,
) -> str | None:
    """Look up the source code of the command defining an entity.

    Convenience wrapper around `_get_definition_with_pos` that discards
    the position.
    """
    result = await _get_definition_with_pos(connection, kind, uk, ctxt=ctxt)
    if result is None:
        return None
    source, _ = result
    return source


async def query_by_name_raw(
    connection: Connection,
    kind: EntityKind,
    name: str,
    with_pretty: bool = True,
    ctxt: Any = None,
) -> tuple[str, universal_key]:
    """Look up entity by kind and name, returning ``(semantic_text, universal_key)``.

    Raises `UndefinedEntity`, `IsabelleError`, or `LookupError` (not yet interpreted).
    """
    uk = await universal_key_of(connection, kind, name, ctxt=ctxt)
    sem = Semantic_DB.query(uk, with_pretty=with_pretty)
    if sem is None:
        raise LookupError(
            f'{kind.label} "{name}" has not been interpreted yet. '
            'Try using `mcp__proof__search_isabelle` to find what you need.')
    return (sem, uk)


async def _append_definition(
    sem: str, connection: Connection, kind: EntityKind, uk: universal_key,
    name: str, log: Any, ctxt: Any = None,
) -> str:
    """Append the source of the command defining *uk* to *sem* (best-effort)."""
    try:
        src = await _get_definition_source(connection, kind, uk, ctxt=ctxt)
    except Exception:
        log.debug("show_defs failed for %r", name, exc_info=True)
        return sem
    if src is None:
        return sem
    return sem + f"\n\nDefinition:\n{src}"


async def _try_resolve_syntax_token(
    connection: Connection, name: str, ctxt: Any,
    with_pretty: bool, show_defs: bool, log: Any,
) -> str | None:
    """Try to resolve a name as a syntax/notation token via resolve_notation.

    Returns a formatted result string if the token resolves to a known constant,
    otherwise None. When *show_defs* is set, the defining command of the
    underlying constant is appended.
    """
    _LHS = "syntax_probe_x"
    _RHS = "syntax_probe_y"
    syntax_patterns = [
        (f"({name})", f"({name})"),                   # operator section, e.g. (≤)
        (f"{_LHS} {name} {_RHS}", f"_ {name} _"),    # infix
        (f"{name} {_LHS}", f"{name} _"),              # prefix
        (f"{_LHS} {name}", f"_ {name}"),              # postfix
        (f"{name} {_LHS}. {_RHS}", f"{name} _. _"),  # binder
    ]
    for pattern, display in syntax_patterns:
        try:
            resolved = await connection.callback(
                "explain_term.resolve_notation", (ctxt, pattern))
        except Exception:
            continue
        if resolved is None:
            continue
        const_name, uk_bytes, compact_str = resolved
        uk: universal_key = bytes(uk_bytes)
        log.debug("resolved syntax token %r -> %s via pattern %r", name, const_name, pattern)
        sem = Semantic_DB.query(uk, with_pretty=with_pretty)
        if sem is not None:
            result = (
                f'"{name}" is a notation (syntax: {display}).\n'
                f'It desugars to: {compact_str}\n'
                f'Underlying constant: {const_name}\n\n{sem}')
        else:
            # Constant found but not yet interpreted — still report the resolution
            result = (
                f'"{name}" is a notation (syntax: {display}).\n'
                f'It desugars to: {compact_str}\n'
                f'Underlying constant: {const_name}\n'
                f'(Not yet interpreted. Use `query` with name="{const_name}" '
                f'or `desugar_and_explain` for more details.)')
        if show_defs:
            result = await _append_definition(
                result, connection, EntityKind.CONSTANT, uk, const_name, log,
                ctxt=ctxt)
        return result
    return None


def mk_query_by_name_tool(
    connection: Connection, working_names: list[str], with_pretty: bool = True,
    file_path: str | None = None,
) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("semantics")
    description = "Look up the English translation of a dependency from parent theories."
    if working_names:
        description += (
            " Do not query entries you have been asked to interpret"
            " — interpret those from the source file yourself.")
    @tool(
        "query",
        description,
        input_schema=_mk_query_by_name_schema(working_names),
    )
    async def query_by_name_tool(args: dict[str, Any]) -> ToolCall_ret:
        t = args.get("type", "")
        name = args.get("name", "")
        log.debug("query_by_name: type=%r name=%r", t, name)
        if not isinstance(t, str) or not isinstance(name, str):
            return _mk_ret("Invalid arguments: 'type' and 'name' must be strings.", is_error=True)
        try:
            tag = EntityKind.from_label(t)
        except KeyError:
            return _mk_ret(f"Invalid type: {t!r}. Must be one of {[k.label for k in EntityKind if k != EntityKind.THEORY]}.", is_error=True)
        if not name:
            return _mk_ret("Invalid name: must be a non-empty string.", is_error=True)
        # Normalize Unicode glyphs the agent may type (e.g. ≤, ∀, subscripts)
        # into Isabelle's ASCII-escape form (\<le>, \<forall>, ...). Syntax.read_term
        # and the name-space only recognize the escape form; raw UTF-8 fails.
        name = ascii_of_unicode(name)

        # Resolve optional context_at position to (file, symbol_offset)
        ctxt, ctxt_note = resolve_context_at(args.get("context_at"), file_path, log)

        def _noted(s: str) -> str:
            return f"{ctxt_note}\n\n{s}" if ctxt_note else s

        try:
            if working_names and name in working_names:
                log.debug("Entity name %r is in working_names; cannot query entities assigned for interpretation.", name)
                return _mk_ret(
                    f"Cannot query \"{name}\" — it is or will be your task to interpret it from the source.",
                    is_error=True,
                )
            sem, uk = await query_by_name_raw(connection, tag, name, with_pretty=with_pretty, ctxt=ctxt)
            if args.get("show_defs", False):
                sem = await _append_definition(sem, connection, tag, uk, name, log,
                                               ctxt=ctxt)
            return _mk_ret(_noted(sem))
        except LookupError as e:
            return _mk_ret(_noted(str(e)))
        except UndefinedEntity as e:
            if "." in name:
                short = name.rsplit(".", 1)[1]
                try:
                    sem, uk = await query_by_name_raw(connection, tag, short, with_pretty=with_pretty)
                    if args.get("show_defs", False):
                        sem = await _append_definition(sem, connection, tag, uk, short, log)
                    return _mk_ret(_noted(f"The {name} is undefined, but we find:\n{sem}"))
                except (IsabelleError, UndefinedEntity, LookupError):
                    pass
            # Try resolving as a syntax/notation token via resolve_notation
            if tag == EntityKind.CONSTANT:
                resolved = await _try_resolve_syntax_token(
                    connection, name, ctxt, with_pretty,
                    args.get("show_defs", False), log)
                if resolved is not None:
                    return _mk_ret(_noted(resolved))
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(
                _noted(str(e) + " Try using `mcp__proof__semantic_search` to find what you need."),
                is_error=True,
            )
        except IsabelleError as e:
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(_noted(str(e)), is_error=True)
        except Exception:
            log.exception("query_by_name: error")
            raise
    return query_by_name_tool


# --- Other utilities ---

async def interpret_theories_by_names(connection: Connection, names: list[str]) -> None:
    """Interpret theories by name (short or long).
    Resolves names, skips already-interpreted theories, and interprets the rest.
    Calls back into Isabelle ML via the Semantic_Store.interpret_theories callback."""
    await connection.callback("Semantic_Store.interpret_theories", (None, names))


_RERANK_FETCH_MULTIPLIER = 4


class Semantic_Vector_Store(Vector_Store):

    class ExtraKey(NamedTuple):
        """A caller-supplied extra search key for ContextExtended: its universal
        key, display name, and whether it is local (proof-context-local) to the
        query. `is_local` drives the no-embedding default score in lookup."""
        key: universal_key
        name: str
        is_local: bool = False

    class Domain:
        """Domain of entities to search over in lookup."""
        pass

    class Restricted(Domain):
        """Search only within the given keys."""
        def __init__(self, keys: list[universal_key]):
            self.keys = keys

    class _ContextAll(Domain):
        """All available entities at the connection's context. Singleton."""
        pass
    ContextAll = _ContextAll()  # singleton instance

    class ContextExtended(Domain):
        """All available entities at the connection's context, plus additional keys.
        Each extra is an ExtraKey carrying its name and locality."""
        def __init__(self, extra: 'list[Semantic_Vector_Store.ExtraKey]'):
            self.extra = extra
    @staticmethod
    def clean_all_wip_in_created_dbs() -> None:
        """Remove all WIP (non-persistent) entries from every vector store on disk."""
        from Isabelle_RPC_Host.theory_hash import is_persistent
        for env in _iter_vector_store_envs():
            to_delete: list[bytes] = []
            with env.begin(write=True) as txn:
                for key, _ in txn.cursor():
                    if not is_persistent(key):
                        to_delete.append(bytes(key))
                for key in to_delete:
                    txn.delete(key)

    @staticmethod
    def created_embedding_models() -> list[str]:
        """Return names of all embedding models that have LMDB stores on disk."""
        cache_dir = semantic_DB_dir()
        if not os.path.isdir(cache_dir):
            return []
        prefix = "vector_"
        suffix = ".lmdb"
        return [unsanitize_model(entry[len(prefix):-len(suffix)])
                for entry in os.listdir(cache_dir)
                if entry.startswith(prefix) and entry.endswith(suffix)
                and os.path.isdir(os.path.join(cache_dir, entry))]

    def __init__(
        self,
        emb_provider: Embedding_Provider | None = None,
        connection: Connection | None = None,
    ):
        # When no provider is supplied, build one from the env-resolved
        # (driver, base_url, model) triple (no connection config available here).
        if emb_provider is None:
            driver, base_url, model = _resolve_embedding_config_env()
            emb_provider = make_embedding_provider(driver, base_url, model)
        # Identity = the canonical (HuggingFace) model name; the LMDB store
        # directory uses a filesystem-safe form of it.
        model_name = emb_provider.canonical_model
        cache_dir = semantic_DB_dir()
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"vector_{sanitize_model(model_name)}.lmdb")
        super().__init__(path, emb_provider, connection)
        self.model_name = model_name
        if connection is not None:
            with _svs_lock:
                stores = getattr(connection, '_semantic_vector_stores', None)
                if stores is None:
                    stores = {}
                    connection._semantic_vector_stores = stores # type: ignore
                if model_name in stores:
                    raise ValueError(f"Semantic_Vector_Store for {model_name!r} already registered on this connection")
                stores[model_name] = self

    async def _get_reranker(self) -> 'Reranker_Provider | None':
        """Lazily resolve the reranker from current config/env at call time."""
        reranker_name = await _resolve_reranker_model(self.connection)
        if reranker_name is None:
            return None
        return reranker_provider(reranker_name)

    def is_thy_embedded(self, theory_key: universal_key) -> bool:
        """Check whether a theory's entities are all embedded in this vector store."""
        with self._env.begin() as txn:
            raw = txn.get(theory_key)
        if raw is None:
            return False
        return unpack_thy_status(raw).get(b"finished", False)

    def thy_embed_tokens(self, theory_key: universal_key) -> int | None:
        """Look up the total tokens used to embed a theory. Returns None if not found."""
        with self._env.begin() as txn:
            raw = txn.get(theory_key)
        if raw is None:
            return None
        return unpack_thy_status(raw).get(b"total_tokens", 0)

    def mark_thy_embedded(self, theory_key: universal_key, total_tokens: int = 0) -> None:
        """Mark a theory as fully embedded in this vector store, recording token usage.
        Skips WIP (non-persistent) theories unless persist_wip is enabled."""
        if is_WIP(theory_key) and not persist_wip:
            return
        with self._env.begin(write=True) as txn:
            raw = txn.get(theory_key)
            if raw is not None:
                data = unpack_thy_status(raw)
            else:
                data = {}
            data[b"finished"] = True
            if total_tokens > 0:
                data[b"total_tokens"] = data.get(b"total_tokens", 0) + total_tokens
            txn.put(theory_key, msgpack.packb(data))  # type: ignore

    async def _auto_embed(self, missing: list[key]) -> list[key]:
        if self.connection is None:
            return []
        gate = await self.connection.config_lookup("auto_interpret_for_embedding")
        confirmed = False
        theory_hashes: set[bytes] = set()
        # (1) When the gate permits it, auto-interpret the uninterpreted theories of the
        # non-xor missing entities so they become embeddable.  XOR-prefixed keys
        # (theorem/rule AND experience) are skipped: their prefix is an XOR pseudo-theory,
        # not a locatable theory.  EXPERIENCE keys are xor-prefixed AND carry their own
        # interpretation, so they never need this step -- the gate governs ONLY this
        # interpretation, not the embedding in (2) (S1-a).
        if gate:
            from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key
            for k in missing:
                if is_xor_prefixed_key(k):
                    continue
                entity = destruct_key(k)
                if not self.is_thy_embedded(entity.theory):
                    theory_hashes.add(entity.theory)
            await self.connection.tracing(
                f"[Semantic_Embedding] {len(missing)} entities missing embeddings, "
                f"spanning {len(theory_hashes)} un-embedded theories")
            # Filter to uninterpreted theories, excluding the current theory and skipped theories
            from Isabelle_RPC_Host.context import theory_long_name
            current_thy = await theory_long_name(self.connection)
            uninterpreted_theories: list[str] = []
            for th in theory_hashes:
                if not Semantic_DB.is_thy_interpreted(th):
                    name = await self.connection.callback("Theory_Hash.theory_name_of", th)
                    if name is not None and name != current_thy and not is_thy_skipped(name):
                        uninterpreted_theories.append(name)
            if uninterpreted_theories:
                if len(uninterpreted_theories) > 5:
                    import Isabelle_RPC_Host.dialogue
                    answer = await self.connection.dialogue(
                        f"[Semantic Embedding] {len(uninterpreted_theories)} uninterpreted theories "
                        f"need interpretation before embedding. "
                        f"This may consume a significant amount of API tokens. Proceed?",
                        ["Yes", "No"])
                    if answer != "Yes":
                        return []
                    confirmed = True
                await self.connection.tracing(
                    f"[Semantic_Embedding] {len(uninterpreted_theories)} of {len(theory_hashes)} theories "
                    f"not yet interpreted, running interpretation for: "
                    + ", ".join(uninterpreted_theories))
                await interpret_theories_by_names(self.connection, uninterpreted_theories)
        # (2) Keys embeddable NOW: record present AND document_text_of != None -- every
        # EXPERIENCE, plus entities whose theory is interpreted (including any just
        # interpreted in (1)).  This set embeds regardless of the gate (S1-a): the gate
        # is about spending LLM tokens to interpret, and these need no interpretation.
        # ONE read txn for the whole batch (get_many), not one per key: `missing` can be
        # 10^5 long on a library not yet embedded for this model, and we are on the event
        # loop.  This runs before the gate check below because the gate governs only the
        # interpretation in (1) -- but it must not cost 10^5 begin/commit pairs to find
        # out there is nothing ready.
        ready: 'list[tuple[key, SemanticRecord]]' = [
            (k, rec) for k, rec in zip(missing, Semantic_DB.get_many(missing))
            if rec is not None and document_text_of(rec) is not None]
        # (3) Gate off: warn about the entities we could NOT embed (they would need
        # interpretation, which the gate forbids).  The already-ready set still embeds.
        if not gate:
            n_blocked = len(missing) - len(ready)
            if n_blocked:
                await self.connection.warning(
                    f"[Semantic_Embedding] {n_blocked} entities missing semantic embeddings, "
                    f"but auto_interpret_for_embedding is disabled. "
                    f"Set [[auto_interpret_for_embedding = true]] to enable automatic interpretation and embedding.")
        if not ready:                                    # C1: nothing embeddable -> do NOT mark theories
            if gate:
                await self.connection.tracing(
                    f"[Semantic_Embedding] no semantic interpretations found for the missing entities, skipping")
            return []
        if len(ready) > 42 and not confirmed:
            import Isabelle_RPC_Host.dialogue
            answer = await self.connection.dialogue(
                f"[Semantic Embedding] {len(ready)} entities to embed. "
                f"This may consume a significant amount of API tokens. Proceed?",
                ["Yes", "No"])
            if answer != "Yes":
                return []
        await self.connection.tracing(
            f"[Semantic_Embedding] embedding {len(ready)} of {len(missing)} missing entities into vectors")
        tokens = await self.embed_records(ready, force=True)
        # Mark processed theories as embedded, recording cost.  theory_hashes is empty
        # when the gate is off, so a gate-off ready-only embed marks nothing.
        for th in theory_hashes:
            self.mark_thy_embedded(th, tokens)
        return [k for k, _ in ready]

    async def _experience_hits(self, term_patterns: 'list[str]',
                               ctxt: Any) -> 'dict[universal_key, float]':
        """Available experience memories → hit_rate.

        Availability (Python-driven, §3): an experience is available iff every
        theory in its constituent list is loaded.  We get the loaded theory
        hashes from ML (Context.loaded_theory_hashes), take the inverted-index
        candidates that mention any loaded theory, then keep those whose FULL
        constituent set is loaded.  hit_rate is the fraction of query
        ``term_patterns`` matched (ML Context.experiences, relaxed bidirectional
        subterm; only >0 returned).  With no term_patterns every available
        experience gets 1.0 (pure semantic ranking; the boost is disabled
        downstream)."""
        from .experience_index import Experience_Index
        conn = self.connection
        if conn is None:
            return {}
        raw = await conn.callback("Context.loaded_theory_hashes", ctxt)
        loaded = {bytes(h) for h in raw}
        available: list[tuple[universal_key, list[str]]] = []
        for uk in Experience_Index.candidates(loaded):
            rec = Semantic_DB[uk]
            if rec is None or rec.theory_constituents is None:
                continue
            if all(h in loaded for _, h in rec.theory_constituents):
                try:
                    pats = json.loads(rec.expr) if rec.expr else []
                except (json.JSONDecodeError, TypeError):
                    pats = []
                available.append((uk, pats))
        if not available:
            return {}
        if not term_patterns:
            return {uk: 1.0 for uk, _ in available}
        hit_raw = await conn.callback(
            "Context.experiences", (ctxt, (term_patterns, available)))
        return {bytes(uk): float(hr) for uk, hr in hit_raw}

    async def lookup(
        self,
        query: np.ndarray | str,
        k: int,
        kinds: list[EntityKind],
        domain: 'Semantic_Vector_Store.Domain' = ContextAll,
        term_patterns: list[str] = [],
        type_patterns: list[str] = [],
        theories_include: list[str] = [],
        name_contains: list[str] = [],
        target_type: str = "",
        ctxt: Any = None,
    ) -> tuple[list[tuple[float, 'SemanticRecord']], list[str], int]:
        """Search the k closest entities to query, filtered by kinds and domain.
        Returns (results, warnings, total) where results are (score, record)
        pairs sorted by similarity, warnings include notices about undeclared
        free variables in term patterns, and total is the number of entities
        matching the filters (the full candidate pool before the top-k cut).
        Domain controls the search scope:
          ContextAll (default): all context entities of the given kinds
          ContextExtended(extra): context entities + additional keys
          Restricted(keys): only the given keys, filtered by kinds
        Pattern/theory filters (empty = no restriction):
          term_patterns: Isabelle term pattern strings (structural subterm matching)
          type_patterns: Isabelle type pattern strings (type matching)
          target_type: induction/case-split rule target type (silently ignored
            for other kinds; bidirectional Sign.typ_instance, wildcards allowed)
          theories_include: case-insensitive substrings matched against each
            entity's fully-qualified theory name (any match keeps it)
        """
        warnings: list[str] = []
        if not kinds:
            return [], warnings, 0
        # candidate_names: uk → full name (for synthesizing placeholder records)
        candidate_names: dict[universal_key, str] = {}
        # is_local_map: uk → proof-context-local? Drives the no-embedding default
        # score below. Theorems get it from the ML callback (via entities_of);
        # ContextExtended.extra records carry their own; everything else non-local.
        is_local_map: dict[universal_key, bool] = {}
        if domain is Semantic_Vector_Store.ContextAll:
            if self.connection is None:
                return [], warnings, 0
            from Isabelle_RPC_Host.context import entities_of
            entries, branch_local, warnings = await entities_of(self.connection, kinds,
                                     theories_not_include=_SKIP_THEORY_LONG_NAMES,
                                     term_patterns=term_patterns,
                                     type_patterns=type_patterns,
                                     theories_include=theories_include,
                                     name_contains=name_contains,
                                     target_type=target_type,
                                     ctxt=ctxt)
            candidates = [uk for uk, _, _ in entries]
            for uk, name, _ in entries:
                candidate_names[uk] = name
            is_local_map.update(branch_local)
        elif isinstance(domain, Semantic_Vector_Store.ContextExtended):
            if self.connection is None:
                return [], warnings, 0
            from Isabelle_RPC_Host.context import entities_of
            entries, branch_local, warnings = await entities_of(self.connection, kinds,
                                     theories_not_include=_SKIP_THEORY_LONG_NAMES,
                                     term_patterns=term_patterns,
                                     type_patterns=type_patterns,
                                     theories_include=theories_include,
                                     name_contains=name_contains,
                                     target_type=target_type,
                                     ctxt=ctxt)
            candidates = [uk for uk, _, _ in entries]
            for uk, name, _ in entries:
                candidate_names[uk] = name
            is_local_map.update(branch_local)
            seen = set(candidates)
            kind_set = set(kinds)
            for rec in domain.extra:
                ek = rec.key
                if ek in seen or destruct_key(ek).kind not in kind_set:
                    continue
                # Honor the query-aware cross-kind dedup (decision 8): entities_of has
                # already suppressed the Theorem(0x02) sibling of every rule member it
                # surfaced as a rule.  If this extra is that suppressed Theorem face
                # (its rule sibling is already present), do NOT re-add it as a theorem —
                # otherwise the member would surface twice (rule face + theorem face).
                if (len(ek) == 32 and ek[16] == int(EntityKind.THEOREM)
                        and any(ek[:16] + bytes([t]) + ek[17:] in seen
                                for t in RULE_ONLY_TAG_BYTES)):
                    continue
                candidates.append(ek)
                seen.add(ek)
                candidate_names[ek] = rec.name
                is_local_map[ek] = rec.is_local
        elif isinstance(domain, Semantic_Vector_Store.Restricted):
            # NB: this branch does NOT populate `candidate_names`, so results
            # here display the DB-stored name (no live `coll(i)` override, and no
            # drop of a stale dynamic-collection member). That is safe only
            # because the Restricted domain is never constructed on the AoA query
            # path (model.py builds only ContextAll / ContextExtended), so
            # dynamic-collection-member uks cannot reach here. If a caller ever
            # routes member uks through Restricted, populate `candidate_names`
            # from a live ML enumeration here, mirroring the Context* branches.
            kind_set = set(kinds)
            candidates = [dk for dk in domain.keys if destruct_key(dk).kind in kind_set]
        else:
            raise TypeError(f"Unknown domain type: {type(domain)}")
        # Experience-memory track: availability + hit_rate (Python-driven, §3/§5).
        # Only when EXPERIENCE is among the requested kinds and the query is a
        # string (experiences are ranked by their own embedded query).
        exp_hit: dict[universal_key, float] = {}
        if (EntityKind.EXPERIENCE in kinds and self.connection is not None
                and isinstance(query, str)):
            exp_hit = await self._experience_hits(term_patterns, ctxt)
        if not candidates and not exp_hit:
            return [], warnings, 0
        total = len(candidates) + len(exp_hit)

        def _apply_live_name(uk: universal_key, rec: SemanticRecord) -> SemanticRecord:
            """Prefer the live, context-resolved name (e.g. 'coll(i)' for a member
            of a dynamic collection, computed at enumeration time) over the stored
            bare name. For static facts the live name equals the stored name."""
            name = candidate_names.get(uk)
            return rec._replace(name=name) if name is not None else rec

        def _resolve(uk: universal_key) -> SemanticRecord | None:
            """Look up SemanticRecord, falling back to a placeholder if name is known."""
            rec = Semantic_DB[uk]
            if rec is not None:
                return _apply_live_name(uk, rec)
            name = candidate_names.get(uk)
            if name is not None:
                return SemanticRecord(EntityKind(uk[16]), name, None, None)
            return None

        def _default_score(uk: universal_key) -> float:
            """Fallback score for an entity with no embedding vector (no
            interpretation): provider-supplied, higher for proof-context-local
            entities. Replaces the old hardcoded 0.0; values are model-dependent."""
            prov = self.emb_provider
            return (prov.default_local_score if is_local_map.get(uk, False)
                    else prov.default_score)

        # Save original query string for potential reranking (topk embeds it)
        query_str = query if isinstance(query, str) else None
        reranker = (await self._get_reranker()) if query_str else None
        fetch_k = k * _RERANK_FETCH_MULTIPLIER if reranker else k
        # Stage-1: entity track (embedded with the {kinds} task) merged with the
        # experience track (embedded with the experience task). EXPERIENCE never
        # reaches entities_of (no per-kind callback), so `candidates` is entity-only.
        entity_kinds = [kk for kk in kinds if kk != EntityKind.EXPERIENCE]
        top: list[tuple[universal_key, float]] = []
        if candidates:
            top = await self.topk(query, candidates, fetch_k,
                                   kinds_phrase=render_kinds(entity_kinds))
        if exp_hit and query_str is not None:
            from . import embedding_config as _ecfg
            exp_qvec = (await self.emb_provider.embed(
                [query_str], role="query",
                task_override=_ecfg.experience_task_description())).vectors[0]
            top = top + await self.topk(exp_qvec, list(exp_hit.keys()), fetch_k)
        # Stage-1 relevance boost (§6): a convex pull of the cosine toward 1 by
        # hit_rate, applied uniformly (entities default hit_rate 1). Disabled when
        # the query has <= 1 term pattern (every survivor then has hit_rate 1, so
        # the boost is a uniform monotone shift that cannot reorder). ONLY Stage 1:
        # the reranker's own scores are never touched. Merge-sort the two tracks by
        # (boosted) score and keep fetch_k.
        if len(term_patterns) > 1:
            _BETA = 0.25
            top = [(uk, s + (1.0 - s) * _BETA * exp_hit.get(uk, 1.0)) for uk, s in top]
        top.sort(key=lambda x: x[1], reverse=True)
        top = top[:fetch_k]
        # Rerank if configured and query was a string
        if reranker is not None and query_str is not None and len(top) > 1:
            doc_entries: list[tuple[universal_key, SemanticRecord, str]] = []
            for uk, _score in top:
                rec = Semantic_DB[uk]
                if rec is not None:
                    # The reranker scores the SAME canonical document that was embedded
                    # (document_text_of), per kind -- not a third, divergent text (D1).
                    doc_text = document_text_of(rec)
                    if doc_text:
                        doc_entries.append((uk, _apply_live_name(uk, rec), doc_text))
            if doc_entries:
                try:
                    rr = await reranker.rerank(
                        query_str, [e[2] for e in doc_entries], min(k, len(doc_entries)))
                    reranked: list[tuple[float, SemanticRecord]] = [
                        (rr.scores[i], doc_entries[idx][1])
                        for i, idx in enumerate(rr.indices)]
                    # Append candidates that had no reranking text (no interpretation)
                    reranked_set = {e[1].name for e in reranked}
                    for uk, score in top:
                        if len(reranked) >= k:
                            break
                        rec = _resolve(uk)
                        if rec is not None and rec.name not in reranked_set:
                            reranked.append((score, rec))
                    return reranked[:k], warnings, total
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Reranker failed, falling back to embedding scores: %s", e)
        # Non-reranker path: merge embedded entities (real KNN score) with
        # no-embedding entities (provider default score by locality), then sort by
        # final score and take top-k ("全量按最终分重排"). Embedded entities keep their
        # real score — the default only replaces the old 0.0 fallback, never floors a
        # real score. Bounded resolve: at most k embedded + k local + k non-local
        # no-embedding records are materialized (each no-embedding tier is tied at one
        # default value, so the first k of each in candidate order suffice for top-k).
        top_scores = dict(top)
        scored: list[tuple[float, SemanticRecord]] = []
        for uk, score in top:                       # embedded: real KNN scores (≤ k)
            rec = _resolve(uk)
            if rec is not None:
                scored.append((score, rec))
        local_quota = nonlocal_quota = k
        for uk in candidates:                       # no-embedding: default by locality
            if local_quota <= 0 and nonlocal_quota <= 0:
                break
            if uk in top_scores:
                continue
            is_loc = is_local_map.get(uk, False)
            if is_loc and local_quota <= 0:
                continue
            if not is_loc and nonlocal_quota <= 0:
                continue
            rec = _resolve(uk)
            if rec is None:
                continue
            scored.append((_default_score(uk), rec))
            if is_loc:
                local_quota -= 1
            else:
                nonlocal_quota -= 1
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k], warnings, total

    async def embed_records(self, items: 'list[tuple[universal_key, SemanticRecord]]',
                            *, force: bool = False) -> int:
        """THE single choke point that turns records into stored vectors, using this
        store's bound (model, provider).  Text comes only from ``document_text_of`` --
        the one authority, dispatched on kind -- so every write path shares one
        convention.  Records with no embeddable text (interpretation missing, or a
        corrupt experience ``expr``) are skipped.  With ``force=False`` skips keys
        already present in this store.  Returns tokens consumed.

        Works on in-memory records (no DB round-trip needed), which is what lets
        write_memory embed with the very text a later re-embed reconstructs -- byte
        identical by construction.  connection is optional (offline embed passes None);
        only the tracing line uses it, guarded."""
        if not force:
            keys = [k for k, _ in items]
            exist = self.contains(keys)
            items = [it for it, ex in zip(items, exist) if not ex]
        kv = [(k, t) for k, rec in items if (t := document_text_of(rec)) is not None]
        if not kv:
            return 0
        if self.connection is not None:
            await self.connection.tracing(
                f"[Semantic_Embedding] embedding {len(kv)} records ({sum(len(t) for _, t in kv)} chars)")
        return await self.embed(kv)

    async def embed_keys(self, keys: list[universal_key], *, force: bool = False) -> int:
        """Convenience over :meth:`embed_records`: fetch each key's record from
        Semantic_DB, then delegate.  Callers that already hold the records should call
        embed_records directly (this re-reads them).  Returns tokens consumed.

        The vector-store ``contains`` filter runs FIRST (one read txn on this store), so
        only the not-yet-embedded keys cost a Semantic_DB read (one read txn + msgpack
        decode each).  On a mostly-embedded theory that is the difference between one
        txn and N."""
        if not force:
            exist = self.contains(keys)
            keys = [k for k, ex in zip(keys, exist) if not ex]
        items = [(k, rec) for k, rec in zip(keys, Semantic_DB.get_many(keys))
                 if rec is not None]
        # force=True: `keys` is already exactly the set to write, so don't re-check.
        return await self.embed_records(items, force=True)

    async def embed_entities(self, keys: list[universal_key]) -> None:
        """Embed the given entity keys, skipping those already embedded."""
        await self.embed_keys(keys, force=False)

    async def put_experience(self, key: universal_key, rec: 'SemanticRecord') -> None:
        """Create/overwrite an experience across all three of its stores (record +
        this model's vector + availability index), embed-first.  Ergonomic entry to
        ``experience_store.put_experience``; the tri-store transaction lives there."""
        from .experience_store import put_experience
        await put_experience(self, key, rec)

    def delete_experience(self, key: universal_key) -> None:
        """Remove an experience from every store (record + vectors in ALL models +
        index).  Ergonomic entry to ``experience_store.delete_experience``; ``self`` is
        just the handle -- deletion is model-global (see Del1)."""
        from .experience_store import delete_experience
        delete_experience(key)

    async def embed_all_entities_in_theories(self, theories: list[str | universal_key],
                                              *, force: bool = False) -> None:
        """Embed semantic interpretations into vectors for the given theories.

        For each theory, collects all entity keys, embeds missing ones,
        and marks the theory as fully embedded.

        Args:
            theories: Long theory names (str) or universal keys to embed.
            force: If True, re-visit theories already marked as fully embedded.

        Raises:
            RuntimeError: If no active connection is available.
            ValueError: If a universal key is not found in the active Isabelle runtime.
        """
        if self.connection is None:
            raise RuntimeError("embed_all_entities_in_theories requires an active connection")
        from Isabelle_RPC_Host.context import entities_of

        for thy in theories:
            if isinstance(thy, str):
                thy_name = thy
                try:
                    thy_key = await universal_key_of(self.connection, EntityKind.THEORY, thy)
                except UndefinedEntity:
                    await self.connection.warning(
                        f"[Semantic_Embedding] skipping unknown theory {thy!r}")
                    continue
            else:
                thy_key = bytes(thy)
                thy_name = await self.connection.callback("Theory_Hash.theory_name_of", thy_key)
                if thy_name is None:
                    raise ValueError(
                        f"Theory key {thy_key.hex()} not found in the active Isabelle runtime; "
                        f"the theory may not be loaded")
            if not force and self.is_thy_embedded(thy_key):
                continue
            entries, _is_local, _warnings = await entities_of(self.connection, EntityKind.ALL, # type: ignore
                               theory=thy_name, the_theory_only=True)
            keys = [k for k, _, _ in entries]
            wip = is_WIP(thy_key) and not persist_wip
            await self.embed_keys(keys, force=force)
            if not wip:
                self.mark_thy_embedded(thy_key)


_svs_lock = threading.Lock()


_DEFAULT_EMBEDDING_DRIVER = "OpenAI_Embedding_Provider"
_DEFAULT_EMBEDDING_BASE_URL = "https://api.fireworks.ai/inference"
_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def _resolve_embedding_config_env() -> tuple[str, str, str]:
    """Resolve (driver, base_url, model) from env vars, falling back to defaults."""
    return (os.getenv("EMBEDDING_DRIVER", _DEFAULT_EMBEDDING_DRIVER),
            os.getenv("EMBEDDING_BASE_URL", _DEFAULT_EMBEDDING_BASE_URL),
            os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL))


async def _resolve_embedding_config(connection: Connection | None) -> tuple[str, str, str]:
    """Resolve (driver, base_url, model), each by: ML config -> env -> default."""
    driver, base_url, model = _resolve_embedding_config_env()
    if connection is not None:
        driver = (await connection.config_lookup("Semantic_Embedding.embedding_driver")) or driver
        base_url = (await connection.config_lookup("Semantic_Embedding.embedding_base_url")) or base_url
        model = (await connection.config_lookup("Semantic_Embedding.embedding_model")) or model
    return driver, base_url, model


async def _resolve_reranker_model(connection: Connection | None) -> str | None:
    """Resolve reranker model name from config, env, or None (disabled)."""
    if connection is not None:
        name = await connection.config_lookup("Semantic_Embedding.reranker_model")
        if name:
            return name
    model = os.getenv("RERANKER_MODEL", "")
    return model if model else None


async def _conn_semantic_vector_store(self: Connection, embedding_model: str | None = None) -> Semantic_Vector_Store:
    """Get or create a Semantic_Vector_Store for the active (or given) embedding model.

    With no ``embedding_model``, uses the fully resolved (driver, base_url, model)
    triple. When ``embedding_model`` is given, it overrides only the model while
    reusing the active driver + base_url.

    LIMITATION: a single run has exactly one active driver + base_url, so embedding
    several models in one run (e.g. semantics_manage --embed-models) only works for
    models served by that same endpoint (fireworks-hosted qwen3/harrier/nv-embed
    are fine; mixing e.g. fireworks + mistral in one run is not supported).
    """
    driver, base_url, model = await _resolve_embedding_config(self)
    if embedding_model is not None:
        model = embedding_model
    with _svs_lock:
        stores = getattr(self, '_semantic_vector_stores', None)
        if stores is not None and model in stores:
            return stores[model]
    provider = make_embedding_provider(driver, base_url, model)
    return Semantic_Vector_Store(emb_provider=provider, connection=self)

Connection.semantic_vector_store = _conn_semantic_vector_store  # type: ignore


# --- RPC wrappers ---

@isabelle_remote_procedure("Semantic_Store.query")
async def _query(arg: Any, connection: Connection) -> str | None:
    key, with_pretty = arg
    return Semantic_DB.query(bytes(key), bool(with_pretty))


@isabelle_remote_procedure("Semantic_Store.is_interpreted")
async def _is_interpreted(arg: Any, connection: Connection) -> bool:
    key = bytes(arg)
    if Semantic_DB.is_thy_interpreted(key):
        return True
    if migrate_on_hash_change and Semantic_DB._try_migrate(key):
        return True
    return False


@isabelle_remote_procedure("Semantic_Store.mark_interpreted")
async def _mark_interpreted(arg: Any, connection: Connection) -> None:
    Semantic_DB.mark_interpreted(bytes(arg))


@isabelle_remote_procedure("Semantic_Store.clean_wip")
async def _clean_wip(arg: Any, connection: Connection) -> int:
    return clean_wip()


@isabelle_remote_procedure("Semantic_Store.contains")
async def _contains(arg: Any, connection: Connection) -> list[bool]:
    keys = [bytes(k) for k in arg]
    return Semantic_DB.contains(keys)


@isabelle_remote_procedure("Semantic_Embedding.query_knn")
async def _query_knn(arg: Any, connection: Connection) -> tuple[
        list[tuple[float, tuple[int, str]]], list[str]]:
    (query_str, k, kind_ints, domain_raw,
     term_patterns, type_patterns, theories_include, name_contains,
     target_type) = arg
    kinds = [EntityKind(ki) for ki in kind_ints]
    # Decode domain tagged union: (tag, payload)
    domain_tag, domain_payload = domain_raw
    if domain_tag == 0:
        domain: Semantic_Vector_Store.Domain = Semantic_Vector_Store.ContextAll
    elif domain_tag == 1:
        # Wire-supplied extras carry (key, name); they are not known to be local.
        domain = Semantic_Vector_Store.ContextExtended(
            [Semantic_Vector_Store.ExtraKey(key=bytes(k_), name=name, is_local=False)
             for k_, name in domain_payload])
    elif domain_tag == 2:
        domain = Semantic_Vector_Store.Restricted([bytes(uk) for uk in domain_payload])
    else:
        raise ValueError(f"Unknown domain tag: {domain_tag}")
    store = await connection.semantic_vector_store()  # type: ignore
    results, warnings, _total = await store.lookup(query_str, k, kinds, domain,
        term_patterns=list(term_patterns),
        type_patterns=list(type_patterns),
        theories_include=list(theories_include),
        name_contains=list(name_contains),
        target_type=str(target_type))
    return ([(score, (int(rec.kind), rec.name)) for score, rec in results],
            list(warnings))


@isabelle_remote_procedure("Semantic_Embedding.embed_all_entities_in_theories")
async def _embed_all_entities_in_theories(arg: Any, connection: Connection) -> None:
    theory_names, model_name, force = arg
    store = await connection.semantic_vector_store(model_name or None)  # type: ignore
    await store.embed_all_entities_in_theories(theory_names, force=force)


@isabelle_remote_procedure("Semantic_Embedding.embed_entities")
async def _embed_entities(arg: Any, connection: Connection) -> None:
    keys = [bytes(k) for k in arg]
    store = await connection.semantic_vector_store()  # type: ignore
    await store.embed_entities(keys)


@isabelle_remote_procedure("Semantic_Embedding.contains")
async def _embedding_contains(arg: Any, connection: Connection) -> list[bool]:
    keys = [bytes(k) for k in arg]
    store = await connection.semantic_vector_store()  # type: ignore
    return store.contains(keys)


@isabelle_remote_procedure("Semantic_Embedding.is_thy_embedded")
async def _is_thy_embedded_rpc(arg: Any, connection: Connection) -> bool:
    theory_name, model_name = arg
    store = await connection.semantic_vector_store(model_name)  # type: ignore
    try:
        thy_key = await universal_key_of(connection, EntityKind.THEORY, theory_name)
    except UndefinedEntity:
        return False
    return store.is_thy_embedded(thy_key)


@isabelle_remote_procedure("Vector_Arith.library_path")
async def _vector_library_path_rpc(arg: Any, connection: Connection) -> str:
    """Tell Isabelle/ML where libisabelle_vector.so is.

    Tools/simd_vector.ML dlopens the same object, and a wheel install puts it under
    site-packages, which no path hard-coded in a theory can find. Answering here
    also guarantees ML and Python load the same file: library_path() reports the
    library only after loading it and checking it exports the kernel.

    It lives in this module rather than in _vecarith so that the numeric core stays
    importable without Isabelle_RPC_Host -- the migration script depends on that.
    """
    return _vector_library_path()