"""Regression tests for the gate-off branch of ``Semantic_Vector_Store._auto_embed``.

This branch (auto_interpret_for_embedding = false) emits the "not interpreted"
warning.  It is reachable only from a live Isabelle RPC lookup, so it is driven
here with a stub Connection and a monkeypatched Semantic_DB.

Regression guarded: the warning used to fire on ``n_blocked`` -- a count that
includes keys the theory-name filter deliberately drops (the current theory,
skipped infra theories such as Pure, unknown theory hashes, and xor-prefixed
theorem keys).  In steady state, when every nameable theory is interpreted, only
those undroppable keys remain, and the warning fired forever as
"0 theories and 4 entities ... : " -- dangling colon, empty list, and a remedy
that resolves nothing.
"""
import asyncio

import pytest

from Isabelle_RPC_Host.universal_key import EntityKind, xor_theory_prefix
import Isabelle_Semantic_Embedding.semantics as S


def _thy(n: int) -> bytes:
    return bytes([n]) + b"\x00" * 15


def _ent(th: bytes, name: str) -> bytes:
    return th + bytes([int(EntityKind.CONSTANT)]) + name.encode()


def _thm(ths: list) -> bytes:
    return xor_theory_prefix(ths) + bytes([int(EntityKind.THEOREM)]) + b"\x00" * 15


T_CUR, T_SKIP, T_UNK = _thy(1), _thy(2), _thy(3)
CUR_NAME = "Scratch.Current"


class _StubConn:
    """Minimal Connection: gate off, ML-Option-like theory_name_of, captured output."""

    def __init__(self, names: dict):
        self.names = names
        self.warnings: list = []
        self.tracings: list = []

    async def config_lookup(self, name, ctxt=None):
        assert name == "auto_interpret_for_embedding"
        return False

    async def callback(self, name, arg):
        if name == "Context.the_theory_long_name":
            return CUR_NAME
        if name == "Theory_Hash.theory_name_of":
            return self.names.get(arg)      # None mirrors the ML NONE
        raise AssertionError(f"unexpected callback {name}")

    async def warning(self, msg):
        self.warnings.append(msg)

    async def tracing(self, msg):
        self.tracings.append(msg)


def _store(names, missing, ready_key=None):
    S.Semantic_DB.get_many = lambda ks: [
        ({"interpretation": "x"} if k == ready_key else None) for k in ks]
    S.Semantic_DB.is_thy_interpreted = lambda k: False
    S.document_text_of = lambda rec: "doc"
    store = object.__new__(S.Semantic_Vector_Store)
    store.connection = _StubConn(names)

    async def _embed(records, force=False):
        return 0
    store.embed_records = _embed
    store.mark_thy_embedded = lambda th, tok: None
    return store


def test_warning_names_theories_and_counts_only_their_entities():
    """12 nameable theories + 4 keys the filter drops: both numbers must be 12."""
    plain = [_thy(10 + i) for i in range(12)]
    names = {T_CUR: CUR_NAME, T_SKIP: "Pure", T_UNK: None}
    names.update({t: f"HOL.Thy{i:02d}" for i, t in enumerate(plain)})

    ready_key = _ent(plain[0], "already_ready")
    missing = (
        [_ent(T_CUR, "cur"), _ent(T_SKIP, "skip"), _ent(T_UNK, "unk")]
        + [_ent(t, f"c{i}") for i, t in enumerate(plain)]
        + [_thm([plain[0], plain[1]])]
        + [ready_key]
    )
    store = _store(names, missing, ready_key)
    assert asyncio.run(store._auto_embed(missing)) == [ready_key]

    (warn,) = store.connection.warnings
    # The entity count must match the named set, not len(missing) - len(ready) == 16.
    assert "12 theories and 12 entities" in warn
    assert "HOL.Thy00" in warn and "(and 2 more)" in warn
    # Nothing the filter drops may be named.
    assert CUR_NAME not in warn and "Pure" not in warn


def test_no_warning_when_nothing_can_be_named():
    """Steady state: only undroppable keys are blocked -> stay silent."""
    names = {T_CUR: CUR_NAME, T_SKIP: "Pure", T_UNK: None}
    missing = [_ent(T_CUR, "cur"), _ent(T_SKIP, "skip"), _ent(T_UNK, "unk"),
               _thm([_thy(10)])]
    store = _store(names, missing)
    assert asyncio.run(store._auto_embed(missing)) == []
    assert store.connection.warnings == []
