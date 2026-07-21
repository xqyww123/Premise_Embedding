"""Tests for the shared vector-store completion routine (complete_vector_store),
the Semantic_Embedding.embed_all_missing RPC handler, the refactored CLI
_embed_models, and the Semantic_Store_verbose tracing gate.

Hermetic in the style of test_auto_embed_gate_off.py: stores are object.__new__
shells with stubbed contains/embed_records; no LMDB, no network, no live DB dir.
"""
import asyncio
import sys

import pytest

import Isabelle_Semantic_Embedding.semantics as S
import Isabelle_Semantic_Embedding.semantic_embedding as SE


def _mk_store(present=frozenset(), tokens_per_batch=7):
    store = object.__new__(S.Semantic_Vector_Store)
    store.connection = None
    store.calls = []

    def contains(keys):
        return [k in present for k in keys]

    async def embed_records(items, force=False):
        store.calls.append((list(items), force))
        return tokens_per_batch

    store.contains = contains
    store.embed_records = embed_records
    return store


class _Recorder:
    def __init__(self):
        self.reports, self.warns = [], []

    async def report(self, m):
        self.reports.append(m)

    async def warn(self, m):
        self.warns.append(m)


@pytest.fixture
def doc_text(monkeypatch):
    """document_text_of stub: records are dicts, {'text': ...} or {} (unrenderable)."""
    monkeypatch.setattr(S, "document_text_of", lambda rec: rec.get("text"))


def test_missing_only_filter_and_force(doc_text):
    cands = [(b"k1", {"text": "a"}), (b"k2", {"text": "b"}), (b"k3", {"text": "c"})]

    store = _mk_store(present={b"k2"})
    r = _Recorder()
    out = asyncio.run(S.complete_vector_store(
        store, cands, force=False, label="M", report=r.report, warn=r.warn))
    assert [k for items, _ in store.calls for k, _ in items] == [b"k1", b"k3"]
    assert all(force for _, force in store.calls)
    assert out == (2, 0, 7)

    store2 = _mk_store(present={b"k2"})
    store2.contains = lambda keys: pytest.fail("force must not consult contains")
    out2 = asyncio.run(S.complete_vector_store(
        store2, cands, force=True, label="M", report=r.report, warn=r.warn))
    assert [k for items, _ in store2.calls for k, _ in items] == [b"k1", b"k2", b"k3"]
    assert out2 == (3, 0, 7)


def test_already_complete_single_line(doc_text):
    cands = [(b"k1", {"text": "a"}), (b"k2", {"text": "b"})]
    store = _mk_store(present={b"k1", b"k2"})
    r = _Recorder()
    out = asyncio.run(S.complete_vector_store(
        store, cands, force=False, label="M", report=r.report, warn=r.warn))
    assert r.reports == ["M: already complete (2 entities)."]
    assert r.warns == [] and store.calls == []
    assert out == (0, 0, 0)


def test_unrenderable_warned_skipped_counted(doc_text):
    cands = [(b"k1", {"text": "aa"}), (b"\x01" * 17, {}), (b"k3", {"text": "bb"})]
    store = _mk_store()
    r = _Recorder()
    out = asyncio.run(S.complete_vector_store(
        store, cands, force=False, label="M", report=r.report, warn=r.warn))
    assert len(r.warns) == 1
    assert "1 record(s) have no embeddable" in r.warns[0]
    assert (b"\x01" * 17).hex()[:16] in r.warns[0]
    assert [k for items, _ in store.calls for k, _ in items] == [b"k1", b"k3"]
    assert out == (2, 1, 7)
    assert "M: 2 of 3 entities need vectors (4 chars)." in r.reports


def test_nothing_embeddable_single_line(doc_text):
    cands = [(b"k1", {}), (b"k2", {})]
    store = _mk_store()
    r = _Recorder()
    out = asyncio.run(S.complete_vector_store(
        store, cands, force=False, label="M", report=r.report, warn=r.warn))
    assert r.reports == ["M: nothing embeddable."]
    assert len(r.warns) == 1 and store.calls == []
    assert out == (0, 2, 0)


def test_batching_and_token_sum(doc_text):
    cands = [(b"k%03d" % i, {"text": "t"}) for i in range(600)]
    store = _mk_store(tokens_per_batch=7)
    r = _Recorder()
    out = asyncio.run(S.complete_vector_store(
        store, cands, force=True, label="M", report=r.report, warn=r.warn))
    assert [len(items) for items, _ in store.calls] == [256, 256, 88]
    assert [m for m in r.reports if m.startswith("  M: embedded ")] == [
        "  M: embedded 256/600", "  M: embedded 512/600", "  M: embedded 600/600"]
    assert r.reports[-1] == "M: done (600 embedded, 21 tokens)."
    assert out == (600, 0, 21)


def test_confirm_hook(doc_text):
    cands = [(b"k1", {"text": "a"})]

    async def refuse(n, total, chars):
        assert (n, total, chars) == (1, 1, 1)
        return False

    store = _mk_store()
    r = _Recorder()
    out = asyncio.run(S.complete_vector_store(
        store, cands, force=False, label="M", report=r.report, warn=r.warn,
        confirm=refuse))
    assert r.reports[-1] == "Aborted." and store.calls == []
    assert out == (0, 0, 0)

    store2 = _mk_store()
    out2 = asyncio.run(S.complete_vector_store(
        store2, cands, force=False, label="M", report=r.report, warn=r.warn,
        confirm=None))
    assert len(store2.calls) == 1
    assert out2 == (1, 0, 7)


class _HandlerConn:
    """Stub Connection for the embed_all_missing handler."""

    def __init__(self, store=None, exc=None, verbose=False):
        self.store, self.exc, self.verbose = store, exc, verbose
        self.writelns, self.warnings, self.tracings = [], [], []

    async def semantic_vector_store(self):
        if self.exc is not None:
            raise self.exc
        return self.store

    async def config_lookup(self, name, ctxt=None):
        assert name == "Semantic_Store_verbose"
        return self.verbose

    async def writeln(self, m):
        self.writelns.append(m)

    async def warning(self, m):
        self.warnings.append(m)

    async def tracing(self, m):
        self.tracings.append(m)


def test_handler_reports_via_writeln(doc_text, monkeypatch):
    cands = [(b"k1", {"text": "a"}), (b"k2", {})]
    monkeypatch.setattr(S, "_collect_embed_candidates", lambda kinds=None: cands)
    store = _mk_store()
    store.model_name = "Qwen/Qwen3-Embedding-8B"
    conn = _HandlerConn(store=store)
    asyncio.run(S._embed_all_missing(None, conn))
    assert conn.writelns and all(m.startswith("[Semantic_Embedding] ") for m in conn.writelns)
    assert "[Semantic_Embedding] Qwen/Qwen3-Embedding-8B: done (1 embedded, 7 tokens)." \
        in conn.writelns
    assert conn.warnings and conn.warnings[0].startswith(
        "[Semantic_Embedding] Qwen/Qwen3-Embedding-8B: WARNING -- 1 record(s)")
    assert conn.tracings == []


@pytest.mark.parametrize("exc", [
    RuntimeError("no API key configured"),
    ValueError("base_url has no API version segment"),
    ImportError("Embedding driver 'X' not found"),
])
def test_handler_marks_config_errors(exc, monkeypatch):
    from Isabelle_Semantic_Embedding.semantic_interpretation import USER_ERROR_MARKER
    monkeypatch.setattr(S, "_collect_embed_candidates",
                        lambda kinds=None: pytest.fail("must fail before the scan"))
    conn = _HandlerConn(exc=exc)
    with pytest.raises(RuntimeError) as ei:
        asyncio.run(S._embed_all_missing(None, conn))
    assert str(ei.value).startswith(USER_ERROR_MARKER)
    assert str(exc) in str(ei.value)


def test_cli_embed_models_output(doc_text, monkeypatch, capsys):
    import Isabelle_Semantic_Embedding.semantics_manage as M

    class _FakeStore:
        # Stands in for Semantic_Vector_Store: the real constructor would
        # makedirs + lmdb-open a junk vector_<model>.lmdb under the LIVE cache
        # dir, which the purge/enumeration paths then see.
        def __init__(self, emb_provider=None, connection=None):
            assert connection is None

        def contains(self, keys):
            return [False] * len(keys)

        async def embed_records(self, items, force=False):
            return 5

    monkeypatch.setattr(S, "Semantic_Vector_Store", _FakeStore)
    monkeypatch.setattr(S, "_collect_embed_candidates",
                        lambda kinds=None: [(b"k%d" % i, {"text": "dd"}) for i in range(3)])
    monkeypatch.setattr(SE, "make_embedding_provider",
                        lambda d, b, m, k=None: object())
    asyncio.run(M._embed_models(["MyModel"], driver="d", base_url="b",
                                force=False, yes=True))
    out = capsys.readouterr().out
    assert "MyModel: 3 of 3 entities need vectors (6 chars)." in out
    assert "  MyModel: embedded 3/3" in out
    assert "MyModel: done (3 embedded, 5 tokens)." in out


def test_tracing_gate_log_and_warn():
    class _Conn:
        def __init__(self):
            self.traced, self.warned = [], []

        async def tracing(self, m):
            self.traced.append(m)

        async def warning(self, m):
            self.warned.append(m)

    async def go():
        from Isabelle_RPC_Host.rpc import Connection
        conn = _Conn()
        Connection.set_current(conn)
        prov = object.__new__(SE.OpenAI_Embedding_Provider)
        await SE.Embedding_Provider._log(prov, "hello")
        assert conn.traced == ["hello"]
        tok = SE._embed_tracing_gated.set(True)
        try:
            await SE.Embedding_Provider._log(prov, "quiet")
            await SE.Embedding_Provider._warn(prov, "warned")
        finally:
            SE._embed_tracing_gated.reset(tok)
        assert conn.traced == ["hello"]      # _log gated
        assert conn.warned == ["warned"]     # _warn never gated
        await SE.Embedding_Provider._log(prov, "again")
        assert conn.traced == ["hello", "again"]

    asyncio.run(go())


def test_tracing_gate_embed_records(monkeypatch):
    monkeypatch.setattr(S, "document_text_of", lambda rec: "t")

    class _Conn:
        def __init__(self):
            self.traced = []

        async def tracing(self, m):
            self.traced.append(m)

    async def go():
        store = object.__new__(S.Semantic_Vector_Store)
        conn = _Conn()
        store.connection = conn

        async def emb(kv):
            return 1

        store.embed = emb
        await store.embed_records([(b"k", {})], force=True)
        assert conn.traced == ["[Semantic_Embedding] embedding 1 records (1 chars)"]
        tok = SE._embed_tracing_gated.set(True)
        try:
            await store.embed_records([(b"k", {})], force=True)
        finally:
            SE._embed_tracing_gated.reset(tok)
        assert len(conn.traced) == 1         # gated: no second line

    asyncio.run(go())


def _gate_recording_store(seen):
    store = _mk_store()

    async def embed_records(items, force=False):
        seen.append(SE._embed_tracing_gated.get())
        return 0

    store.embed_records = embed_records
    return store


def test_gate_set_iff_not_verbose(doc_text):
    cands = [(b"k1", {"text": "a"})]
    seen = []
    r = _Recorder()
    asyncio.run(S.complete_vector_store(
        _gate_recording_store(seen), cands, force=True, label="M",
        report=r.report, warn=r.warn))
    asyncio.run(S.complete_vector_store(
        _gate_recording_store(seen), cands, force=True, label="M",
        report=r.report, warn=r.warn, verbose=True))
    assert seen == [True, False]


def test_gate_reset_in_same_task_context(doc_text):
    # The reset assertions MUST run inside the same task context that awaited
    # complete_vector_store: asyncio.run executes the coroutine in a COPY of the
    # caller's context (PEP 567), so asserting after asyncio.run returns reads
    # the default value whether or not the finally-reset exists.
    cands = [(b"k1", {"text": "a"})]
    r = _Recorder()

    async def go():
        await S.complete_vector_store(
            _mk_store(), cands, force=True, label="M",
            report=r.report, warn=r.warn)
        assert SE._embed_tracing_gated.get() is False   # reset after normal return

        await S.complete_vector_store(
            _mk_store(), cands, force=True, label="M",
            report=r.report, warn=r.warn, verbose=True)
        assert SE._embed_tracing_gated.get() is False   # reset after a verbose run

        boom_store = _mk_store()

        async def boom(items, force=False):
            raise RuntimeError("provider down")

        boom_store.embed_records = boom
        with pytest.raises(RuntimeError):
            await S.complete_vector_store(
                boom_store, cands, force=True, label="M",
                report=r.report, warn=r.warn)
        assert SE._embed_tracing_gated.get() is False   # reset on the exception path

    asyncio.run(go())


@pytest.mark.parametrize("verbose,expected_gate", [(False, True), (True, False)])
def test_handler_forwards_verbose_to_gate(doc_text, monkeypatch, verbose, expected_gate):
    # Guards the L6 opt-in end to end: a handler that drops, hard-codes, or
    # inverts the Semantic_Store_verbose lookup fails one of the two cases.
    cands = [(b"k1", {"text": "a"})]
    monkeypatch.setattr(S, "_collect_embed_candidates", lambda kinds=None: cands)
    seen = []
    store = _gate_recording_store(seen)
    store.model_name = "M"
    conn = _HandlerConn(store=store, verbose=verbose)
    asyncio.run(S._embed_all_missing(None, conn))
    assert seen == [expected_gate]
