"""Unit tests for the env-resolution chain behind Connection.getenv.

Covers RPC_GETENV_PLAN.md (MLML) §8.1: Isabelle-side values win, "" means
unset, degradation on both version-skew directions, the var-major API-key
cascade, config-beats-env ordering, post-construction api_key assignment,
and the reranker's bind_connection_env.

Run:  python test_getenv_resolution.py    (no framework, plain asserts)
"""
import asyncio
import os
from typing import Any, cast

from Isabelle_RPC_Host.rpc import Connection, IsabelleError
from Isabelle_Semantic_Embedding import semantic_embedding as SE
from Isabelle_Semantic_Embedding.semantic_embedding import _resolve_env
from Isabelle_Semantic_Embedding import semantics as SEM


# --- fakes -------------------------------------------------------------------

class FakeIsaConn:
    """Duck-typed connection exposing getenv/config_lookup like the real one."""
    def __init__(self, env: dict[str, str], config: dict[str, object] | None = None):
        self.env = env
        self.config = config or {}

    async def getenv(self, name, default=None):
        # mirrors Connection.getenv semantics: "" = unset -> os.environ -> default
        val = self.env.get(name, "")
        if val:
            return val
        return os.environ.get(name, default)

    async def config_lookup(self, name, ctxt=None):
        return self.config.get(name, "")


class PreGetenvConn:
    """A connection object with NO getenv attribute (old Connection class)."""


class _Logger:
    def __init__(self):
        self.warnings = []
    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


class _Server:
    def __init__(self):
        self.logger = _Logger()


def make_real_connection(callback) -> Any:
    """A real Connection object without a socket: only what getenv touches."""
    conn = cast(Any, object.__new__(Connection))
    conn._getenv_unavailable = False
    conn.server = _Server()
    conn.callback = callback           # instance attribute shadows the method
    return conn


# --- tests -------------------------------------------------------------------

async def test_resolve_env_prefers_isabelle():
    conn = FakeIsaConn({"SOME_VAR": "isa-value"})
    os.environ["SOME_VAR"] = "py-value"
    try:
        assert await _resolve_env(conn, "SOME_VAR") == "isa-value"
    finally:
        del os.environ["SOME_VAR"]


async def test_resolve_env_empty_falls_to_process_env():
    conn = FakeIsaConn({})            # Isabelle side: unset ("")
    os.environ["SOME_VAR2"] = "py-value"
    try:
        assert await _resolve_env(conn, "SOME_VAR2") == "py-value"
    finally:
        del os.environ["SOME_VAR2"]
    assert await _resolve_env(conn, "SET_NOWHERE_X1") is None


async def test_resolve_env_no_connection_uses_process_env():
    os.environ["SOME_VAR3"] = "py-only"
    try:
        assert await _resolve_env(None, "SOME_VAR3") == "py-only"
    finally:
        del os.environ["SOME_VAR3"]


async def test_resolve_env_pre_getenv_connection_degrades():
    # Reverse version skew: a still-running server that imported this new code
    # but holds the pre-getenv Connection class must degrade, not AttributeError.
    os.environ["SOME_VAR4"] = "py-fallback"
    try:
        assert await _resolve_env(PreGetenvConn(), "SOME_VAR4") == "py-fallback"
    finally:
        del os.environ["SOME_VAR4"]


async def test_connection_getenv_isabelle_wins_and_empty_falls_back():
    async def cb(name, arg):
        assert name == "getenv"
        return {"FRESH": "from-isabelle"}.get(arg, "")
    conn = make_real_connection(cb)
    os.environ["FRESH"] = "stale-python"
    os.environ["ONLY_PY"] = "py-side"
    try:
        assert await Connection.getenv(conn, "FRESH") == "from-isabelle"
        assert await Connection.getenv(conn, "ONLY_PY") == "py-side"
        assert await Connection.getenv(conn, "NOWHERE_Z", "dflt") == "dflt"
        assert await Connection.getenv(conn, "NOWHERE_Z") is None
    finally:
        del os.environ["FRESH"], os.environ["ONLY_PY"]


async def test_connection_getenv_skew_warns_once_and_memoizes():
    # Forward version skew: old Isabelle heap without Tools/getenv.ML.
    calls = {"n": 0}
    async def cb(name, arg):
        calls["n"] += 1
        raise IsabelleError(["Unknown callback: getenv"], None)
    conn = make_real_connection(cb)
    os.environ["SKEWVAR"] = "py-value"
    try:
        assert await Connection.getenv(conn, "SKEWVAR") == "py-value"
        assert await Connection.getenv(conn, "SKEWVAR") == "py-value"
        assert calls["n"] == 1, "second call must skip the wire"
        assert len(conn.server.logger.warnings) == 1, "warn exactly once"
        assert conn._getenv_unavailable is True
    finally:
        del os.environ["SKEWVAR"]


async def test_key_cascade_var_major_order():
    # Gemini: EMBEDDING_API_KEY (isa > py) beats GEMINI_API_KEY (isa > py),
    # matching the hint copy: the alternate works only while the primary is unset.
    key_vars = SE.Gemini_Embedding.API_KEY_ENV_VARS
    assert key_vars == ("EMBEDDING_API_KEY", "GEMINI_API_KEY")

    async def resolve(conn):
        for var in key_vars:
            v = await _resolve_env(conn, var)
            if v:
                return v
        return None

    for name in ("EMBEDDING_API_KEY", "GEMINI_API_KEY"):
        assert name not in os.environ, f"{name} set in this shell; unset it to test"

    # primary set on the Isabelle side only
    assert await resolve(FakeIsaConn({"EMBEDDING_API_KEY": "isa-primary",
                                      "GEMINI_API_KEY": "isa-alt"})) == "isa-primary"
    # primary unset everywhere -> alternate (isa)
    assert await resolve(FakeIsaConn({"GEMINI_API_KEY": "isa-alt"})) == "isa-alt"
    # primary in the PYTHON env still beats alternate on the Isabelle side
    os.environ["EMBEDDING_API_KEY"] = "py-primary"
    try:
        assert await resolve(FakeIsaConn({"GEMINI_API_KEY": "isa-alt"})) == "py-primary"
    finally:
        del os.environ["EMBEDDING_API_KEY"]


async def test_resolve_one_config_beats_env():
    conn = cast(Any, FakeIsaConn({"EMBEDDING_MODEL": "isa-model"},
                                 config={"Semantic_Embedding.embedding_model": "config-model"}))
    got = await SEM._resolve_one(conn, "Semantic_Embedding.embedding_model",
                                 "EMBEDDING_MODEL", "default-model")
    assert got == "config-model"
    conn2 = cast(Any, FakeIsaConn({"EMBEDDING_MODEL": "isa-model"}))   # config empty
    got2 = await SEM._resolve_one(conn2, "Semantic_Embedding.embedding_model",
                                  "EMBEDDING_MODEL", "default-model")
    assert got2 == "isa-model"
    got3 = await SEM._resolve_one(cast(Any, FakeIsaConn({})),
                                  "Semantic_Embedding.embedding_model",
                                  "EMBEDDING_MODEL", "default-model")
    assert got3 == "default-model"


def test_make_embedding_provider_post_assignment():
    FIREWORKS = "https://api.fireworks.ai/inference/v1"
    QWEN = "Qwen/Qwen3-Embedding-8B"
    p = SE.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS, QWEN,
                                   api_key="explicit-key")
    assert p.api_key == "explicit-key", "explicit api_key must win"
    os.environ["EMBEDDING_API_KEY"] = "env-key"
    try:
        p2 = SE.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS, QWEN)
        assert p2.api_key == "env-key", "None must keep the constructor's env fallback"
    finally:
        del os.environ["EMBEDDING_API_KEY"]
    assert SE.resolve_embedding_driver_class("No_Such_Driver_X9") is None
    try:
        SE.make_embedding_provider("No_Such_Driver_X9", FIREWORKS, QWEN)
        raise AssertionError("unknown driver must raise ImportError")
    except ImportError:
        pass


async def test_reranker_bind_connection_env():
    conn = FakeIsaConn({"QWEN3_RERANKER_API_KEY": "isa-rk",
                        "QWEN3_RERANKER_MODEL": "isa-model"})
    p = await SE.reranker_provider("qwen3-reranker-8b", conn)
    assert p.api_key == "isa-rk"
    assert p.model == "isa-model"
    assert p.base_url == "https://api.fireworks.ai/inference/v1", \
        "unset everywhere -> constructor default survives"
    p2 = await SE.reranker_provider("qwen3-reranker-8b")     # no connection
    assert p2.api_key == os.getenv("QWEN3_RERANKER_API_KEY")


async def main():
    await test_resolve_env_prefers_isabelle()
    await test_resolve_env_empty_falls_to_process_env()
    await test_resolve_env_no_connection_uses_process_env()
    await test_resolve_env_pre_getenv_connection_degrades()
    await test_connection_getenv_isabelle_wins_and_empty_falls_back()
    await test_connection_getenv_skew_warns_once_and_memoizes()
    await test_key_cascade_var_major_order()
    await test_resolve_one_config_beats_env()
    test_make_embedding_provider_post_assignment()
    await test_reranker_bind_connection_env()
    print("ALL GETENV RESOLUTION TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
