"""Unit tests for embedding/reranker config resolution.

Covers the cascade ML config option > env > default (_resolve_embedding_config),
the var-major API-key order (the alternate variable only matters while the
primary is unset, matching the hint copy), and make_embedding_provider's
post-construction api_key assignment.

Run:  python test_config_resolution.py    (no framework, plain asserts)
"""
import asyncio
import os
from typing import Any, cast

from Isabelle_Semantic_Embedding import semantic_embedding as SE
from Isabelle_Semantic_Embedding import semantics as SEM


class FakeIsaConn:
    """Duck-typed connection exposing config_lookup like the real one."""
    def __init__(self, config: dict[str, object] | None = None):
        self.config = config or {}

    async def config_lookup(self, name, ctxt=None):
        return self.config.get(name, "")


async def test_config_beats_env_beats_default():
    # EMBEDDING_API_KEY keeps the missing-key guard quiet in the default cases.
    os.environ["EMBEDDING_API_KEY"] = "k"
    os.environ["EMBEDDING_MODEL"] = "env-model"
    try:
        conn = cast(Any, FakeIsaConn(
            {"Semantic_Embedding.embedding_model": "config-model"}))
        _, _, model, _ = await SEM._resolve_embedding_config(conn)
        assert model == "config-model"
        _, _, model2, _ = await SEM._resolve_embedding_config(
            cast(Any, FakeIsaConn()))                            # config empty
        assert model2 == "env-model"
        del os.environ["EMBEDDING_MODEL"]
        _, _, model3, key = await SEM._resolve_embedding_config(
            cast(Any, FakeIsaConn()))
        assert model3 == SEM._DEFAULT_EMBEDDING_MODEL
        assert key == "k"
    finally:
        os.environ.pop("EMBEDDING_MODEL", None)
        del os.environ["EMBEDDING_API_KEY"]


async def test_key_cascade_var_major_order():
    # Gemini: EMBEDDING_API_KEY beats GEMINI_API_KEY, matching the hint copy:
    # the alternate works only while the primary is unset.
    key_vars = SE.Gemini_Embedding.API_KEY_ENV_VARS
    assert key_vars == ("EMBEDDING_API_KEY", "GEMINI_API_KEY")

    for name in key_vars:
        assert name not in os.environ, f"{name} set in this shell; unset it to test"

    conn = cast(Any, FakeIsaConn(
        {"Semantic_Embedding.embedding_driver": "Gemini_Embedding"}))
    os.environ["GEMINI_API_KEY"] = "alt-key"
    try:
        _, _, _, key = await SEM._resolve_embedding_config(conn)
        assert key == "alt-key", "primary unset -> alternate"
        os.environ["EMBEDDING_API_KEY"] = "primary-key"
        _, _, _, key = await SEM._resolve_embedding_config(conn)
        assert key == "primary-key", "primary set -> beats alternate"
    finally:
        os.environ.pop("EMBEDDING_API_KEY", None)
        del os.environ["GEMINI_API_KEY"]


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


async def main():
    await test_config_beats_env_beats_default()
    await test_key_cascade_var_major_order()
    test_make_embedding_provider_post_assignment()
    print("ALL CONFIG RESOLUTION TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
