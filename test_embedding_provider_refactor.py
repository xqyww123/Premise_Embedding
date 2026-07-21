"""Offline tests for the parametrized embedding provider + YAML config refactor.

Run: EMBEDDING_CONFIG_PATH defaults here to the bundled template, so no user
config or network is needed.

    python3 test_embedding_provider_refactor.py
    # or: pytest test_embedding_provider_refactor.py
"""
import os

# Point the config loader at the bundled template before importing the module.
os.environ.setdefault(
    "EMBEDDING_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__),
                 "Isabelle_Semantic_Embedding", "embedding_config_template.yaml"))
os.environ["EMBEDDING_API_KEY"] = "test-key-123"

from Isabelle_Semantic_Embedding import semantic_embedding as se  # noqa: E402

FIREWORKS = "https://api.fireworks.ai/inference/v1"


def test_fireworks_qwen():
    p = se.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS,
                                   "Qwen/Qwen3-Embedding-8B")
    assert p.canonical_model == "Qwen/Qwen3-Embedding-8B"
    assert p.model == "fireworks/qwen3-embedding-8b"   # domain normalization
    assert p.dimension == 4096
    assert (p.default_score, p.default_local_score) == (0.3, 0.5)
    assert p.normalize is True
    assert p.supports_batch is False                   # fireworks: no batch entry
    assert p.api_key == "test-key-123"                 # EMBEDDING_API_KEY only


def test_openai_batch():
    p = se.OpenAI_Embedding_Provider("https://api.openai.com/v1", "text-embedding-3-large")
    assert p.model == "text-embedding-3-large"         # no normalization needed
    assert p.dimension == 3072
    assert (p.default_score, p.default_local_score) == (0.0, 0.0)
    assert p.normalize is False
    assert p.supports_batch is True
    assert p.max_batch_size == 50000
    assert p._batch_endpoint == "/v1/batches"
    line = p._format_batch_line(2, "hi")
    assert line["method"] == "POST" and line["body"]["model"] == "text-embedding-3-large"


def test_mistral_batch_dialect():
    p = se.OpenAI_Embedding_Provider("https://api.mistral.ai/v1", "codestral-embed")
    assert p.dimension == 1536
    assert p.supports_batch is True
    assert p.max_batch_size == 1000000
    assert p._batch_endpoint == "/v1/batch/jobs"
    assert p._batch_completed_status == "SUCCESS"
    line = p._format_batch_line(0, "x")
    assert "method" not in line and line["body"]["input"] == "x"   # mistral shape
    req = p._create_batch_request("fid")
    assert req["input_files"] == ["fid"]


def test_aliyun_request_cap():
    p = se.make_embedding_provider(
        "OpenAI_Embedding_Provider",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "text-embedding-v4")
    assert p.dimension == 1024
    assert p.max_request_size == 10
    assert p.supports_batch is False


def test_gemini_native_driver():
    p = se.Gemini_Embedding(FIREWORKS, "gemini-embedding-2-preview")
    assert p.dimension == 3072
    # base_url is the (non-gemini) fireworks default -> falls back to gemini endpoint
    assert "generativelanguage" in p._endpoint_base


def test_sanitize_roundtrip():
    for m in ["Qwen/Qwen3-Embedding-8B", "text-embedding-3-large", "a/b/c"]:
        assert se.unsanitize_model(se.sanitize_model(m)) == m


def test_missing_dimension_errors():
    import pytest
    with pytest.raises(KeyError):
        se.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS,
                                   "no-such-model-xyz")


# --- Per-model query/document template refactor -----------------------------

QWEN = "Qwen/Qwen3-Embedding-8B"
NVEMBED = "llama-nv-embed-reasoning-3b"


def _task():
    # task_description rendered with NO kinds filter (the default phrase) -- the
    # value embed(role="query") produces when no kinds_phrase is passed.
    from Isabelle_Semantic_Embedding import embedding_config as cfg
    return cfg.task_description().replace("{kinds}", cfg._DEFAULT_KINDS_PHRASE)


def _qwen():
    return se.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS, QWEN)


def _nvembed():
    return se.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS, NVEMBED)


def test_template_accessors():
    from Isabelle_Semantic_Embedding import embedding_config as cfg
    assert cfg.query_template(QWEN) == "Instruct: {task}\nQuery: {text}"
    assert cfg.document_template(QWEN) == "{text}"
    assert cfg.query_template(NVEMBED) == "query: {text}"
    assert cfg.document_template(NVEMBED) == "passage: {text}"
    # unlisted model -> identity templates (raw, fully backward-compatible)
    assert cfg.query_template("text-embedding-3-large") == "{text}"
    assert cfg.document_template("text-embedding-3-large") == "{text}"
    # task_description carries the dynamic {kinds} slot (filled per query) but
    # never the literal {text}/{task}
    td = cfg.task_description()
    assert "{kinds}" in td
    assert "{text}" not in td and "{task}" not in td
    assert "Isabelle/HOL" in td


def test_apply_template_query_and_document():
    p = _qwen()
    task = _task()
    assert p._apply_template(["find me a lemma"], "query") == \
        ["Instruct: " + task + "\nQuery: find me a lemma"]
    # Qwen3 document template is identity -> existing corpus vectors stay valid
    assert p._apply_template(["a theorem about lists"], "document") == \
        ["a theorem about lists"]


def test_apply_template_nvembed():
    p = _nvembed()
    # nv-embed: query and document get DISTINCT prefixes (docs cannot be raw)
    assert p._apply_template(["q"], "query") == ["query: q"]
    assert p._apply_template(["d"], "document") == ["passage: d"]


def test_apply_template_brace_safety():
    # Set-builder / record braces must pass through verbatim (no str.format).
    p = _qwen()
    task = _task()
    text = "{x. x > 0} and (| a = 1 |)"
    assert p._apply_template([text], "query") == \
        ["Instruct: " + task + "\nQuery: " + text]
    assert p._apply_template([text], "document") == [text]


def test_apply_template_unlisted_model_is_raw():
    p = se.make_embedding_provider("OpenAI_Embedding_Provider",
                                   "https://api.openai.com/v1", "text-embedding-3-large")
    assert p._apply_template(["raw {q}"], "query") == ["raw {q}"]
    assert p._apply_template(["raw {q}"], "document") == ["raw {q}"]


def test_apply_template_bad_role():
    p = _qwen()
    raised = False
    try:
        p._apply_template(["x"], "neither")
    except ValueError:
        raised = True
    assert raised


def test_task_description_guard():
    # A hand-edited task_description with literal {text}/{task} must fail fast,
    # not silently splice the query into the instruction sentence.
    from Isabelle_Semantic_Embedding import embedding_config as cfg
    p = _qwen()
    orig = cfg.task_description
    cfg.task_description = lambda: "leak {text} into instruction"
    try:
        raised = False
        try:
            p._apply_template(["q"], "query")
        except ValueError:
            raised = True
        assert raised, "expected ValueError for {text} in task_description"
        # document role does not use task_description -> not guarded
        assert p._apply_template(["d"], "document") == ["d"]
    finally:
        cfg.task_description = orig


def test_embed_role_templates_before_cache():
    # embed(role=...) must apply the template BEFORE the per-string cache/backend,
    # so the cache key and HTTP body see the *templated* text.
    import asyncio
    import numpy as np
    p = _qwen()
    task = _task()
    captured = {}

    async def fake_cached(text, backend):
        captured["text"] = list(text)
        return se.EmbedResult(np.zeros((len(text), p.dimension), dtype=np.float32), 0)

    p._embed_cached = fake_cached  # bypass the real diskcache + network
    asyncio.run(p.embed(["hello {set}"], role="query"))
    assert captured["text"] == ["Instruct: " + task + "\nQuery: hello {set}"]
    asyncio.run(p.embed(["hello {set}"], role="document"))
    assert captured["text"] == ["hello {set}"]   # Qwen3 document = identity
    # default role is "document" (all existing corpus callers keep behaving raw)
    asyncio.run(p.embed(["plain"]))
    assert captured["text"] == ["plain"]


def test_render_kinds():
    from Isabelle_RPC_Host.universal_key import EntityKind as EK
    from Isabelle_Semantic_Embedding.semantics import render_kinds
    from Isabelle_Semantic_Embedding import embedding_config as cfg
    assert render_kinds([EK.CONSTANT]) == "constants"
    assert render_kinds([EK.CLASS]) == "type classes"
    assert render_kinds([EK.THEOREM_COLLECTION]) == "theorem collections"
    assert render_kinds([EK.METHOD]) == "proof methods"
    # Oxford-style join for 2 and 3 kinds
    assert render_kinds([EK.CONSTANT, EK.THEOREM]) == "constants and theorems"
    assert render_kinds([EK.CONSTANT, EK.THEOREM, EK.TYPE]) == "constants, theorems and types"
    # all four rule kinds collapse to a single phrase; THEOREM is NOT a rule
    assert render_kinds([EK.INTRODUCTION_RULE, EK.ELIMINATION_RULE,
                         EK.INDUCTION_RULE, EK.CASE_SPLIT_RULE]) == "inference rules"
    assert render_kinds([EK.THEOREM]) == "theorems"
    assert render_kinds([EK.CONSTANT, EK.CONSTANT]) == "constants"      # dedup
    # empty / unknown -> default phrase, never KeyError
    assert render_kinds([]) == cfg._DEFAULT_KINDS_PHRASE
    assert render_kinds([EK.THEORY]) == cfg._DEFAULT_KINDS_PHRASE


def test_apply_template_query_with_kinds():
    from Isabelle_Semantic_Embedding import embedding_config as cfg
    p = _qwen()
    rendered = cfg.task_description().replace("{kinds}", "constants and theorems")
    assert p._apply_template(["find X"], "query", "constants and theorems") == \
        ["Instruct: " + rendered + "\nQuery: find X"]
    # document role ignores kinds_phrase entirely
    assert p._apply_template(["d"], "document", "constants and theorems") == ["d"]


# --- base_url /v1 convention (2026-07) ---------------------------------------


def test_embed_posts_to_embeddings_suffix():
    # base_url includes the version segment; _embed appends only /embeddings.
    import asyncio
    p = _qwen()
    captured = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"data": [{"embedding": [0.0] * p.dimension}], "usage": {}}

    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            captured["url"] = url
            return _Resp()

    orig = se.httpx.AsyncClient
    se.httpx.AsyncClient = _Client
    try:
        asyncio.run(p._embed(["x"]))
    finally:
        se.httpx.AsyncClient = orig
    assert captured["url"] == "https://api.fireworks.ai/inference/v1/embeddings"


def test_batch_urls_join_origin():
    # Batch traffic joins onto the URL origin, so /v1 never doubles even though
    # base_url now carries it; the JSONL body `url` stays a protocol constant.
    import asyncio
    import json as _json
    p = se.OpenAI_Embedding_Provider("https://api.openai.com/v1", "text-embedding-3-large")
    urls = []

    class _Resp:
        def __init__(self, data=None, text=""):
            self._data, self.text = data, text
        def raise_for_status(self): pass
        def json(self): return self._data

    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            urls.append(url)
            if url.endswith("/files"):
                return _Resp({"id": "f1"})
            return _Resp({"id": "b1"})
        async def get(self, url, **kw):
            urls.append(url)
            if "/files/" in url:
                line = _json.dumps({"custom_id": "0", "response": {"body": {
                    "data": [{"embedding": [0.0] * p.dimension}], "usage": {}}}})
                return _Resp(text=line)
            return _Resp({"status": "completed", "output_file_id": "of1",
                          "request_counts": {}})

    orig = se.httpx.AsyncClient
    se.httpx.AsyncClient = _Client
    try:
        asyncio.run(p._embed_batch(["x"]))
    finally:
        se.httpx.AsyncClient = orig
    assert urls == [
        "https://api.openai.com/v1/files",
        "https://api.openai.com/v1/batches",
        "https://api.openai.com/v1/batches/b1",
        "https://api.openai.com/v1/files/of1/content",
    ]
    assert p._format_batch_line(0, "t")["url"] == "/v1/embeddings"
    assert p._create_batch_request("f")["endpoint"] == "/v1/embeddings"


def test_legacy_base_url_rejected():
    # Version-less base_url for a providers-listed domain = pre-change value.
    for legacy, model in [("https://api.fireworks.ai/inference", QWEN),
                          ("https://api.openai.com", "text-embedding-3-large"),
                          ("https://api.mistral.ai", "codestral-embed")]:
        try:
            se.OpenAI_Embedding_Provider(legacy, model)
            raise AssertionError(f"expected ValueError for {legacy}")
        except ValueError as e:
            assert "/v1" in str(e)


def test_selfhosted_versionless_root_allowed():
    # Unlisted domains (self-hosted server roots) stay exempt from the check.
    p = se.OpenAI_Embedding_Provider("http://localhost:8000", "text-embedding-3-large")
    assert p.supports_batch is False


def test_domain_normalization_case_and_port():
    # Netloc-spelling variants must hit the same YAML providers entry: uppercase
    # host and a scheme-default port get model normalization AND the legacy check.
    p = se.OpenAI_Embedding_Provider("https://API.FIREWORKS.AI:443/inference/v1", QWEN)
    assert p.model == "fireworks/qwen3-embedding-8b"
    try:
        se.OpenAI_Embedding_Provider("https://API.FIREWORKS.AI/inference", QWEN)
        raise AssertionError("expected ValueError for uppercase legacy spelling")
    except ValueError:
        pass
    # Explicit non-default ports survive, so a `myhost:8080` providers key works.
    assert se._endpoint_domain("http://myhost:8080/v1") == "myhost:8080"
    assert se._endpoint_domain("https://api.openai.com/v1") == "api.openai.com"


def test_404_hint_suggests_version_segment():
    class _R:
        status_code = 404
    e = Exception()
    e.response = _R()   # type: ignore[attr-defined]
    p = se.OpenAI_Embedding_Provider("http://localhost:8000", "text-embedding-3-large")
    assert "version segment" in p._http_error_hint(e)
    p2 = se.OpenAI_Embedding_Provider("http://localhost:8000/v1", "text-embedding-3-large")
    assert "version segment" not in p2._http_error_hint(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if fn.__name__ == "test_missing_dimension_errors":
            try:
                fn()
            except ImportError:
                # pytest not importable in this run; check KeyError directly
                try:
                    se.make_embedding_provider("OpenAI_Embedding_Provider", FIREWORKS,
                                               "no-such-model-xyz")
                    raise AssertionError("expected KeyError")
                except KeyError:
                    pass
        else:
            fn()
        print(f"ok: {fn.__name__}")
    print("ALL PASSED")
