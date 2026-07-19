from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import importlib.util
import json
import os
import pathlib
import tempfile
import time
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Awaitable, Callable, ClassVar, NamedTuple, cast
if TYPE_CHECKING:
    from Isabelle_RPC_Host.rpc import Connection
import httpx
import numpy as np
import lmdb
import diskcache
from ._paths import semantic_DB_dir

from ._vecarith import encode_q15, gather_addrs, top_k_q15_gather, Q15_SCALE

_EMBED_CACHE_TTL = 3 * 86400  # 3 days


class EmbedResult(NamedTuple):
    """Result of an embedding call: vectors + total tokens used."""
    vectors: np.ndarray
    total_tokens: int = 0


class Embedding_Provider(ABC):
    type name = str
    canonical_model: str  # identity: HuggingFace name where one exists, else canonical id
    model: str            # model id actually sent to the API
    base_url: str
    api_key: str | None
    dimension: int
    max_request_size: int = 2048
    supports_batch: bool = False
    normalize: bool = False  # L2-normalize returned vectors
    # Fallback scores for entities with no embedding vector (no interpretation),
    # used by Semantic_Vector_Store.lookup in place of the old hardcoded 0.0. The
    # meaningful value depends on the model's score distribution (these sit on the
    # same axis as 1 - L2/2 ≈ cosine), so it is configured per model in the YAML
    # embedding config. Base 0.0 preserves the previous behavior when unset.
    default_score: float = 0.0        # no-embedding, non-local fallback
    default_local_score: float = 0.0  # no-embedding, proof-context-local fallback
    DRIVERS: dict[name, type['Embedding_Provider']] = {}
    _cache: diskcache.Cache | None = None

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        """Configure a provider from the three user parameters + the YAML config.

        ``model`` is the canonical (identity) name; per-model metadata
        (dimension, default scores, normalization, request size) is resolved
        from the embedding config keyed by it.
        """
        from . import embedding_config as cfg
        self.canonical_model = model
        self.base_url = base_url
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("EMBEDDING_API_KEY")
        self.dimension = cfg.dimension(model)
        self.default_score, self.default_local_score = cfg.default_scores(model)
        self.normalize = cfg.normalize(model)
        self.max_request_size = cfg.max_request_size(model)

    @property
    def _cache_key_prefix(self) -> str:
        return self.canonical_model

    @abstractmethod
    async def _embed(self, text: list[str]) -> EmbedResult:
        """Embed a list of texts into vectors.

        Must return an ``EmbedResult`` where ``vectors`` is a 2D float32 matrix
        with shape ``(len(text), dimension)``.
        """
        ...

    async def _embed_batch(self, text: list[str]) -> EmbedResult:
        """Embed a large batch of texts. Defaults to ``_embed``.

        Override to implement chunked or rate-limited batching.
        """
        return await self._embed(text)

    @staticmethod
    def _get_cache() -> diskcache.Cache:
        if Embedding_Provider._cache is None:
            cache_dir = semantic_DB_dir()
            os.makedirs(cache_dir, exist_ok=True)
            Embedding_Provider._cache = diskcache.Cache(
                os.path.join(cache_dir, "embed_cache"),
                size_limit=2 * 1024 * 1024 * 1024)
        return Embedding_Provider._cache

    async def _embed_cached(self, text: list[str],
                      backend: 'Callable[[list[str]], Awaitable[EmbedResult]]') -> EmbedResult:
        """Per-string cache lookup, chunk misses, call backend with retry, cache incrementally."""
        cache = self._get_cache()
        results: list[np.ndarray | None] = [None] * len(text)
        misses: list[tuple[int, str]] = []
        for i, t in enumerate(text):
            cached = cache.get((self._cache_key_prefix, t))
            if cached is not None:
                results[i] = np.frombuffer(cast(bytes, cached), dtype=np.float32)
            else:
                misses.append((i, t))
        hits = len(text) - len(misses)
        if not misses:
            await self._log(f"[Embedding Cache] {self.model}: all {hits} texts cached")
            return EmbedResult(np.stack(results), 0)  # type: ignore
        if hits > 0:
            await self._log(f"[Embedding Cache] {self.model}: {hits} hits, {len(misses)} misses")
        total_tokens = 0
        chunk_size = self.max_request_size
        total_chunks = (len(misses) + chunk_size - 1) // chunk_size
        for ci in range(0, len(misses), chunk_size):
            chunk = misses[ci:ci + chunk_size]
            chunk_texts = [t for _, t in chunk]
            if total_chunks > 1:
                await self._log(f"[Embedding] {self.model}: chunk {ci // chunk_size + 1}/{total_chunks} "
                          f"({len(chunk)} texts)")
            last_err: Exception | None = None
            for attempt in range(10):
                try:
                    embed_result = await backend(chunk_texts)
                    break
                except Exception as e:
                    last_err = e
                    await self._log(f"[Embedding] {self.model}: chunk {ci // chunk_size + 1}/{total_chunks} "
                              f"attempt {attempt + 1}/10 failed: {e}")
                    await asyncio.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Embedding chunk {ci // chunk_size + 1}/{total_chunks} "
                    f"failed after 10 retries") from last_err
            total_tokens += embed_result.total_tokens
            for (i, t), vec in zip(chunk, embed_result.vectors):
                cache.set((self._cache_key_prefix, t), vec.tobytes(), expire=_EMBED_CACHE_TTL)
                results[i] = vec
        return EmbedResult(np.stack(results), total_tokens)  # type: ignore

    def _normalize(self, result: EmbedResult) -> EmbedResult:
        if self.normalize:
            # numpy, not faiss.normalize_L2. This was the only faiss call left in the
            # project -- retrieval is the SIMD LMDB scan, not a faiss index -- and it
            # cost a dependency on faiss + libfaiss for one row-wise divide.
            #
            # Semantics match fvec_renorm_L2 exactly, including the part that is easy to
            # get wrong: faiss scales a row only when its norm is > 0, leaving an
            # all-zero row untouched rather than producing NaN. Hence `where=`.
            # In-place, like faiss: callers rely on result.vectors being mutated.
            v = result.vectors
            n = np.linalg.norm(v, axis=1, keepdims=True)
            np.divide(v, n, out=v, where=n != 0)
        return result

    def _apply_template(self, texts: list[str], role: str,
                        kinds_phrase: str | None = None,
                        task_override: str | None = None) -> list[str]:
        """Wrap each text in the model's query/document template before embedding.

        Uses literal str.replace (NOT str.format) so raw text containing '{' or
        '}' -- routine in Isabelle interpretation text -- is safe. A model with
        no template entry, or the identity template '{text}', leaves the text
        unchanged (fully backward-compatible). For role='query', ``kinds_phrase``
        fills the task_description's {kinds} slot (None -> the default phrase);
        ``task_override``, when given, REPLACES the whole task sentence (used by
        the experience-memory query pass, whose sentence has no {kinds} slot).
        """
        from . import embedding_config as cfg
        model = self.canonical_model
        if role == "document":
            tmpl = cfg.document_template(model)
            return [tmpl.replace("{text}", t) for t in texts]
        if role == "query":
            tmpl = cfg.query_template(model)
            if task_override is not None:
                task = task_override
            else:
                phrase = kinds_phrase if kinds_phrase is not None else cfg._DEFAULT_KINDS_PHRASE
                task = cfg.task_description().replace("{kinds}", phrase)
            # Enforce the config constraint AFTER {kinds} substitution: a
            # hand-edited task_description with a literal {text}/{task} would
            # otherwise be spliced by the .replace below (corrupting the query).
            # Fail fast rather than silently garble. (render_kinds' phrase is
            # brace-free, so it can never introduce {text}/{task}.)
            if "{text}" in task or "{task}" in task:
                raise ValueError(
                    "task_description must not contain '{text}' or '{task}': "
                    f"{task!r}")
            # {task} first, then {text}; the inserted t is never re-scanned, so
            # any braces inside t are inert.
            return [tmpl.replace("{task}", task).replace("{text}", t) for t in texts]
        raise ValueError(f"unknown embed role {role!r}")

    async def _log(self, msg: str) -> None:
        from Isabelle_RPC_Host.rpc import Connection
        conn = Connection.current()
        if conn is not None:
            await conn.tracing(msg)

    async def embed(self, text: list[str], *, role: str = "document",
                    kinds_phrase: str | None = None,
                    task_override: str | None = None) -> EmbedResult:
        """Embed texts, always using cache. ``role`` ('document' for corpus text,
        'query' for a search query) selects the per-model template applied first;
        ``kinds_phrase`` fills a query template's {kinds} slot (ignored for documents);
        ``task_override`` replaces the whole query task sentence (experience pass)."""
        text = self._apply_template(text, role, kinds_phrase, task_override)
        total_chars = sum(len(t) for t in text)
        await self._log(f"[Embedding] {self.model}: embedding {len(text)} texts, {total_chars} chars")
        result = await self._embed_cached(text, self._embed)
        await self._log(f"[Embedding] {self.model}: done, total_tokens={result.total_tokens}")
        return self._normalize(result)

    async def _embed_batch_or_fallback(self, text: list[str]) -> EmbedResult:
        """Route to _embed_batch if supported, otherwise fall back to _embed."""
        if self.supports_batch:
            return await self._embed_batch(text)
        return await self._embed(text)

    async def embed_batch(self, text: list[str], *, role: str = "document",
                          kinds_phrase: str | None = None) -> EmbedResult:
        """Embed a large batch of texts, always using cache. See ``embed`` for ``role``/``kinds_phrase``."""
        text = self._apply_template(text, role, kinds_phrase)
        total_chars = sum(len(t) for t in text)
        await self._log(f"[Embedding Batch] {self.model}: embedding {len(text)} texts, {total_chars} chars")
        result = await self._embed_cached(text, self._embed_batch_or_fallback)
        await self._log(f"[Embedding Batch] {self.model}: done, total_tokens={result.total_tokens}")
        return self._normalize(result)

def sanitize_model(model: str) -> str:
    """Filesystem-safe form of a canonical model name (for LMDB store dirnames)."""
    return model.replace("/", "__")

def unsanitize_model(name: str) -> str:
    """Inverse of sanitize_model: recover the canonical model name from a dirname."""
    return name.replace("__", "/")

def register_embedding_driver(name: str):
    """Class decorator to register an Embedding_Provider driver class by name."""
    def decorator(cls: type[Embedding_Provider]) -> type[Embedding_Provider]:
        Embedding_Provider.DRIVERS[name] = cls
        return cls
    return decorator

def make_embedding_provider(driver: str, base_url: str, model: str) -> Embedding_Provider:
    """Instantiate an embedding driver class with the (base_url, model) parameters.
    Checks the DRIVERS registry first, then dynamically loads drivers/{driver}.py
    (whose module must expose an ``Embedding_Provider`` driver class)."""
    cls = Embedding_Provider.DRIVERS.get(driver)
    if cls is None:
        path = pathlib.Path(__file__).parent / "drivers" / f"{driver}.py"
        if not path.exists():
            raise ImportError(f"Embedding driver {driver!r} not found in DRIVERS or at {path}")
        spec = importlib.util.spec_from_file_location(driver, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load embedding driver {driver!r} from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.Embedding_Provider
    return cls(base_url, model)

@register_embedding_driver("OpenAI_Embedding_Provider")
class OpenAI_Embedding_Provider(Embedding_Provider):
    """Generic provider for any OpenAI-compatible ``/v1/embeddings`` endpoint.

    All model/endpoint specifics come from the YAML embedding config: the API
    model id (per-domain normalization), the dimension/scores/normalize/request
    size (per model), and the optional Batch API shape (per domain). Batch is
    enabled iff the base_url's domain has a ``batch`` entry; ``dialect`` selects
    the request/response shape (``openai`` or ``mistral``).
    """
    max_batch_size: int = 50000

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        from . import embedding_config as cfg
        super().__init__(base_url, model, api_key)
        domain = urlsplit(base_url).netloc
        self.model = cfg.api_model_name(domain, model)  # name sent to the API
        self._batch: dict = cfg.batch_config(domain) or {}
        self.supports_batch = bool(self._batch)
        if self._batch:
            self.max_batch_size = int(self._batch.get("max_batch_size", 50000))

    # --- Batch API hooks (shape selected by the per-domain `dialect`) ---

    @property
    def _batch_endpoint(self) -> str:
        return self._batch["endpoint"]

    @property
    def _batch_completed_status(self) -> str:
        return self._batch["completed"]

    @property
    def _batch_failed_statuses(self) -> set[str]:
        return set(self._batch["failed"])

    @property
    def _batch_output_file_key(self) -> str:
        return self._batch["output_file_key"]

    def _format_batch_line(self, i: int, text: str) -> dict:
        """One JSONL line for the batch input file."""
        if self._batch["dialect"] == "mistral":
            return {"custom_id": str(i),
                    "body": {"input": text, "encoding_format": "float"}}
        return {"custom_id": str(i), "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": self.model, "input": text,
                         "encoding_format": "float"}}

    def _create_batch_request(self, file_id: str) -> dict:
        """JSON body for the create-batch POST."""
        if self._batch["dialect"] == "mistral":
            return {"input_files": [file_id], "model": self.model,
                    "endpoint": "/v1/embeddings", "timeout_hours": 24}
        return {"input_file_id": file_id,
                "endpoint": "/v1/embeddings",
                "completion_window": "24h"}

    def _batch_progress(self, data: dict) -> tuple[int, int]:
        """Extract (completed, total) from poll response."""
        if self._batch["dialect"] == "mistral":
            return data.get("completed_requests", 0), data.get("total_requests", 0)
        counts = data.get("request_counts", {})
        return counts.get("completed", 0), counts.get("total", 0)

    async def _embed(self, text: list[str]) -> EmbedResult:
        url = self.base_url.rstrip("/") + "/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={
                "input": text,
                "model": self.model,
                "encoding_format": "float",
            }, headers=headers, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            vectors = np.asarray(
                [item["embedding"] for item in data["data"]],
                dtype=np.float32
            )
            usage = data.get("usage", {})
            return EmbedResult(vectors, usage.get("total_tokens", 0))

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _submit_one_batch(self, client: httpx.AsyncClient, texts: list[str],
                          url_base: str, auth_headers: dict[str, str]) -> str:
        """Write JSONL, upload, create batch. Returns batch_id."""
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jsonl') as f:
            for i, t in enumerate(texts):
                f.write(json.dumps(self._format_batch_line(i, t)).encode())
                f.write(b'\n')
            f.seek(0)
            resp = await client.post(f"{url_base}/v1/files",
                headers=auth_headers,
                files={"file": f}, data={"purpose": "batch"})
        resp.raise_for_status()
        file_id = resp.json()["id"]
        resp = await client.post(f"{url_base}{self._batch_endpoint}",
            headers={"Content-Type": "application/json", **auth_headers},
            json=self._create_batch_request(file_id))
        resp.raise_for_status()
        return resp.json()["id"]

    async def _poll_and_download(self, client: httpx.AsyncClient, batch_id: str,
                           n: int, url_base: str, auth_headers: dict[str, str],
                           conn: Connection | None) -> tuple[np.ndarray, int]:
        """Poll until batch completes, download results. Returns (vectors, total_tokens)."""
        while True:
            resp = await client.get(f"{url_base}{self._batch_endpoint}/{batch_id}",
                headers=auth_headers)
            resp.raise_for_status()
            batch_data = resp.json()
            status = batch_data["status"]
            if status == self._batch_completed_status:
                output_file_id = batch_data[self._batch_output_file_key]
                if conn is not None:
                    await conn.tracing(
                        f"[Embedding Batch] batch {batch_id} completed, downloading results")
                break
            elif status in self._batch_failed_statuses:
                raise RuntimeError(f"Batch {batch_id} {status}: {batch_data}")
            if conn is not None:
                completed, total = self._batch_progress(batch_data)
                await conn.tracing(
                    f"[Embedding Batch] batch {batch_id}: status={status}, "
                    f"completed={completed}/{total}, "
                    f"waiting...")
            await asyncio.sleep(10)

        resp = await client.get(f"{url_base}/v1/files/{output_file_id}/content",
            headers=auth_headers)
        resp.raise_for_status()
        results: list[list[float] | None] = [None] * n
        total_tokens = 0
        for line in resp.text.strip().split('\n'):
            obj = json.loads(line)
            idx = int(obj["custom_id"])
            body = obj["response"]["body"]
            results[idx] = body["data"][0]["embedding"]
            total_tokens += body.get("usage", {}).get("total_tokens", 0)
        return np.asarray(results, dtype=np.float32), total_tokens

    async def _embed_batch(self, text: list[str]) -> EmbedResult:
        """Embed via Batch API (async, 50% cost). Splits into sub-batches if needed."""
        from Isabelle_RPC_Host.rpc import Connection
        conn = Connection.current()
        url_base = self.base_url.rstrip("/")
        auth_headers = self._auth_headers()
        total_chars = sum(len(t) for t in text)

        chunks = [text[i:i + self.max_batch_size]
                  for i in range(0, len(text), self.max_batch_size)]
        if conn is not None:
            await conn.tracing(
                f"[Embedding Batch] submitting {len(text)} texts ({total_chars} chars) "
                f"to {self.model} via Batch API"
                + (f" in {len(chunks)} sub-batches" if len(chunks) > 1 else ""))

        async with httpx.AsyncClient() as client:
            # Submit all sub-batches
            batch_ids: list[str] = []
            for ci, chunk in enumerate(chunks):
                bid = await self._submit_one_batch(client, chunk, url_base, auth_headers)
                batch_ids.append(bid)
                if conn is not None:
                    await conn.tracing(
                        f"[Embedding Batch] sub-batch {ci+1}/{len(chunks)} submitted: {bid} "
                        f"({len(chunk)} texts)")

            # Poll and download all
            all_results: list[np.ndarray] = []
            grand_total_tokens = 0
            for ci, (bid, chunk) in enumerate(zip(batch_ids, chunks)):
                vectors, tokens = await self._poll_and_download(
                    client, bid, len(chunk), url_base, auth_headers, conn)
                all_results.append(vectors)
                grand_total_tokens += tokens

        if conn is not None:
            await conn.tracing(
                f"[Embedding Batch] all {len(chunks)} sub-batch(es) finished, "
                f"total_tokens={grand_total_tokens}")

        combined = np.concatenate(all_results) if len(all_results) > 1 else all_results[0]
        return EmbedResult(combined, grand_total_tokens)

_GEMINI_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

@register_embedding_driver("Gemini_Embedding")
class Gemini_Embedding(Embedding_Provider):
    """Native Google Gemini embeddings (``batchEmbedContents``), not OpenAI-shaped.

    ``base_url`` overrides the Gemini endpoint base only when it looks like a
    Gemini URL (contains ``generativelanguage``); otherwise the standard
    endpoint is used (the default base_url is the OpenAI-compatible one, which
    does not apply here). api_key falls back to ``GEMINI_API_KEY``.
    """
    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        super().__init__(base_url, model, api_key)
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
        self._endpoint_base = (base_url if base_url and "generativelanguage" in base_url
                               else _GEMINI_DEFAULT_ENDPOINT)

    async def _embed(self, text: list[str]) -> EmbedResult:
        url = f"{self._endpoint_base.rstrip('/')}/models/{self.model}:batchEmbedContents"
        requests = [{"model": f"models/{self.model}",
                     "content": {"parts": [{"text": t}]}}
                    for t in text]
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, params={"key": self.api_key},
                json={"requests": requests}, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            vectors = np.asarray(
                [emb["values"] for emb in data["embeddings"]],
                dtype=np.float32)
            return EmbedResult(vectors, 0)


# --- Reranker Provider ---

class RerankResult(NamedTuple):
    """Reranker output: document indices and their relevance scores, sorted by relevance."""
    indices: list[int]
    scores: list[float]


class Reranker_Provider(ABC):
    type name = str
    _registration_name: ClassVar[str]
    model: str
    max_documents: int = 200
    PROVIDERS: dict[name, type['Reranker_Provider']] = {}

    @abstractmethod
    async def rerank(self, query: str, documents: list[str], top_n: int) -> RerankResult:
        """Rerank documents by relevance to query. Returns top_n results sorted by relevance."""
        ...


def register_reranker_provider(name: str):
    """Class decorator to register a Reranker_Provider subclass by name."""
    def decorator(cls: type[Reranker_Provider]) -> type[Reranker_Provider]:
        cls._registration_name = name
        Reranker_Provider.PROVIDERS[name] = cls
        return cls
    return decorator


def reranker_provider(name: Reranker_Provider.name) -> Reranker_Provider:
    """Instantiate a Reranker_Provider by name.
    Checks PROVIDERS registry first, then dynamically loads from drivers/{name}.py."""
    if name in Reranker_Provider.PROVIDERS:
        return Reranker_Provider.PROVIDERS[name]()
    path = pathlib.Path(__file__).parent / "drivers" / f"{name}.py"
    if not path.exists():
        raise ImportError(f"Reranker driver {name!r} not found in PROVIDERS or at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reranker driver {name!r} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Reranker_Provider()


class OpenAI_Reranker_Provider(Reranker_Provider):
    """Reranker using OpenAI-compatible /v1/rerank endpoint."""
    base_url: str
    api_key: str | None = None

    async def rerank(self, query: str, documents: list[str], top_n: int) -> RerankResult:
        url = self.base_url.rstrip("/") + "/v1/rerank"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }, headers=headers, timeout=600)
            resp.raise_for_status()
            data = resp.json()
        results = sorted(data["results"], key=lambda r: r["relevance_score"], reverse=True)
        return RerankResult(
            [r["index"] for r in results[:top_n]],
            [r["relevance_score"] for r in results[:top_n]],
        )


@register_reranker_provider("qwen3-reranker-8b")
class Qwen3_Reranker_8B(OpenAI_Reranker_Provider):
    base_url = os.getenv("QWEN3_RERANKER_BASE_URL", "https://api.fireworks.ai/inference")
    api_key: str | None = os.getenv("QWEN3_RERANKER_API_KEY")
    model = os.getenv("QWEN3_RERANKER_MODEL", "fireworks/qwen3-reranker-8b")
    max_documents = 200


type key = bytes

import threading
import atexit

_lmdb_envs: dict[str, lmdb.Environment] = {}
_lmdb_lock = threading.Lock()

# Write ceiling for a vector_<model>.lmdb store.  LMDB does not preallocate: the
# value only reserves virtual address space, and the one passed at open() is the
# hard limit on *writes* by that process (past it, MapFullError).  Read-only
# openers adopt the file's real size and ignore it.  Lives here rather than in
# semantics.py because semantics imports this module, not the other way round.
# Every writer of a vector store must open with this one value.
VECTOR_MAP_SIZE: int = 1 << 34      # 16 GiB


def _get_lmdb_env(path: str) -> lmdb.Environment:
    with _lmdb_lock:
        env = _lmdb_envs.get(path)
        if env is None:
            env = lmdb.open(path, map_size=VECTOR_MAP_SIZE)
            _lmdb_envs[path] = env
        return env

def _close_all_lmdb_envs() -> None:
    with _lmdb_lock:
        for env in _lmdb_envs.values():
            env.close()
        _lmdb_envs.clear()

atexit.register(_close_all_lmdb_envs)


def _decode_q15(raw, D: int, k: key) -> np.ndarray:
    """Dequantize one stored Q1.15 value to float32.

    The value must be exactly D*2 bytes. A D*4 one is a leftover float32 record:
    reinterpreting it as int16 would yield a plausible-looking but wrong vector,
    so say what happened instead.
    """
    if len(raw) == D * 4:
        raise ValueError(
            f"vector for key {k.hex()[:16]}… is {D * 4} bytes (float32); this store "
            f"has not been migrated to Q1.15. Run the migration script first.")
    if len(raw) != D * 2:
        raise ValueError(
            f"vector for key {k.hex()[:16]}… is {len(raw)} bytes, expected {D * 2}")
    return np.frombuffer(bytes(raw), dtype="<i2").astype(np.float32) / Q15_SCALE


class Vector_Store(ABC):
    emb_provider : Embedding_Provider

    def __init__(self, path: str, emb_provider: Embedding_Provider,
                 connection: Connection | None = None):
        self.path = path
        self.emb_provider = emb_provider
        self.dimension = self.emb_provider.dimension
        self.connection = connection

    @property
    def _env(self) -> lmdb.Environment:
        return _get_lmdb_env(self.path)

    def __getitem__(self, k: key) -> np.ndarray | None:
        """Return the vector for key k, dequantized to float32, or None.

        What comes back is the Q1.15 round-trip: a unit vector scaled to
        TARGET_NORM, not the provider's raw output. Nothing reads magnitudes back
        out of the store today; a caller that needs the original must re-embed.
        """
        with self._env.begin(buffers=True) as txn:
            raw = txn.get(k)
            if raw is None:
                return None
            return _decode_q15(raw, self.dimension, k)

    def __contains__(self, k: key) -> bool:
        """Check if key k has a stored vector."""
        with self._env.begin() as txn:
            return txn.cursor().set_key(k)

    def __setitem__(self, k: key, vector: np.ndarray) -> None:
        """Store a vector for key k, quantized to Q1.15."""
        with self._env.begin(write=True) as txn:
            txn.put(k, encode_q15(vector).tobytes())

    def put(self, k: key, vector: np.ndarray) -> None:
        """Store a vector for key k. Alias for __setitem__."""
        self[k] = vector

    def delete(self, k: key) -> bool:
        """Delete the stored vector for key k. Returns True if it existed.
        Used e.g. to overwrite an experience memory (see write_memory)."""
        with self._env.begin(write=True) as txn:
            return txn.delete(k)

    def contains(self, keys: list[key]) -> list[bool]:
        """Check existence for a batch of keys in a single transaction."""
        with self._env.begin() as txn:
            cursor = txn.cursor()
            return [cursor.set_key(k) for k in keys]

    async def embed(self, kv_pairs: list[tuple[key, str]]) -> int:
        """Embed texts via emb_provider and store the resulting vectors. Returns total tokens used."""
        texts = [text for _, text in kv_pairs]
        result = await self.emb_provider.embed(texts)
        with self._env.begin(write=True) as txn:
            for (k, _), vec in zip(kv_pairs, result.vectors):
                txn.put(k, encode_q15(vec).tobytes())
        return result.total_tokens

    async def _auto_embed(self, missing: list[key]) -> list[key]:
        """Override to obtain and persist vectors for keys absent from the store.

        Called by topk before it opens its read transaction, so implementations
        are free to await. Whatever they persist is picked up by the subsequent
        gather; they need not hand the vectors back.
        Returns the keys that are now stored. Default: none recovered."""
        return []

    async def topk(self, query: np.ndarray | str, domain: list[key], k: int,
                   *, kinds_phrase: str | None = None) -> list[tuple[key, float]]:
        """Return the top-k (key, cosine) pairs from domain most similar to query.

        If query is a string, it is embedded via emb_provider first (role="query",
        with ``kinds_phrase`` filling the instruction's {kinds} slot).
        Keys missing from LMDB are passed to _auto_embed for recovery.

        Split in two phases. Everything that awaits — embedding the query, and
        recovering missing vectors — happens here, on the event loop. The scan
        itself runs in a worker thread, because py-lmdb read transactions are
        bound to the thread that opened them and must not span an await.
        """
        if isinstance(query, str):
            if self.connection is not None:
                await self.connection.tracing(f"[Semantic_Embedding] embedding query: {query!r}")
            # A string input to topk is a search query -> use the query template
            # (Instruct/query: prefix) with the caller's kinds_phrase. Corpus text
            # is embedded as role="document" elsewhere. lookup() keeps passing the
            # *string* here, so its reranker gate (isinstance(query, str)) is unaffected.
            query = (await self.emb_provider.embed(
                [query], role="query", kinds_phrase=kinds_phrase)).vectors[0]

        query_q15 = encode_q15(query)
        # The scan runs in a worker thread; ctypes releases the GIL for the kernel
        # call, so the event loop keeps running instead of freezing for the scan.
        # Missing keys fall out of the gather itself — probing for them up front
        # would mean a second pass over the whole domain on the event loop.
        results, missing = await asyncio.to_thread(self._topk_sync, query_q15, domain, k)
        if missing and await self._auto_embed(missing):
            results, _ = await asyncio.to_thread(self._topk_sync, query_q15, domain, k)
        return results

    def _topk_sync(self, query_q15: np.ndarray, domain: list[key],
                   k: int) -> tuple[list[tuple[key, float]], list[key]]:
        """Gather the domain's vectors straight out of the LMDB mmap and scan them.

        Nothing is copied: each address points at a value inside the transaction's
        MVCC snapshot, and the SIMD kernel reads it in place. Two consequences —
        the kernel must run before the transaction closes, and a value whose length
        is not exactly D*2 (a pre-migration float32 record, say) would be read as a
        truncated vector, so those are skipped rather than trusted.
        """
        D = self.dimension
        expected = D * 2
        # Sorting turns ~10^5 scattered b-tree lookups into a near-sequential walk
        # of the file; measured 2.9 -> 7.95 GB/s on the production store. Duplicates
        # then sit next to each other, so dropping them costs a comparison rather
        # than the ~48ms a set() of 10^5 keys takes.
        ordered = sorted(domain)
        keys = [dk for i, dk in enumerate(ordered) if i == 0 or dk != ordered[i - 1]]
        with self._env.begin(buffers=True) as txn:
            # The memoryviews must outlive the kernel call: their addresses point
            # into this transaction's snapshot of the mmap, and nothing is copied.
            buffers = [txn.get(dk) for dk in keys]
            addrs, kept, missing_at, skipped = gather_addrs(buffers, expected)
            if skipped:
                print(f"[Semantic_Embedding] topk skipped {skipped} record(s) whose size "
                      f"is not {expected} bytes; the store may not be migrated to Q1.15")
            missing = [keys[int(i)] for i in missing_at]
            if addrs.size == 0:
                return [], missing
            idx, cos = top_k_q15_gather(addrs, query_q15, D, min(k, int(addrs.size)))
            results = [(keys[int(kept[int(i)])], float(c)) for i, c in zip(idx, cos)]
        return results, missing