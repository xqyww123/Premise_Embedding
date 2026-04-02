from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import importlib.util
import json
import os
import pathlib
import tempfile
import time
from typing import TYPE_CHECKING, Awaitable, Callable, NamedTuple, cast
if TYPE_CHECKING:
    from Isabelle_RPC_Host.rpc import Connection
import httpx
import numpy as np
import lmdb
import faiss
import diskcache
import platformdirs

_EMBED_CACHE_TTL = 3 * 86400  # 3 days


class EmbedResult(NamedTuple):
    """Result of an embedding call: vectors + total tokens used."""
    vectors: np.ndarray
    total_tokens: int = 0


class Embedding_Provider(ABC):
    type name = str
    dimension : int
    model : str
    max_request_size: int = 2048
    supports_batch: bool = False
    normalize: bool = False  # L2-normalize returned vectors
    PROVIDERS: dict[name, type['Embedding_Provider']] = {}
    _cache: diskcache.Cache | None = None

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
            cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
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
            cached = cache.get((self.model, t))
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
                cache.set((self.model, t), vec.tobytes(), expire=_EMBED_CACHE_TTL)
                results[i] = vec
        return EmbedResult(np.stack(results), total_tokens)  # type: ignore

    def _normalize(self, result: EmbedResult) -> EmbedResult:
        if self.normalize:
            faiss.normalize_L2(result.vectors)
        return result

    async def _log(self, msg: str) -> None:
        from Isabelle_RPC_Host.rpc import Connection
        conn = Connection.current()
        if conn is not None:
            await conn.tracing(msg)

    async def embed(self, text: list[str]) -> EmbedResult:
        """Embed texts, always using cache."""
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

    async def embed_batch(self, text: list[str]) -> EmbedResult:
        """Embed a large batch of texts, always using cache."""
        total_chars = sum(len(t) for t in text)
        await self._log(f"[Embedding Batch] {self.model}: embedding {len(text)} texts, {total_chars} chars")
        result = await self._embed_cached(text, self._embed_batch_or_fallback)
        await self._log(f"[Embedding Batch] {self.model}: done, total_tokens={result.total_tokens}")
        return self._normalize(result)

def register_embedding_provider(name: str):
    """Class decorator to register an Embedding_Provider subclass by name."""
    def decorator(cls: type[Embedding_Provider]) -> type[Embedding_Provider]:
        Embedding_Provider.PROVIDERS[name] = cls
        return cls
    return decorator

def embedding_provider(name: Embedding_Provider.name) -> Embedding_Provider:
    """Instantiate an Embedding_Provider by name.
    Checks PROVIDERS registry first, then dynamically loads from drivers/{name}.py."""
    if name in Embedding_Provider.PROVIDERS:
        return Embedding_Provider.PROVIDERS[name]()
    path = pathlib.Path(__file__).parent / "drivers" / f"{name}.py"
    if not path.exists():
        raise ImportError(f"Embedding driver {name!r} not found in PROVIDERS or at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load embedding driver {name!r} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Embedding_Provider()

class OpenAI_Embedding_Provider(Embedding_Provider):
    base_url: str = "https://api.openai.com"
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    model: str
    max_batch_size: int = 50000
    supports_batch: bool = True

    # --- Batch API hooks (override for Mistral, etc.) ---

    _batch_endpoint: str = "/v1/batches"
    _batch_completed_status: str = "completed"
    _batch_failed_statuses: set[str] = {"failed", "expired", "cancelled"}
    _batch_output_file_key: str = "output_file_id"

    def _format_batch_line(self, i: int, text: str) -> dict:
        """One JSONL line for the batch input file."""
        return {"custom_id": str(i), "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": self.model, "input": text,
                         "encoding_format": "float"}}

    def _create_batch_request(self, file_id: str) -> dict:
        """JSON body for the create-batch POST."""
        return {"input_file_id": file_id,
                "endpoint": "/v1/embeddings",
                "completion_window": "24h"}

    def _batch_progress(self, data: dict) -> tuple[int, int]:
        """Extract (completed, total) from poll response."""
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
            }, headers=headers, timeout=60)
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

@register_embedding_provider("oai.text-embedding-3-large")
class Text_Embedding_3_Large(OpenAI_Embedding_Provider):
    model = "text-embedding-3-large"
    dimension = 3072

@register_embedding_provider("oai.text-embedding-3-small")
class Text_Embedding_3_Small(OpenAI_Embedding_Provider):
    model = "text-embedding-3-small"
    dimension = 1536

@register_embedding_provider("codestral-embed")
class Codestral_Embed(OpenAI_Embedding_Provider):
    base_url = "https://api.mistral.ai"
    api_key: str | None = os.getenv("MISTRAL_API_KEY")
    model = "codestral-embed"
    dimension = 1536
    max_request_size = 50
    max_batch_size = 1000000
    supports_batch = False

    _batch_endpoint = "/v1/batch/jobs"
    _batch_completed_status = "SUCCESS"
    _batch_failed_statuses = {"FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}
    _batch_output_file_key = "output_file"

    def _format_batch_line(self, i: int, text: str) -> dict:
        return {"custom_id": str(i),
                "body": {"input": text, "encoding_format": "float"}}

    def _create_batch_request(self, file_id: str) -> dict:
        return {"input_files": [file_id], "model": self.model,
                "endpoint": "/v1/embeddings", "timeout_hours": 24}

    def _batch_progress(self, data: dict) -> tuple[int, int]:
        return data.get("completed_requests", 0), data.get("total_requests", 0)

@register_embedding_provider("aliyun.text-embedding-v4")
class Qwen3_Embedding_by_Aliyun(OpenAI_Embedding_Provider):
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode"
    api_key: str | None = os.getenv("ALIYUN_API_KEY")
    model = "text-embedding-v4"
    dimension = 1024
    max_request_size = 10
    supports_batch = False

@register_embedding_provider("fw.qwen3-embedding-8b")
class Qwen3_Embedding_8B_by_Fireworks(OpenAI_Embedding_Provider):
    base_url = "https://api.fireworks.ai/inference"
    api_key: str | None = os.getenv("FIREWORKS_API_KEY")
    model = "fireworks/qwen3-embedding-8b"
    dimension = 4096
    max_request_size = 2048
    supports_batch = False
    normalize = True

type key = bytes

import threading
import atexit

_lmdb_envs: dict[str, lmdb.Environment] = {}
_lmdb_lock = threading.Lock()

def _get_lmdb_env(path: str) -> lmdb.Environment:
    with _lmdb_lock:
        env = _lmdb_envs.get(path)
        if env is None:
            env = lmdb.open(path, map_size=1 << 30)
            _lmdb_envs[path] = env
        return env

def _close_all_lmdb_envs() -> None:
    with _lmdb_lock:
        for env in _lmdb_envs.values():
            env.close()
        _lmdb_envs.clear()

atexit.register(_close_all_lmdb_envs)


class Vector_Store(ABC):
    emb_provider : Embedding_Provider

    def __init__(self, path: str, emb_provider: Embedding_Provider.name | Embedding_Provider,
                 connection: Connection | None = None):
        self.path = path
        if isinstance(emb_provider, str):
            self.emb_provider = embedding_provider(emb_provider)
        else:
            self.emb_provider = emb_provider
        self.dimension = self.emb_provider.dimension
        self.connection = connection

    @property
    def _env(self) -> lmdb.Environment:
        return _get_lmdb_env(self.path)

    def __getitem__(self, k: key) -> np.ndarray | None:
        """Return the vector for key k, or None if not stored."""
        with self._env.begin(buffers=True) as txn:
            raw = txn.get(k)
            if raw is None:
                return None
            return np.array(np.frombuffer(raw, dtype=np.float32))

    def __contains__(self, k: key) -> bool:
        """Check if key k has a stored vector."""
        with self._env.begin() as txn:
            return txn.cursor().set_key(k)

    def __setitem__(self, k: key, vector: np.ndarray) -> None:
        """Store a vector for key k."""
        with self._env.begin(write=True) as txn:
            txn.put(k, vector.astype(np.float32).tobytes())

    def put(self, k: key, vector: np.ndarray) -> None:
        """Store a vector for key k. Alias for __setitem__."""
        self[k] = vector

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
                txn.put(k, vec.astype(np.float32).tobytes())
        return result.total_tokens

    async def _auto_embed(self, missing: list[key], matrix: np.ndarray, row: int) -> list[key]:
        """Override to automatically obtain and store vectors for missing keys during topk.
        Writes recovered vectors directly into matrix starting at the given row.
        Returns the list of recovered keys (in the same order as written to matrix).
        Default: returns [], i.e., missing keys are skipped."""
        return []

    async def topk(self, query: np.ndarray | str, domain: list[key], k: int) -> list[tuple[key, float]]:
        """Return the top-k (key, score) pairs from domain most similar to query.
        If query is a string, it is embedded via emb_provider first.
        Keys missing from LMDB are passed to _auto_embed for recovery.
        Uses faiss.knn directly on the assembled matrix (no index)."""
        if isinstance(query, str):
            if self.connection is not None:
                await self.connection.tracing(f"[Semantic_Embedding] embedding query: {query!r}")
            query = (await self.emb_provider.embed([query])).vectors[0]
        matrix = np.empty((len(domain), self.dimension), dtype=np.float32)
        valid_keys: list[key] = []
        missing_keys: list[key] = []
        i = 0
        with self._env.begin(buffers=True) as txn:
            for dk in domain:
                raw = txn.get(dk)
                if raw is not None:
                    matrix[i] = np.frombuffer(raw, dtype=np.float32)
                    valid_keys.append(dk)
                    i += 1
                else:
                    missing_keys.append(dk)
        if missing_keys:
            recovered = await self._auto_embed(missing_keys, matrix, i)
            valid_keys.extend(recovered)
            i += len(recovered)
        matrix = matrix[:i]
        if i == 0:
            return []
        distances, indices = faiss.knn(query.reshape(1, -1).astype(np.float32), matrix, min(k, i)) # type: ignore
        return [(valid_keys[j], 1.0 - float(distances[0][ri]) / 2.0)
                for ri, j in enumerate(indices[0]) if j >= 0]
