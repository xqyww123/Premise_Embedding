"""Semantic query tools for looking up interpretations from parent theories."""

import os
import threading
from typing import Any, NamedTuple

import lmdb
import msgpack
import numpy as np
import platformdirs
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from Isabelle_RPC_Host.rpc import IsabelleError
from Isabelle_RPC_Host.position import AsciiPosition, UnicodePosition, IsabellePosition
from Isabelle_RPC_Host.universal_key import EntityKind, UndefinedEntity, universal_key, universal_key_of, destruct_key, is_WIP
from claude_agent_sdk import SdkMcpTool, tool

from .semantic_embedding import Vector_Store, Embedding_Provider, embedding_provider, key

from .base import ToolCall_ret, mk_ret as _mk_ret
from .hover import _position_schema, _validate_position_args, _resolve_thy_path

_PIDE_KIND_TO_TAG: dict[str, EntityKind] = {
    "const": EntityKind.CONSTANT, "type": EntityKind.TYPE, "thm": EntityKind.THEOREM,
    "class": EntityKind.CLASS, "locale": EntityKind.LOCALE,
}

# Long theory names to exclude from interpretation and entity enumeration.
_SKIP_THEORY_LONG_NAMES = ["Pure", "Tools.Code_Generator", "HOL.Code_Evaluation", "HOL.Typerep"]

# Base names (after the last dot) for matching theories under different session qualifiers.
_SKIP_THEORY_BASES = {n.rsplit(".", 1)[-1] for n in _SKIP_THEORY_LONG_NAMES}

def is_thy_skipped(name: str) -> bool:
    """Check whether a theory should be skipped from interpretation."""
    base = name.rsplit(".", 1)[-1] if "." in name else name
    return base in _SKIP_THEORY_BASES


class _Semantic_DB:
    _env: lmdb.Environment | None = None
    _lock = threading.Lock()

    class Record(NamedTuple):
        kind: EntityKind
        name: str
        expr: str
        interpretation: str

        @property
        def pretty_print(self) -> str:
            pp = f"{self.kind.label} {self.name}"
            if self.expr:
                pp += f": {self.expr}"
            return pp

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            with self._lock:
                if self._env is None:
                    import atexit
                    cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
                    os.makedirs(cache_dir, exist_ok=True)
                    _Semantic_DB._env = lmdb.open(os.path.join(cache_dir, "semantics.lmdb"), map_size=1 << 30)
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

    def __getitem__(self, key: universal_key) -> 'Record | None':
        with self._ensure_env().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return None
        kind, name, expr, sem = msgpack.unpackb(raw)
        return _Semantic_DB.Record(EntityKind(kind), self._dec(name), self._dec(expr), self._dec(sem))

    def __setitem__(self, key: universal_key, record: 'Record') -> None:
        with self._ensure_env().begin(write=True) as txn:
            txn.put(key, msgpack.packb(tuple(record))) # type: ignore

    def query(self, key: universal_key, with_pretty: bool = False) -> str | None:
        """Look up a semantic interpretation by universal key."""
        rec = self[key]
        if rec is None:
            return None
        if with_pretty:
            return rec.pretty_print + "\n" + rec.interpretation
        return rec.interpretation

    def is_thy_interpreted(self, key: universal_key) -> bool:
        """Check whether a theory has been fully interpreted."""
        with self._ensure_env().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            return False
        return msgpack.unpackb(raw).get(b"finished", False)

    def mark_interpreted(self, key: universal_key) -> None:
        """Mark a theory as interpreted (finished) in the semantic store.
        Skips WIP (non-persistent) theories."""
        if is_WIP(key):
            return
        with self._ensure_env().begin(write=True) as txn:
            raw = txn.get(key)
            if raw is not None:
                data = msgpack.unpackb(raw)
                data[b"finished"] = True
                txn.put(key, msgpack.packb(data))  # type: ignore
            else:
                txn.put(key, msgpack.packb({
                    "input_tokens": 0, "output_tokens": 0,
                    "finished": True,
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

_NAME_DESCRIPTION_BASE = "The short or full name of the entity to look up."
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
                        "introduction rule", "elimination rule"],
                "description": "The kind of entity to query.",
            },
            "name": {
                "type": "string",
                "description": _NAME_DESCRIPTION_INTERP if working_names else _NAME_DESCRIPTION_BASE,
            },
        },
        "required": ["type", "name"],
    }


def mk_query_by_name_tool(
    connection: Connection, working_names: list[str], with_pretty: bool = True
) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("semantics")
    description = "Look up the English translation of a dependency from parent theories."
    if working_names:
        description += (
            " Do not query entries you have been asked to interpret"
            " — interpret those from the source file yourself.")
    @tool(
        "query_by_name",
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
        try:
            if working_names and name in working_names:
                log.debug("Entity name %r is in working_names; cannot query entities assigned for interpretation.", name)
                return _mk_ret(
                    f"Cannot query \"{name}\" — it is or will be your task to interpret it from the source.",
                    is_error=True,
                )
            uk = universal_key_of(connection, tag, name)
            sem = Semantic_DB.query(uk, with_pretty=with_pretty)
            if sem is None:
                return _mk_ret(f"{t} \"{name}\" has not been interpreted yet.")
            return _mk_ret(sem)
        except UndefinedEntity as e:
            if "." in name:
                short = name.rsplit(".", 1)[1]
                try:
                    uk = universal_key_of(connection, tag, short)
                    sem = Semantic_DB.query(uk, with_pretty=with_pretty)
                    if sem is not None:
                        return _mk_ret(f"The {name} is undefined, but we find:\n{sem}")
                except (IsabelleError, UndefinedEntity):
                    pass
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(str(e), is_error=True)
        except IsabelleError as e:
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(str(e), is_error=True)
        except Exception:
            log.exception("query_by_name: error")
            raise
    return query_by_name_tool


def mk_query_by_position_tool(
    connection: Connection, working_names: list[str],
    unicode: bool = False, with_pretty: bool = True
) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("semantics")
    description = "Look up the semantic interpretation of the entity at a given source position."
    if working_names:
        description += (
            " Do not query entries you have been asked to interpret"
            " — interpret those from the source file yourself.")
    @tool(
        "query_by_position",
        description,
        input_schema=_position_schema,
    )
    async def query_by_position_tool(args: dict[str, Any]) -> ToolCall_ret:
        try:
            log.debug("query_by_position: %s:%s:%s",
                       args.get("file"), args.get("line"), args.get("column"))
            err = _validate_position_args(args)
            if err is not None:
                log.warning("query_by_position: validation error: %s", err)
                return _mk_ret(err, is_error=True)
            thy_path = _resolve_thy_path(args["file"])
            if unicode:
                isa_pos = UnicodePosition(args["line"], args["column"], thy_path).to_isabelle_position()
            else:
                isa_pos = AsciiPosition(args["line"], args["column"], thy_path).to_isabelle_position()
            entity = connection.callback(
                "pide_state.entity_at_position", (isa_pos.file, isa_pos.raw_offset))
            if entity is None:
                log.debug("query_by_position: no entity found")
                return _mk_ret("No entity found at this position.")
            kind, name = entity
            log.debug("query_by_position: found %s %r", kind, name)
            tag = _PIDE_KIND_TO_TAG.get(kind)
            if tag is None:
                return _mk_ret(f"Entity kind \"{kind}\" ({name}) is not queryable.")
            if working_names and name in working_names:
                return _mk_ret(
                    f"Cannot query \"{name}\" — it is your task to interpret it from the source.",
                    is_error=True,
                )
            uk = universal_key_of(connection, tag, name)
            sem = Semantic_DB.query(uk, with_pretty=with_pretty)
            if sem is None:
                return _mk_ret(f"{tag.label} \"{name}\" has not been interpreted yet.")
            return _mk_ret(sem)
        except (IsabelleError, UndefinedEntity) as e:
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(str(e), is_error=True)
        except Exception:
            log.exception("query_by_position: error")
            raise
    return query_by_position_tool


# --- Other utilities ---

def interpret_theories_by_names(connection: Connection, names: list[str]) -> None:
    """Interpret theories by name (short or long).
    Resolves names, skips already-interpreted theories, and interprets the rest.
    Calls back into Isabelle ML via the Semantic_Store.interpret_theories callback."""
    connection.callback("Semantic_Store.interpret_theories", names)


class Semantic_Vector_Store(Vector_Store):
    @staticmethod
    def clean_all_wip_in_created_dbs() -> None:
        """Remove all WIP (non-persistent) entries from every vector store on disk."""
        from Isabelle_RPC_Host.theory_hash import is_persistent
        cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
        if not os.path.isdir(cache_dir):
            return
        prefix = "vector_"
        suffix = ".lmdb"
        for entry in os.listdir(cache_dir):
            if entry.startswith(prefix) and entry.endswith(suffix):
                path = os.path.join(cache_dir, entry)
                if os.path.isdir(path):
                    from .semantic_embedding import _get_lmdb_env
                    env = _get_lmdb_env(path)
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
        cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
        if not os.path.isdir(cache_dir):
            return []
        prefix = "vector_"
        suffix = ".lmdb"
        return [entry[len(prefix):-len(suffix)]
                for entry in os.listdir(cache_dir)
                if entry.startswith(prefix) and entry.endswith(suffix)
                and os.path.isdir(os.path.join(cache_dir, entry))]

    def __init__(
        self,
        emb_provider: Embedding_Provider.name | Embedding_Provider | None = None,
        connection: Connection | None = None,
    ):
        if emb_provider is None:
            emb_provider = _resolve_embedding_model(connection, None)
        if isinstance(emb_provider, str):
            model_name = emb_provider
        else:
            model_name = getattr(emb_provider, 'model', 'custom')
        cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"vector_{model_name}.lmdb")
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

    def is_thy_embedded(self, theory_key: universal_key) -> bool:
        """Check whether a theory's entities are all embedded in this vector store."""
        with self._env.begin() as txn:
            raw = txn.get(theory_key)
        if raw is None:
            return False
        return msgpack.unpackb(raw).get(b"finished", False)

    def thy_embed_tokens(self, theory_key: universal_key) -> int | None:
        """Look up the total tokens used to embed a theory. Returns None if not found."""
        with self._env.begin() as txn:
            raw = txn.get(theory_key)
        if raw is None:
            return None
        return msgpack.unpackb(raw).get(b"total_tokens", 0)

    def mark_thy_embedded(self, theory_key: universal_key, total_tokens: int = 0) -> None:
        """Mark a theory as fully embedded in this vector store, recording token usage.
        Skips WIP (non-persistent) theories."""
        if is_WIP(theory_key):
            return
        with self._env.begin(write=True) as txn:
            raw = txn.get(theory_key)
            if raw is not None:
                data = msgpack.unpackb(raw)
            else:
                data = {}
            data[b"finished"] = True
            if total_tokens > 0:
                data[b"total_tokens"] = data.get(b"total_tokens", 0) + total_tokens
            txn.put(theory_key, msgpack.packb(data))  # type: ignore

    def _auto_embed(self, missing: list[key], matrix: np.ndarray, row: int) -> list[key]:
        if self.connection is None:
            return []
        if not self.connection.config_lookup("auto_interpret_for_embedding"):
            self.connection.warning(
                f"[Semantic_Embedding] {len(missing)} entities missing semantic embeddings, "
                f"but auto_interpret_for_embedding is disabled. "
                f"Set [[auto_interpret_for_embedding = true]] to enable automatic interpretation and embedding.")
            return []
        # Extract theory hashes from missing keys, skipping already-embedded theories
        theory_hashes: set[bytes] = set()
        for k in missing:
            entity = destruct_key(k)
            if not self.is_thy_embedded(entity.theory):
                theory_hashes.add(entity.theory)
        self.connection.tracing(
            f"[Semantic_Embedding] {len(missing)} entities missing embeddings, "
            f"spanning {len(theory_hashes)} un-embedded theories")
        # # DEBUG: show missing entities (requires debug_key_name from context.py)
        # from Isabelle_RPC_Host.context import debug_key_name
        # for k in missing[:50]:
        #     readable = debug_key_name(k) or f"<unknown {k.hex()[:16]}…>"
        #     self.connection.tracing(f"  MISSING: {readable}")
        # if len(missing) > 50:
        #     self.connection.tracing(f"  ... and {len(missing) - 50} more")
        # Filter to uninterpreted theories, excluding the current theory and skipped theories
        from Isabelle_RPC_Host.context import theory_long_name
        current_thy = theory_long_name(self.connection)
        uninterpreted_theories: list[str] = []
        for th in theory_hashes:
            if not Semantic_DB.is_thy_interpreted(th):
                name = self.connection.callback("Theory_Hash.theory_name_of", th)
                if name is not None and name != current_thy and not is_thy_skipped(name):
                    uninterpreted_theories.append(name)
        confirmed = False
        if uninterpreted_theories:
            if len(uninterpreted_theories) > 5:
                import Isabelle_RPC_Host.dialogue
                answer = self.connection.dialogue(
                    f"[Semantic Embedding] {len(uninterpreted_theories)} uninterpreted theories "
                    f"need interpretation before embedding. "
                    f"This may consume a significant amount of API tokens. Proceed?",
                    ["Yes", "No"])
                if answer != "Yes":
                    return []
                confirmed = True
            self.connection.tracing(
                f"[Semantic_Embedding] {len(uninterpreted_theories)} of {len(theory_hashes)} theories "
                f"not yet interpreted, running interpretation for: "
                + ", ".join(uninterpreted_theories))
            interpret_theories_by_names(self.connection, uninterpreted_theories)
        # Query semantic store for interpretations and embed them
        texts: list[str] = []
        text_keys: list[key] = []
        for k in missing:
            sem = Semantic_DB.query(k, with_pretty=True)
            if sem is not None:
                texts.append(sem)
                text_keys.append(k)
        if not texts:
            self.connection.tracing(
                f"[Semantic_Embedding] no semantic interpretations found for the missing entities, skipping")
            return []
        if len(texts) > 42 and not confirmed:
            import Isabelle_RPC_Host.dialogue
            answer = self.connection.dialogue(
                f"[Semantic Embedding] {len(texts)} entities to embed. "
                f"This may consume a significant amount of API tokens. Proceed?",
                ["Yes", "No"])
            if answer != "Yes":
                return []
        total_chars = sum(len(t) for t in texts)
        self.connection.tracing(
            f"[Semantic_Embedding] embedding {len(texts)} of {len(missing)} missing entities "
            f"({total_chars} chars total) into vectors")
        embed_result = self.emb_provider.embed(texts)
        # Write vectors into matrix and store in LMDB
        with self._env.begin(write=True) as txn:
            for j, (k, vec) in enumerate(zip(text_keys, embed_result.vectors)):
                v = vec.astype(np.float32)
                matrix[row + j] = v
                txn.put(k, v.tobytes())
        # Mark processed theories as embedded, recording cost
        for th in theory_hashes:
            self.mark_thy_embedded(th, embed_result.total_tokens)
        return text_keys

    def lookup(
        self,
        query: np.ndarray | str,
        k: int,
        kinds: list[EntityKind],
        domain: list[universal_key] | None = None,
    ) -> list[tuple[float, 'SemanticRecord']]:
        """Search the k closest entities to query, filtered by kinds and domain.
        Returns (score, record) pairs sorted by similarity.
        If domain is None, uses all entities of the given kinds from the context.
        If both kinds and domain are provided, takes their intersection."""
        if not kinds:
            return []
        if domain is None:
            if self.connection is None:
                return []
            from Isabelle_RPC_Host.context import entities_of
            candidates = entities_of(self.connection, kinds,
                                     theories_not_include=_SKIP_THEORY_LONG_NAMES)
        else:
            kind_set = set(kinds)
            candidates = [dk for dk in domain if destruct_key(dk).kind in kind_set]
        if not candidates:
            return []
        top = self.topk(query, candidates, k)
        return [(score, rec) for uk, score in top if (rec := Semantic_DB[uk]) is not None]

    def embed_theories(self, theories: list[str | universal_key]) -> None:
        """Embed semantic interpretations into vectors for the given theories.

        Collects all missing entities across all theories and embeds them
        in a single batch call.

        Args:
            theories: Long theory names (str) or universal keys to embed.

        Raises:
            RuntimeError: If no active connection is available.
            ValueError: If a universal key is not found in the active Isabelle runtime.
        """
        if self.connection is None:
            raise RuntimeError("embed_theories requires an active connection")
        from Isabelle_RPC_Host.context import entities_of

        # Phase 1: collect all missing texts across all theories
        theory_jobs: list[tuple[universal_key, str, bool, int, int]] = []
        all_texts: list[str] = []
        all_text_keys: list[key] = []
        for thy in theories:
            if isinstance(thy, str):
                thy_name = thy
                try:
                    thy_key = universal_key_of(self.connection, EntityKind.THEORY, thy)
                except UndefinedEntity:
                    self.connection.warning(
                        f"[Semantic_Embedding] skipping unknown theory {thy!r}")
                    continue
            else:
                thy_key = bytes(thy)
                thy_name = self.connection.callback("Theory_Hash.theory_name_of", thy_key)
                if thy_name is None:
                    raise ValueError(
                        f"Theory key {thy_key.hex()} not found in the active Isabelle runtime; "
                        f"the theory may not be loaded")
            if self.is_thy_embedded(thy_key):
                continue
            keys = entities_of(self.connection, EntityKind.ALL, # type: ignore
                               theory=thy_name, the_theory_only=True)
            missing = [k for k in keys if k not in self]
            wip = is_WIP(thy_key)
            if not missing:
                if not wip:
                    self.mark_thy_embedded(thy_key)
                continue
            texts: list[str] = []
            text_keys: list[key] = []
            for k in missing:
                sem = Semantic_DB.query(k, with_pretty=True)
                if sem is not None:
                    texts.append(sem)
                    text_keys.append(k)
            if not texts:
                if not wip:
                    self.mark_thy_embedded(thy_key)
                continue
            start = len(all_texts)
            all_texts.extend(texts)
            all_text_keys.extend(text_keys)
            theory_jobs.append((thy_key, thy_name, wip, start, start + len(texts)))

        if not all_texts:
            return

        # Phase 2: one batch embed call
        self.connection.tracing(
            f"[Semantic_Embedding] embed_theories: embedding {len(all_texts)} entities "
            f"from {len(theory_jobs)} theories in one batch")
        result = self.emb_provider.embed_batch(all_texts)

        # Phase 3: store vectors
        with self._env.begin(write=True) as txn:
            for k, vec in zip(all_text_keys, result.vectors):
                txn.put(k, vec.astype(np.float32).tobytes())

        # Phase 4: mark theories, distributing token count proportionally
        total = len(all_texts)
        for thy_key, thy_name, wip, start, end in theory_jobs:
            if not wip:
                n = end - start
                tokens = result.total_tokens * n // total if total > 0 else 0
                self.mark_thy_embedded(thy_key, tokens)


_svs_lock = threading.Lock()


def _resolve_embedding_model(connection: Connection | None, emb_provider: str | None) -> str:
    """Resolve embedding model name from config, env, or default."""
    if emb_provider is not None:
        return emb_provider
    if connection is not None:
        try:
            name = connection.config_lookup("Semantic_Embedding.embedding_model")
            if name:
                return name
        except Exception:
            pass
    return os.getenv("EMBEDDING_MODEL", "fw.qwen3-embedding-8b")


def _conn_semantic_vector_store(self: Connection, embedding_model: str | None = None) -> Semantic_Vector_Store:
    """Get or create a Semantic_Vector_Store for the given embedding model."""
    resolved = _resolve_embedding_model(self, embedding_model)
    with _svs_lock:
        stores = getattr(self, '_semantic_vector_stores', None)
        if stores is not None and resolved in stores:
            return stores[resolved]
    return Semantic_Vector_Store(emb_provider=resolved, connection=self)

Connection.semantic_vector_store = _conn_semantic_vector_store  # type: ignore


# --- RPC wrappers ---

@isabelle_remote_procedure("Semantic_Store.query")
def _query(arg: Any, connection: Connection) -> str | None:
    key, with_pretty = arg
    return Semantic_DB.query(bytes(key), bool(with_pretty))


@isabelle_remote_procedure("Semantic_Store.is_interpreted")
def _is_interpreted(arg: Any, connection: Connection) -> bool:
    return Semantic_DB.is_thy_interpreted(bytes(arg))


@isabelle_remote_procedure("Semantic_Store.mark_interpreted")
def _mark_interpreted(arg: Any, connection: Connection) -> None:
    Semantic_DB.mark_interpreted(bytes(arg))


@isabelle_remote_procedure("Semantic_Store.clean_wip")
def _clean_wip(arg: Any, connection: Connection) -> int:
    return clean_wip()


@isabelle_remote_procedure("Semantic_Embedding.query_knn")
def _query_knn(arg: Any, connection: Connection) -> list[tuple[float, tuple[int, str]]]:
    query_str, k, kind_ints, domain_raw = arg
    kinds = [EntityKind(ki) for ki in kind_ints]
    domain = [bytes(uk) for uk in domain_raw] if domain_raw is not None else None
    store = connection.semantic_vector_store()  # type: ignore
    results = store.lookup(query_str, k, kinds, domain)
    return [(score, (int(rec.kind), rec.name))
            for score, rec in results]


@isabelle_remote_procedure("Semantic_Embedding.embed_semantics")
def _embed_semantics(arg: Any, connection: Connection) -> None:
    theory_names, model_name = arg
    store = connection.semantic_vector_store(model_name or None)  # type: ignore
    store.embed_theories(theory_names)


@isabelle_remote_procedure("Semantic_Embedding.is_thy_embedded")
def _is_thy_embedded_rpc(arg: Any, connection: Connection) -> bool:
    theory_name, model_name = arg
    store = connection.semantic_vector_store(model_name)  # type: ignore
    try:
        thy_key = universal_key_of(connection, EntityKind.THEORY, theory_name)
    except UndefinedEntity:
        return False
    return store.is_thy_embedded(thy_key)
