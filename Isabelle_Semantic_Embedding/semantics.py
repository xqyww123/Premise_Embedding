"""Semantic query tools for looking up interpretations from parent theories."""

import asyncio
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

from .semantic_embedding import Vector_Store, Embedding_Provider, embedding_provider, Reranker_Provider, reranker_provider, key

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


EXPR_DISPLAY_LIMIT = 500

def trunc_expr(s: str, limit: int = EXPR_DISPLAY_LIMIT) -> str:
    """Truncate an expression string to the given limit (default EXPR_DISPLAY_LIMIT)."""
    return s[:limit] + "…" if len(s) > limit else s


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
                pp += f": {trunc_expr(self.expr)}"
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

    def __contains__(self, key: universal_key) -> bool:
        with self._ensure_env().begin() as txn:
            return txn.get(key) is not None

    def contains(self, keys: list[universal_key]) -> list[bool]:
        """Check existence for a batch of keys in a single transaction."""
        with self._ensure_env().begin() as txn:
            return [txn.get(k) is not None for k in keys]

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
                    "input_tokens": 0, "cache_creation_tokens": 0,
                    "cache_read_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "finished": True,
                    "model": "",
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
            "show_defs": {
                "type": "boolean",
                "description": "If true, include the Isabelle source code of the command defining the entity.",
                "default": False,
            },
        },
        "required": ["type", "name"],
    }


async def _get_definition_with_pos(
    connection: Connection, kind: EntityKind, uk: universal_key
) -> tuple[str, IsabellePosition] | None:
    """Look up the source code and position of the command defining an entity.

    Uses cached entity enumeration to find the definition position,
    then calls command_at_position to retrieve the source.

    Returns ``(source, cmd_pos)`` where *cmd_pos* is an
    `IsabellePosition` for the command start (symbol offset), or ``None``
    if the position or source is unavailable.
    """
    from Isabelle_RPC_Host.context import entities_of
    from .hover import command_at_position
    entries, _ = await entities_of(connection, [kind])
    pos = None
    for key, p in entries:
        if key == uk:
            pos = p
            break
    if pos is None:
        return None
    cmd = await command_at_position(pos, connection)
    if cmd is None:
        return None
    source, start_offset, _ = cmd
    cmd_pos = IsabellePosition(0, start_offset, pos.file)
    return (source, cmd_pos)


async def _get_definition_source(
    connection: Connection, kind: EntityKind, uk: universal_key
) -> str | None:
    """Look up the source code of the command defining an entity.

    Convenience wrapper around `_get_definition_with_pos` that discards
    the position.
    """
    result = await _get_definition_with_pos(connection, kind, uk)
    if result is None:
        return None
    source, _ = result
    return source


async def query_by_name_raw(
    connection: Connection,
    kind: EntityKind,
    name: str,
    with_pretty: bool = True,
) -> tuple[str, universal_key]:
    """Look up entity by kind and name, returning ``(semantic_text, universal_key)``.

    Raises `UndefinedEntity`, `IsabelleError`, or `LookupError` (not yet interpreted).
    """
    uk = await universal_key_of(connection, kind, name)
    sem = Semantic_DB.query(uk, with_pretty=with_pretty)
    if sem is None:
        raise LookupError(
            f'{kind.label} "{name}" has not been interpreted yet. '
            'Try using `mcp__proof__search_isabelle` to find what you need.')
    return (sem, uk)


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
            sem, uk = await query_by_name_raw(connection, tag, name, with_pretty=with_pretty)
            if args.get("show_defs", False):
                src = await _get_definition_source(connection, tag, uk)
                if src is not None:
                    sem += f"\n\nDefinition:\n{src}"
            return _mk_ret(sem)
        except LookupError as e:
            return _mk_ret(str(e))
        except UndefinedEntity as e:
            if "." in name:
                short = name.rsplit(".", 1)[1]
                try:
                    sem, uk = await query_by_name_raw(connection, tag, short, with_pretty=with_pretty)
                    if args.get("show_defs", False):
                        src = await _get_definition_source(connection, tag, uk)
                        if src is not None:
                            sem += f"\n\nDefinition:\n{src}"
                    return _mk_ret(f"The {name} is undefined, but we find:\n{sem}")
                except (IsabelleError, UndefinedEntity, LookupError):
                    pass
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(
                str(e) + " Try using `mcp__proof__semantic_search` to find what you need.",
                is_error=True,
            )
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
    _query_by_position_schema = {
        **_position_schema,
        "properties": {
            **_position_schema["properties"],
            "show_defs": {
                "type": "boolean",
                "description": "If true, include the Isabelle source code of the command defining the entity.",
                "default": False,
            },
        },
    }
    @tool(
        "query_by_position",
        description,
        input_schema=_query_by_position_schema,
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
            entity = await connection.callback(
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
            uk = await universal_key_of(connection, tag, name)
            sem = Semantic_DB.query(uk, with_pretty=with_pretty)
            if sem is None:
                return _mk_ret(f"{tag.label} \"{name}\" has not been interpreted yet.")
            if args.get("show_defs", False):
                src = await _get_definition_source(connection, tag, uk)
                if src is not None:
                    sem += f"\n\nDefinition:\n{src}"
            return _mk_ret(sem)
        except (IsabelleError, UndefinedEntity) as e:
            log.warning("%s: %s", type(e).__name__, e)
            return _mk_ret(str(e), is_error=True)
        except Exception:
            log.exception("query_by_position: error")
            raise
    return query_by_position_tool


# --- Other utilities ---

async def interpret_theories_by_names(connection: Connection, names: list[str]) -> None:
    """Interpret theories by name (short or long).
    Resolves names, skips already-interpreted theories, and interprets the rest.
    Calls back into Isabelle ML via the Semantic_Store.interpret_theories callback."""
    await connection.callback("Semantic_Store.interpret_theories", names)


_RERANK_FETCH_MULTIPLIER = 4


class Semantic_Vector_Store(Vector_Store):

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
        """All available entities at the connection's context, plus additional keys."""
        def __init__(self, extra: list[universal_key]):
            self.extra = extra
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
            emb_provider = os.getenv("EMBEDDING_MODEL", "fw.qwen3-embedding-8b")
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

    async def _auto_embed(self, missing: list[key], matrix: np.ndarray, row: int) -> list[key]:
        if self.connection is None:
            return []
        if not await self.connection.config_lookup("auto_interpret_for_embedding"):
            await self.connection.warning(
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
        await self.connection.tracing(
            f"[Semantic_Embedding] {len(missing)} entities missing embeddings, "
            f"spanning {len(theory_hashes)} un-embedded theories")
        # # DEBUG: show missing entities (requires debug_key_name from context.py)
        # from Isabelle_RPC_Host.context import debug_key_name
        # for k in missing[:50]:
        #     readable = debug_key_name(k) or f"<unknown {k.hex()[:16]}…>"
        #     await self.connection.tracing(f"  MISSING: {readable}")
        # if len(missing) > 50:
        #     await self.connection.tracing(f"  ... and {len(missing) - 50} more")
        # Filter to uninterpreted theories, excluding the current theory and skipped theories
        from Isabelle_RPC_Host.context import theory_long_name
        current_thy = await theory_long_name(self.connection)
        uninterpreted_theories: list[str] = []
        for th in theory_hashes:
            if not Semantic_DB.is_thy_interpreted(th):
                name = await self.connection.callback("Theory_Hash.theory_name_of", th)
                if name is not None and name != current_thy and not is_thy_skipped(name):
                    uninterpreted_theories.append(name)
        confirmed = False
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
        # Query semantic store for interpretations and embed them
        texts: list[str] = []
        text_keys: list[key] = []
        for k in missing:
            sem = Semantic_DB.query(k, with_pretty=True)
            if sem is not None:
                texts.append(sem)
                text_keys.append(k)
        if not texts:
            await self.connection.tracing(
                f"[Semantic_Embedding] no semantic interpretations found for the missing entities, skipping")
            return []
        if len(texts) > 42 and not confirmed:
            import Isabelle_RPC_Host.dialogue
            answer = await self.connection.dialogue(
                f"[Semantic Embedding] {len(texts)} entities to embed. "
                f"This may consume a significant amount of API tokens. Proceed?",
                ["Yes", "No"])
            if answer != "Yes":
                return []
        total_chars = sum(len(t) for t in texts)
        await self.connection.tracing(
            f"[Semantic_Embedding] embedding {len(texts)} of {len(missing)} missing entities "
            f"({total_chars} chars total) into vectors")
        embed_result = await self.emb_provider.embed(texts)
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
    ) -> tuple[list[tuple[float, 'SemanticRecord']], list[str]]:
        """Search the k closest entities to query, filtered by kinds and domain.
        Returns (results, warnings) where results are (score, record) pairs
        sorted by similarity, and warnings include notices about undeclared
        free variables in term patterns.
        Domain controls the search scope:
          ContextAll (default): all context entities of the given kinds
          ContextExtended(extra): context entities + additional keys
          Restricted(keys): only the given keys, filtered by kinds
        Pattern/theory filters (empty = no restriction):
          term_patterns: Isabelle term pattern strings (structural subterm matching)
          type_patterns: Isabelle type pattern strings (type matching)
          theories_include: only entities from these theories
        """
        warnings: list[str] = []
        if not kinds:
            return [], warnings
        if domain is Semantic_Vector_Store.ContextAll:
            if self.connection is None:
                return [], warnings
            from Isabelle_RPC_Host.context import entities_of
            entries, warnings = await entities_of(self.connection, kinds,
                                     theories_not_include=_SKIP_THEORY_LONG_NAMES,
                                     term_patterns=term_patterns,
                                     type_patterns=type_patterns,
                                     theories_include=theories_include,
                                     name_contains=name_contains)
            candidates = [k for k, _ in entries]
        elif isinstance(domain, Semantic_Vector_Store.ContextExtended):
            if self.connection is None:
                return [], warnings
            from Isabelle_RPC_Host.context import entities_of
            entries, warnings = await entities_of(self.connection, kinds,
                                     theories_not_include=_SKIP_THEORY_LONG_NAMES,
                                     term_patterns=term_patterns,
                                     type_patterns=type_patterns,
                                     theories_include=theories_include,
                                     name_contains=name_contains)
            candidates = [k for k, _ in entries]
            seen = set(candidates)
            kind_set = set(kinds)
            for ek in domain.extra:
                if ek not in seen and destruct_key(ek).kind in kind_set:
                    candidates.append(ek)
                    seen.add(ek)
        elif isinstance(domain, Semantic_Vector_Store.Restricted):
            kind_set = set(kinds)
            candidates = [dk for dk in domain.keys if destruct_key(dk).kind in kind_set]
        else:
            raise TypeError(f"Unknown domain type: {type(domain)}")
        if not candidates:
            return [], warnings
        # Save original query string for potential reranking (topk embeds it)
        query_str = query if isinstance(query, str) else None
        reranker = (await self._get_reranker()) if query_str else None
        fetch_k = k * _RERANK_FETCH_MULTIPLIER if reranker else k
        top = await self.topk(query, candidates, fetch_k)
        # Rerank if configured and query was a string
        if reranker is not None and query_str is not None and len(top) > 1:
            doc_entries: list[tuple[universal_key, SemanticRecord, str]] = []
            for uk, _score in top:
                rec = Semantic_DB[uk]
                if rec is not None:
                    doc_text = Semantic_DB.query(uk, with_pretty=True)
                    if doc_text:
                        doc_entries.append((uk, rec, doc_text))
            if doc_entries:
                try:
                    rr = await reranker.rerank(
                        query_str, [e[2] for e in doc_entries], min(k, len(doc_entries)))
                    results = [(rr.scores[i], doc_entries[idx][1])
                               for i, idx in enumerate(rr.indices)]
                    return results, warnings
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Reranker failed, falling back to embedding scores: %s", e)
        results = [(score, rec) for uk, score in top if (rec := Semantic_DB[uk]) is not None]
        if len(results) < k:
            # Pad with entities that had no embedding, assigned score 0
            top_set = {uk for uk, _ in top}
            for uk in candidates:
                if len(results) >= k:
                    break
                if uk not in top_set:
                    rec = Semantic_DB[uk]
                    if rec is not None:
                        results.append((0.0, rec))
        return results[:k], warnings

    async def _embed_keys(self, keys: list[universal_key]) -> int:
        """Embed the given keys that are missing from the vector store.
        Looks up semantic texts from Semantic_DB, embeds them, and stores vectors.
        Returns the number of entities actually embedded."""
        # Filter to keys not already in the store
        exists = self.contains(keys)
        missing = [k for k, ex in zip(keys, exists) if not ex]
        if not missing:
            return 0
        # Collect semantic texts
        texts: list[str] = []
        text_keys: list[universal_key] = []
        for k in missing:
            sem = Semantic_DB.query(k, with_pretty=True)
            if sem is not None:
                texts.append(sem)
                text_keys.append(k)
        if not texts:
            return 0
        # Embed and store
        if self.connection is not None:
            await self.connection.tracing(
                f"[Semantic_Embedding] embedding {len(texts)} entities ({sum(len(t) for t in texts)} chars)")
        result = await self.emb_provider.embed(texts)
        with self._env.begin(write=True) as txn:
            for k, vec in zip(text_keys, result.vectors):
                txn.put(k, vec.astype(np.float32).tobytes())
        return len(text_keys)

    async def embed_entities(self, keys: list[universal_key]) -> None:
        """Embed the given entity keys, skipping those already embedded."""
        await self._embed_keys(keys)

    async def embed_all_entities_in_theories(self, theories: list[str | universal_key]) -> None:
        """Embed semantic interpretations into vectors for the given theories.

        For each theory, collects all entity keys, embeds missing ones,
        and marks the theory as fully embedded.

        Args:
            theories: Long theory names (str) or universal keys to embed.

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
            if self.is_thy_embedded(thy_key):
                continue
            entries, _warnings = await entities_of(self.connection, EntityKind.ALL, # type: ignore
                               theory=thy_name, the_theory_only=True)
            keys = [k for k, _ in entries]
            wip = is_WIP(thy_key)
            await self._embed_keys(keys)
            if not wip:
                self.mark_thy_embedded(thy_key)


_svs_lock = threading.Lock()


async def _resolve_embedding_model(connection: Connection | None, emb_provider: str | None) -> str:
    """Resolve embedding model name from config, env, or default."""
    if emb_provider is not None:
        return emb_provider
    if connection is not None:
        name = await connection.config_lookup("Semantic_Embedding.embedding_model")
        if name:
            return name
    return os.getenv("EMBEDDING_MODEL", "fw.qwen3-embedding-8b")


async def _resolve_reranker_model(connection: Connection | None) -> str | None:
    """Resolve reranker model name from config, env, or None (disabled)."""
    if connection is not None:
        name = await connection.config_lookup("Semantic_Embedding.reranker_model")
        if name:
            return name
    model = os.getenv("RERANKER_MODEL", "")
    return model if model else None


async def _conn_semantic_vector_store(self: Connection, embedding_model: str | None = None) -> Semantic_Vector_Store:
    """Get or create a Semantic_Vector_Store for the given embedding model."""
    resolved = await _resolve_embedding_model(self, embedding_model)
    with _svs_lock:
        stores = getattr(self, '_semantic_vector_stores', None)
        if stores is not None and resolved in stores:
            return stores[resolved]
    return Semantic_Vector_Store(emb_provider=resolved, connection=self)

Connection.semantic_vector_store = _conn_semantic_vector_store  # type: ignore


# --- RPC wrappers ---

@isabelle_remote_procedure("Semantic_Store.query")
async def _query(arg: Any, connection: Connection) -> str | None:
    key, with_pretty = arg
    return Semantic_DB.query(bytes(key), bool(with_pretty))


@isabelle_remote_procedure("Semantic_Store.is_interpreted")
async def _is_interpreted(arg: Any, connection: Connection) -> bool:
    return Semantic_DB.is_thy_interpreted(bytes(arg))


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
async def _query_knn(arg: Any, connection: Connection) -> list[tuple[float, tuple[int, str]]]:
    query_str, k, kind_ints, domain_raw = arg
    kinds = [EntityKind(ki) for ki in kind_ints]
    domain = Semantic_Vector_Store.Restricted([bytes(uk) for uk in domain_raw]) if domain_raw is not None else Semantic_Vector_Store.ContextAll
    store = await connection.semantic_vector_store()  # type: ignore
    results, _warnings = await store.lookup(query_str, k, kinds, domain)
    return [(score, (int(rec.kind), rec.name))
            for score, rec in results]


@isabelle_remote_procedure("Semantic_Embedding.embed_all_entities_in_theories")
async def _embed_all_entities_in_theories(arg: Any, connection: Connection) -> None:
    theory_names, model_name = arg
    store = await connection.semantic_vector_store(model_name or None)  # type: ignore
    await store.embed_all_entities_in_theories(theory_names)


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
