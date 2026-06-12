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
from Isabelle_RPC_Host.position import AsciiPosition, IsabellePosition
from Isabelle_RPC_Host.unicode import pretty_unicode, ascii_of_unicode
from Isabelle_RPC_Host.universal_key import EntityKind, UndefinedEntity, universal_key, universal_key_of, destruct_key, is_WIP
from claude_agent_sdk import SdkMcpTool, tool

from .semantic_embedding import Vector_Store, Embedding_Provider, embedding_provider, Reranker_Provider, reranker_provider, key

from .base import ToolCall_ret, mk_ret as _mk_ret
from .hover import _resolve_thy_path

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
        kind: EntityKind
        name: str
        expr: str | None
        interpretation: str | None
        # locale-interpretation provenance; None for ordinary entries and for
        # legacy 4-tuple records (read compatibly)
        provenance: 'Provenance | None' = None

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

    @staticmethod
    def _decode(raw: bytes) -> 'Record':
        """Decode a stored record; legacy 4-tuples read with provenance = None."""
        vals = list(msgpack.unpackb(raw))
        vals += [None] * (5 - len(vals))
        kind, name, expr, sem, prov_raw = vals[:5]
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
        return _Semantic_DB.Record(EntityKind(kind), d(name), d(expr), d(sem), prov)

    @staticmethod
    def _encode(record: 'Record') -> bytes:
        prov_map = None
        if record.provenance is not None:
            prov_map = {}
            if record.provenance.template_uk is not None:
                prov_map["template_uk"] = record.provenance.template_uk
            if record.provenance.locale_uk is not None:
                prov_map["locale_uk"] = record.provenance.locale_uk
            if record.provenance.qualifier is not None:
                prov_map["qualifier"] = record.provenance.qualifier
        return msgpack.packb((int(record.kind), record.name, record.expr,
                              record.interpretation, prov_map))  # type: ignore[return-value]

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

    def __setitem__(self, key: universal_key, record: 'Record') -> None:
        with self._ensure_env().begin(write=True) as txn:
            txn.put(key, self._encode(record))

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
        Skips WIP (non-persistent) theories unless persist_wip is enabled."""
        if is_WIP(key) and not persist_wip:
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

    @staticmethod
    def _copy_prefix(env: lmdb.Environment, old_prefix: bytes, new_prefix: bytes) -> int:
        assert len(old_prefix) == 16 and len(new_prefix) == 16
        count = 0
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(old_prefix):
                while True:
                    key = bytes(cursor.key())
                    if not key.startswith(old_prefix):
                        break
                    txn.put(new_prefix + key[16:], bytes(cursor.value()))
                    count += 1
                    if not cursor.next():
                        break
        return count

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
            if raw is not None and msgpack.unpackb(raw).get(b"finished", False):
                old_hash = candidate_hash
                break

        if old_hash is None:
            return False

        n = self._copy_prefix(sem_env, old_hash, new_hash)

        cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
        if os.path.isdir(cache_dir):
            from .semantic_embedding import _get_lmdb_env
            for entry in os.listdir(cache_dir):
                if entry.startswith("vector_") and entry.endswith(".lmdb"):
                    path = os.path.join(cache_dir, entry)
                    if os.path.isdir(path):
                        self._copy_prefix(_get_lmdb_env(path), old_hash, new_hash)

        print(f"Migrated {n} entries for {new_name} from {old_hash.hex()[:12]}… to {new_hash.hex()[:12]}…")
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
                        "named theorems", "proof method",
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
                    "column": {
                        "type": "integer",
                        "description": "1-based column number. If omitted, uses the end of the line.",
                    },
                },
                "required": ["line"],
            },
        },
        "required": ["type", "name"],
    }


def _end_of_line_column(file_path: str, line: int) -> int:
    """Return the last column (1-based) of the given line in the file."""
    try:
        with open(file_path, 'r') as f:
            for i, text in enumerate(f, 1):
                if i == line:
                    return max(1, len(text.rstrip('\n')))
        return 1
    except OSError:
        return 1


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
    entries, _ = await entities_of(connection, [kind], ctxt=ctxt)
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
        ctxt = None
        context_at = args.get("context_at")
        if isinstance(context_at, dict):
            line = context_at.get("line")
            if isinstance(line, int) and line >= 1:
                file = context_at.get("file") or file_path
                if isinstance(file, str):
                    file = _resolve_thy_path(file)
                if file:
                    column = context_at.get("column")
                    if not isinstance(column, int) or column < 1:
                        column = _end_of_line_column(file, line)
                    try:
                        from Isabelle_RPC_Host.position import AsciiPosition
                        isa_pos = AsciiPosition(line, column, file).to_isabelle_position()
                        ctxt = (isa_pos.file, isa_pos.raw_offset)
                    except Exception:
                        log.debug("position conversion failed for %s:%d:%d",
                                  file, line, column, exc_info=True)

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
            return _mk_ret(sem)
        except LookupError as e:
            return _mk_ret(str(e))
        except UndefinedEntity as e:
            if "." in name:
                short = name.rsplit(".", 1)[1]
                try:
                    sem, uk = await query_by_name_raw(connection, tag, short, with_pretty=with_pretty)
                    if args.get("show_defs", False):
                        sem = await _append_definition(sem, connection, tag, uk, short, log)
                    return _mk_ret(f"The {name} is undefined, but we find:\n{sem}")
                except (IsabelleError, UndefinedEntity, LookupError):
                    pass
            # Try resolving as a syntax/notation token via resolve_notation
            if tag == EntityKind.CONSTANT:
                resolved = await _try_resolve_syntax_token(
                    connection, name, ctxt, with_pretty,
                    args.get("show_defs", False), log)
                if resolved is not None:
                    return _mk_ret(resolved)
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


# --- Other utilities ---

async def interpret_theories_by_names(connection: Connection, names: list[str]) -> None:
    """Interpret theories by name (short or long).
    Resolves names, skips already-interpreted theories, and interprets the rest.
    Calls back into Isabelle ML via the Semantic_Store.interpret_theories callback."""
    await connection.callback("Semantic_Store.interpret_theories", (None, names))


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
        def __init__(self, extra: list[universal_key],
                     extra_names: dict[universal_key, str] | None = None):
            self.extra = extra
            self.extra_names = extra_names or {}
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
            emb_provider = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-8b")
        if isinstance(emb_provider, str):
            model_name = emb_provider
        else:
            model_name = getattr(emb_provider, '_registration_name', getattr(emb_provider, 'model', 'custom'))
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
        Skips WIP (non-persistent) theories unless persist_wip is enabled."""
        if is_WIP(theory_key) and not persist_wip:
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
        if domain is Semantic_Vector_Store.ContextAll:
            if self.connection is None:
                return [], warnings, 0
            from Isabelle_RPC_Host.context import entities_of
            entries, warnings = await entities_of(self.connection, kinds,
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
        elif isinstance(domain, Semantic_Vector_Store.ContextExtended):
            if self.connection is None:
                return [], warnings, 0
            from Isabelle_RPC_Host.context import entities_of
            entries, warnings = await entities_of(self.connection, kinds,
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
            seen = set(candidates)
            kind_set = set(kinds)
            for ek in domain.extra:
                if ek not in seen and destruct_key(ek).kind in kind_set:
                    candidates.append(ek)
                    seen.add(ek)
                    if ek in domain.extra_names:
                        candidate_names[ek] = domain.extra_names[ek]
                    else:
                        extra_rec = Semantic_DB[ek]
                        if extra_rec is not None:
                            candidate_names[ek] = extra_rec.name
                        else:
                            entity = destruct_key(ek)
                            if isinstance(entity.name, str):
                                candidate_names[ek] = entity.name
        elif isinstance(domain, Semantic_Vector_Store.Restricted):
            kind_set = set(kinds)
            candidates = [dk for dk in domain.keys if destruct_key(dk).kind in kind_set]
        else:
            raise TypeError(f"Unknown domain type: {type(domain)}")
        if not candidates:
            return [], warnings, 0
        total = len(candidates)

        def _resolve(uk: universal_key) -> SemanticRecord | None:
            """Look up SemanticRecord, falling back to a placeholder if name is known."""
            rec = Semantic_DB[uk]
            if rec is not None:
                return rec
            name = candidate_names.get(uk)
            if name is not None:
                return SemanticRecord(EntityKind(uk[16]), name, None, None)
            return None

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
        results: list[tuple[float, SemanticRecord] | None] = [None] * k
        n = 0
        for uk, score in top:
            if n >= k:
                break
            rec = _resolve(uk)
            if rec is not None:
                results[n] = (score, rec)
                n += 1
        if n < k:
            # Pad with entities that had no embedding, assigned score 0
            top_set = {uk for uk, _ in top}
            for uk in candidates:
                if n >= k:
                    break
                if uk not in top_set:
                    rec = _resolve(uk)
                    if rec is not None:
                        results[n] = (0.0, rec)
                        n += 1
        return results[:n], warnings, total  # type: ignore[list-item]

    async def _embed_keys(self, keys: list[universal_key], *,
                          force: bool = False) -> int:
        """Embed the given keys, storing vectors in the vector store.
        Looks up semantic texts from Semantic_DB, embeds them, and stores vectors.
        When force=True, re-embeds all keys even if they already have vectors.
        Returns the number of entities actually embedded."""
        if force:
            to_embed = keys
        else:
            exists = self.contains(keys)
            to_embed = [k for k, ex in zip(keys, exists) if not ex]
        if not to_embed:
            return 0
        # Collect semantic texts
        texts: list[str] = []
        text_keys: list[universal_key] = []
        for k in to_embed:
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
        await self._embed_keys(keys, force=False)

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
            entries, _warnings = await entities_of(self.connection, EntityKind.ALL, # type: ignore
                               theory=thy_name, the_theory_only=True)
            keys = [k for k, _, _ in entries]
            wip = is_WIP(thy_key) and not persist_wip
            await self._embed_keys(keys, force=force)
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
    return os.getenv("EMBEDDING_MODEL", "qwen3-embedding-8b")


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
        extras = [(bytes(k_), name) for k_, name in domain_payload]
        domain = Semantic_Vector_Store.ContextExtended(
            [k_ for k_, _ in extras],
            extra_names={k_: name for k_, name in extras})
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
