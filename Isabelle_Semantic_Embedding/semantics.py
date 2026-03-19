"""Semantic query tools for looking up interpretations from parent theories."""

import os
from typing import Any

import lmdb
import platformdirs
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from Isabelle_RPC_Host.position import AsciiPosition, UnicodePosition, IsabellePosition
from Isabelle_RPC_Host.universal_key import EntityKind, universal_key, universal_key_of
from claude_agent_sdk import SdkMcpTool, tool

from .base import ToolCall_ret, mk_ret as _mk_ret
from .hover import _position_schema, _validate_position_args, _resolve_thy_path

_QUERY_TAGS = {"constant": 1, "lemma": 2, "type": 3, "typeclass": 4, "locale": 5}

# PIDE Markup.Entity kinds → Semantic_Store query tags
_PIDE_KIND_TO_TAG = {
    "const": 1, "type": 3, "thm": 2,
    "class": 4, "locale": 5,
}


def open_semantic_store() -> lmdb.Environment:
    """Open the shared LMDB semantic store."""
    cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
    os.makedirs(cache_dir, exist_ok=True)
    return lmdb.open(os.path.join(cache_dir, "semantics.lmdb"), map_size=1 << 30)

_PIDE_KIND_TO_LABEL = {
    "const": "constant", "type": "type", "thm": "lemma",
    "class": "typeclass", "locale": "locale",
}

_query_by_name_schema = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["constant", "lemma", "type", "typeclass", "locale"],
            "description": "The kind of entity to query.",
        },
        "name": {
            "type": "string",
            "description": "The name of the entity to look up. "
            "For multi-variant theorems, include a '(idx)' suffix (e.g. 'conjI(2)').",
        },
    },
    "required": ["type", "name"],
}


def mk_query_by_name_tool(
    connection: Connection, lmdb_env: lmdb.Environment, working_names: list[str]
) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("semantics")
    @tool(
        "query_by_name",
        "Look up the semantic interpretation of a dependency from parent theories. "
        "Do not query entries you have been asked to interpret — interpret those from the source file yourself.",
        input_schema=_query_by_name_schema,
    )
    async def query_by_name_tool(args: dict[str, Any]) -> ToolCall_ret:
        try:
            t = args.get("type", "")
            name = args.get("name", "")
            log.debug("query_by_name: type=%r name=%r", t, name)
            if not isinstance(t, str) or not isinstance(name, str):
                return _mk_ret("Invalid arguments: 'type' and 'name' must be strings.", is_error=True)
            tag = _QUERY_TAGS.get(t)
            if tag is None:
                return _mk_ret(f"Invalid type: {t!r}. Must be one of {list(_QUERY_TAGS)}.", is_error=True)
            if not name:
                return _mk_ret("Invalid name: must be a non-empty string.", is_error=True)
            if name in working_names:
                return _mk_ret(
                    f"Cannot query \"{name}\" — it is or will be your task to interpret it from the source.",
                    is_error=True,
                )
            import msgpack
            key = universal_key_of(connection, EntityKind(tag), name)
            with lmdb_env.begin() as txn:
                val = txn.get(key)
            if val is None:
                return _mk_ret(f"{t} \"{name}\" has not been interpreted yet.")
            pp, sem = msgpack.unpackb(val)
            if isinstance(sem, bytes):
                sem = sem.decode()
            return _mk_ret(sem)
        except Exception:
            log.exception("query_by_name: error")
            raise
    return query_by_name_tool


def mk_query_by_position_tool(
    connection: Connection, lmdb_env: lmdb.Environment,
    working_names: list[str], unicode: bool = False
) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("semantics")
    @tool(
        "query_by_position",
        "Look up the semantic interpretation of the entity at a given source position.",
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
            # Resolve entity at position
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
            if name in working_names:
                return _mk_ret(
                    f"Cannot query \"{name}\" — it is your task to interpret it from the source.",
                    is_error=True,
                )
            import msgpack
            label = _PIDE_KIND_TO_LABEL.get(kind, kind)
            key = universal_key_of(connection, EntityKind(tag), name)
            with lmdb_env.begin() as txn:
                val = txn.get(key)
            if val is None:
                return _mk_ret(f"{label} \"{name}\" has not been interpreted yet.")
            pp, sem = msgpack.unpackb(val)
            if isinstance(sem, bytes):
                sem = sem.decode()
            return _mk_ret(sem)
        except Exception:
            log.exception("query_by_position: error")
            raise
    return query_by_position_tool


# --- Store operations ---

def query(key: universal_key, with_pretty: bool = False) -> str | None:
    """Look up a semantic interpretation by universal key."""
    import msgpack
    env = open_semantic_store()
    with env.begin() as txn:
        val = txn.get(key)
    env.close()
    if val is None:
        return None
    pp, sem = msgpack.unpackb(val)
    if isinstance(pp, bytes):
        pp = pp.decode()
    if isinstance(sem, bytes):
        sem = sem.decode()
    if with_pretty and pp:
        return pp + "\n" + sem
    return sem


def is_interpreted(key: universal_key) -> bool:
    """Check whether a theory has been fully interpreted."""
    import msgpack
    env = open_semantic_store()
    with env.begin() as txn:
        raw = txn.get(key)
    env.close()
    if raw is None:
        return False
    return msgpack.unpackb(raw).get(b"finished", False)


def mark_interpreted(key: universal_key) -> None:
    """Mark a theory as interpreted (finished) in the semantic store."""
    import msgpack
    env = open_semantic_store()
    with env.begin(write=True) as txn:
        raw = txn.get(key)
        if raw is not None:
            data = msgpack.unpackb(raw)
            data[b"finished"] = True
            txn.put(key, msgpack.packb(data))
        else:
            txn.put(key, msgpack.packb({
                "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "finished": True,
            }))
    env.close()


def clean_wip() -> int:
    """Remove all entries with non-persistent theory hashes."""
    from Isabelle_RPC_Host.theory_hash import is_persistent
    env = open_semantic_store()
    deleted = 0
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            if not is_persistent(key):
                cursor.delete()
                deleted += 1
    env.close()
    return deleted


# --- RPC wrappers ---

@isabelle_remote_procedure("Semantic_Store.query")
def _query(arg: Any, connection: Connection) -> str | None:
    key, with_pretty = arg
    return query(bytes(key), bool(with_pretty))


@isabelle_remote_procedure("Semantic_Store.is_interpreted")
def _is_interpreted(arg: Any, connection: Connection) -> bool:
    return is_interpreted(bytes(arg))


@isabelle_remote_procedure("Semantic_Store.mark_interpreted")
def _mark_interpreted(arg: Any, connection: Connection) -> None:
    mark_interpreted(bytes(arg))


@isabelle_remote_procedure("Semantic_Store.clean_wip")
def _clean_wip(arg: Any, connection: Connection) -> int:
    return clean_wip()
