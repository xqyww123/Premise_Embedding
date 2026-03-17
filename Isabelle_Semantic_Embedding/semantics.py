"""Semantic query tools for looking up interpretations from parent theories."""

from typing import Any

from Isabelle_RPC_Host import Connection
from Isabelle_RPC_Host.position import AsciiPosition, UnicodePosition, IsabellePosition
from claude_agent_sdk import SdkMcpTool, tool

from .base import ToolCall_ret, mk_ret as _mk_ret
from .hover import _position_schema, _validate_position_args, _resolve_thy_path

_QUERY_TAGS = {"constant": 0, "lemma": 1, "type": 2, "typeclass": 3, "locale": 4}

# PIDE Markup.Entity kinds → Semantic_Store query tags
_PIDE_KIND_TO_TAG = {
    "const": 0, "type": 2, "thm": 1,
    "class": 3, "locale": 4,
}

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


def mk_query_by_name_tool(connection: Connection, working_names: list[str]) -> SdkMcpTool[Any]:
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
            result = connection.callback("Semantic_Store.query", (tag, name))
            if result is None:
                return _mk_ret(f"{t} \"{name}\" has not been interpreted yet.")
            return _mk_ret(result)
        except Exception:
            log.exception("query_by_name: error")
            raise
    return query_by_name_tool


def mk_query_by_position_tool(
    connection: Connection, working_names: list[str], unicode: bool = False
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
            label = _PIDE_KIND_TO_LABEL.get(kind, kind)
            result = connection.callback("Semantic_Store.query", (tag, name))
            if result is None:
                return _mk_ret(f"{label} \"{name}\" has not been interpreted yet.")
            return _mk_ret(result)
        except Exception:
            log.exception("query_by_position: error")
            raise
    return query_by_position_tool
