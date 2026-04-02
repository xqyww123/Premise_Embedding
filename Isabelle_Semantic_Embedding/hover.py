"""
Go-to-definition and hover message for Isabelle entities.

Calls into Isabelle/ML which delegates to Scala. The Scala side tries the live
PIDE runtime first, then falls back to session export databases.
"""

import os
from typing import Any, Optional

from Isabelle_RPC_Host import Connection
from Isabelle_RPC_Host.position import AsciiPosition, UnicodePosition, IsabellePosition
from Isabelle_RPC_Host.unicode import pretty_unicode
from claude_agent_sdk import SdkMcpTool, tool

from .base import ToolCall_ret, mk_ret as _mk_ret


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def goto_definition(
    pos: IsabellePosition,
    connection: Optional[Connection] = None,
) -> Optional[tuple[str, int, int, int]]:
    """Given a cursor position, return the definition position.

    Returns (def_file, def_line, def_offset, def_end_offset) or None.
    The Scala side tries the live PIDE runtime first, then falls back to
    session export databases for compiled theories.
    """
    if connection is None:
        return None
    try:
        result = await connection.callback(
            "pide_state.goto_definition", (pos.file, pos.raw_offset))
        if result is not None:
            def_file, def_line, def_offset, def_end_offset = result
            return (def_file, def_line, def_offset, def_end_offset)
    except Exception:
        pass
    return None


async def hover_message(
    pos: IsabellePosition,
    connection: Optional[Connection] = None,
) -> Optional[str]:
    """Given a cursor position, return the hover tooltip text.

    The Scala side tries the live PIDE runtime first, then falls back to
    session export databases for compiled theories.
    """
    if connection is None:
        return None
    try:
        result = await connection.callback(
            "pide_state.hover_message", (pos.file, pos.raw_offset))
        if result is not None and result != "":
            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# MCP tools for Claude Code agent
# ---------------------------------------------------------------------------


def _validate_position_args(args: dict[str, Any]) -> str | None:
    """Validate position tool arguments. Returns error message or None."""
    file = args.get("file")
    line = args.get("line")
    column = args.get("column")
    if not isinstance(file, str) or not file:
        return "Invalid argument: 'file' must be a non-empty string."
    if not isinstance(line, int) or line < 1:
        return "Invalid argument: 'line' must be a positive integer."
    if not isinstance(column, int) or column < 1:
        return "Invalid argument: 'column' must be a positive integer."
    if not os.path.isfile(file):
        return f"File not found: {file}"
    return None


_position_schema = {
    "type": "object",
    "properties": {
        "file": {"type": "string", "description": "Absolute path to the .thy file"},
        "line": {"type": "integer", "description": "1-based line number"},
        "column": {"type": "integer", "description": "1-based column"},
    },
    "required": ["file", "line", "column"],
    "additionalProperties": False,
}


def _resolve_thy_path(file: str) -> str:
    """Map .unicode.thy back to original .thy; pass through otherwise."""
    if file.endswith(".unicode.thy"):
        return file[:-len(".unicode.thy")] + ".thy"
    return file


def mk_definition_tool(connection: Connection, unicode: bool = False) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("hover")
    @tool(
        "definition",
        "Go to the definition of the symbol at a given source position.",
        input_schema=_position_schema,
    )
    async def definition_tool(args: dict[str, Any]) -> ToolCall_ret:
        try:
            log.debug("definition: %s:%d:%d", args.get("file"), args.get("line"), args.get("column"))
            err = _validate_position_args(args)
            if err is not None:
                log.warning("definition: validation error: %s", err)
                return _mk_ret(err, is_error=True)
            thy_path = _resolve_thy_path(args["file"])
            if unicode:
                isa_pos = UnicodePosition(args["line"], args["column"], thy_path).to_isabelle_position()
            else:
                isa_pos = AsciiPosition(args["line"], args["column"], thy_path).to_isabelle_position()
            result = await goto_definition(isa_pos, connection)
            if result is None:
                log.debug("definition: not found")
                return _mk_ret("No definition found at this position.")
            def_file, def_line, def_offset, def_end_offset = result
            def_isa = IsabellePosition(def_line, def_offset, def_file)
            if unicode:
                from .theory_structure import mk_unicode_file
                out = def_isa.to_unicode_position()
                out_file = mk_unicode_file(out.file)
            else:
                out = def_isa.to_ascii_position()
                out_file = out.file
            ret = f"{out_file}:{out.line}:{out.column}"
            log.debug("definition: -> %s", ret)
            return _mk_ret(ret)
        except Exception:
            log.exception("definition: error")
            raise
    return definition_tool


def mk_hover_tool(connection: Connection, unicode: bool = False) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("hover")
    @tool(
        "hover",
        "Get hover information for the symbol at a given source position.",
        input_schema=_position_schema,
    )
    async def hover_tool(args: dict[str, Any]) -> ToolCall_ret:
        try:
            log.debug("hover: %s:%d:%d", args.get("file"), args.get("line"), args.get("column"))
            err = _validate_position_args(args)
            if err is not None:
                log.warning("hover: validation error: %s", err)
                return _mk_ret(err, is_error=True)
            thy_path = _resolve_thy_path(args["file"])
            if unicode:
                isa_pos = UnicodePosition(args["line"], args["column"], thy_path).to_isabelle_position()
            else:
                isa_pos = AsciiPosition(args["line"], args["column"], thy_path).to_isabelle_position()
            result = await hover_message(isa_pos, connection)
            if result is None:
                log.debug("hover: not found")
                return _mk_ret("No hover information at this position.")
            if unicode:
                result = pretty_unicode(result)
            log.debug("hover: -> %s", result)
            return _mk_ret(result)
        except Exception:
            log.exception("hover: error")
            raise
    return hover_tool

