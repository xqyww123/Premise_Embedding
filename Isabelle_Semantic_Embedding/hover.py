"""
Go-to-definition and hover message for Isabelle entities.

Calls into Isabelle/ML which delegates to Scala. The Scala side tries the live
PIDE runtime first, then falls back to session export databases.
"""

import os
from typing import Any, Optional

from Isabelle_RPC_Host import Connection
from Isabelle_RPC_Host.position import IsabellePosition, get_file_index
from Isabelle_RPC_Host.tokens import find_symbol_token_indices
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


async def command_at_position(
    pos: IsabellePosition,
    connection: Optional[Connection] = None,
) -> Optional[tuple[str, int, int]]:
    """Given a cursor position, return the source code of the command there.

    Returns (source, start_offset, end_offset) or None.
    Offsets are 1-based Isabelle symbol offsets within the file.
    Tries live PIDE state first, falls back to session export DB.
    """
    if connection is None:
        return None
    try:
        result = await connection.callback(
            "pide_state.command_at_position", (pos.file, pos.raw_offset))
        if result is not None:
            source, start_offset, end_offset = result
            return (source, start_offset, end_offset)
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
    symbol = args.get("symbol")
    if not isinstance(file, str) or not file:
        return "Invalid argument: 'file' must be a non-empty string."
    if not isinstance(line, int) or line < 1:
        return "Invalid argument: 'line' must be a positive integer."
    if not isinstance(symbol, str) or not symbol:
        return "Invalid argument: 'symbol' must be a non-empty string."
    if not os.path.isfile(file):
        return f"File not found: {file}"
    return None


_position_schema = {
    "type": "object",
    "properties": {
        "file": {"type": "string", "description": "Absolute path to the .thy file"},
        "line": {"type": "integer",
                 "description": "1-based line number where the entity/command is"},
        "symbol": {"type": "string",
                   "description":
                       "The token to look up — pass a SINGLE token, ASCII or"
                       " Unicode both work (e.g. \"Suc\", \"\\<Longrightarrow>\","
                       " \"⟹\"). For an identifier carrying a subscript,"
                       " pass it whole (e.g. \"foo\\<^sub>1\" or \"foo⇩1\")."
                       " The whole command at this line is searched, so the token"
                       " may be on a continuation line of a multi-line command."},
    },
    "required": ["file", "line", "symbol"],
    "additionalProperties": False,
}

# Cap on how many occurrences we issue backend queries for (counting is uncapped).
_MAX_OCCURRENCES = 10


def _resolve_thy_path(file: str) -> str:
    """Map .unicode.thy back to original .thy; pass through otherwise."""
    if file.endswith(".unicode.thy"):
        return file[:-len(".unicode.thy")] + ".thy"
    return file


async def _resolve_symbol_offsets(
    connection: Connection, thy_path: str, line: int, symbol: str,
) -> tuple[list[int], int, str]:
    """Locate ``symbol`` near ``line``.

    Returns ``(raw_offsets, total, scope)`` where ``raw_offsets`` are 1-based
    file-global Isabelle symbol offsets (capped to ``_MAX_OCCURRENCES``),
    ``total`` is the uncapped match count, and ``scope`` is ``"command"`` when
    the whole enclosing command span was searched (handles multi-line commands)
    or ``"line"`` when we could not resolve the command and fell back to the
    single line.
    """
    idx = get_file_index(thy_path)
    probe = IsabellePosition(line, idx.end_of_line_offset(line), thy_path)
    cmd = await command_at_position(probe, connection)
    if cmd is not None:
        src, start_off, _end_off = cmd
        raws = [start_off + k for k in find_symbol_token_indices(src, symbol)]
        scope = "command"
    else:
        raws = idx.symbol_offsets(line, symbol)
        scope = "line"
    return raws[:_MAX_OCCURRENCES], len(raws), scope


def _not_found_msg(symbol: str, line: int, scope: str) -> str:
    if scope == "command":
        return (f"Symbol {symbol!r} not found in the command at line {line}."
                " Check the spelling, or — if it belongs to a different"
                " command — adjust `line`.")
    return (f"Symbol {symbol!r} not found on line {line}"
            " (could not resolve the enclosing command, so only this single"
            " line was searched; try adjusting `line`).")


def resolve_context_at(
    context_at: Any, default_file: str | None, log: Any,
) -> tuple[tuple[str, int] | None, str | None]:
    """Resolve an optional ``context_at`` hint to ``(ctxt, note)``.

    ``context_at`` is ``{file?, line, symbol?}``. ``ctxt`` is ``(thy_path,
    raw_offset)`` (a 1-based Isabelle symbol offset) or ``None`` when the hint is
    absent/unusable. When ``symbol`` is given it pins the offset to that token's
    first occurrence on ``line``; when omitted the end of the line is used.

    If an explicit ``symbol`` is given but not found, we fall back to the end of
    the line AND return a ``note`` (also logged) so the caller can surface that
    the position hint missed — we do not silently relocate the context.
    """
    if not isinstance(context_at, dict):
        return None, None
    line = context_at.get("line")
    if not isinstance(line, int) or line < 1:
        return None, None
    file = context_at.get("file") or default_file
    if not isinstance(file, str) or not file:
        return None, None
    thy_path = _resolve_thy_path(file)
    try:
        idx = get_file_index(thy_path)
    except Exception:
        log.debug("context_at: cannot index %s", thy_path, exc_info=True)
        return None, None
    symbol = context_at.get("symbol")
    note: str | None = None
    if isinstance(symbol, str) and symbol:
        offs = idx.symbol_offsets(line, symbol)
        if offs:
            raw_offset = offs[0]
        else:
            raw_offset = idx.end_of_line_offset(line)
            note = (f"⚠ symbol {symbol!r} not found on line {line};"
                    " using end-of-line context instead.")
            log.warning("context_at: symbol %r not found on line %d;"
                        " falling back to end of line", symbol, line)
    else:
        raw_offset = idx.end_of_line_offset(line)
    return (thy_path, raw_offset), note


def mk_definition_tool(connection: Connection, unicode: bool = False) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("hover")

    def _render_location(def_file: str, def_line: int, def_offset: int) -> str:
        def_isa = IsabellePosition(def_line, def_offset, def_file)
        if not unicode:
            out = def_isa.to_ascii_position()
            return f"{out.file}:{out.line}:{out.column}"
        try:
            from .theory_structure import mk_unicode_file
            out = def_isa.to_unicode_position()
            return f"{mk_unicode_file(out.file)}:{out.line}:{out.column}"
        except Exception:
            # Cross-file target may be heap-only / unreadable: degrade to
            # file:line rather than failing the whole query.
            log.debug("definition: unicode render failed for %s; degrading", def_file,
                      exc_info=True)
            return f"{def_file}:{def_line}"

    @tool(
        "definition",
        "Go to the definition of a symbol. Give the line and the token text;"
        " the whole command at that line is searched (so the token may be on a"
        " continuation line). Pass a single token, ASCII or Unicode.",
        input_schema=_position_schema,
    )
    async def definition_tool(args: dict[str, Any]) -> ToolCall_ret:
        try:
            log.debug("definition: %s:%s %r", args.get("file"), args.get("line"),
                      args.get("symbol"))
            err = _validate_position_args(args)
            if err is not None:
                log.warning("definition: validation error: %s", err)
                return _mk_ret(err, is_error=True)
            thy_path = _resolve_thy_path(args["file"])
            line, symbol = args["line"], args["symbol"]
            raws, _total, scope = await _resolve_symbol_offsets(
                connection, thy_path, line, symbol)
            if not raws:
                return _mk_ret(_not_found_msg(symbol, line, scope))
            seen: set[tuple[str, int, int]] = set()
            locations: list[str] = []
            for raw in raws:
                result = await goto_definition(
                    IsabellePosition(line, raw, thy_path), connection)
                if result is None:
                    continue
                def_file, def_line, def_offset, _def_end = result
                key = (def_file, def_line, def_offset)
                if key in seen:
                    continue
                seen.add(key)
                locations.append(_render_location(def_file, def_line, def_offset))
            if not locations:
                return _mk_ret(f"No definition found for {symbol!r} at line {line}.")
            if len(locations) == 1:
                return _mk_ret(locations[0])
            return _mk_ret("Definitions:\n" + "\n".join(f"  {loc}" for loc in locations))
        except Exception:
            log.exception("definition: error")
            raise
    return definition_tool


def mk_hover_tool(connection: Connection, unicode: bool = False) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("hover")

    @tool(
        "hover",
        "Get hover information (type, kind, documentation) for a symbol. Give the"
        " line and the token text; the whole command at that line is searched (so"
        " the token may be on a continuation line). Pass a single token, ASCII or"
        " Unicode.",
        input_schema=_position_schema,
    )
    async def hover_tool(args: dict[str, Any]) -> ToolCall_ret:
        try:
            log.debug("hover: %s:%s %r", args.get("file"), args.get("line"),
                      args.get("symbol"))
            err = _validate_position_args(args)
            if err is not None:
                log.warning("hover: validation error: %s", err)
                return _mk_ret(err, is_error=True)
            thy_path = _resolve_thy_path(args["file"])
            line, symbol = args["line"], args["symbol"]
            idx = get_file_index(thy_path)
            raws, total, scope = await _resolve_symbol_offsets(
                connection, thy_path, line, symbol)
            if not raws:
                return _mk_ret(_not_found_msg(symbol, line, scope))
            # Group occurrences by hover text; record each occurrence's line:col.
            groups: dict[str, list[tuple[int, int]]] = {}
            for raw in raws:
                txt = await hover_message(
                    IsabellePosition(line, raw, thy_path), connection)
                if txt is None:
                    continue
                if unicode:
                    txt = pretty_unicode(txt)
                u_line, u_col = idx.isabelle_to_unicode(raw)
                groups.setdefault(txt, []).append((u_line, u_col))
            if not groups:
                return _mk_ret(
                    "No hover information at this position."
                    " If you are interpreting a syntax or notation,"
                    " call `desugar_and_explain` instead.")
            header = f"Symbol {symbol!r} — {total} occurrence(s)"
            if total > len(raws):
                header += f" (showing first {len(raws)})"
            out_lines = [header + ":"]
            for txt, locs in groups.items():
                at = ", ".join(f"{ln}:{co}" for ln, co in locs)
                out_lines.append(f"  at {at}:")
                for tl in (txt.splitlines() or [""]):
                    out_lines.append(f"    {tl}")
            return _mk_ret("\n".join(out_lines))
        except Exception:
            log.exception("hover: error")
            raise
    return hover_tool

