"""Desugar-and-explain tool: strips syntax sugar and annotates constants."""

import re
from typing import Any

from Isabelle_RPC_Host import Connection
from Isabelle_RPC_Host.universal_key import universal_key
from Isabelle_RPC_Host.unicode import ascii_of_unicode
from claude_agent_sdk import SdkMcpTool, tool

from .base import ToolCall_ret, mk_ret as _mk_ret
from .hover import resolve_context_at
from .semantics import Semantic_DB


def _clean_error(msg: str) -> str:
    if msg.startswith("Ambiguous input"):
        first = msg.split('\n', 1)[0]
        first = re.sub(r'\s*\(\d+ displayed\)', '', first)
        return first.rstrip(':')
    return msg.split('\n', 1)[0]


_schema = {
    "type": "object",
    "properties": {
        "term": {
            "type": "string"
        },
        "context_at": {
            "type": "object",
            "description":
                "Parse the term under the proof context at this source position."
                " Omit to use the theory's global context.",
            "properties": {
                "file": {
                    "type": "string",
                    "description":
                        "Path to the theory file."
                        " Defaults to the current theory file.",
                },
                "line": {
                    "type": "integer",
                    "description": "1-based line number.",
                },
                "symbol": {
                    "type": "string",
                    "description":
                        "A token on that line to pin the context to (ASCII or"
                        " Unicode). Defaults to the end of the line.",
                },
            },
            "required": ["line"],
            "additionalProperties": False,
        },
    },
    "required": ["term"],
}


def mk_desugar_and_explain_tool(
    connection: Connection,
    file_path: str | None = None,
    seen_constants: set[str] | None = None,
) -> SdkMcpTool[Any]:
    log = connection.server.logger.getChild("desugar")
    if seen_constants is None:
        seen_constants = set()

    @tool(
        "desugar_and_explain",
        "Desugar an Isabelle term and annotate each constant"
        " with its English semantic interpretation. **IMPORTANT:** Call this tool when you are uncertain about a syntax/notation!",
        input_schema=_schema,
    )
    async def desugar_and_explain_tool(args: dict[str, Any]) -> ToolCall_ret:
        term_str = args.get("term", "")
        if not isinstance(term_str, str) or not term_str.strip():
            return _mk_ret("Invalid argument: 'term' must be a non-empty string.",
                           is_error=True)
        log.debug("desugar_and_explain: term=%r", term_str)
        # Normalize Unicode glyphs (≤, ∀, subscripts, ...) to Isabelle's
        # ASCII-escape form; Syntax.read_term does not recognize raw UTF-8.
        term_str = ascii_of_unicode(term_str)

        ctxt, ctxt_note = resolve_context_at(args.get("context_at"), file_path, log)

        try:
            compact_str, constants = await connection.callback(
                "explain_term.desugar_and_explain", (ctxt, term_str))
        except Exception as e:
            msg = _clean_error(str(e))
            log.debug("desugar_and_explain: error: %s", msg)
            if ctxt_note is not None:
                msg = f"{ctxt_note}\n\n{msg}"
            return _mk_ret(msg, is_error=True)

        lines: list[str] = []
        if ctxt_note is not None:
            lines.append(ctxt_note)
            lines.append("")
        lines.append(compact_str)
        new_annotations: list[str] = []
        for full_name, uk_bytes in constants:
            uk: universal_key = bytes(uk_bytes)
            if full_name in seen_constants:
                continue
            sem = Semantic_DB.query(uk, with_pretty=False)
            if sem is None:
                continue
            base = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
            new_annotations.append(f"  {base}: {sem}")
            seen_constants.add(full_name)

        if new_annotations:
            lines.append("")
            lines.append("Constants:")
            lines.extend(new_annotations)

        return _mk_ret("\n".join(lines))

    return desugar_and_explain_tool
