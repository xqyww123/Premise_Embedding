"""Shared utilities for MCP tool implementations."""

from typing import Any

type ToolCall_ret = dict[str, Any]

def mk_ret(text: str, is_error: bool = False) -> ToolCall_ret:
    ret: ToolCall_ret = {"content": [{"type": "text", "text": text}]}
    if is_error:
        ret["is_error"] = True
    return ret
