"""Semantic interpretation of Isabelle constants and theorems via Claude Code agent."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, cast

import platformdirs
import xxhash
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from Isabelle_RPC_Host.unicode import pretty_unicode
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import (
    HookInput,
    HookContext,
    HookJSONOutput,
    PreToolUseHookInput,
    ResultMessage,
)
from rocksdict import Options, Rdict

from .base import ToolCall_ret, mk_ret as _mk_ret

# --- Thread-local state ---

_KIND_CONSTANT = 0
_KIND_THEOREM = 1
_KIND_TYPE = 2
_KIND_CLASS = 3
_KIND_LOCALE = 4
_KIND_PROMPT_LABELS = {
    _KIND_CONSTANT: "constant",
    _KIND_THEOREM: "lemma",
    _KIND_TYPE: "type",
    _KIND_CLASS: "typeclass",
    _KIND_LOCALE: "locale",
}
_BATCH_SIZE = 20

class InterpretationTask:
    def __init__(self, connection: Connection, file_path: str,
                 names: list[str], kinds: list[int],
                 prop_strs: list[str], line_numbers: list[int],
                 file_hash: bytes, orig_names: list[str],
                 orig_kinds: list[int], uncached: list[int]):
        self.connection = connection
        self.file_path = file_path
        self.names = names
        self.kinds = kinds
        self.prop_strs = prop_strs
        self.line_numbers = line_numbers
        self.file_hash = file_hash
        self.orig_names = orig_names
        self.orig_kinds = orig_kinds
        self.uncached = uncached
        self.results: dict[str, str | None] = {
            f"{_KIND_PROMPT_LABELS[kinds[i]]} {names[i]}": None
            for i in range(len(names))
        }
        self._keys = list(self.results.keys())
        self.batches: list[tuple[str, range]] = []
        self.current_batch: int = 0
        self.batch_range: range = range(0)
        self._db: Rdict | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def __enter__(self) -> InterpretationTask:
        self._db = _open_cache()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def write_answer(self, task_idx: int, sem: str) -> None:
        """Write a single answer to the RocksDB cache."""
        if self._db is not None:
            orig_idx = self.uncached[task_idx]
            key = _cache_key(self.file_hash, self.orig_kinds[orig_idx], self.orig_names[orig_idx])
            self._db[key] = sem.encode("utf-8")

    def historical_cost(self) -> tuple[int, int, float]:
        """Read cumulative cost from RocksDB (without modifying it)."""
        import json
        if self._db is not None:
            raw = self._db.get(self.file_hash + b"\xff:cost")
            if raw is not None:
                prev = json.loads(raw.decode("utf-8"))
                return (prev.get("input_tokens", 0),
                        prev.get("output_tokens", 0),
                        prev.get("cost_usd", 0.0))
        return (0, 0, 0.0)

    def write_cost(self) -> tuple[int, int, float]:
        """Accumulate cost into the RocksDB cache. Returns updated cumulative totals."""
        import json
        prev_in, prev_out, prev_cost = self.historical_cost()
        total = (prev_in + self.total_input_tokens,
                 prev_out + self.total_output_tokens,
                 prev_cost + self.total_cost_usd)
        if self._db is not None:
            key = self.file_hash + b"\xff:cost"
            self._db[key] = json.dumps({
                "input_tokens": total[0],
                "output_tokens": total[1],
                "cost_usd": total[2],
            }).encode("utf-8")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        return total

    def advance_batch(self) -> str | None:
        self.current_batch += 1
        if self.current_batch >= len(self.batches):
            return None
        prompt, self.batch_range = self.batches[self.current_batch]
        return prompt

    def format_entries(self, indices: range) -> str:
        lines = []
        for i in indices:
            label = _KIND_PROMPT_LABELS.get(self.kinds[i], "unknown")
            lineno = self.line_numbers[i]
            line = (f"  [line {lineno}] " if lineno > 0 else "  ") + f"{label} {self.names[i]}"
            if self.prop_strs[i]:
                line += f": {self.prop_strs[i]}"
            lines.append(line)
        return "\n".join(lines)

    def build_prompt(self, file_path: str, theory_longname: str, indices: range | None = None) -> str:
        if indices is None:
            indices = range(len(self.names))
        entries_text = self.format_entries(indices)

        if indices.start != 0:
            return (
                f"Continue to translate the following entities into concise plain English. "
                f"State only what the entity defines or asserts. "
                f"Do NOT explain how it is derived or why it is useful. "
                f"The formal statement is already shown; describe its meaning rather than transcribing it. "
                f"Prefer plain English over formulas. Wrap formulas in backticks (e.g., `x`, `x + 1`).\n\n"
                f"Entries:\n{entries_text}\n\n"
                f"Submit translations via `mcp__isabelle_semantics__answer`."
            )

        return f"""\
Informalize the following entities from Isabelle theory "{theory_longname}" (location: {file_path})

Entries:
{entries_text}

Line numbers in brackets (e.g. [line 42]) indicate where each entity appears in the source file.

For each entry, translate the formal statement into a concise plain-English description (1\u20133 sentences). \
State only what the entity defines or asserts. \
Do NOT explain how it is derived or why it is useful. \
The formal statement is already shown; describe its meaning rather than transcribing it. \
Prefer plain English over formulas. Wrap formulas in backticks (e.g., `x`, `x + 1`).

Examples of good translations:
- constant Nat.add: The addition function on natural numbers, taking two natural numbers and returning their sum.
- lemma List.length_append: The length of the concatenation of two lists equals the sum of their individual lengths.
- lemma List.map_comp: Mapping `f` then `g` over a list is the same as mapping their composition `g \u2218 f`.
- type Prod: The product type, consisting of a pair of two values of possibly different types.

Translation hints:
- Suc n \u2192 "the successor of n" or "n + 1"

When you encounter an entity whose meaning is unclear, use `mcp__isabelle_semantics__query_by_name`, \
`mcp__isabelle_semantics__query_by_position`, `mcp__isabelle_semantics__hover`, or \
`mcp__isabelle_semantics__definition` to look it up before translating. \
However, you cannot query entries you have been asked to translate \u2014 do it yourself.

Submit all translations via `mcp__isabelle_semantics__answer` for each entry."""

class _LocalState(threading.local):
    task: InterpretationTask

_local = _LocalState()


# --- Persistent cache ---

def _open_cache() -> Rdict:
    cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
    os.makedirs(cache_dir, exist_ok=True)
    return Rdict(os.path.join(cache_dir, "semantics.db"),
                 options=Options(raw_mode=True))

def _cache_key(file_hash: bytes, kind: int, name: str) -> bytes:
    return file_hash + kind.to_bytes(1, "little") + b":" + name.encode("utf-8")


_log = logging.getLogger(__name__)


# --- MCP Tool: answer ---

_answer_schema = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["constant", "lemma", "type", "typeclass", "locale"]
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the entity, e.g. 'Groups.abel_semigroup'.",
                    },
                    "translation": {
                        "type": "string",
                        "description": "The plain-English translation of this entity.",
                    },
                },
                "required": ["type", "name", "translation"],
            },
            "description": "List of English translations to submit.",
        },
    },
    "required": ["interpretations"],
}


@tool(
    "answer",
    "Submit English translations for one or more of the listed entries. "
    "Each translation should be a concise plain-English description of what the entity defines or asserts. "
    "You may also resubmit an entry to correct a previous answer.",
    input_schema=_answer_schema,
)
async def _answer_tool(args: dict[str, Any]) -> ToolCall_ret:
    task = _local.task
    interpretations = args["interpretations"]
    errors = []
    count = 0
    for item in interpretations:
        key = f"{item['type']} {item['name']}"
        if key not in task.results:
            errors.append(f"Unknown entry: {key!r}")
            continue
        task.results[key] = item["translation"]
        task.write_answer(task._keys.index(key), item["translation"])
        _log.info("answer: %s = %s", key, item["translation"])
        count += 1
    batch_remaining = sum(1 for i in task.batch_range if task.results[task._keys[i]] is None)
    cs = "" if count == 1 else "s"
    total_answered = sum(1 for v in task.results.values() if v is not None)
    _log.info("answer: submitted %d, batch_remaining %d, %d/%d done",
               count, batch_remaining, total_answered, len(task.results))
    if batch_remaining == 0:
        next_prompt = task.advance_batch()
        if next_prompt is None:
            msg = "All done! Stop immediately without any further output."
        else:
            msg = f"Good job! You can resubmit corrections later using the `mcp__isabelle_semantics__answer` tool if needed.\n\n{next_prompt}"
    else:
        msg = f"Answered {count} translation{cs}, remaining {batch_remaining} in this batch."
    if errors:
        msg += "\nErrors:\n" + "\n".join(errors)
    return _mk_ret(msg)


# --- Permission control ---

_TOOL_WHITELIST = {
    "Read",
    "Grep",
    "Skill",
    "Agent",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "WebFetch",
    "WebSearch",
    "ExitPlanMode",
    "MCPSearch",
    "ToolSearch",
    "mcp__isabelle_semantics__query_by_name",
    "mcp__isabelle_semantics__query_by_position",
    "mcp__isabelle_semantics__definition",
    "mcp__isabelle_semantics__hover",
    "mcp__isabelle_semantics__answer",
}


def _deny(reason: str) -> HookJSONOutput:
    return {
        "continue_": False,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


async def _permission_control(
    hook_input: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    pre = cast(PreToolUseHookInput, hook_input)
    tool_name = pre["tool_name"]
    tool_input = pre.get("tool_input") or {}

    if tool_name not in _TOOL_WHITELIST:
        _log.warning("tool denied: %s %s", tool_name, tool_input)
        return _deny(f"Tool {tool_name!r} is not allowed.")

    _log.info("tool allowed: %s %s", tool_name, tool_input)
    return {}


# --- Agent runner ---

def _log_message(message: Any) -> None:
    """Log model output and thinking from a response message."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            _log.info("[model] %s", text)
        thinking = getattr(block, "thinking", None)
        if isinstance(thinking, str) and thinking:
            _log.debug("[thinking] %s", thinking)


async def _list_tools(client: ClaudeSDKClient) -> None:
    """Ask the model to list all available tools, for debugging."""
    await client.query("List all available tools you have access to.")
    async for message in client.receive_response():
        _log_message(message)


async def _run_agent(options: ClaudeAgentOptions) -> None:
    task = _local.task
    first_prompt, task.batch_range = task.batches[0]
    async with ClaudeSDKClient(options=options) as client:
        _log.info("agent: starting batch 0 (%d–%d)",
                   task.batch_range.start, task.batch_range.stop - 1)
        await client.query(first_prompt)
        async for message in client.receive_response():
            _log_message(message)
            if isinstance(message, ResultMessage):
                if message.usage:
                    task.total_input_tokens += message.usage.get("input_tokens", 0)
                    task.total_output_tokens += message.usage.get("output_tokens", 0)
                if message.total_cost_usd:
                    task.total_cost_usd += message.total_cost_usd
                _log.info("round usage: %s, cost: $%.4f",
                           message.usage, message.total_cost_usd or 0)
        # Retry any globally missing entries
        while True:
            missing = [k for k, v in task.results.items() if v is None]
            if not missing:
                break
            _log.info("agent: retrying %d missing entries", len(missing))
            missing_lines = "\n".join(f"  {k}" for k in missing)
            await client.query(
                f"You still have {len(missing)} unanswered entries. "
                f"Use the `mcp__isabelle_semantics__answer` tool to submit interpretations for:\n{missing_lines}"
            )
            async for message in client.receive_response():
                _log_message(message)
                if isinstance(message, ResultMessage):
                    if message.usage:
                        task.total_input_tokens += message.usage.get("input_tokens", 0)
                        task.total_output_tokens += message.usage.get("output_tokens", 0)
                    if message.total_cost_usd:
                        task.total_cost_usd += message.total_cost_usd
                    _log.info("round usage: %s, cost: $%.4f",
                               message.usage, message.total_cost_usd or 0)
    _log.info("total usage: input=%d output=%d tokens, cost=$%.4f",
               task.total_input_tokens, task.total_output_tokens, task.total_cost_usd)


# --- RPC procedure ---

@isabelle_remote_procedure("Semantic_Store.interpret_file")
def interpret_file(arg: Any, connection: Connection) -> list[str | None]:
    (file_path, theory_longname, thy_hash, deps_longname, raw_entries, context_info) = arg
    file_hash = bytes(thy_hash)
    kinds = [kind for (kind, _, _, _) in raw_entries]
    prop_strs = [pretty_unicode(prop) for (_, _, prop, _) in raw_entries]
    line_numbers = [lineno for (_, _, _, lineno) in raw_entries]
    names = [pretty_unicode(name) for (_, name, _, _) in raw_entries]
    n = len(names)

    # Build Unicode pretty-prints for all entries
    pretty_prints = []
    for i in range(n):
        label = _KIND_PROMPT_LABELS.get(kinds[i], "unknown")
        pp = f"{label} {names[i]}"
        if prop_strs[i]:
            pp += f": {prop_strs[i]}"
        pretty_prints.append(pp)

    # Inherit RPC server's logging configuration (idempotent, no race)
    if not _log.handlers and connection.server.logger.handlers:
        for h in connection.server.logger.handlers:
            _log.addHandler(h)
        _log.setLevel(connection.server.logger.level)
    _log.info("interpret_file: %s (%s), %d entries", theory_longname, file_path, n)

    # Check cache
    results: dict[int, str] = {}
    with _open_cache() as db:
        for i, name in enumerate(names):
            v = db.get(_cache_key(file_hash, kinds[i], name))
            if v is not None:
                results[i] = v.decode("utf-8")

    uncached = [i for i in range(n) if i not in results]
    _log.info("interpret_file: %d cached, %d to interpret", len(results), len(uncached))
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    cum_input_tokens = 0
    cum_output_tokens = 0
    cum_cost_usd = 0.0

    if uncached:
        from .hover import mk_definition_tool, mk_hover_tool
        from .semantics import mk_query_by_name_tool, mk_query_by_position_tool
        from .theory_structure import mk_unicode_file

        unicode_file_path = mk_unicode_file(file_path)

        with InterpretationTask(
            connection, file_path,
            names=[names[i] for i in uncached],
            kinds=[kinds[i] for i in uncached],
            prop_strs=[prop_strs[i] for i in uncached],
            line_numbers=[line_numbers[i] for i in uncached],
            file_hash=file_hash,
            orig_names=names,
            orig_kinds=kinds,
            uncached=uncached,
        ) as task:
            _local.task = task
            m = len(task.names)

            for start in range(0, m, _BATCH_SIZE):
                batch_range = range(start, min(start + _BATCH_SIZE, m))
                task.batches.append((
                    task.build_prompt(unicode_file_path, theory_longname, batch_range),
                    batch_range,
                ))

            query_by_name_tool = mk_query_by_name_tool(connection, task.names)
            query_by_position_tool = mk_query_by_position_tool(
                connection, task.names, unicode=True)
            definition_tool = mk_definition_tool(connection, unicode=True)
            hover_tool = mk_hover_tool(connection, unicode=True)
            mcp = create_sdk_mcp_server("isabelle_semantics", tools=[
                query_by_name_tool, query_by_position_tool,
                definition_tool, hover_tool, _answer_tool])
            options = ClaudeAgentOptions(
                model="claude-sonnet-4-6",
                cwd=os.path.dirname(unicode_file_path),
                permission_mode="default",
                allowed_tools=list(_TOOL_WHITELIST),
                mcp_servers={"isabelle_semantics": mcp},
                hooks={
                    "PreToolUse": [
                        HookMatcher(matcher="*", hooks=[_permission_control]),
                    ]
                },
            )

            _log.info("interpret_file: starting agent with %d batches", len(task.batches))
            asyncio.run(_run_agent(options))
            answered = sum(1 for v in task.results.values() if v is not None)
            _log.info("interpret_file: agent finished, %d/%d interpreted",
                       answered, len(task.names))
            cum_input, cum_output, cum_cost = task.write_cost()
            total_input_tokens = task.total_input_tokens
            total_output_tokens = task.total_output_tokens
            total_cost_usd = task.total_cost_usd
            cum_input_tokens = cum_input
            cum_output_tokens = cum_output
            cum_cost_usd = cum_cost

            # Remap agent results to original indices (cache already written incrementally)
            for i, sem in enumerate(task.results.values()):
                if sem is not None:
                    results[uncached[i]] = sem
    else:
        # All cached — read cumulative cost from DB
        with InterpretationTask(
            connection, file_path,
            names=[], kinds=[], prop_strs=[], line_numbers=[],
            file_hash=file_hash, orig_names=names, orig_kinds=kinds, uncached=[],
        ) as task:
            cum_input_tokens, cum_output_tokens, cum_cost_usd = task.historical_cost()

    return ([results.get(i) for i in range(n)],
            pretty_prints,
            (total_input_tokens, total_output_tokens, total_cost_usd),
            (cum_input_tokens, cum_output_tokens, cum_cost_usd))
