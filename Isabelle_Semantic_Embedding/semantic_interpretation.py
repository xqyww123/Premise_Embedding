"""Semantic interpretation of Isabelle constants and theorems via Claude Code agent."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import threading
from typing import Any, NamedTuple, cast

from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from Isabelle_RPC_Host.universal_key import EntityKind, universal_key
from Isabelle_RPC_Host.unicode import pretty_unicode
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ThinkingConfigAdaptive,
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

from .base import ToolCall_ret, mk_ret as _mk_ret
from .semantics import Semantic_DB, SemanticRecord

# --- Thread-local state ---

_KIND_CONSTANT = 1
_KIND_THEOREM = 2
_KIND_TYPE = 3
_KIND_CLASS = 4
_KIND_LOCALE = 5
_KIND_INTRODUCTION_RULE = 0x12
_KIND_ELIMINATION_RULE = 0x22
_KIND_PROMPT_LABELS = {
    _KIND_CONSTANT: "constant",
    _KIND_THEOREM: "lemma",
    _KIND_TYPE: "type",
    _KIND_CLASS: "typeclass",
    _KIND_LOCALE: "locale",
    _KIND_INTRODUCTION_RULE: "introduction rule",
    _KIND_ELIMINATION_RULE: "elimination rule",
}
_BATCH_SIZE = 20


def _pretty_print_entry(e: Entry) -> str:
    label = _KIND_PROMPT_LABELS.get(e.kind, "unknown")
    pp = f"{label} {e.name}"
    if e.prop_str:
        pp += f": {e.prop_str}"
    return pp


class Entry(NamedTuple):
    """A single entity to interpret."""
    kind: int            # _KIND_CONSTANT, _KIND_THEOREM, etc.
    name: str            # fully qualified name (Unicode)
    prop_str: str        # printed proposition / type signature (Unicode)
    line_number: int     # source line (-1 if unavailable)
    universal_key: universal_key


class CostSummary(NamedTuple):
    """Token usage and dollar cost for an interpretation run."""
    input_tokens: int
    output_tokens: int
    cost_usd: float


class InterpretationResult(NamedTuple):
    """Result of interpreting a theory file."""
    interpretations: list[str | None]   # per-entry semantic interpretation (None if unanswered)
    pretty_prints: list[str]            # per-entry Unicode pretty-print
    current_cost: CostSummary           # cost incurred in this call
    cumulative_cost: CostSummary        # total cost including historical


class InterpretationTask:
    def __init__(self, connection: Connection, file_path: str,
                 theory_longname: str, theory_key: universal_key,
                 entries: list[Entry]):
        self.connection = connection
        self.file_path = file_path
        self.theory_longname = theory_longname
        self.theory_key = theory_key
        self.entries = entries
        self.results: dict[str, str | None] = {
            f"{_KIND_PROMPT_LABELS[e.kind]} {e.name}": None
            for e in entries
        }
        self._keys = list(self.results.keys())
        self.batches: list[tuple[str, range]] = []
        self.current_batch: int = 0
        self.batch_range: range = range(0)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def __enter__(self) -> InterpretationTask:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def write_answer(self, task_idx: int, sem: str) -> None:
        """Write a single answer to the LMDB store."""
        entry = self.entries[task_idx]
        Semantic_DB[entry.universal_key] = SemanticRecord(
            EntityKind(entry.kind), entry.name, entry.prop_str, sem)

    def historical_cost(self) -> tuple[int, int, float]:
        """Read cumulative cost from LMDB (without modifying it)."""
        import msgpack
        env = Semantic_DB._ensure_env()
        with env.begin() as txn:
            raw = txn.get(self.theory_key)
        if raw is None:
            return (0, 0, 0.0)
        prev = msgpack.unpackb(raw)
        return (prev.get(b"input_tokens", 0),
                prev.get(b"output_tokens", 0),
                prev.get(b"cost_usd", 0.0))

    def write_cost(self) -> tuple[int, int, float]:
        """Accumulate cost into the LMDB store. Returns updated cumulative totals."""
        import msgpack
        env = Semantic_DB._ensure_env()
        with env.begin(write=True) as txn:
            raw = txn.get(self.theory_key)
            prev_in, prev_out, prev_cost, finished = 0, 0, 0.0, False
            if raw is not None:
                prev = msgpack.unpackb(raw)
                prev_in = prev.get(b"input_tokens", 0)
                prev_out = prev.get(b"output_tokens", 0)
                prev_cost = prev.get(b"cost_usd", 0.0)
                finished = prev.get(b"finished", False)
            total = (prev_in + self.total_input_tokens,
                     prev_out + self.total_output_tokens,
                     prev_cost + self.total_cost_usd)
            packed: bytes = msgpack.packb({  # type: ignore[assignment]
                "input_tokens": total[0],
                "output_tokens": total[1],
                "cost_usd": total[2],
                "finished": finished,
            })
            txn.put(self.theory_key, packed)
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
            e = self.entries[i]
            label = _KIND_PROMPT_LABELS.get(e.kind, "unknown")
            line = (f"  [line {e.line_number}] " if e.line_number > 0 else "  ") + f"{label} {e.name}"
            if e.prop_str:
                line += f": {e.prop_str}"
            lines.append(line)
        return "\n".join(lines)

    def build_prompt(self, file_path: str, theory_longname: str, indices: range | None = None) -> str:
        if indices is None:
            indices = range(len(self.entries))
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
Load the skill `isabelle-intro-elim-rules`.
Informalize the following entities from Isabelle theory "{theory_longname}" (location: {file_path})

Entries:
{entries_text}

Line numbers in brackets (e.g. [line 42]) indicate where each entity appears in the source file.

For each entry, translate the formal statement into a concise plain-English description (1\u20133 sentences). \
State only what the entity defines or asserts. \
Do NOT explain how it is derived or why it is useful. \
The formal statement is already shown; describe its meaning **rather than** transcribing it. \
Prefer plain English over formulas. Wrap formulas in backticks (e.g., `x`, `x + 1`). \
When a lemma/rule/term has a well-known name (e.g., proof by contradiction), you MUST mention it explicitly in the translation.

Examples of good translations:
- constant Nat.add: The addition operator on natural numbers, taking two natural numbers and returning their sum.
- lemma List.length_append: The length of the concatenation of two lists equals the sum of their individual lengths.
- lemma List.map_comp: Mapping `f` then `g` over a list is the same as mapping their composition `g \u2218 f`.
- type Prod: The product type, consisting of a pair of two values of possibly different types.
- introduction rule notI `(P \u27f9 False) \u27f9 \u00acP`: The rule of proof by contradiction \u2014 to prove `\u00acP`, assume `P` and derive `False`.

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
                        "enum": ["constant", "lemma", "type", "typeclass", "locale",
                                 "introduction rule", "elimination rule"]
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
            msg = "All done! If you noticed any mistakes in your earlier translations, correct them now using `mcp__isabelle_semantics__answer`. Otherwise, stop immediately without any further output."
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


class ReachLimitError(Exception):
    """Usage cap hit (e.g. 'You've hit your limit')."""
    pass

class RateLimitError(Exception):
    """API rate limit (429)."""
    pass

async def _run_agent(options: ClaudeAgentOptions) -> None:
    task = _local.task
    first_prompt, task.batch_range = task.batches[0]
    try:
        async with ClaudeSDKClient(options=options) as client:
            _log.info("agent: starting batch 0 (%d–%d)",
                    task.batch_range.start, task.batch_range.stop - 1)
            await client.query(first_prompt)
            async for message in client.receive_response():
                _log_message(message)
                content = getattr(message, "content", None)
                if content is not None and isinstance(content, list) and content:
                    block = content[0]
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        if text.startswith("You've hit your limit"):
                            raise ReachLimitError()
                        if "Rate limit" in text:
                            raise RateLimitError()
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
    except ReachLimitError:
        _log.info("agent: reached usage limit, waiting 20min to retry")
        await asyncio.sleep(1200)
        return await _run_agent(options)
    except RateLimitError:
        _log.info("agent: API rate limit, waiting 2s to retry")
        await asyncio.sleep(2)
        return await _run_agent(options)


# --- Public API ---

def interpret_file(
    connection: Connection,
    file_path: str,
    theory_longname: str,
    theory_key: universal_key,
    entries: list[Entry],
) -> InterpretationResult:
    """Interpret entities from an Isabelle theory file.

    Looks up cached interpretations in LMDB. For uncached entries, launches
    a Claude agent to generate plain-English translations.

    Args:
        connection: Active Isabelle RPC connection.
        file_path: Path to the theory source file.
        theory_longname: Fully qualified theory name (e.g. "HOL.List").
        theory_key: Universal key for the theory (used for cost tracking).
        entries: Entities to interpret, each with kind, name, prop_str,
            line_number, and universal_key.

    Returns:
        InterpretationResult with per-entry interpretations, pretty-prints,
        and cost summaries (current run + cumulative).
    """
    n = len(entries)

    # Build Unicode pretty-prints for all entries
    pretty_prints = [_pretty_print_entry(e) for e in entries]

    # Inherit RPC server's logging configuration (idempotent, no race)
    if not _log.handlers and connection.server.logger.handlers:
        for h in connection.server.logger.handlers:
            _log.addHandler(h)
        _log.setLevel(connection.server.logger.level)
    _log.info("interpret_file: %s (%s), %d entries", theory_longname, file_path, n)

    # Check LMDB cache
    results: list[str | None] = [None] * n
    for i, e in enumerate(entries):
        sem = Semantic_DB.query(e.universal_key)
        if sem is not None:
            results[i] = sem

    uncached = [i for i, r in enumerate(results) if r is None]
    _log.info("interpret_file: %d cached, %d to interpret", n - len(uncached), len(uncached))
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
            connection, file_path, theory_longname, theory_key,
            entries=[entries[i] for i in uncached],
        ) as task:
            _local.task = task

            m = len(task.entries)

            for start in range(0, m, _BATCH_SIZE):
                batch_range = range(start, min(start + _BATCH_SIZE, m))
                task.batches.append((
                    task.build_prompt(unicode_file_path, theory_longname, batch_range),
                    batch_range,
                ))

            working_names = [e.name for e in task.entries]
            query_by_name_tool = mk_query_by_name_tool(
                connection, working_names)
            query_by_position_tool = mk_query_by_position_tool(
                connection, working_names, unicode=True)
            definition_tool = mk_definition_tool(connection, unicode=True)
            hover_tool = mk_hover_tool(connection, unicode=True)
            mcp = create_sdk_mcp_server("isabelle_semantics", tools=[
                query_by_name_tool, query_by_position_tool,
                definition_tool, hover_tool, _answer_tool])
            options = ClaudeAgentOptions(
                model="claude-opus-4-6",
                cwd=str(Path(__file__).parent / "Agent_Interpretation_Dir"),
                setting_sources=["project"],
                permission_mode="default",
                allowed_tools=list(_TOOL_WHITELIST),
                mcp_servers={"isabelle_semantics": mcp},
                thinking=ThinkingConfigAdaptive(type="adaptive"),
                effort="high",
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
                       answered, len(task.entries))
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
            connection, file_path, theory_longname, theory_key,
            entries=[],
        ) as task:
            cum_input_tokens, cum_output_tokens, cum_cost_usd = task.historical_cost()

    return InterpretationResult(
        results,
        pretty_prints,
        CostSummary(total_input_tokens, total_output_tokens, total_cost_usd),
        CostSummary(cum_input_tokens, cum_output_tokens, cum_cost_usd),
    )


# --- RPC shim ---

@isabelle_remote_procedure("Semantic_Store.interpret_file")
def _interpret_file(arg: Any, connection: Connection) -> InterpretationResult:
    (file_path, theory_longname, theory_key, raw_entries) = arg
    entries = [
        Entry(
            kind=kind,
            name=pretty_unicode(name),
            prop_str=pretty_unicode(prop),
            line_number=lineno,
            universal_key=bytes(uk),
        )
        for kind, name, prop, lineno, uk in raw_entries
    ]
    return interpret_file(
        connection, file_path, theory_longname, bytes(theory_key), entries
    )

