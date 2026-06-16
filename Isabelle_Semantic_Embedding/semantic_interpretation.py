"""Semantic interpretation of Isabelle constants and theorems via Claude Code agent."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from pathlib import Path
from collections.abc import Iterable
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
try:
    from claude_agent_sdk import RateLimitEvent
except ImportError:
    RateLimitEvent = None
from claude_agent_sdk.types import (
    HookInput,
    HookContext,
    HookJSONOutput,
    PreToolUseHookInput,
    ResultMessage,
)

from .base import ToolCall_ret, mk_ret as _mk_ret
from .desugar import mk_desugar_and_explain_tool
from .semantics import Provenance, Semantic_DB, SemanticRecord, unpack_thy_status

# --- Module-level configuration ---

interpretation_model: str = "claude-opus-4-8[1m]"
"""LLM model used for semantic interpretation. Set before calling interpret_file."""

# --- Context-local state ---

_KIND_CONSTANT = 1
_KIND_THEOREM = 2
_KIND_TYPE = 3
_KIND_CLASS = 4
_KIND_LOCALE = 5
_KIND_THEOREM_COLLECTION = 6
_KIND_METHOD = 7
_KIND_INTRODUCTION_RULE = 0x12
_KIND_ELIMINATION_RULE = 0x22
_KIND_INDUCTION_RULE = 0x32
_KIND_CASE_SPLIT_RULE = 0x42
# NB: these label strings must stay identical to the `type` enum in
# _answer_schema below — the agent echoes the label back and it is matched
# against the keys built from _KIND_PROMPT_LABELS; any drift silently drops
# every answer for that kind ("Unknown entry").
_KIND_PROMPT_LABELS = {
    _KIND_CONSTANT: "constant",
    _KIND_THEOREM: "lemma",
    _KIND_TYPE: "type",
    _KIND_CLASS: "typeclass",
    _KIND_LOCALE: "locale",
    _KIND_THEOREM_COLLECTION: "named theorem bundles",
    _KIND_METHOD: "proof method",
    _KIND_INTRODUCTION_RULE: "introduction rule",
    _KIND_ELIMINATION_RULE: "elimination rule",
    _KIND_INDUCTION_RULE: "induction rule",
    _KIND_CASE_SPLIT_RULE: "case-split rule",
}

# Module-load invariants for the agent addressing scheme (see `_label`).  The
# ML side (Universal_Key.entity_kind_int) and this dict must agree: every
# interpretable entity kind needs exactly one title, and titles must be
# injective — the agent echoes the title back to address an entry, so two kinds
# sharing a title would make answers ambiguous, and a missing kind would route
# to the dead "unknown" branch.  Asserting here makes such drift fail fast at
# import rather than only on a coincidental runtime collision.  THEORY (0) is
# excluded: theory entities are never interpreted.
assert len(set(_KIND_PROMPT_LABELS.values())) == len(_KIND_PROMPT_LABELS), (
    "_KIND_PROMPT_LABELS titles must be injective (the agent addresses entries "
    "by title + name)")
assert set(_KIND_PROMPT_LABELS) == {
    k.value for k in EntityKind if k is not EntityKind.THEORY
}, ("_KIND_PROMPT_LABELS keys must cover exactly the interpretable EntityKind "
    "ints (all kinds except THEORY); it has drifted from Universal_Key.entity_kind_int")

_BATCH_SIZE = 20

# Standing instructions for the interpretation agent.  These are batch- and
# file-independent, so they live in the system prompt rather than the per-batch
# user messages: the system prompt is re-sent verbatim on every request and is
# never folded into a context-compaction summary, so the agent keeps the full
# translation spec even after the original batch prompts have been compacted
# away.  Only the per-batch work (which theory/file, which entries) stays in
# the user messages built by `build_prompt`.
_SYSTEM_PROMPT = """\
You informalize entities from Isabelle theory files: you translate each formal \
statement you are given into a thorough, self-contained plain-English description.

For each entry, aim for 2–5 sentences. \
State only what the entity defines or asserts. \
Do NOT explain how it is derived or why it is useful. \
The formal statement is already shown; describe its meaning **rather than** transcribing it. \
Prefer plain English over formulas. Wrap formulas in backticks (e.g., `x`, `x + 1`). \
When a lemma/rule/term has a well-known name (e.g., proof by contradiction), you MUST mention it explicitly in the translation. \
Every translation must be **self-contained**: assume the reader has no prior context and knows no notation. \
Do not assume they know what any symbol means — for instance, do not assume they know that `x # l` prepends `x` \
to the list `l`; spell out such notation wherever you use it. \
Make sure that every nonstandard notion has been clearly explained somewhere in each of your translations. \
Be thorough rather than terse: fully unfold what the statement means — name and explain every variable, symbol, \
and sub-expression it involves — instead of compressing it into a single line (still without explaining its \
derivation or usefulness).

- For a `named theorem bundles` entry, describe what kind of facts the collection gathers and its purpose; \
you may use the listed current members to infer this, but do NOT enumerate the members in your answer. \
The declared comment in the command (if any) is often terse, inaccurate, or incomplete, so check it \
against the members: copy it verbatim only when it is genuinely complete and accurate, otherwise \
correct and expand it into a full description using the members. \
- For a `proof method` entry, describe the proof strategy or tactic it performs, when it should be used, \
and what kinds of proof goals it is meant to solve; if its description is empty, \
draw on the surrounding context and its uses in other files to learn what it does.

Line numbers in brackets (e.g. [line 42]) indicate where each entity appears in the source file.

Examples of good translations:
- constant Nat.add: The addition operator on natural numbers, taking two natural numbers and returning their sum.
- lemma List.length_append: The length of the concatenation of two lists equals the sum of their individual lengths.
- lemma List.map_comp: Mapping `f` then `g` over a list is the same as mapping their composition `g ∘ f`.
- type Prod: The product type, consisting of a pair of two values of possibly different types.
- introduction rule notI `(P ⟹ False) ⟹ ¬P`: The rule of proof by contradiction — to prove `¬P`, assume `P` and derive `False`.
- named theorem bundles Groups.algebra_simps: A collection of rewrite rules that normalise expressions over groups, rings and related structures — multiplying products out and ordering sums and products into a canonical form — so the simplifier can decide algebraic equalities and help discharge inequalities.
- proof method Presburger.presburger: An automatic decision procedure for first-order linear arithmetic over integers and naturals (Presburger arithmetic) — it eliminates quantifiers and handles divisibility and modulo constraints via Cooper's algorithm.

Translation hints:
- Suc n → "the successor of n" or "n + 1"

When you encounter an entity whose meaning is unclear, use `mcp__isabelle_semantics__query`, \
`mcp__isabelle_semantics__hover`, or \
`mcp__isabelle_semantics__definition` to look it up before translating. \
However, you cannot query entries you have been asked to translate — do it yourself.

Submit all translations via `mcp__isabelle_semantics__answer`."""


def _label(e: Entry) -> str:
    """The agent-facing addressing label for an entry: kind-title + name.

    This is the ONLY handle the agent has to address an entry (it echoes
    ``{type, name}`` back through the ``answer`` tool).  It must be IDENTICAL
    everywhere it is formed — the prompt the agent reads (`format_entries` /
    `_pretty_print_entry`), the `results` key, and the answer-routing map — so
    the label the agent echoes round-trips to the right entry.  Use `.get(...,
    "unknown")` (not `[...]`) so the key matches what `format_entries` shows."""
    return f"{_KIND_PROMPT_LABELS.get(e.kind, 'unknown')} {e.name}"


def _pretty_print_entry(e: Entry) -> str:
    pp = _label(e)
    if e.prop_str:
        pp += f": {e.prop_str}"
    return pp


class Entry(NamedTuple):
    """A single entity to interpret."""
    kind: int            # _KIND_CONSTANT, _KIND_THEOREM, etc.
    name: str            # fully qualified name (Unicode)
    prop_str: str        # printed proposition / type signature (Unicode); stored as expr
    line_number: int     # source line (-1 if unavailable)
    universal_key: universal_key
    prompt_extra: str = ""  # extra context shown to the agent only, NOT stored as expr
                            # (e.g. current members of a named_theorems collection,
                            # or locale-interpretation provenance)
    # locale-interpretation provenance (None for ordinary entries); stored
    # alongside the interpretation in the semantic DB
    locale_provenance: "Provenance | None" = None
    # constituent theories of theorem/rule entities — sorted (theory long
    # name, 16-byte theory hash) list whose XOR is the key's theory prefix;
    # None for non-theorem kinds.  Stored in the semantic DB record.
    theory_constituents: "list[tuple[str, bytes]] | None" = None


class CostSummary(NamedTuple):
    """Token usage and dollar cost for an interpretation run."""
    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
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
        # results / _keys / _label_to_idx are strictly 1:1 with `entries`, in
        # order.  Because the agent addresses an entry only by its label (see
        # `_label` and `_answer_tool`), two entries sharing a label are mutually
        # un-addressable; a label-keyed dict comprehension would SILENTLY
        # collapse them, desyncing _keys vs entries and mis-routing write_answer
        # (the "name != content" LMDB corruption).  Build it with a loop that
        # RAISES on the first duplicate instead — the ML side
        # (Semantic_Store, (entity-kind, name) assert) guarantees uniqueness, so
        # this only ever fires on a genuine regression.
        self.results: dict[str, str | None] = {}
        self._label_to_idx: dict[str, int] = {}
        for i, e in enumerate(entries):
            key = _label(e)
            if key in self.results:
                j = self._label_to_idx[key]
                raise ValueError(
                    f"duplicate interpretation label {key!r} at entries {j} and {i} "
                    f"(uks {bytes(entries[j].universal_key).hex()} and "
                    f"{bytes(e.universal_key).hex()}); (kind,name) labels must be "
                    f"unique to be addressable by the agent")
            self.results[key] = None
            self._label_to_idx[key] = i
        self._keys = list(self.results.keys())
        self.batches: list[tuple[str, range]] = []
        self.current_batch: int = 0
        self.batch_range: range = range(0)
        # `total_*` is the pending delta not yet flushed to LMDB; write_cost()
        # accumulates it into the theory record and resets it to 0.  Cost is
        # flushed per agent round (see _accumulate_usage), mirroring how answers
        # are written per-answer (write_answer) — so an interrupt (the parallel
        # scheduler's by-design hard-crash) cannot drop cost for answers that
        # are already cached.
        self.total_input_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        # `run_*` is the cumulative cost of THIS interpret_file invocation; it is
        # never reset by write_cost(), so it survives the per-round flushes and
        # is reported as `current_cost`.
        self.run_input_tokens = 0
        self.run_cache_creation_tokens = 0
        self.run_cache_read_tokens = 0
        self.run_output_tokens = 0
        self.run_cost_usd = 0.0

    def __enter__(self) -> InterpretationTask:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def write_answer(self, task_idx: int, sem: str) -> None:
        """Write a single answer to the LMDB store."""
        entry = self.entries[task_idx]
        Semantic_DB[entry.universal_key] = SemanticRecord(
            EntityKind(entry.kind), entry.name, entry.prop_str, sem,
            entry.locale_provenance, entry.theory_constituents)

    def historical_cost(self) -> tuple[int, int, int, int, float]:
        """Read cumulative cost from LMDB (without modifying it)."""
        env = Semantic_DB._ensure_env()
        with env.begin() as txn:
            raw = txn.get(self.theory_key)
        if raw is None:
            return (0, 0, 0, 0, 0.0)
        prev = unpack_thy_status(raw)
        return (prev.get(b"input_tokens", 0),
                prev.get(b"cache_creation_tokens", 0),
                prev.get(b"cache_read_tokens", 0),
                prev.get(b"output_tokens", 0),
                prev.get(b"cost_usd", 0.0))

    def write_cost(self) -> tuple[int, int, int, int, float]:
        """Accumulate cost into the LMDB store. Returns updated cumulative totals."""
        import msgpack
        env = Semantic_DB._ensure_env()
        with env.begin(write=True) as txn:
            raw = txn.get(self.theory_key)
            prev_in, prev_cw, prev_cr, prev_out, prev_cost, finished = 0, 0, 0, 0, 0.0, False
            if raw is not None:
                prev = unpack_thy_status(raw)
                prev_in = prev.get(b"input_tokens", 0)
                prev_cw = prev.get(b"cache_creation_tokens", 0)
                prev_cr = prev.get(b"cache_read_tokens", 0)
                prev_out = prev.get(b"output_tokens", 0)
                prev_cost = prev.get(b"cost_usd", 0.0)
                finished = prev.get(b"finished", False)
            total = (prev_in + self.total_input_tokens,
                     prev_cw + self.total_cache_creation_tokens,
                     prev_cr + self.total_cache_read_tokens,
                     prev_out + self.total_output_tokens,
                     prev_cost + self.total_cost_usd)
            packed: bytes = msgpack.packb({  # type: ignore[assignment]
                b"input_tokens": total[0],
                b"cache_creation_tokens": total[1],
                b"cache_read_tokens": total[2],
                b"output_tokens": total[3],
                b"cost_usd": total[4],
                b"finished": finished,
                b"model": interpretation_model,
            })
            txn.put(self.theory_key, packed)
        self.total_input_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        return total

    def advance_batch(self) -> str | None:
        self.current_batch += 1
        if self.current_batch >= len(self.batches):
            return None
        prompt, self.batch_range = self.batches[self.current_batch]
        return prompt

    def format_entries(self, indices: Iterable[int]) -> str:
        lines = []
        for i in indices:
            e = self.entries[i]
            label = _KIND_PROMPT_LABELS.get(e.kind, "unknown")
            line = (f"  [line {e.line_number}] " if e.line_number > 0 else "  ") + f"{label} {e.name}"
            if e.prop_str:
                line += f": {e.prop_str}"
            if e.prompt_extra:
                # indent the extra context block under the entry line
                line += "\n    " + e.prompt_extra.replace("\n", "\n    ")
            lines.append(line)
        return "\n".join(lines)

    def build_prompt(self, file_path: str, theory_longname: str, indices: range | None = None) -> str:
        if indices is None:
            indices = range(len(self.entries))
        entries_text = self.format_entries(indices)

        if indices.start != 0:
            return (
                f'Continue with the following entities from Isabelle theory "{theory_longname}" (location: {file_path}).\n\n'
                f"Entries:\n{entries_text}\n\n"
                f"Submit translations via `mcp__isabelle_semantics__answer`."
            )

        return (
            f"Load the skills `isabelle-intro-elim-rules`, `isabelle-datatype`, and `isabelle-record`.\n"
            f'Informalize the following entities from Isabelle theory "{theory_longname}" (location: {file_path}).\n\n'
            f"Entries:\n{entries_text}\n\n"
            f"Submit translations via `mcp__isabelle_semantics__answer`."
        )

_local_task: contextvars.ContextVar[InterpretationTask] = contextvars.ContextVar('_local_task')


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
                                 "named theorem bundles", "proof method",
                                 "introduction rule", "elimination rule",
                                 "induction rule", "case-split rule"]
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
    "You may also resubmit an entry to correct a previous answer. "
    "To see the remaining unanswered entries in the current batch, call this tool with an empty list [].",
    input_schema=_answer_schema,
)
async def _answer_tool(args: dict[str, Any]) -> ToolCall_ret:
    task = _local_task.get()
    interpretations = args["interpretations"]
    errors = []
    count = 0
    for item in interpretations:
        key = f"{item['type']} {item['name']}"
        if key not in task.results:
            errors.append(f"Unknown entry: {key!r}")
            continue
        task.results[key] = item["translation"]
        # Address by the precomputed label->entry-index map (O(1), and indexes
        # the FULL `entries` list correctly).  The old `_keys.index(key)` indexed
        # the deduped key list against the full entries list — the misalignment
        # that wrote translations onto neighbouring entries' universal_keys.
        task.write_answer(task._label_to_idx[key], item["translation"])
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
        remaining_indices = [i for i in task.batch_range if task.results[task._keys[i]] is None]
        remaining_text = task.format_entries(remaining_indices)
        msg = (f"Answered {count} translation{cs}, remaining {batch_remaining} in this batch.\n\n"
               f"Unanswered entries:\n{remaining_text}\n"
               f"In file: {task.file_path}\n\n"
               f"Submit translations via `mcp__isabelle_semantics__answer`.")
    if errors:
        msg += "\nErrors:\n" + "\n".join(errors)
    return _mk_ret(msg)


# --- Permission control ---

_TOOL_WHITELIST = {
    "Read",
    "Grep",
    "Glob",
    "LS",
    "Bash",
    "Skill",
    "Agent",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "TaskOutput",
    "TaskStop",
    "WebFetch",
    "WebSearch",
    "ExitPlanMode",
    "MCPSearch",
    "ToolSearch",
    "mcp__isabelle_semantics__query",
    "mcp__isabelle_semantics__definition",
    "mcp__isabelle_semantics__hover",
    "mcp__isabelle_semantics__answer",
    "mcp__isabelle_semantics__desugar_and_explain",
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


def _accumulate_usage(task: InterpretationTask, message: ResultMessage) -> None:
    if message.usage:
        d_in = message.usage.get("input_tokens", 0)
        d_cw = message.usage.get("cache_creation_input_tokens", 0)
        d_cr = message.usage.get("cache_read_input_tokens", 0)
        d_out = message.usage.get("output_tokens", 0)
    else:
        d_in = d_cw = d_cr = d_out = 0
    d_cost = message.total_cost_usd or 0.0
    # pending delta to be flushed into the theory record this round
    task.total_input_tokens += d_in
    task.total_cache_creation_tokens += d_cw
    task.total_cache_read_tokens += d_cr
    task.total_output_tokens += d_out
    task.total_cost_usd += d_cost
    # run-level accumulator (never reset by write_cost) — reported as current_cost
    task.run_input_tokens += d_in
    task.run_cache_creation_tokens += d_cw
    task.run_cache_read_tokens += d_cr
    task.run_output_tokens += d_out
    task.run_cost_usd += d_cost
    _log.info("round usage: %s, cost: $%.4f", message.usage, message.total_cost_usd or 0)
    # Flush immediately so an interrupt cannot drop this round's cost: answers are
    # persisted per-answer (write_answer), and on resume they become free cache
    # hits, so their cost must be persisted just as eagerly.
    task.write_cost()


class ReachLimitError(Exception):
    """Usage cap hit (e.g. 'You've hit your limit')."""
    pass

class RateLimitError(Exception):
    """API rate limit (429)."""
    pass

async def _run_agent(options: ClaudeAgentOptions) -> None:
    task = _local_task.get()
    first_prompt, task.batch_range = task.batches[0]
    try:
        async with ClaudeSDKClient(options=options) as client:
            _log.info("agent: starting batch 0 (%d–%d)",
                    task.batch_range.start, task.batch_range.stop - 1)
            await client.query(first_prompt)
            async for message in client.receive_response():
                if RateLimitEvent is not None and isinstance(message, RateLimitEvent):
                    if message.rate_limit_info.status == "rejected":
                        raise ReachLimitError()
                    continue
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
                    _accumulate_usage(task, message)
                    if message.is_error and message.result:
                        if message.result.startswith("You've hit your limit"):
                            raise ReachLimitError()
                        if "Rate limit" in message.result:
                            raise RateLimitError()
            # Retry any globally missing entries.  Re-send the full entry text
            # (line, kind, name, proposition) via format_entries, NOT bare names:
            # after context compaction the original batch prompts are gone, and
            # for facts generated by locale interpretations the proposition is
            # unrecoverable from the source file — a names-only list forces the
            # agent to answer from memory, inviting mispaired translations.
            # Retries go in _BATCH_SIZE chunks: a pathological run can leave
            # hundreds of entries unanswered, and a single message carrying all
            # their propositions + provenance hints would dwarf the normal
            # batch prompts and immediately re-trigger compaction.
            while True:
                missing_idx = [i for i, k in enumerate(task._keys)
                               if task.results[k] is None]
                if not missing_idx:
                    break
                chunk = missing_idx[:_BATCH_SIZE]
                _log.info("agent: retrying %d of %d missing entries",
                          len(chunk), len(missing_idx))
                missing_text = task.format_entries(chunk)
                header = (
                    f"You still have {len(missing_idx)} unanswered entries from theory "
                    f'"{task.theory_longname}" (location: {task.file_path})'
                    + (f"; here are the first {len(chunk)}"
                       if len(missing_idx) > len(chunk) else "")
                )
                await client.query(
                    f"{header}:\n"
                    f"{missing_text}\n\n"
                    f"Submit their translations via the `mcp__isabelle_semantics__answer` tool. "
                    f"Each translation must describe the formal statement shown above next to that exact name."
                )
                async for message in client.receive_response():
                    _log_message(message)
                    if isinstance(message, ResultMessage):
                        _accumulate_usage(task, message)
        _log.info("total usage: input=%d cache_write=%d cache_read=%d output=%d tokens, cost=$%.4f",
                task.total_input_tokens, task.total_cache_creation_tokens,
                task.total_cache_read_tokens, task.total_output_tokens, task.total_cost_usd)
    except ReachLimitError:
        _log.info("agent: reached usage limit, waiting 20min to retry")
        await asyncio.sleep(1200)
        return await _run_agent(options)
    except RateLimitError:
        _log.info("agent: API rate limit, waiting 2s to retry")
        await asyncio.sleep(2)
        return await _run_agent(options)


# --- Public API ---

async def interpret_file(
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
        rec = Semantic_DB[e.universal_key]
        if rec is not None and rec.interpretation is not None:
            results[i] = rec.interpretation
            if e.prop_str and rec.expr != e.prop_str:
                Semantic_DB.update_expr(e.universal_key, e.prop_str)

    uncached = [i for i, r in enumerate(results) if r is None]
    _log.info("interpret_file: %d cached, %d to interpret", n - len(uncached), len(uncached))
    current_cost = CostSummary(0, 0, 0, 0, 0.0)
    cumulative_cost = CostSummary(0, 0, 0, 0, 0.0)

    if uncached:
        from .hover import mk_definition_tool, mk_hover_tool
        from .semantics import mk_query_by_name_tool
        from .theory_structure import mk_unicode_file

        unicode_file_path = mk_unicode_file(file_path)

        with InterpretationTask(
            connection, file_path, theory_longname, theory_key,
            entries=[entries[i] for i in uncached],
        ) as task:
            _local_task.set(task)

            m = len(task.entries)

            for start in range(0, m, _BATCH_SIZE):
                batch_range = range(start, min(start + _BATCH_SIZE, m))
                task.batches.append((
                    task.build_prompt(unicode_file_path, theory_longname, batch_range),
                    batch_range,
                ))

            working_names = [e.name for e in task.entries]
            seen_constants: set[str] = set()
            query_by_name_tool = mk_query_by_name_tool(
                connection, working_names, file_path=file_path)
            definition_tool = mk_definition_tool(connection, unicode=True)
            hover_tool = mk_hover_tool(connection, unicode=True)
            desugar_tool = mk_desugar_and_explain_tool(
                connection, file_path=file_path, seen_constants=seen_constants)
            mcp = create_sdk_mcp_server("isabelle_semantics", tools=[
                query_by_name_tool,
                definition_tool, hover_tool, desugar_tool, _answer_tool])

            async def _on_compact(
                hook_input: HookInput,
                tool_use_id: str | None,
                context: HookContext,
            ) -> HookJSONOutput:
                seen_constants.clear()
                return {}

            options = ClaudeAgentOptions(
                model=interpretation_model,
                system_prompt=_SYSTEM_PROMPT,
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
                    ],
                    "PreCompact": [
                        HookMatcher(matcher="*", hooks=[_on_compact]),
                    ],
                },
            )

            _log.info("interpret_file: starting agent with %d batches", len(task.batches))
            await _run_agent(options)
            answered = sum(1 for v in task.results.values() if v is not None)
            _log.info("interpret_file: agent finished, %d/%d interpreted",
                       answered, len(task.entries))
            # Cost is flushed per round in _accumulate_usage, so this is normally
            # a no-op flush; it still returns the up-to-date cumulative totals.
            cum = task.write_cost()
            # current_cost = cost of THIS run; read from the run-level
            # accumulator (write_cost resets total_*, but never run_*).
            current_cost = CostSummary(
                task.run_input_tokens, task.run_cache_creation_tokens,
                task.run_cache_read_tokens, task.run_output_tokens,
                task.run_cost_usd)
            cumulative_cost = CostSummary(*cum)

            # Remap agent results to original indices (cache already written
            # incrementally). Iterate _keys by position — it is 1:1 with
            # task.entries and with `uncached` — instead of relying on
            # results.values() insertion order.
            for i, key in enumerate(task._keys):
                sem = task.results[key]
                if sem is not None:
                    results[uncached[i]] = sem
    else:
        # All cached — read cumulative cost from DB
        with InterpretationTask(
            connection, file_path, theory_longname, theory_key,
            entries=[],
        ) as task:
            cumulative_cost = CostSummary(*task.historical_cost())

    return InterpretationResult(
        results,
        pretty_prints,
        current_cost,
        cumulative_cost,
    )


# --- RPC shim ---

@isabelle_remote_procedure("Semantic_Store.interpret_file")
async def _interpret_file(arg: Any, connection: Connection) -> InterpretationResult:
    from Isabelle_RPC_Host.universal_key import THM_RULE_KINDS
    (file_path, theory_longname, theory_key, raw_entries) = arg
    entries = [
        Entry(
            kind=kind,
            name=pretty_unicode(name),
            prop_str=pretty_unicode(prop),
            line_number=lineno,
            universal_key=bytes(uk),
            prompt_extra=pretty_unicode(hint),
            locale_provenance=(Provenance(
                template_uk=bytes(prov[0]) if prov[0] is not None else None,
                locale_uk=bytes(prov[1]) if prov[1] is not None else None,
                qualifier=prov[2],
            ) if prov is not None else None),
            theory_constituents=(
                [(n, bytes(h)) for n, h in consts]
                if EntityKind(kind) in THM_RULE_KINDS else None),
        )
        for kind, name, prop, lineno, uk, hint, prov, consts in raw_entries
    ]
    return await interpret_file(
        connection, file_path, theory_longname, bytes(theory_key), entries
    )

