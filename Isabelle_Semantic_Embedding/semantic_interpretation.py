"""Semantic interpretation of Isabelle constants and theorems via Claude Code agent."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

import platformdirs
import xxhash
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)
from rocksdict import Options, Rdict
import IsaREPL

# --- Thread-local state ---

_KIND_CONSTANT = 0
_KIND_THEOREM = 1
_KIND_PROMPT_LABELS = {_KIND_CONSTANT: "constant", _KIND_THEOREM: "lemma"}

class _LocalState(threading.local):
    connection: Connection
    results: dict[int, str]       # index -> semantics
    names: list[str]
    kinds: list[int]              # _KIND_CONSTANT or _KIND_THEOREM

_local = _LocalState()


# --- Persistent cache ---

def _open_cache() -> Rdict:
    cache_dir = platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")
    os.makedirs(cache_dir, exist_ok=True)
    return Rdict(os.path.join(cache_dir, "semantics.db"),
                 options=Options(raw_mode=True))

def _cache_key(file_hash: bytes, name: str) -> bytes:
    return file_hash + b":" + name.encode("utf-8")


# --- MCP tool return helper ---

type ToolCall_ret = dict[str, Any]

def _mk_ret(text: str) -> ToolCall_ret:
    return {"content": [{"type": "text", "text": text}]}


# --- MCP Tool 1: query_semantics ---

_query_schema = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["constant", "theorem"],
            "description": "The kind of entity to query: 'constant' or 'theorem'.",
        },
        "name": {
            "type": "string",
            "description": "The name of the constant or theorem to look up. "
            "For multi-variant theorems, include a '(idx)' suffix (e.g. 'conjI(2)').",
        },
    },
    "required": ["type", "name"],
}


@tool(
    "query_semantics",
    "Query the semantic interpretation of a previously interpreted constant or theorem from Isabelle. "
    "Use this to look up the meaning of dependencies that the current entries rely on.",
    input_schema=_query_schema,
)
async def _query_tool(args: dict[str, Any]) -> ToolCall_ret:
    t = args.get("type")
    name = args.get("name")
    if t not in ("constant", "theorem"):
        return _mk_ret(f"Invalid type: {t!r}. Must be 'constant' or 'theorem'.")
    if not isinstance(name, str) or not name:
        return _mk_ret("Invalid name: must be a non-empty string.")
    if name in _local.names:
        return _mk_ret(
            f"\"{name}\" is one of the entries you are currently interpreting. "
            "You should interpret it yourself based on the source file, not query it."
        )
    tag = 0 if t == "constant" else 1
    result = _local.connection.callback("Semantic_Store.query", (tag, name))
    if result is None:
        return _mk_ret(f"No semantics found for {t} \"{name}\"")
    return _mk_ret(result)


# --- MCP Tool 2: answer ---

_answer_schema = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The 0-based index of the entry to interpret.",
                    },
                    "semantics": {
                        "type": "string",
                        "description": "The English semantic interpretation of this entity.",
                    },
                },
                "required": ["index", "semantics"],
            },
            "description": "List of semantic interpretations to submit.",
        },
    },
    "required": ["interpretations"],
}


@tool(
    "answer",
    "Submit semantic interpretations for one or more constants/theorems. "
    "Each interpretation should be a concise English description of what the entity means or states.",
    input_schema=_answer_schema,
)
async def _answer_tool(args: dict[str, Any]) -> ToolCall_ret:
    interpretations = args["interpretations"]
    for item in interpretations:
        _local.results[item["index"]] = item["semantics"]
    count = len(interpretations)
    remaining = len(_local.names) - len(_local.results)
    cs = "" if count == 1 else "s"
    rs = "" if remaining == 1 else "s"
    return _mk_ret(f"Answered {count} interpretation{cs}, remaining {remaining} to answer.")


# --- Prompt builder ---

def _build_prompt(
    file_path: str,
    theory_longname: str,
    deps_longname: list[str],
    kinds: list[int],
    names: list[str],
    prop_strs: list[str],
    line_numbers: list[int],
) -> str:
    entry_lines = []
    for i, (kind, name, prop, lineno) in enumerate(zip(kinds, names, prop_strs, line_numbers)):
        label = _KIND_PROMPT_LABELS.get(kind, "unknown")
        if lineno > 0:
            line = f"  {i}. [line {lineno}] {label} {name}"
        else:
            line = f"  {i}. {label} {name}"
        if prop:
            line += f": {prop}"
        entry_lines.append(line)
    entries_text = "\n".join(entry_lines)

    deps_text = ", ".join(deps_longname) if deps_longname else "(none)"

    ret = f"""\
You are interpreting the semantics of constants and theorems defined in the Isabelle theory "{theory_longname}".

The theory source file is at: {file_path}
Parent theories: {deps_text}

Your task:
1. Read the source file to understand the definitions.
2. For each entry listed below, write a concise English interpretation of what it means or states.
3. Use the `query_semantics` tool to look up interpretations of dependency constants/theorems from parent theories if needed.
4. Submit ALL your interpretations using the `mcp__isabelle_semantics__answer` tool, identifying each entry by its index.

Entries to interpret:
{entries_text}

Guidelines:
- For constants: describe what the constant represents, its type, and its purpose.
- For theorems: describe what the theorem states in plain English.
  Some auto-generated theorems (not introduced by a goal keyword in the source) include
  their pretty-printed proposition below the entry — use it to understand the statement.
- Be concise but precise. One to three sentences per entry is typical.
- You MUST submit interpretations for ALL entries listed above.
- Use the 0-based index shown before each entry when answering.
- When querying dependencies, use `query_semantics` with the name of the constant or theorem.
- Read the source file first before interpreting."""
    return IsaREPL.Client.pretty_unicode(ret)


# --- Agent runner ---

async def _run_agent(options: ClaudeAgentOptions, prompt: str) -> None:
    n = len(_local.names)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            pass
        while True:
            missing = [i for i in range(n) if i not in _local.results]
            if not missing:
                break
            missing_lines = "\n".join(
                f"  {i}. {_KIND_PROMPT_LABELS.get(_local.kinds[i], 'unknown')} {_local.names[i]}"
                for i in missing
            )
            await client.query(
                f"You still have {len(missing)} unanswered entries. "
                f"Use the `mcp__isabelle_semantics__answer` tool to submit interpretations for:\n{missing_lines}"
            )
            async for message in client.receive_response():
                pass


# --- RPC procedure ---

@isabelle_remote_procedure("Semantic_Store.interpret_file")
def interpret_file(arg: Any, connection: Connection) -> list[str | None]:
    (file_path, theory_longname, deps_longname, raw_entries) = arg
    kinds = [kind for (kind, _, _, _) in raw_entries]
    prop_strs = [prop for (_, _, prop, _) in raw_entries]
    line_numbers = [lineno for (_, _, _, lineno) in raw_entries]
    names = [name for (_, name, _, _) in raw_entries]
    n = len(names)

    # Compute file content hash
    with open(file_path, "rb") as f:
        file_hash = xxhash.xxh128(f.read()).digest()

    # Check cache
    results: dict[int, str] = {}
    with _open_cache() as db:
        for i, name in enumerate(names):
            v = db.get(_cache_key(file_hash, name))
            if v is not None:
                results[i] = v.decode("utf-8")

    uncached = [i for i in range(n) if i not in results]

    if uncached:
        # Set up thread-local state for only uncached entries
        _local.connection = connection
        _local.results = {}
        _local.names = [names[i] for i in uncached]
        _local.kinds = [kinds[i] for i in uncached]

        from .theory_structure import mk_unicode_file
        unicode_file_path = mk_unicode_file(file_path)

        prompt = _build_prompt(
            unicode_file_path, theory_longname, deps_longname,
            _local.kinds, _local.names,
            [prop_strs[i] for i in uncached],
            [line_numbers[i] for i in uncached],
        )

        mcp = create_sdk_mcp_server("isabelle_semantics", tools=[_query_tool, _answer_tool])
        options = ClaudeAgentOptions(
            cwd=os.path.dirname(file_path),
            permission_mode="default",
            allowed_tools=[
                "Read",
                "Grep",
                "Glob",
                "mcp__isabelle_semantics__query_semantics",
                "mcp__isabelle_semantics__answer",
            ],
            mcp_servers={"isabelle_semantics": mcp},
        )

        asyncio.run(_run_agent(options, prompt))

        # Remap agent results to original indices and write to cache
        with _open_cache() as db:
            for agent_idx, sem in _local.results.items():
                orig_idx = uncached[agent_idx]
                results[orig_idx] = sem
                db[_cache_key(file_hash, names[orig_idx])] = sem.encode("utf-8")

    return [results.get(i) for i in range(n)]
