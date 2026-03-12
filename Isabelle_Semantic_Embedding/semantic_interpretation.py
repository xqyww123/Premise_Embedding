"""Semantic interpretation of Isabelle constants and theorems via Claude Code agent."""

from __future__ import annotations

import asyncio
import os
import re
import threading
from typing import Any

from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)
import IsaREPL

# --- Thread-local state ---

_KIND_CONSTANT = 0
_KIND_THEOREM = 1
_KIND_PROMPT_LABELS = {_KIND_CONSTANT: "constant", _KIND_THEOREM: "lemma"}

class _LocalState(threading.local):
    connection: Connection
    results: dict[int, str]       # index -> semantics
    display_names: list[str]
    kinds: list[int]              # _KIND_CONSTANT or _KIND_THEOREM

_local = _LocalState()


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
    if name in _local.display_names:
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
    remaining = len(_local.display_names) - len(_local.results)
    cs = "" if count == 1 else "s"
    rs = "" if remaining == 1 else "s"
    return _mk_ret(f"Answered {count} interpretation{cs}, remaining {remaining} to answer.")


# --- Prompt builder ---

def _build_prompt(
    file_path: str,
    theory_longname: str,
    deps_longname: list[str],
    kinds: list[int],
    display_names: list[str],
    prop_strs: list[str],
) -> str:
    entry_lines = []
    for i, (kind, name, prop) in enumerate(zip(kinds, display_names, prop_strs)):
        label = _KIND_PROMPT_LABELS.get(kind, "unknown")
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
    n = len(_local.display_names)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            pass
        while True:
            missing = [i for i in range(n) if i not in _local.results]
            if not missing:
                break
            missing_lines = "\n".join(
                f"  {i}. {_KIND_PROMPT_LABELS.get(_local.kinds[i], 'unknown')} {_local.display_names[i]}"
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
    kinds = [kind for (kind, _, _) in raw_entries]
    prop_strs = [prop for (_, _, prop) in raw_entries]

    # Strip the leading theory-name qualifier from fully qualified names.
    # E.g. if theory_longname is "List", "List.append_def" becomes "append_def".
    prefix = theory_longname + "."
    display_names = [
        name[len(prefix):] if name.startswith(prefix) else name
        for (_, name, _) in raw_entries
    ]

    # Set up thread-local state for MCP tool handlers
    _local.connection = connection
    _local.results = {}
    _local.display_names = display_names
    _local.kinds = kinds

    # Build prompt
    prompt = _build_prompt(file_path, theory_longname, deps_longname, kinds, display_names, prop_strs)

    # Create MCP server and agent options
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

    # Run the agent
    asyncio.run(_run_agent(options, prompt))

    # Collect results in index order
    return [_local.results.get(i) for i in range(len(display_names))]


_GOAL_KEYWORDS = ("lemma", "theorem", "proposition", "corollary", "schematic_goal")

# Precompiled patterns for single-pass file preprocessing
_WORD_RE = re.compile(r"\b\w+(?:\.\w+)*\b")
_GOAL_RE = re.compile(
    r"(?:" + "|".join(_GOAL_KEYWORDS) + r")\s+(\w+(?:\.\w+)*)\b"
)
_GOAL_CODE_RE = re.compile(
    r"(?:" + "|".join(_GOAL_KEYWORDS) + r")\s+(\w+(?:\.\w+)*)\s*\[[^\]]*\bcode\b[^\]]*\]"
)


@isabelle_remote_procedure("Semantic_Store.check_theorem_name_in_file")
def check_theorem_name_in_file(arg: Any, connection: Connection) -> list[tuple[int, int, bool]]:
    """Check each name against the file content and return a triple per name.

    The file is preprocessed once into lookup structures, then each name is
    resolved via dict/set lookups (O(1) per name instead of O(file_size)).

    Returns a list of (word_offset, goal_keyword_offset, has_code_attr) tuples:

    - word_offset: byte offset of the first occurrence of the name as a whole word
      (using ``\\w+(?:\\.\\w+)*`` to capture dotted Isabelle names like ``foo.simps``),
      or -1 if the name does not appear at all.

    - goal_keyword_offset: byte offset of the first occurrence where the name
      immediately follows one of the Isar goal keywords
      (lemma, theorem, proposition, corollary, schematic_goal),
      or -1 if no such occurrence exists.  This distinguishes explicitly stated
      theorems from auto-generated ones (e.g. datatype .simps / .induct).

    - has_code_attr: True if the name follows a goal keyword and is accompanied
      by an attribute list containing the word ``code``, i.e. matches
      ``(lemma|theorem|...)\\s+name\\s*\\[...code...]``.

    Names are expected to be fully qualified (e.g. ``Foo.bar_def``, ``Foo.bar.simps(2)``).
    Before lookup, each name is reduced to its base form: the ``(N)`` index suffix is
    stripped, then the last dot-separated component is taken (e.g. ``bar_def``, ``simps``).
    """
    (file_path, names) = arg
    content = open(file_path).read()

    # 1. Build word_offsets: first occurrence of each word-boundary token
    word_offsets: dict[str, int] = {}
    for m in _WORD_RE.finditer(content):
        word = m.group()
        if word not in word_offsets:
            word_offsets[word] = m.start()

    # 2. Build goal_offsets: first occurrence of name after a goal keyword
    goal_offsets: dict[str, int] = {}
    for m in _GOAL_RE.finditer(content):
        name = m.group(1)
        if name not in goal_offsets:
            goal_offsets[name] = m.start()

    # 3. Build code_names: names following a goal keyword with [code] attribute
    code_names: set[str] = set()
    for m in _GOAL_CODE_RE.finditer(content):
        code_names.add(m.group(1))

    # 4. O(1) lookups per name, using base name (last dot-separated component,
    #    with any trailing "(N)" index suffix stripped)
    results: list[tuple[int, int, bool]] = []
    for name in names:
        # Strip "(N)" suffix if present, then take last dot-separated component
        base = re.sub(r"\(\d+\)$", "", name).rsplit(".", 1)[-1]
        results.append((
            word_offsets.get(base, -1),
            goal_offsets.get(base, -1),
            base in code_names,
        ))
    return results
