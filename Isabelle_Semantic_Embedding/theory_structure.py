import os
import re
from typing import TYPE_CHECKING, Any

from Isabelle_RPC_Host import isabelle_remote_procedure
from Isabelle_RPC_Host.unicode import pretty_unicode

if TYPE_CHECKING:
    from Isabelle_RPC_Host import Connection


async def theory_info(connection: "Connection", theory_name: str) -> tuple[str, str]:
    """Resolve a theory name to (long_name, file_path) via Isabelle callback."""
    return tuple(await connection.callback("Context.theory_long_name_and_path", theory_name))


async def get_session_databases(connection: "Connection") -> list[tuple[str, str]]:
    """Return all loaded (ancestor) sessions and their export database paths.

    Returns a list of (session_name, db_path) pairs.
    """
    return await connection.callback("pide_state.get_session_databases", None)


def mk_unicode_file(path: str) -> str:
    if not path.endswith(".thy"):
        raise ValueError(f"Expected .thy file, got: {path}")
    unicode_path = path[:-4] + ".unicode.thy"
    if os.path.exists(unicode_path) and os.path.getmtime(unicode_path) > os.path.getmtime(path):
        return unicode_path
    with open(path, "r") as f:
        content = f.read()
    unicode_content = pretty_unicode(content)
    with open(unicode_path, "w") as f:
        f.write(unicode_content)
    return unicode_path


# --- check_theorem_name_in_file RPC ---

_GOAL_KEYWORDS = ("lemma", "theorem", "proposition", "corollary", "schematic_goal")

# Precompiled patterns for single-pass file preprocessing
_WORD_RE = re.compile(r"\b\w+(?:\.\w+)*\b")
_GOAL_RE = re.compile(
    r"(?:" + "|".join(_GOAL_KEYWORDS) + r")\s+(\w+(?:\.\w+)*)\b"
)

@isabelle_remote_procedure("Semantic_Store.check_theorem_name_in_file")
async def check_theorem_name_in_file(arg: Any, connection: "Connection") -> list[tuple[int, int]]:
    """Check each name against the file content and return a pair per name.

    The file is preprocessed once into lookup structures, then each name is
    resolved via dict/set lookups (O(1) per name instead of O(file_size)).

    Returns a list of (word_offset, goal_keyword_offset) tuples:

    - word_offset: byte offset of the first occurrence of the name as a whole word
      (using ``\\w+(?:\\.\\w+)*`` to capture dotted Isabelle names like ``foo.simps``),
      or -1 if the name does not appear at all.

    - goal_keyword_offset: byte offset of the first occurrence where the name
      immediately follows one of the Isar goal keywords
      (lemma, theorem, proposition, corollary, schematic_goal),
      or -1 if no such occurrence exists.  This distinguishes explicitly stated
      theorems from auto-generated ones (e.g. datatype .simps / .induct).

    Names are short names with theory prefix stripped (e.g. ``append_def``, ``list.simps(2)``).
    Before lookup, only the ``(N)`` index suffix is stripped.
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

    # 3. O(1) lookups per name.  Names have theory prefix stripped
    #    (e.g. "append_def", "list.simps(2)").  Strip "(N)" index suffix for lookup.
    results: list[tuple[int, int]] = []
    for name in names:
        base = re.sub(r"\(\d+\)$", "", name.split(".")[-1])
        results.append((
            word_offsets.get(base, -1),
            goal_offsets.get(base, -1),
        ))
    return results
