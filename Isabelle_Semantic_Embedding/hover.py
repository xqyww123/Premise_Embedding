"""
Go-to-definition and hover message for Isabelle entities.

Calls into Isabelle/ML which delegates to Scala. The Scala side tries the live
PIDE runtime first, then falls back to session export databases.
"""

import os
from typing import Optional

from Isabelle_RPC_Host import Connection
from Isabelle_RPC_Host.position import IsabellePosition


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def goto_definition(
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
        result = connection.callback(
            "pide_state.goto_definition", (pos.file, pos.raw_offset))
        if result is not None:
            def_file, def_line, def_offset, def_end_offset = result
            return (_resolve_path(def_file), def_line, def_offset, def_end_offset)
    except Exception:
        pass
    return None


def hover_message(
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
        result = connection.callback(
            "pide_state.hover_message", (pos.file, pos.raw_offset))
        if result is not None and result != "":
            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """Expand ~~ prefix to ISABELLE_HOME."""
    if path.startswith("~~"):
        isabelle_home = os.environ.get("ISABELLE_HOME", "")
        if isabelle_home:
            return isabelle_home + path[2:]
    return path
