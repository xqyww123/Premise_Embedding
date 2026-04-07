"""
Isabelle Semantic Embedding module for premise selection and semantic search.
"""

from .premise_selection import (
    embed,
    embed_goal,
    embed_premises,
    embed_goal_and_premises,
    encode_goal,
    encode_premise,
)

from .theory_structure import mk_unicode_file, theory_info
from .hover import goto_definition, hover_message, command_at_position
from .semantic_interpretation import interpret_file, _interpret_file
from .semantics import (
    Semantic_DB,
    mk_query_by_name_tool as query_by_name_tool,
    mk_query_by_position_tool as query_by_position_tool,
    _query,
    _is_interpreted,
    _mark_interpreted,
    _clean_wip,
)
from . import semantics

__all__ = [
    "embed",
    "embed_goal",
    "embed_premises",
    "embed_goal_and_premises",
    "encode_goal",
    "encode_premise",
    "goto_definition",
    "hover_message",
    "command_at_position",
    "interpret_file",
    "Semantic_DB",
    "query_by_name_tool",
    "query_by_position_tool",
    "theory_info",
]
