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
from .hover import goto_definition, hover_message
from .semantic_interpretation import interpret_file, _interpret_file
from .semantics import (
    mk_query_by_name_tool as query_by_name_tool,
    mk_query_by_position_tool as query_by_position_tool,
    _query as query_semantic_store,
    _is_interpreted as is_interpreted,
    _mark_interpreted as mark_interpreted,
    _clean_wip as clean_wip,
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
    "interpret_file",
    "is_interpreted",
    "mark_interpreted",
    "open_semantic_store",
    "query_by_name_tool",
    "query_by_position_tool",
    "clean_wip",
    "query_semantic_store",
    "theory_info",
    "is_persistent_theory_hash",
    "theory_xxhash128",
]
