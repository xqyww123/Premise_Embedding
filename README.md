# Semantic Embedding

Generates natural-language semantic interpretations for Isabelle mathematical entities (theorems, constants, types, classes, locales) via LLM. Results are cached in LMDB. Invoked from Isabelle/ML via RPC to Python.

## LMDB store: `semantics.lmdb`

- **Path:** `~/.cache/Isabelle_Semantic_Embedding/Qiyuan/semantics.lmdb`
  - Constructed via `platformdirs.user_cache_dir("Isabelle_Semantic_Embedding", "Qiyuan")` in `Isabelle_Semantic_Embedding/semantics.py:open_semantic_store()`
- Uses default unnamed database (no named sub-databases)
- **Key:** `universal_key` bytes (EntityKind tag + name + theory hash)
- **Value:** msgpack-encoded, two record types sharing the same keyspace:
  - **Entity record** (constants, theorems, types, classes, locales): `(pretty_str, semantics_str)` — a 2-tuple of the pretty-printed signature and the LLM-generated interpretation
  - **Theory record** (keyed by theory's `universal_key`): `{input_tokens: int, output_tokens: int, cost_usd: float, finished: bool}` — cumulative LLM cost and whether interpretation is complete
