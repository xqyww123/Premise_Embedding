# Semantic Embedding

Semantic interpretation and vector search for Isabelle entities (constants, theorems, types, type classes, locales). A Claude agent translates formal definitions into plain English, which are then embedded into vectors for similarity search.

## Semantic DB

The semantic database (`Semantic_DB` singleton in `semantics.py`) stores interpretations in an LMDB file at `~/.cache/Isabelle_Semantic_Embedding/semantics.lmdb`.

### Entity Records

Each entity is stored as a msgpack tuple keyed by its universal key:

```
(kind: int, name: str, expr: str, interpretation: str)
```

- `kind`: EntityKind value (1=constant, 2=lemma, 3=type, 4=typeclass, 5=locale)
- `name`: fully qualified name (e.g. `"HOL.List.append"`)
- `expr`: pretty-printed type signature or proposition
- `interpretation`: plain English description

Access via `Semantic_DB[key]` (returns `SemanticRecord | None`) or `Semantic_DB[key] = record`.

### Theory Status

Theory keys (16-byte theory hashes) store cost tracking and interpretation status as a msgpack dict:

```python
{"input_tokens": int, "output_tokens": int, "cost_usd": float, "finished": bool}
```

`Semantic_DB.is_thy_interpreted(key)` checks the `finished` flag.

### Persistent vs WIP Theories

Theory hashes use an LSB convention (defined in `Isabelle_RPC/Tools/theory_hash.ML`):
- **Persistent** (LSB=0): from built heap images, content-hashed — stable across sessions
- **WIP** (LSB=1): theories being edited in jEdit, name-hashed — change on every reload

WIP entities are stored persistently just like persistent ones. However, when source files are edited and entities change semantics, the stored interpretations and embeddings are **not** automatically updated — they become stale. You must explicitly call `clean_wip()` (Python) or `Semantic_Store.clean_wip()` (ML) to purge all WIP data from both the semantic DB and all vector stores, so that subsequent queries regenerate fresh interpretations and embeddings.

WIP theories are never marked as "interpreted" or "embedded finished" — `mark_interpreted` and `mark_thy_embedded` silently skip them. This ensures WIP theories are always re-processed on the next request, since their content may have changed.

### Skipped Theories

Certain infrastructure theories are never interpreted: `Pure`, `Code_Generator`, `Code_Evaluation`, `Typerep`. These are defined in `_SKIP_THEORY_LONG_NAMES` and filtered in both interpretation and entity enumeration.

## Semantic Interpretation

`semantic_interpretation.py` uses a Claude agent (`claude-opus-4-6`) to translate Isabelle entities into plain English. The flow:

1. **ML side** (`semantic_store.ML:interpret'`): extracts constants, theorems, types, classes, locales from a theory, computes universal keys, and calls Python via RPC
2. **Python side** (`interpret_file`): checks LMDB cache for existing interpretations, launches Claude agent for uncached entries
3. **Agent**: receives batches of entities (20 per batch), uses MCP tools (`query_by_name`, `query_by_position`, `definition`, `hover`) to understand dependencies, submits translations via the `answer` tool
4. **Storage**: each answer is immediately written to LMDB via `Semantic_DB[key] = SemanticRecord(...)`

`Semantic_Store.interpret(context)` interprets all uninterpreted ancestor theories plus the current proof context. `interpret_theories_by_names(connection, names)` interprets specific theories by name.

## Embedding Providers

`Embedding_Provider` (in `semantic_embedding.py`) is the abstract base for text embedding services.

### Registered Providers

| Name | Model | Dimension | API |
|------|-------|-----------|-----|
| `oai.text-embedding-3-large` | text-embedding-3-large | 3072 | OpenAI |
| `oai.text-embedding-3-small` | text-embedding-3-small | 1536 | OpenAI |
| `mistral.codestral-embed` | codestral-embed | 1536 | Mistral |

Custom providers can be added via `@register_embedding_provider("name")` or placed as `drivers/{name}.py`.

### Caching

Embedding results are cached per-string in a DiskCache database at `~/.cache/Isabelle_Semantic_Embedding/embed_cache/` (100MB limit, 3-day TTL). Only short queries (total text length <= 512 chars) are cached by `embed()`. `embed_batch()` always uses the cache.

### Batch API

`OpenAI_Embedding_Provider._embed_batch` uses the OpenAI Batch API for 50% cost reduction. It uploads a JSONL file, creates an async batch job, polls for completion, and downloads results. Large inputs are split into sub-batches of `max_batch_size` (default 50,000) and submitted in parallel on the server side.

## Vector Store & Semantic Vector Store

### Vector Store

`Vector_Store` (in `semantic_embedding.py`) stores embedding vectors in LMDB, keyed by universal keys. Multiple stores share a global LMDB environment pool (thread-safe).

Key operations:
- `store[key]` / `store.put(key, vector)`: get/set vectors
- `store.topk(query, domain, k)`: return top-k `(key, score)` pairs using `faiss.knn`
- `store._auto_embed(missing, matrix, row)`: hook for subclasses to recover missing vectors

### Semantic Vector Store

`Semantic_Vector_Store` (in `semantics.py`) extends `Vector_Store` with:
- **Auto-interpretation**: when vectors are missing, automatically interprets uninterpreted theories (if `auto_interpret_for_embedding` is enabled), fetches semantic texts from `Semantic_DB`, embeds them, and stores the vectors
- **Model resolution**: reads from Isabelle config `Semantic_Embedding.embedding_model`, then env var `EMBEDDING_MODEL`, then defaults to `oai.text-embedding-3-small`
- **Per-connection registry**: each `Connection` maintains a dict of stores by model name, accessed via `connection.semantic_vector_store(model_name)`

`lookup(query, k, kinds, domain)` combines entity filtering with k-NN search:
- `kinds`: filter by `EntityKind` (e.g. `[ConstantK, TheoremK]`)
- `domain=None`: search all entities of those kinds (excluding skipped theories)
- Returns `list[tuple[float, SemanticRecord]]` sorted by similarity

## Config Options

Set in Isabelle via `declare [[option = value]]`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_interpret_for_embedding` | bool | `true` | Auto-trigger interpretation when embeddings are missing |
| `Semantic_Embedding.embedding_model` | string | `""` | Override embedding model (empty = use env var or default) |
| `Semantic_Embedding.reranker_model` | string | `""` | Reranker model for re-ranking search results (empty = disabled; env var `RERANKER_MODEL`) |

These are accessible from Python via `connection.config_lookup("option_name")`.

## Usage

### Isabelle/ML

```sml
(* Interpret all ancestor theories *)
Semantic_Store.interpret \<^context>

(* Query semantic interpretation *)
Semantic_Store.query_semantics \<^context> (Universal_Key.Constant "List.append") false

(* k-NN search for similar entities *)
Semantic_Store.query_knn \<^context>
  "concatenation of two lists"
  10
  [Universal_Key.ConstantK, Universal_Key.TheoremK]
  NONE
```

### Python (within RPC handler)

```python
from Isabelle_Semantic_Embedding.semantics import Semantic_DB, SemanticRecord

# Query interpretation
rec = Semantic_DB[some_universal_key]
if rec is not None:
    print(rec.pretty_print)       # "constant List.append: 'a list => ..."
    print(rec.interpretation)     # "Appends two lists together."

# Vector search
store = connection.semantic_vector_store()
results = store.lookup("addition on natural numbers", k=5,
                       kinds=[EntityKind.CONSTANT, EntityKind.THEOREM])
for score, rec in results:
    print(f"{score:.3f} {rec.pretty_print}: {rec.interpretation}")
```
