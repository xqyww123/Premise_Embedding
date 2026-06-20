# Semantic Embedding

Semantic interpretation and vector search for Isabelle entities (constants, theorems, types, type classes, locales). A Claude agent translates formal definitions into plain English, which are then embedded into vectors for similarity search.

## Semantic DB

The semantic database (`Semantic_DB` singleton in `semantics.py`) stores interpretations in an LMDB file at `~/.cache/Isabelle_Semantic_Embedding/semantics.lmdb`.

### Entity Records

Each entity is stored as a msgpack tuple keyed by its universal key:

```
(kind: int, name: str, expr: str, interpretation: str)
```

- `kind`: EntityKind value (1=constant, 2=lemma, 3=type, 4=typeclass, 5=locale,
  6=named_theorems, 7=proof method; plus 0x12/0x22/0x32/0x42 for intro/elim/induction/case-split rules)
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
3. **Agent**: receives batches of entities (20 per batch), uses MCP tools (`query`, `definition`, `hover`) to understand dependencies, submits translations via the `answer` tool
4. **Storage**: each answer is immediately written to LMDB via `Semantic_DB[key] = SemanticRecord(...)`

`Semantic_Store.interpret(context)` interprets all uninterpreted ancestor theories plus the current proof context. `interpret_theories_by_names(connection, names)` interprets specific theories by name.

## Embedding Providers

`Embedding_Provider` (in `semantic_embedding.py`) is the abstract base for text embedding services. A provider is selected by **three parameters** — `driver`, `base_url`, `model` — rather than a per-model registered subclass:

- **driver**: a registered driver class. `OpenAI_Embedding_Provider` serves any OpenAI-compatible `/v1/embeddings` endpoint (Fireworks, OpenAI, Mistral, Aliyun DashScope, …); `Gemini_Embedding` is the native Google Gemini API. Default `OpenAI_Embedding_Provider`.
- **base_url**: the API endpoint. Default `https://api.fireworks.ai/inference`.
- **model**: the **canonical** model name — the HuggingFace name where one exists (e.g. `Qwen/Qwen3-Embedding-8B`), else the provider's id (e.g. `text-embedding-3-large`). Default `Qwen/Qwen3-Embedding-8B`.

Build one with `make_embedding_provider(driver, base_url, model)`. The API key is read **only** from the `EMBEDDING_API_KEY` environment variable (Gemini additionally falls back to `GEMINI_API_KEY`).

### YAML config

All model/endpoint specifics live in a YAML config at `$ISABELLE_HOME_USER/etc/embedding_config`, seeded on first run from the bundled `embedding_config_template.yaml`. It is keyed by the canonical model name and by the base_url domain (netloc):

- per model: `dimension` (**required** — determines the LMDB matrix shape; a missing entry is a hard error), `default_scores` (`{score, local}` fallbacks for un-embedded entities), `normalize` (L2-normalize returned vectors), `max_request_size`, `templates` (per-model `query`/`document` text wrappers applied before embedding; default identity `"{text}"`, literal `str.replace` over `{text}`/`{task}` so Isabelle `{ }` text is safe)
- `providers.<domain>.normalization`: maps the canonical name to the id this endpoint expects (e.g. on `api.fireworks.ai`, `Qwen/Qwen3-Embedding-8B` → `fireworks/qwen3-embedding-8b`)
- `providers.<domain>.batch`: the Batch API shape (`dialect: openai | mistral`, endpoint, status strings, `max_batch_size`); its presence is what enables batch for that domain
- `task_description`: the sentence injected into a query template's `{task}` slot; its `{kinds}` slot is filled per query from the `EntityKind` filter (e.g. "constants and theorems", or "constructs" when unfiltered). Shared across models; must **not** contain literal `{text}`/`{task}`

Add a model by editing the YAML; add a new driver class via `@register_embedding_driver("Name")` or `drivers/{Name}.py`. Configs are seeded once and never auto-merged, so a machine that already has `etc/embedding_config` must add `templates`/`task_description` by hand.

### Caching

Embedding results are cached per-string (keyed by the canonical model name + text) in a DiskCache database at `~/.cache/Isabelle_Semantic_Embedding/embed_cache/` (2 GB limit, 3-day TTL). Both `embed()` and `embed_batch()` use the cache. This cache is purely local and is **excluded** from the published DB snapshot.

### Batch API

When the base_url's domain has a `batch` entry in the YAML config, `OpenAI_Embedding_Provider._embed_batch` uses the OpenAI-style Batch API (e.g. for 50% cost reduction on OpenAI). It uploads a JSONL file, creates an async batch job, polls for completion, and downloads results. Large inputs are split into sub-batches of `max_batch_size` (from the YAML `batch` config) and submitted in parallel on the server side. The `dialect` (`openai`/`mistral`) selects the request/response shape.

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
- **Provider resolution**: resolves `(driver, base_url, model)`, each by Isabelle config (`Semantic_Embedding.embedding_driver` / `embedding_base_url` / `embedding_model`) → env var (`EMBEDDING_DRIVER` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`) → default (`OpenAI_Embedding_Provider` / Fireworks / `Qwen/Qwen3-Embedding-8B`)
- **Store identity**: the LMDB vector store lives at `vector_<canonical model>.lmdb`, with the canonical name made filesystem-safe (`/` → `__`), e.g. `vector_Qwen__Qwen3-Embedding-8B.lmdb`
- **Per-connection registry**: each `Connection` maintains a dict of stores by canonical model name, accessed via `connection.semantic_vector_store(model_name)`. NOTE: one run has a single active driver+base_url, so embedding several models at once only works for models served by that same endpoint.

`lookup(query, k, kinds, domain)` combines entity filtering with k-NN search:
- `kinds`: filter by `EntityKind` (e.g. `[ConstantK, TheoremK]`)
- `domain=None`: search all entities of those kinds (excluding skipped theories)
- Returns `list[tuple[float, SemanticRecord]]` sorted by similarity

## Config Options

Set in Isabelle via `declare [[option = value]]`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_interpret_for_embedding` | bool | `true` | Auto-trigger interpretation when embeddings are missing |
| `Semantic_Embedding.embedding_driver` | string | `""` | Embedding driver class (empty = env `EMBEDDING_DRIVER` or `OpenAI_Embedding_Provider`) |
| `Semantic_Embedding.embedding_base_url` | string | `""` | Embedding endpoint base_url (empty = env `EMBEDDING_BASE_URL` or the Fireworks endpoint) |
| `Semantic_Embedding.embedding_model` | string | `""` | Canonical (HuggingFace) embedding model name (empty = env `EMBEDDING_MODEL` or `Qwen/Qwen3-Embedding-8B`) |
| `Semantic_Embedding.reranker_model` | string | `""` | Reranker model for re-ranking search results (empty = disabled; env var `RERANKER_MODEL`) |

These are accessible from Python via `connection.config_lookup("option_name")`.

## Marking entities as infrastructure (excluded from retrieval)

Some entities are plumbing that should never surface in semantic retrieval. Built-in
heuristics already drop most of them (concealed/hidden names, ADT/record/BNF machinery,
`Minilang.*`, tool-internal collections, …). To mark additional ones by hand:

```isabelle
declare [[infra_constant Foo.bar Baz.qux]]   (* constants (also cascades to theorems
                                                 whose statement mentions them) *)
declare [[infra_type   Foo.t]]               (* types *)
declare some_lemma[infra_thm]                 (* a theorem — attached fact attribute *)

declare [[infra_constant del Foo.bar]]        (* `del` undoes any of the above *)
declare some_lemma[infra_thm del]
```

Notes:
- `infra_thm` matches **by proposition** (like `named_theorems`): it stores the theorem
  itself, so two lemmas with the same statement are treated alike, and a declaration inside
  a `locale` correctly suppresses the exported instance at each `interpretation`. Marking a
  trivially-shaped lemma (`x = x`, `True`) is honoured but warns, since it would suppress
  every same-statement lemma.
- `infra_constant`/`infra_type` resolve their (global) argument at parse time; `infra_thm`
  is attached to the fact (`lemma[infra_thm]`, not `[[infra_thm name]]`).
- Whole `named_theorems` *collections* are suppressed via the static blacklist in
  `Tools/infra_filter.ML`, not via `infra_thm`.
- Effect is read at collection time: re-collect to drop entries already written to the DB.

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
