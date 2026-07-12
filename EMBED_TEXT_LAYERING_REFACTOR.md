# Refactor: one authority for "record → embedding document text"

Status: **required, not yet done.** Written 2026-07-11. Cross-repo
(`Semantic_Embedding` + `Isa-Mini/AoA`), touches a public interface.

---

## 1. The defect

A record's *embedding document text* — the string that gets turned into its
Stage-1 bi-encoder document vector — is assembled in **several places, at two
different layers, with no single authority**. Nothing dispatches on
`Record.kind`, so the EXPERIENCE convention and the entity convention have drifted
apart and land in the *same* per-model vector store under the *same* keys.

There **is** a record abstraction (`_Semantic_DB.Record`, `semantics.py:174`, with
a `kind` field) but **no embedding interface built on it**. The closest thing,
`Semantic_DB.query(with_pretty=True)`, hardcodes the entity convention and never
looks at `kind`.

### The convention, per site

| # | site | text it embeds | correct for EXPERIENCE? |
|---|---|---|---|
| 1 | `semantics.py:341-342` `query(with_pretty=True)` | `rec.pretty_print + "\n" + rec.interpretation` | ❌ (entity template) |
| 2 | `semantics.py:1120` `_auto_embed` | calls #1 (`query`, `:1182`) | ❌ |
| 3 | `semantics.py:~1493` `_embed_keys` / `embed_entities` | calls #1 (`query`, `:1501`) | ❌ |
| 4 | `Isa-Mini/AoA/mcp_http_server.py:2342` write_memory | `_experience_document_text(pats, desc)` (`:2152`) | ✅ |
| 5 | `Isa-Mini/AoA/mcp_http_server.py:2378` write_memory dedup | `_experience_document_text(pats, desc)` | ✅ |

So the entity path has a *near*-authority (#1, reused by #2 #3), but it is
kind-blind; the EXPERIENCE authority (#4/#5) is a separate function stranded in the
**upper** (AoA) layer. They never meet.

### Independently, the *write* path is also triplicated

Three separate "get text → `emb_provider.embed` → put vector" implementations:
`_embed_keys` (`semantics.py:~1511`), `_auto_embed` (`:1202`), and
`Vector_Store.embed` (`semantic_embedding.py:670-677`). Each re-derives text and
writes vectors itself — which is *why* the convention can diverge: no single choke
point forces "where does the text come from".

## 2. Concrete failure

EXPERIENCE records (`SemanticRecord`, `semantics.py:174-203`) reuse entity fields:
`interpretation` = goal_description, `expr` = goal patterns (JSON). The write path
(#4) embeds `_experience_document_text` = a framing line + `- <pattern>` per goal
pattern + a situation line + the description. The entity path (#1/#2/#3) embeds
`pretty_print + interpretation` = `"experience <name>: <patterns-json>\n<desc>"`.

Trigger: **switch or add an embedding model.** The new `vector_M2` store has no
experience vectors, so at retrieval time each experience key is `missing` →
`_auto_embed` (#2) backfills it via `query(with_pretty=True)` → **entity template,
wrong**. Meanwhile any live `write_memory` writes the framing template into the
same store. One store, two conventions for the same kind → Stage-1 experience
cosine scores are no longer apples-to-apples → experiences mis-ranked or missed.

Also: this repo's own offline `embed` (semantics_manage.py, commit a31ad48) hit the
*same* trap with a third text (`rec.interpretation` alone); commit f63b0bb worked
around it by **excluding experiences entirely** — a stopgap that this refactor
should supersede so offline embed can (correctly) cover experiences again.

Severity: not corruption; retrieval-quality degradation, gated on model change.
Real, and structural.

## 3. Root cause

The `embed(kv_pairs: list[(key, text)])` interface (`semantic_embedding.py:670`)
sits **below** the record abstraction: it demands a pre-assembled string, pushing
the "what text does a record of this kind embed as" decision onto every caller.
The decision is domain knowledge that belongs **inside** `Semantic_Embedding`,
keyed on `Record.kind` — not replicated in three callers and one upper-layer
function.

Note the query side already puts its experience template in the lower layer:
`embedding_config.experience_task_description()` (`embedding_config.py:149`). The
*document* template `_experience_document_text` is the outlier still living in AoA.
Bringing it down is consistent with where the query-side template already is.

## 4. Target design

### 4.1 One authority: `document_text_of(record)`

Add to `_Semantic_DB` (next to `query`, `semantics.py:334`), dispatching on `kind`:

```python
def document_text_of(self, rec: 'Record') -> str | None:
    if rec.interpretation is None:
        return None
    if rec.kind == EntityKind.EXPERIENCE:
        # patterns come from rec.expr (JSON goal patterns), desc from interpretation
        return experience_document_text(_patterns_of(rec), rec.interpretation)
    return rec.pretty_print + "\n" + rec.interpretation
```

Move `_experience_document_text` **down** from `Isa-Mini/AoA/mcp_http_server.py:2152`
into `Semantic_Embedding` (it is pure string assembly; no AoA dependency). Natural
home: `embedding_config.py` (beside `experience_task_description`) or a small
`document_text.py`. Re-export/rename as `experience_document_text`.

`query(with_pretty=True)` should then be **defined in terms of** `document_text_of`
(or vice versa) so the entity convention also has exactly one definition. Preserve
`with_pretty=False` behavior (returns bare `interpretation`) — audit its callers
(e.g. `desugar.py:108`) so the refactor is behavior-preserving for them.

### 4.2 One write path: `Semantic_Vector_Store.embed_keys(keys, *, force)`

Add a single key-driven embed on `Semantic_Vector_Store` (`semantics.py:998`) that
is the *only* place that turns keys into vectors:

```python
async def embed_keys(self, keys, *, force=False) -> int:
    todo = keys if force else [k for k, ex in zip(keys, self.contains(keys)) if not ex]
    kv = []
    for k in todo:
        rec = Semantic_DB[k]
        t = Semantic_DB.document_text_of(rec) if rec else None
        if t is not None:
            kv.append((k, t))
    return await self.embed(kv)   # the low-level (key,text) primitive stays
```

Then collapse the callers onto it:
- `_auto_embed` (#2): build `missing` as today, then `await self.embed_keys(missing)`.
  Deletes its own text loop + `emb_provider.embed` + `txn.put` block.
- `_embed_keys` / `embed_entities` (#3): become thin wrappers over `embed_keys`.
- offline `embed` (this repo, `_collect_embed_candidates` + `_embed_models`):
  enumerate keys via `iter_entity_records`, hand them to `embed_keys`; **drop the
  f63b0bb experience-exclusion** — experiences now embed correctly.
- write_memory (#4, `mcp_http_server.py:2342`): keep its explicit
  `store.embed([(key, experience_document_text(...))])` OR switch to
  `store.embed_keys([key])` **after** it has written the record. Either is fine as
  long as the text comes from the shared `experience_document_text`. The dedup call
  (#5, `:2378`) must use the same `experience_document_text` (now the shared one).

The low-level `embed(kv_pairs)` (`semantic_embedding.py:670`) stays as the private
primitive `embed_keys` calls; nothing else should call it directly for corpus
documents.

## 5. Migration (existing stores)

Because the convention changes for whichever experiences were written by the wrong
path, existing experience vectors are of mixed provenance. After the refactor:
- delete all EXPERIENCE vectors from every `vector_*.lmdb` (they are a derived
  cache), then let them be rebuilt by `embed_keys` / `_auto_embed` under the single
  convention. (This repo already deleted the 5 it had mis-written, f63b0bb.)
- entity vectors are unchanged (convention #1 preserved), so no entity re-embed.

A one-shot `semantics_manage.py` helper (or `embed --kinds experience --force`)
can do the targeted re-embed.

## 6. Verification

- Unit: `document_text_of` on an EXPERIENCE record == `experience_document_text` of
  its patterns+desc; on an entity == `pretty_print + "\n" + interpretation`;
  `interpretation is None` → None.
- Equivalence: for a corpus of entities, `embed_keys` produces byte-identical
  vectors to today's `query(with_pretty=True)` path (entity convention unchanged).
- Consistency: an experience written by `write_memory` and the same key re-embedded
  by `_auto_embed`/offline `embed` now yield the **same** document text (assert the
  strings are equal — the core regression that must never come back).
- Cross-repo: AoA write_memory + dedup still pass their tests
  (`Isa-Mini/IsaMini/AoA/Tests/WriteMemoryGate.yml`).

## 7. Scope / order

Cross-repo, changes a public-ish interface, touches `_auto_embed`, `_embed_keys`,
`Semantic_Vector_Store`, and AoA `write_memory`. Read
`Isa-Mini/AoA/docs/EXPERIENCE_MEMORY.md` (§8.1 is cited at the doc-text function)
before moving the template. Suggested order:
1. Land `document_text_of` + move `experience_document_text` down; make `query`
   delegate. (No behavior change for entities; adds the correct experience text.)
2. Add `embed_keys`; migrate `_auto_embed`, `_embed_keys`, offline `embed` onto it;
   drop the experience-exclusion stopgap.
3. Point AoA write_memory/dedup at the shared `experience_document_text`.
4. Migrate stores (§5) + run §6 checks.
