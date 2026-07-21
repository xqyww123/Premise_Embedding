# Isabelle Semantic Embedding

This library provides semantic search for Isabelle: it allows users retrieve, in natural language, the semantically closest entities — including constants, theorems, theorem collections (so-called `named_theorems`), types, type classes, locales, and tactics.

The process consists of two stages: semantic interpretation and embedding. In the interpretation stage, a Claude/Codex agent automatically explains each entity in an Isabelle theory into plain English and stores the obtained English explanations in a database. In the embedding stage, an embedding model (typically an LLM of several billion parameters) embeds each explanation into a vector (usually a few thousand dimensions), which is likewise stored in the local database.
At query time, the user's natural-language query is embedded into a vector by the same model. The system then ranks all stored entity vectors by their cosine similarity to the query vector and returns the top $k$ — the entities whose explanations are closest to the query under the model's semantic encoding.


## 1. Installation

This library ships as part of the Isabelle-AI package. If you already have Isabelle-AI installed, no further installation is needed; otherwise:
```bash
conda create -n <YOUR_ENV> -c https://conda.qiyuan.me -c conda-forge isabelle-ai
```

**Note**: this system currently supports Isabelle 2025-2 only. 
Installing without the conda manager is possible, but is an expert-only path: detailed documentation is not ready for now; the author suggests asking Claude/Codex to work out a standalone installation.

## 2. Quick start

This library is not normally used directly. A typical entry point is [AoA](https://github.com/xqyww123/Isa-Mini/blob/main/IsaMini/AoA/Readme.md).

## 3. Fetching the prebuilt database

The author provides pre-computed semantic interpretations, together with their Qwen3-8B embeddings, covering part of Isabelle/HOL 2025-2 and `afp-2026-05-13`. Download them for immediate use with the following `bash`/`powershell` commands:
```bash
conda activate <YOUR_ENV>
isabelle-semantics pull      # download the prebuilt database and merge it into the local DB
isabelle-semantics status    # compare the local database against the prebuilt one
```
Note: complete pre-computed interpretations of the AFP, together with the exact `afp-2026-05-13` version they correspond to, will be published soon; this section will be updated accordingly.


## 4. Running semantic interpretation on your own theories

The prebuilt database covers only part of Isabelle/HOL and `afp-2026-05-13`. To interpret and embed your own theories, use the `run_semantic_interpretation` command:

```isabelle
theory My_Wonderful_Theory
  imports "Semantic_Embedding.Semantic_Embedding" (* DONT'T FORGET ME ;-) *)
          Some_Wonderful_Dependencies
begin

(* A lot of wonderful formalizations here *)

run_semantic_interpretation                           (* interpret and embed the current theory and its ancestors *)
run_semantic_interpretation Theory_Name1 Theory_Name2 (* interpret and embed exactly these two theories, not My_Wonderful_Theory itself *)

end
```

Be aware that interpretation and embedding together can take a long time; please let the run finish. If you do interrupt it, most intermediate results are saved, and the next run resumes rather than starting from scratch.


## 5. Configuring the embedding provider

By default, the system uses Claude Code with `claude-opus-4-8[1m]` for semantic interpretation, and `Qwen/Qwen3-Embedding-8B` for embedding. A Codex-based interpreter, along with the ability to switch the interpretation LLM, is under development.

The embedding model can be changed through three settings — the driver, the endpoint, and the model name:
```isabelle
declare [[Semantic_Embedding.embedding_driver   = "OpenAI_Embedding_Provider",   (* most embedding providers speak the OpenAI-compatible protocol, so you will rarely need to change this *)
          Semantic_Embedding.embedding_base_url = "https://api.fireworks.ai/inference",
          Semantic_Embedding.embedding_model    = "Qwen/Qwen3-Embedding-8B"]]
```
Alternatively, the same three settings can be given as environment variables in `$(isabelle getenv -b ISABELLE_HOME_USER)/etc/settings`:
```
EMBEDDING_DRIVER=...
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
```
The Isabelle options take precedence: an environment variable is consulted only where the corresponding option is unset.

If you self-host an embedding model (e.g. via vLLM or TGI), set `embedding_base_url` to your server's address and `embedding_model` to the model name it serves.

The system guarantees that the query and the stored entities are embedded by the same model. If you change `embedding_model`, all contextual entities will therefore be re-embedded with the new model before the next semantic search — this typically costs little. Changing only `embedding_base_url` while keeping `embedding_model` unchanged triggers no re-embedding.

The following subsections cover each configuration option and its available choices in turn.

### 5.1 `embedding_driver`

Two drivers are currently available:
- `OpenAI_Embedding_Provider` (default) — speaks to any OpenAI-compatible /v1/embeddings endpoint: Fireworks, OpenAI, Mistral, Aliyun DashScope, ….
- `Gemini_Embedding` — the native Google Gemini API.

### 5.2 `embedding_base_url`

The endpoint the embedding driver contacts, given **without** a `/v1` suffix (the driver appends `/v1/embeddings` itself). Examples:

- `https://api.fireworks.ai/inference` (default)
- `https://api.openai.com`
- `https://api.mistral.ai`

### 5.3 `embedding_model`

The **canonical** model name — the HuggingFace name where one exists, else the provider's model id. The following are configured out of the box:

| Canonical name | Endpoint |
|---|---|
| `Qwen/Qwen3-Embedding-8B` (default) | Fireworks |
| `harrier-oss-v1-27b`, `llama-nv-embed-reasoning-3b` | Fireworks |
| `text-embedding-3-large`, `text-embedding-3-small` | OpenAI |
| `codestral-embed` | Mistral |
| `gemini-embedding-2-preview` | Gemini |

Other models can be added through the embedding config file (§7.1).

### 5.4 API key

The embedding API key is read from the single environment variable `EMBEDDING_API_KEY`, whatever the endpoint. Add the line
```
EMBEDDING_API_KEY=...
```
to `$(isabelle getenv -b ISABELLE_HOME_USER)/etc/settings`, then restart Isabelle.

The `Gemini_Embedding` driver additionally falls back to `GEMINI_API_KEY`.

## 6. The database

All data lives in the following directory:
| OS | Database directory |
|---|---|
| Linux | `~/.cache/Isabelle_Semantic_Embedding` |
| macOS | `~/Library/Caches/Isabelle_Semantic_Embedding` |
| Windows | `C:\Users\<USER NAME>\AppData\Local\Qiyuan\Isabelle_Semantic_Embedding\Cache` |

The directory consists of the semantic database (`semantics.lmdb`), one vector store per embedding model (`vector_<model>.lmdb`), and local caches.

**Warning**: the system uses [LMDB](http://www.lmdb.tech/) for its semantic and vector stores, and LMDB on a networked filesystem (NFS, Lustre) **can corrupt silently**. On compute clusters that use network filesystems, it is highly recommended to relocate the semantic database to a local disk by adding the line
```
SEMANTIC_DB_DIR=<PATH_ON_A_LOCAL_DISK>
```
to `$(isabelle getenv -b ISABELLE_HOME_USER)/etc/settings`. After changing this setting, you **must kill the background RPC process** and fully restart Isabelle. The RPC process can be found by looking for a process whose command line contains `Isabelle_RPC_Host.fork_and_launch__()`, e.g.:
```bash
pkill -f 'Isabelle_RPC_Host\.fork_and_launch__'
```

**Update checks.** At most once a week, the library probes the prebuilt database with a single HEAD request, and prints a notice when a newer version exists. It never downloads anything on its own — updating is always an explicit isabelle-semantics pull. The check can be turned off by setting
```bash
SEMANTIC_EMBEDDING_AUTO_UPDATE=false
```
in `$(isabelle getenv -b ISABELLE_HOME_USER)/etc/settings`


## 7. Technical details

### 7.1 The embedding config file

To use a new embedding model or a new provider endpoint, you must supply the following settings in `$(isabelle getenv -b ISABELLE_HOME_USER)/etc/embedding_config`:

- `dimension` — the vector dimension, **required** for every model in use;
- `normalize` — whether to L2-normalize the vectors the endpoint returns (default `false`). Needed when the endpoint does not return unit vectors.
- `max_request_size`  — the maximum number of texts sent in one embedding request batch (default 2048). Some endpoints enforce hard caps (e.g. Aliyun DashScope allows only 10)
- `default_scores` — the fallback similarity scores `{score, local}` given to entities that have no embedding vector yet. `score` for ordinary entities, `local` for entities local to the current proof context. Default: `{0.0, 0.0}`.
- `templates`, `task_description` — the text wrappers applied to queries and documents before embedding;
- `providers.<domain>.normalization` — maps the canonical model name (i.e., the name seen in HugginFace) to the id that the endpoint expects;
- `providers.<domain>.batch` — its presence enables the OpenAI-style Batch API for that endpoint.

The bundled [template](Isabelle_Semantic_Embedding/embedding_config_template.yaml) is an example.

### 7.2 Excluding entities from retrieval

Some entities are internal infrastructure: they are not meant to be used outside their own formalization, and therefore should not surface in search results. Built-in heuristics already exclude most of them (hidden names, datatype/record internals, …); to mark more by hand:

```isabelle
declare [[infra_constant Foo.bar]]      (* a constant; cascades to theorems mentioning it *)
declare [[infra_type Foo.t]]            (* a type *)
declare some_lemma[infra_thm]           (* a theorem *)
declare [[infra_constant del Foo.bar]]  (* del undoes any of the above *)
```

### 7.3 Reranker

A reranking stage after the vector search is implemented (option `Semantic_Embedding.reranker_model`) but disabled by default and currently unused: in our measurements it actually harms the retrieval performance.
