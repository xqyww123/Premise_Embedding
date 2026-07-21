"""Loader for the embedding provider configuration.

The configuration lives at ``$ISABELLE_HOME_USER/etc/embedding_config`` (YAML).
On first use it is seeded from the bundled template
``embedding_config_template.yaml`` next to this module. It carries, keyed by the
*canonical* model name (HuggingFace name where one exists, else the canonical
id) and by the base_url domain (netloc):

  - ``dimension``:      per-model embedding vector dimension (required in use)
  - ``default_scores``: per-model {score, local} fallback for un-embedded entities
  - ``providers``:      per-domain ``normalization`` (canonical -> API model id)
                        and ``batch`` (Batch API shape) config
  - ``templates`` / ``task_description``: per-model query/document text templates
                        applied (before caching) in Embedding_Provider, via
                        ``query_template`` / ``document_template`` / ``task_description``.

Set ``EMBEDDING_CONFIG_PATH`` to override the config file location (used by tests).
"""
from __future__ import annotations

import os
import pathlib

from ._user_config import User_Config


def _isabelle_home_user() -> pathlib.Path | None:
    """Resolve $ISABELLE_HOME_USER, falling back to ~/.isabelle/$ISABELLE_IDENTIFIER."""
    env = os.getenv("ISABELLE_HOME_USER")
    if env:
        return pathlib.Path(env)
    ident = os.getenv("ISABELLE_IDENTIFIER")
    if ident:
        return pathlib.Path.home() / ".isabelle" / ident
    return None


def _resolve_config_path() -> pathlib.Path | None:
    """Path of the editable config file, or None if it cannot be located."""
    home = _isabelle_home_user()
    return None if home is None else home / "etc" / "embedding_config"


_CONFIG = User_Config("embedding_config_template.yaml", "EMBEDDING_CONFIG_PATH",
                      _resolve_config_path)


def load_embedding_config(force_reload: bool = False) -> dict:
    """Load (and cache) the embedding configuration dict.

    Seeds the user config from the bundled template on first run. If the user
    config location cannot be resolved (e.g. ISABELLE_HOME_USER unset), falls
    back to reading the bundled template read-only.
    """
    return _CONFIG.load(force_reload)


def config_source() -> pathlib.Path | None:
    """Path the active config was loaded from (for diagnostics)."""
    return _CONFIG.source()


def dimension(model: str) -> int:
    """Embedding vector dimension for a canonical model name. Hard error if missing."""
    cfg = load_embedding_config()
    dims = cfg.get("dimension") or {}
    if model not in dims:
        raise KeyError(
            f"No 'dimension' entry for model {model!r} in embedding config "
            f"({config_source()}). Add it under 'dimension:'.")
    return int(dims[model])


def default_scores(model: str) -> tuple[float, float]:
    """(non-local, local) fallback scores for a model; defaults to (0.0, 0.0)."""
    cfg = load_embedding_config()
    entry = (cfg.get("default_scores") or {}).get(model)
    if not entry:
        return (0.0, 0.0)
    return (float(entry.get("score", 0.0)), float(entry.get("local", 0.0)))


def normalize(model: str) -> bool:
    """Whether to L2-normalize the model's returned vectors; defaults to False."""
    cfg = load_embedding_config()
    return bool((cfg.get("normalize") or {}).get(model, False))


def max_request_size(model: str, default: int = 2048) -> int:
    """Max texts per non-batch request for a model; defaults to `default`."""
    cfg = load_embedding_config()
    return int((cfg.get("max_request_size") or {}).get(model, default))


def _provider_entry(domain: str) -> dict:
    cfg = load_embedding_config()
    return (cfg.get("providers") or {}).get(domain) or {}


def api_model_name(domain: str, model: str) -> str:
    """The model id this domain expects in the API body; canonical name if unmapped."""
    norm = _provider_entry(domain).get("normalization") or {}
    return norm.get(model, model)


def batch_config(domain: str) -> dict | None:
    """Batch API config for this domain, or None if batch is not configured."""
    return _provider_entry(domain).get("batch")


def provider_listed(domain: str) -> bool:
    """Whether this domain has a ``providers:`` entry in the embedding config."""
    cfg = load_embedding_config()
    return domain in (cfg.get("providers") or {})


_DEFAULT_TEXT_TEMPLATE = "{text}"
_DEFAULT_TASK_DESCRIPTION = "retrieve the most relevant Isabelle/HOL constructs"
# Phrase filling a query template's {kinds} slot when the query has no (or an
# "all") EntityKind filter; also the value of render_kinds([]). Single source of
# truth -- imported by semantics.render_kinds so the two never drift.
_DEFAULT_KINDS_PHRASE = "constructs"


def query_template(model: str) -> str:
    """Query template for a canonical model name; defaults to '{text}' (raw)."""
    cfg = load_embedding_config()
    entry = (cfg.get("templates") or {}).get(model) or {}
    return entry.get("query", _DEFAULT_TEXT_TEMPLATE)


def document_template(model: str) -> str:
    """Document template for a canonical model name; defaults to '{text}' (raw)."""
    cfg = load_embedding_config()
    entry = (cfg.get("templates") or {}).get(model) or {}
    return entry.get("document", _DEFAULT_TEXT_TEMPLATE)


def task_description() -> str:
    """Static task sentence injected into a query template's {task} slot
    (model- and query-independent). Defaults to a plain retrieval sentence."""
    cfg = load_embedding_config()
    return cfg.get("task_description") or _DEFAULT_TASK_DESCRIPTION


# Task sentence for the experience-memory query pass (§8.2 of AoA's
# EXPERIENCE_MEMORY.md).  Unlike task_description it has NO {kinds} slot:
# experiences are retrieved by their own dedicated instruction, not the
# entity-kind phrase.  Passed to Embedding_Provider.embed via task_override.
_DEFAULT_EXPERIENCE_TASK_DESCRIPTION = (
    "Given a proof situation, retrieve experiences and proof strategies "
    "that help prove the goal")


def experience_task_description() -> str:
    """Task sentence for embedding an experience-memory retrieval query."""
    cfg = load_embedding_config()
    return cfg.get("experience_task_description") or _DEFAULT_EXPERIENCE_TASK_DESCRIPTION
