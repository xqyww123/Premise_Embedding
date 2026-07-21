"""Shared machinery for the package's user-editable YAML configuration files.

Two configs use it, and they deliberately live in different places:

  * ``embedding_config``  -- ``$ISABELLE_HOME_USER/etc/embedding_config``, because
    the embedding provider is only ever driven from inside Isabelle.
  * ``config.yaml``       -- ``platformdirs.user_config_dir(...)``, because the R2
    sync runs offline too (``semantics_manage.py`` needs no Isabelle environment),
    and because it must not sit next to the database it synchronizes.

Both are seeded from a template bundled in the package on first use, then cached
for the life of the process.

SEEDING ONLY EVER CREATES A MISSING FILE.  A key added to a template later does
not reach a user who already has the file, so a template is not a defaults layer.
Read every optional key through a default written in code (see ``r2_sync``'s
``DEFAULT_*``); the template exists to show a new user what is settable.
"""
from __future__ import annotations

import os
import pathlib
import shutil
from collections.abc import Callable

import yaml

_PACKAGE_DIR = pathlib.Path(__file__).parent


class User_Config:
    """A YAML file seeded from a bundled template, loaded once per process."""

    def __init__(self, template_name: str, env_var: str,
                 locate: 'Callable[[], pathlib.Path | None]'):
        """`env_var` overrides the path (for tests); `locate` finds it otherwise,
        returning None when the location cannot be determined at all."""
        self._template = _PACKAGE_DIR / template_name
        self._env_var = env_var
        self._locate = locate
        self._data: dict | None = None
        self._source: pathlib.Path | None = None

    def path(self) -> 'pathlib.Path | None':
        override = os.getenv(self._env_var)
        if override:
            return pathlib.Path(override)
        return self._locate()

    def load(self, force_reload: bool = False) -> dict:
        """Load and cache the config.

        Falls back to reading the bundled template read-only when the user
        location cannot be resolved — the caller then gets template values, not
        an error, which is why no *option that matters* may live only there.
        """
        if self._data is not None and not force_reload:
            return self._data
        path = self.path()
        if path is not None:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self._template, path)
            source = path
        else:
            source = self._template
        with open(source, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._data, self._source = data, source
        return data

    def source(self) -> 'pathlib.Path | None':
        """Path the active config was loaded from (for diagnostics)."""
        return self._source


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def env_bool(name: str) -> 'bool | None':
    """Parse a boolean environment variable.  None when unset (fall through to
    the config file); empty string means an explicit False.

    Anything else is an error.  Do NOT fall back on the ``os.getenv(x, "") != ""``
    idiom used for SEMANTIC_PERSIST_WIP: under it ``SEMANTIC_EMBEDDING_AUTO_UPDATE=0``
    reads as True, so a user turning a switch off would turn it on.  Guessing is worse
    than refusing — a flag nobody can reliably disable is a flag nobody trusts.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean. Use one of "
        f"{', '.join(_TRUE)} or {', '.join(x for x in _FALSE if x)}.")
