"""Synchronize the semantic database with a Cloudflare R2 bucket.

The remote is ONE object: a ``tar.zst`` holding a consistent, compacted copy of
``semantics.lmdb`` and of every ``vector_<model>.lmdb``, plus a ``MANIFEST.json``.
Its ``ETag`` is the version token, so one HEAD request tells you whether the local
copy is current, with no transfer.

READING IS ANONYMOUS, WRITING IS NOT.  The bucket is served publicly at
``public_url``, so ``pull`` / ``status`` / ``check_update`` are plain HTTPS and need
no credentials at all.  Only ``push`` speaks S3, and only ``push`` needs keys.
(The S3 endpoint signs *every* request, GET included, so it cannot be used for the
anonymous half; and the public endpoint returns no ``x-amz-meta-*``, which is why
what a snapshot declares about itself lives in its ``MANIFEST.json`` instead.)

A snapshot is validated after it is unpacked, not before it is fetched: the
manifest, a scan for legacy records, and a measurement of the actual vector byte
lengths.  Rejecting early would need metadata the anonymous endpoint will not
serve, and would buy only the ten seconds the (free) download takes.  A corrupt
download never reaches any of this — the zstd frame carries an XXH64 content
checksum and fails to extract.  Packing and unpacking are pure Python (``tarfile``
+ ``zstandard``), so the sync needs no ``tar``/``zstd`` binary on any platform.

The two directions are deliberately asymmetric:

  * ``pull`` MERGES the remote into the local stores, key by key.
  * ``push`` OVERWRITES the single remote object, wholesale.

So a push from a machine with less data silently narrows what everyone else
pulls.  ``push`` therefore refuses to run when the remote has moved since the
last sync, and never runs on its own — there is no automatic upload path.

``pull`` is only half-reversible: the pre-merge backup is the sole way back.
Everything that can reject a snapshot runs before the first byte is written.

BOTH DIRECTIONS ARE EXPLICIT.  Nothing here ever merges on its own: the only
automatic path is ``check_update``, which probes the remote at most once a week
and prints a line telling you to run ``pull`` yourself.  A background merge would
have to take the LMDB write lock inside somebody else's long-running process, and
undoing it means restoring a 0.7 GiB backup — not a thing to do unattended.

Failure semantics split by caller:

  * ``push_snapshot`` / ``pull_snapshot`` raise. ``semantics_manage.py`` exits non-zero.
  * ``check_update`` swallows everything and logs one line.

The push credentials come only from the environment and have no defaults:

    export R2_ACCESS_KEY_ID=...
    export R2_SECRET_ACCESS_KEY=...

Everything else is read as env > ``config.yaml`` > the ``DEFAULT_*`` below.
"""
from __future__ import annotations

import email.utils
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, NamedTuple

import lmdb
import platformdirs
import zstandard as zstd
from ._paths import semantic_DB_dir
from filelock import FileLock, Timeout as _LockTimeout

from ._user_config import User_Config, env_bool

# The snapshot format.  Bump when the tarball layout or a stored record's shape
# changes, so an older client refuses a snapshot it would silently misread.
SCHEMA_VERSION = "1"
VECTOR_FORMAT = "q15"

DEFAULT_ACCOUNT_ID = "532d99283b5aa1e02486ee3fdcb163d5"
DEFAULT_BUCKET = "mlml"
DEFAULT_OBJECT_KEY = "Isabelle_Semantic_Embedding.tar.zst"
# The bucket's public custom domain: anonymous reads, no signing, no credentials.
DEFAULT_PUBLIC_URL = "https://data.mlml.qiyuan.me"
DEFAULT_CHECK_INTERVAL_HOURS = 168      # weekly

CACHE_DIR = semantic_DB_dir()
MANIFEST_NAME = "MANIFEST.json"
MARKER_PATH = os.path.join(CACHE_DIR, ".r2_snapshot.json")
LOCK_PATH = os.path.join(CACHE_DIR, ".r2_pull.lock")
# Written while a merge is in flight, removed on success. A merge is many batched
# write-txns across stores (semantics.lmdb first), so a crash midway leaves a
# non-empty-but-incomplete DB that `semantic_db_is_empty` would call "present".
# This sentinel surviving means "the last pull did not finish" -> re-pull.
INCOMPLETE_PATH = os.path.join(CACHE_DIR, ".pull_incomplete")

# Peak transient usage, measured 2026-07-09: push needs the compacted copy
# (~1.5 GB) plus the tarball (~0.7 GB); pull adds the backup and the merge's
# growth (~4.4 GB total).  Both thresholds leave roughly a factor of two.
PUSH_MIN_FREE = 4 << 30
PULL_MIN_FREE = 6 << 30

# Values are ~8 KB each in a vector store, so a batch of 10k dirty pages is
# ~80 MB of write transaction — large enough to amortize, small enough to hold.
MERGE_BATCH = 10_000

_MULTIPART_THRESHOLD = 256 << 20
_MULTIPART_CHUNKSIZE = 128 << 20


class R2Error(RuntimeError):
    """Anything that should stop a sync: misconfiguration, an incompatible
    snapshot, a busy database, a full disk."""


class R2Busy(R2Error):
    """Raised when another process already holds the pull lock.  A subclass so the
    manual CLI's `except R2Error` still catches it, while the auto path can tell
    'someone else is downloading' apart from a genuine failure."""


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _user_config_dir() -> str:
    return platformdirs.user_config_dir("Isabelle_Semantic_Embedding", "Qiyuan")


_CONFIG = User_Config(
    "config_template.yaml", "SEMANTIC_EMBEDDING_CONFIG_PATH",
    lambda: pathlib.Path(_user_config_dir()) / "config.yaml")


class Settings(NamedTuple):
    account_id: str
    bucket: str
    endpoint: str          # S3 endpoint; signed, used only by push
    object_key: str
    public_url: str        # anonymous read origin; "" forces reads through S3 too
    auto_check: bool
    check_interval_hours: int

    @property
    def public_object_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/{self.object_key}"


def settings() -> Settings:
    """Resolve the R2 settings: env > config.yaml > the DEFAULT_* constants.

    The template is not a defaults layer — seeding only ever creates a *missing*
    file, so a key added to the template never reaches an existing user.  Every
    optional key is defaulted here, in code.
    """
    r2 = (_CONFIG.load().get("r2") or {})

    def pick(env: str, key: str, default: str) -> str:
        return os.getenv(env) or str(r2.get(key) or default)

    account_id = pick("R2_ACCOUNT_ID", "account_id", DEFAULT_ACCOUNT_ID)
    endpoint = pick("R2_ENDPOINT", "endpoint",
                    f"https://{account_id}.r2.cloudflarestorage.com")
    # Empty is meaningful: it says "no public origin", so reads fall back to the
    # signed S3 endpoint.  `pick` treats "" as unset, hence the explicit lookup.
    public_url = os.getenv("R2_PUBLIC_URL")
    if public_url is None:
        public_url = r2["public_url"] if "public_url" in r2 else DEFAULT_PUBLIC_URL
    public_url = str(public_url or "")

    def flag(env: str, key: str, default: bool) -> bool:
        from_env = env_bool(env)
        if from_env is not None:
            return from_env
        return default if r2.get(key) is None else bool(r2[key])

    hours = os.getenv("R2_CHECK_INTERVAL_HOURS") or r2.get("check_interval_hours")
    return Settings(
        account_id=account_id,
        bucket=pick("R2_BUCKET", "bucket", DEFAULT_BUCKET),
        endpoint=endpoint,
        object_key=pick("R2_OBJECT_KEY", "object_key", DEFAULT_OBJECT_KEY),
        public_url=public_url,
        auto_check=flag("R2_AUTO_CHECK", "auto_check", True),
        check_interval_hours=int(hours if hours is not None else DEFAULT_CHECK_INTERVAL_HOURS),
    )


def manage_script() -> str:
    """Absolute path of semantics_manage.py, for telling a user what to run.

    It sits beside the package directory in a source checkout.  A wheel does not
    ship it (it is a script, not package data), so fall back to the bare name.
    """
    p = pathlib.Path(__file__).resolve().parent.parent / "semantics_manage.py"
    return str(p) if p.exists() else "semantics_manage.py"


def config_path() -> str:
    """Where the settings are (or would be) stored; for diagnostics."""
    p = _CONFIG.path()
    return str(p) if p else "<unresolvable>"


def _client(s: Settings):
    """A boto3 S3 client for R2.  Only `push` needs one; reads go over HTTPS.

    Importing boto3 and building this costs ~0.4s of pure Python, so it stays
    behind the read paths rather than in front of them.

    No `request_checksum_calculation="when_required"` here.  botocore >= 1.36
    computes a CRC32 flexible checksum per request, which is widely reported to
    break S3-compatible backends — but measured against this bucket (botocore
    1.43.44) both a single PUT and a multipart upload succeed under the default.
    """
    import boto3
    from botocore.config import Config

    key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret = os.getenv("R2_SECRET_ACCESS_KEY")
    if not key_id or not secret:
        hint = ("`pull` and `status` need none of this — an unset public_url is "
                "what routed this one through the signed endpoint."
                if not s.public_url else
                "`pull` and `status` need no credentials; only `push` does.")
        raise R2Error(
            "R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY are not set. They have no "
            f"defaults; export them (e.g. from secret.sh). {hint}")
    return boto3.client(
        "s3", endpoint_url=s.endpoint,
        aws_access_key_id=key_id, aws_secret_access_key=secret,
        region_name="auto",
        config=Config(signature_version="s3v4", connect_timeout=5,
                      read_timeout=10, retries={"max_attempts": 2}))


# Short, because `check_update` runs during another process's startup and must
# not stall it: urllib has no default timeout at all, which would hang forever.
_HTTP_TIMEOUT = 15

# Cloudflare answers 403 to urllib's default `Python-urllib/3.x` User-Agent, and
# 200 to anything else -- including no User-Agent at all. Measured against this
# very domain; without this header every anonymous read fails.
_USER_AGENT = "Isabelle_Semantic_Embedding/r2_sync"


def _http(url: str, method: str):
    """Open `url`, or return None on 404.  Anonymous; no signing, no credentials."""
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": _USER_AGENT})
    try:
        return urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code == 404:
            return None
        raise R2Error(f"{method} {url} -> HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise R2Error(f"{method} {url} -> {e.reason}") from e


# ---------------------------------------------------------------------------
# The local marker: what we last saw of the remote
# ---------------------------------------------------------------------------

def read_marker() -> dict:
    try:
        with open(MARKER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_marker(**fields: Any) -> None:
    marker = read_marker()
    marker.update(fields)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = MARKER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)
    os.replace(tmp, MARKER_PATH)


class Remote_Head(NamedTuple):
    etag: str
    size: int
    last_modified: datetime


def remote_head(s: 'Settings | None' = None, client=None) -> 'Remote_Head | None':
    """One HEAD request.  None when the object does not exist yet.

    Anonymous over `public_url` when there is one, so `status` and `check_update`
    need no credentials.  `client` forces the signed S3 path (push already has one).

    The ETag is an opaque version token, NOT a content hash: a multipart upload
    yields `<md5-of-part-md5s>-<nparts>`, which also shifts with the chunk size.
    Only ever compare it for equality.
    """
    s = s or settings()
    if client is None and s.public_url:
        r = _http(s.public_object_url, "HEAD")
        if r is None:
            return None
        with r:
            h = r.headers
        return Remote_Head(
            etag=(h.get("ETag") or "").strip('"'),
            size=int(h.get("Content-Length") or 0),
            last_modified=email.utils.parsedate_to_datetime(h["Last-Modified"]))

    from botocore.exceptions import ClientError
    client = client or _client(s)
    try:
        r = client.head_object(Bucket=s.bucket, Key=s.object_key)
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == 404:
            return None
        # NOT 403: that is "you may not look", not "it is not there". Reading it
        # as absent would let `push` skip its overwrite check and clobber the remote.
        if code == 403:
            raise R2Error(f"R2 refused the HEAD of s3://{s.bucket}/{s.object_key} (403). "
                          f"Check R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY and the "
                          f"token's permissions on this bucket.") from e
        raise
    return Remote_Head(etag=r["ETag"].strip('"'), size=r["ContentLength"],
                       last_modified=r["LastModified"])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

_ZSTD_LEVEL = 3   # zstd's own default, and what `tar --zstd` used before this


def _write_tar_zst(out_path: str, base_dir: str, arcnames: 'list[str]',
                   exclude: 'str | None' = None) -> None:
    """Stream `arcnames` (each relative to `base_dir`) into a tar, zstd-compressed
    with a frame checksum, at `out_path`.  Pure Python -- no `tar`/`zstd` binary,
    so it behaves the same on Linux, macOS and Windows.  `exclude`, if given, drops
    every entry with a path component of that name (like `tar --exclude=NAME`; a
    dir dropped this way is not recursed into).  The result is an ordinary
    `.tar.zst`, interchangeable with one made by `tar --zstd`.
    """
    def _filter(ti: tarfile.TarInfo) -> 'tarfile.TarInfo | None':
        return None if exclude and exclude in ti.name.split("/") else ti

    cctx = zstd.ZstdCompressor(level=_ZSTD_LEVEL, write_checksum=True)
    with open(out_path, "wb") as fh, cctx.stream_writer(fh) as z:
        with tarfile.open(mode="w|", fileobj=z) as tar:
            for name in arcnames:
                tar.add(os.path.join(base_dir, name), arcname=name, filter=_filter)


def _extract_tar_zst(tarball: str, dest: str) -> None:
    """Extract a `.tar.zst` (ours, or one made by `tar --zstd`) into `dest`.  The
    zstd frame checksum makes a corrupt or truncated archive raise here rather than
    reach the merge -- the guarantee the old `tar --zstd` extraction gave us.  The
    `data` filter rejects absolute paths and `..` escapes, since extraction runs
    before the manifest is even read."""
    dctx = zstd.ZstdDecompressor()
    with open(tarball, "rb") as fh, dctx.stream_reader(fh) as z:
        with tarfile.open(mode="r|", fileobj=z) as tar:
            tar.extractall(dest, filter="data")


def _free_bytes(path: str) -> int:
    probe = path
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe).free


def _require_disk(need: int, *paths: str) -> None:
    for p in paths:
        free = _free_bytes(p)
        if free < need:
            raise R2Error(
                f"only {free / 1024 ** 3:.1f} GiB free on the filesystem holding {p}; "
                f"this operation needs {need / 1024 ** 3:.0f} GiB. Free space and retry.")


def _open_handles() -> list[str]:
    """Processes (other than this one) holding a file open under CACHE_DIR.

    LMDB has a single writer and an inter-process lock, so packing or merging
    while a collection run or an AoA agent is live either blocks it or captures a
    torn snapshot.  `lsof` cannot say "writer", only "has it open" — which is the
    conservative reading, since a holder may start writing at any moment.
    """
    exe = shutil.which("lsof")
    if exe is None or not os.path.isdir(CACHE_DIR):
        return []
    r = subprocess.run([exe, "-F", "pcf", "+D", CACHE_DIR],
                       capture_output=True, text=True)   # exit 1 == nothing found
    holders, pid, cmd = {}, None, None
    for line in r.stdout.splitlines():
        tag, val = line[:1], line[1:]
        if tag == "p":
            pid = val
        elif tag == "c":
            cmd = val
        elif tag == "f" and val.isdigit() and pid != str(os.getpid()):
            holders[pid] = cmd                          # a real fd, not cwd/rtd/txt/mem
    return [f"{c} (pid {p})" for p, c in holders.items()]


def _require_idle(force: bool) -> None:
    holders = _open_handles()
    if not holders:
        return
    msg = ("the database is open in another process: " + ", ".join(holders) +
           ". Stop it first — packing or merging under a live writer corrupts "
           "the snapshot.")
    if not force:
        raise R2Error(msg)
    _log(f"WARNING (--force): {msg}")


def _report_blockers(need: int, *paths: str) -> None:
    """Print what would stop the real run.  A dry run must never fail on a
    preflight check — naming the obstacle is the whole point of asking."""
    holders = _open_handles()
    if holders:
        _log(f"  BLOCKED: the database is open in {', '.join(holders)}")
    for p in paths:
        free = _free_bytes(p)
        if free < need:
            _log(f"  BLOCKED: {free / 1024 ** 3:.1f} GiB free on the filesystem "
                 f"holding {p}; {need / 1024 ** 3:.0f} GiB needed")
    if not holders:
        _log("  preflight: no other process holds the database")


@contextmanager
def _pull_lock():
    """Serialise every pull -- auto or manual, same process or not -- against each
    other, so two never merge into the same stores at once.

    `filelock` is a cross-platform advisory lock (fcntl on POSIX, msvcrt on
    Windows); like flock it is an OS-held lock the kernel drops when the holder
    closes the handle OR dies, so a crashed puller never leaves a stale lock -- no
    TTL, no manual recovery.  `timeout=0` is non-blocking: a second puller fails
    fast with R2Busy rather than queueing.  Even two coroutines in one process
    exclude each other (verified: the default FileLock is not re-entrant across
    instances)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    lock = FileLock(LOCK_PATH, timeout=0)
    try:
        lock.acquire()
    except _LockTimeout:
        raise R2Busy("another process is already pulling (holds .r2_pull.lock)")
    try:
        yield
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Snapshot introspection
# ---------------------------------------------------------------------------

def _store_dirs(root: str) -> list[str]:
    """Names of the LMDB stores in a snapshot (or in the live cache)."""
    if not os.path.isdir(root):
        return []
    return [e for e in sorted(os.listdir(root))
            if e.endswith(".lmdb") and os.path.isdir(os.path.join(root, e))
            and (e == "semantics.lmdb" or e.startswith("vector_"))]


def _model_of(store: str) -> 'str | None':
    """Canonical embedding-model name encoded in a vector store's directory name."""
    if not store.startswith("vector_"):
        return None
    from .semantic_embedding import unsanitize_model
    return unsanitize_model(store[len("vector_"):-len(".lmdb")])


def _local_dimension(model: str) -> 'int | None':
    """The dimension this machine expects for `model`, or None if unconfigured."""
    from .embedding_config import dimension
    try:
        return dimension(model)
    except KeyError:
        return None


def _human(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GiB" if n >= 1 << 30 else f"{n / 1024 ** 2:.1f} MiB"


def _progress(label: str, total: int):
    state = {"done": 0, "pct": -5}

    def cb(n: int) -> None:
        state["done"] += n
        pct = int(100 * state["done"] / total) if total else 100
        if pct >= state["pct"] + 5:
            state["pct"] = pct - pct % 5
            _log(f"  {label}: {pct:3d}%  ({_human(state['done'])})")
    return cb


def _download(url: str, dest: str, expected: int) -> None:
    """Fetch `url` to `dest` anonymously, with progress.

    One stream, deliberately.  The origin does serve `Range`, so ranged GETs in a
    thread pool look tempting; measured against it, they are slower (40 MB/s on
    one stream, 20 on four, 10 on eight — the link, not the client, is the
    bottleneck), and one eight-way run was answered with the *whole object* for
    some parts, having quietly ignored the Range header.  Do not "optimize" this.

    Throughput swings between roughly 3 and 40 MB/s minute to minute, so a slow
    pull is the network, not a regression: the snapshot is far past Cloudflare's
    512 MB cacheable-file limit, so every read goes to the origin, uncached.

    A short read means a truncated transfer, which `tar --zstd` would also catch
    (its XXH64 fails to extract) — but saying so here names the cause.
    """
    r = _http(url, "GET")
    if r is None:
        raise R2Error(f"GET {url} -> 404; the object vanished between HEAD and GET")
    cb = _progress("download", expected)
    got = 0
    with r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            cb(len(chunk))
    if expected and got != expected:
        raise R2Error(f"truncated download: got {got} of {expected} bytes")


# ---------------------------------------------------------------------------
# Compatibility gates.  All of them read the unpacked snapshot: what a snapshot
# says about itself is written by whoever pushed it, but its bytes are not.
# ---------------------------------------------------------------------------

def _check_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise R2Error(f"tarball manifest has schema_version "
                      f"{manifest.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
    if manifest.get("vector_format", VECTOR_FORMAT) != VECTOR_FORMAT:
        raise R2Error(f"tarball manifest declares vector_format "
                      f"{manifest['vector_format']!r}, expected {VECTOR_FORMAT!r}")


def _check_no_legacy(semantics_path: str) -> None:
    """Refuse a snapshot carrying pre-XOR records (§4.4, row 4).

    A record with no constituent list cannot be attributed to a theory, so
    `remove` cannot delete it and `list` only warns about it.  Merging one in
    would import a defect we have no way to undo short of the backup.
    """
    from Isabelle_RPC_Host.universal_key import is_xor_prefixed_key
    from .semantics import record_constituent_hashes
    if not os.path.isdir(semantics_path):
        raise R2Error("the snapshot has no semantics.lmdb; refusing to merge it.")
    env = lmdb.open(semantics_path, readonly=True, lock=False)
    try:
        legacy = 0
        with env.begin() as txn:
            for key, val in txn.cursor():
                if is_xor_prefixed_key(bytes(key)) and record_constituent_hashes(bytes(val)) is None:
                    legacy += 1
    finally:
        env.close()
    if legacy:
        raise R2Error(
            f"the remote snapshot holds {legacy} legacy theorem/rule record(s) with "
            f"no constituent list. Purge them on the machine that pushed "
            f"(migrate_xor_thm_keys.py) and push again.")


def _check_vector_format(store_path: str, model: 'str | None') -> None:
    """Sample the store: every vector must be exactly D*2 bytes (Q1.15 int16).

    A D*4-byte value is a leftover float32 record; reinterpreting it as int16
    yields a plausible-looking but wrong vector, which is why this is fatal.
    """
    dim = _local_dimension(model) if model else None
    env = lmdb.open(store_path, readonly=True, lock=False)
    try:
        lengths, sampled = set(), 0
        with env.begin() as txn:
            for key, val in txn.cursor():
                if len(bytes(key)) == 16:
                    continue                       # theory embed-status, not a vector
                lengths.add(len(bytes(val)))
                sampled += 1
                if sampled >= 64:
                    break
    finally:
        env.close()
    if not sampled:
        return
    if len(lengths) > 1:
        raise R2Error(f"{os.path.basename(store_path)}: vectors have inconsistent "
                      f"lengths {sorted(lengths)}; the store is not uniformly encoded.")
    got = lengths.pop()
    if dim is None:
        _log(f"NOTE: {os.path.basename(store_path)}: {got}-byte vectors, "
             f"unverified (no local dimension for {model}).")
    elif got == dim * 4:
        raise R2Error(f"{os.path.basename(store_path)}: {got}-byte vectors are float32, "
                      f"not Q1.15. The pushing machine never ran the q15 migration.")
    elif got != dim * 2:
        raise R2Error(f"{os.path.basename(store_path)}: {got}-byte vectors, "
                      f"expected {dim * 2} for {model}.")


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------

class Merge_Stats(NamedTuple):
    added: int              # keys the local store did not have
    overwritten: int        # keys the remote replaced
    thy_kept_local: int     # 16-byte theory records where local was the finished one


def _keep_local_thy_status(local_raw: bytes, remote_raw: bytes) -> bool:
    """Merge rule for the 16-byte theory records: `finished` is a logical OR.

    Blind overwriting would knock a locally *finished* theory back to WIP, and
    the next collection run would re-interpret and re-embed it — burning API
    money for nothing.  Everything else (cost_usd, model, tokens) follows the
    more-finished side, remote on a tie.
    """
    from .semantics import unpack_thy_status
    local = unpack_thy_status(local_raw)
    remote = unpack_thy_status(remote_raw)
    return bool(local.get(b"finished")) and not bool(remote.get(b"finished"))


def merge_env(src: lmdb.Environment, dst: lmdb.Environment,
              batch: int = MERGE_BATCH) -> Merge_Stats:
    """Merge every key of `src` into `dst`: the remote wins, except on the
    16-byte theory records (see `_keep_local_thy_status`).

    Committed in batches rather than as one transaction — a single write txn
    would have to hold well over a gigabyte of dirty pages.  A crash halfway
    therefore leaves a partial merge; the pre-merge backup is the way back, and
    the merge is idempotent, so re-running it is also safe.
    """
    added = overwritten = kept = 0
    with src.begin() as rtxn:
        cur = rtxn.cursor()
        if not cur.first():
            return Merge_Stats(0, 0, 0)
        exhausted = False
        while not exhausted:
            with dst.begin(write=True) as wtxn:
                for _ in range(batch):
                    key, val = bytes(cur.key()), bytes(cur.value())
                    old = wtxn.get(key)
                    if old is None:
                        wtxn.put(key, val)
                        added += 1
                    elif len(key) == 16 and _keep_local_thy_status(old, val):
                        kept += 1
                    elif old != val:
                        wtxn.put(key, val)
                        overwritten += 1
                    if not cur.next():
                        exhausted = True
                        break
    return Merge_Stats(added, overwritten, kept)


def _merge_snapshot(root: str) -> None:
    """Merge an extracted snapshot into the live stores, then rebuild the index."""
    from .semantic_embedding import _get_lmdb_env
    from .semantics import Semantic_DB

    for store in _store_dirs(root):
        src_path = os.path.join(root, store)
        model = _model_of(store)
        if store != "semantics.lmdb":
            _check_vector_format(src_path, model)

        # Ask before opening: _get_lmdb_env creates the directory it is given.
        existed = os.path.isdir(os.path.join(CACHE_DIR, store))
        # Go through the package's own environments: py-lmdb refuses to open one
        # environment twice in a process, and these are process-wide singletons.
        dst = Semantic_DB._ensure_env() if store == "semantics.lmdb" \
            else _get_lmdb_env(os.path.join(CACHE_DIR, store))
        src = lmdb.open(src_path, readonly=True, lock=False)
        try:
            stats = merge_env(src, dst)
        finally:
            src.close()
        _log(f"  {store}{'' if existed else '  (new)'}: "
             f"+{stats.added} added, {stats.overwritten} overwritten, "
             f"{stats.thy_kept_local} local theory record(s) kept")

    # Not optional.  The experience index is a derived view of the EXPERIENCE
    # records; without this, every experience just merged in is unreachable to
    # retrieval AND invisible to AoA's dedup, which then re-learns it.
    _log("  rebuilding the experience index...")
    n = Semantic_DB.rebuild_experience_index()
    _log(f"  experience index rebuilt from {n} record(s)")

    _warn_on_map_size()


def _warn_on_map_size() -> None:
    from .semantics import SEMANTICS_MAP_SIZE
    data = os.path.join(CACHE_DIR, "semantics.lmdb", "data.mdb")
    if not os.path.exists(data):
        return
    pct = 100.0 * os.path.getsize(data) / SEMANTICS_MAP_SIZE
    if pct > 80:
        _log(f"WARNING: semantics.lmdb is at {pct:.0f}% of its {SEMANTICS_MAP_SIZE >> 30} GiB "
             f"map_size. Writes fail at 100%. Raise SEMANTICS_MAP_SIZE.")


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------

def _pack_snapshot(tmp: str) -> str:
    """Write a consistent, compacted copy of every store into `tmp`, tar it up.

    `env.copy(compact=True)` is a hot backup: it takes a read transaction, so it
    needs no quiescent database, and it drops the free pages a live store
    accumulates.  Never tar an LMDB directory in place — that captures whatever
    torn state the writer happened to be in.
    """
    from .semantic_embedding import _get_lmdb_env
    from .semantics import Semantic_DB

    stores: dict[str, dict] = {}
    for store in _store_dirs(CACHE_DIR):
        env = Semantic_DB._ensure_env() if store == "semantics.lmdb" \
            else _get_lmdb_env(os.path.join(CACHE_DIR, store))
        out = os.path.join(tmp, store)
        os.makedirs(out, exist_ok=True)
        _log(f"  compacting {store}...")
        env.copy(out, compact=True)
        model = _model_of(store)
        stores[store] = {"entries": env.stat()["entries"],
                         "bytes": os.path.getsize(os.path.join(out, "data.mdb")),
                         **({"model": model, "dimension": _local_dimension(model)}
                            if model else {})}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "vector_format": VECTOR_FORMAT,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_by": os.uname().nodename,
        "stores": stores,
    }
    with open(os.path.join(tmp, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    tarball = os.path.join(tmp, "snapshot.tar.zst")
    _log("  packing tar.zst...")
    _write_tar_zst(tarball, tmp, [MANIFEST_NAME, *stores.keys()])
    return tarball


def push_snapshot(*, force: bool = False, dry_run: bool = False) -> None:
    """Pack the local stores and overwrite the single remote object."""
    s = settings()
    client = _client(s)
    head = remote_head(s, client)
    marker = read_marker()
    stale = head is not None and head.etag != marker.get("etag")

    if dry_run:
        _log(f"dry run: would pack {', '.join(_store_dirs(CACHE_DIR)) or '(no stores)'}")
        _log(f"         and upload to s3://{s.bucket}/{s.object_key}")
        _report_blockers(PUSH_MIN_FREE, CACHE_DIR)
        if stale:
            _log("  BLOCKED: the remote moved since this machine last synced; pull first")
        return

    if stale:
        msg = (f"the remote object changed since this machine last synced "
               f"(remote ETag {head.etag}, local marker {marker.get('etag')!r}).\n"  # type: ignore[union-attr]
               f"  push OVERWRITES the whole snapshot, so this would discard whatever "
               f"another machine uploaded.\n  Run `pull` first.")
        if not force:
            raise R2Error(msg)
        _log(f"WARNING (--force): {msg}")

    _require_idle(force)
    _require_disk(PUSH_MIN_FREE, CACHE_DIR)

    tmp = tempfile.mkdtemp(prefix=".r2_push_", dir=os.path.dirname(CACHE_DIR))
    try:
        tarball = _pack_snapshot(tmp)
        size = os.path.getsize(tarball)

        from boto3.s3.transfer import TransferConfig
        _log(f"  uploading to s3://{s.bucket}/{s.object_key}")
        client.upload_file(
            tarball, s.bucket, s.object_key,
            Config=TransferConfig(multipart_threshold=_MULTIPART_THRESHOLD,
                                  multipart_chunksize=_MULTIPART_CHUNKSIZE),
            Callback=_progress("upload", size))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Signed HEAD, not the public one: the CDN may not see the new object yet,
    # and the marker must record what we actually uploaded.
    head = remote_head(s, client)
    _write_marker(etag=head.etag if head else None,
                  pushed_at=time.time(), last_checked_at=time.time())
    _log(f"Pushed {_human(size)}.")


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

def _backup(keep: int = 2) -> str:
    """tar.zst the whole cache directory before a merge, keeping the last `keep`.

    Without rotation a few pulls would eat the free space the next one preflights
    for.  `embed_cache` is a pure API-response cache and is not worth the bytes.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    home = os.path.expanduser("~")
    out = os.path.join(home, f"Isabelle_Semantic_Embedding.backup_{stamp}.tar.zst")
    _log(f"  backing up to {out}")
    _write_tar_zst(out, os.path.dirname(CACHE_DIR), [os.path.basename(CACHE_DIR)],
                   exclude="embed_cache")

    prefix = "Isabelle_Semantic_Embedding.backup_"
    old = sorted(e for e in os.listdir(home)
                 if e.startswith(prefix) and e.endswith(".tar.zst"))
    for e in old[:-keep]:
        os.remove(os.path.join(home, e))
        _log(f"  removed old backup {e}")
    return out


def pull_snapshot(*, backup: bool = True, force: bool = False,
                  dry_run: bool = False, require_idle: bool = True,
                  on_phase: 'Callable[[str], None] | None' = None) -> bool:
    """Download the remote snapshot and merge it into the local stores.

    Anonymous when `public_url` is set, which it is by default: no credentials.
    Returns False when the local copy is already current.  Every check that can
    reject the snapshot runs before the merge writes its first key; past that
    point the backup is the only way back.

    `require_idle=False` skips the "is another process holding the DB open?" gate.
    The auto-pull path passes it: that gate was written for push/pack (whose hot
    copy wants a quiescent DB), but a merge is safe under concurrent readers and
    writers -- LMDB serialises the writers and the merge is idempotent -- so on a
    machine running several RPC hosts off one cache, keeping the gate would only
    wedge cold-start (each host holds the empty env, so every pull refuses).

    `on_phase`, if given, is called with 'downloading' / 'extracting' / 'merging'
    as each stage begins, so a caller on an event loop can report progress.
    """
    def _phase(name: str) -> None:
        if on_phase:
            on_phase(name)
    s = settings()
    with _pull_lock():
        where = s.public_object_url if s.public_url else f"s3://{s.bucket}/{s.object_key}"
        client = None if s.public_url else _client(s)
        head = remote_head(s, client)
        if head is None:
            raise R2Error(f"{where} does not exist. Nothing to pull — `push` first.")

        marker = read_marker()
        if head.etag == marker.get("etag") and not force:
            _log(f"Already up to date (ETag {head.etag}).")
            _write_marker(last_checked_at=time.time())
            return False

        if dry_run:
            _log(f"dry run: would download {_human(head.size)} "
                 f"(ETag {head.etag}, {head.last_modified:%Y-%m-%d %H:%M}) and merge it")
            _report_blockers(PULL_MIN_FREE, CACHE_DIR, os.path.expanduser("~"))
            return True

        _require_disk(PULL_MIN_FREE, CACHE_DIR, os.path.expanduser("~"))
        if require_idle:
            _require_idle(force)

        if backup:
            _backup()

        tmp = tempfile.mkdtemp(prefix=".r2_pull_", dir=os.path.dirname(CACHE_DIR))
        try:
            _phase("downloading")
            tarball = os.path.join(tmp, "snapshot.tar.zst")
            _log(f"  downloading {_human(head.size)} from {where}")
            if client is None:
                _download(s.public_object_url, tarball, head.size)
            else:
                from boto3.s3.transfer import TransferConfig
                client.download_file(
                    s.bucket, s.object_key, tarball,
                    Config=TransferConfig(multipart_threshold=_MULTIPART_THRESHOLD,
                                          multipart_chunksize=_MULTIPART_CHUNKSIZE),
                    Callback=_progress("download", head.size))

            root = os.path.join(tmp, "snapshot")
            os.makedirs(root)
            # The zstd frame carries an XXH64 content checksum, so a corrupt or
            # truncated download fails to extract here rather than reaching the merge.
            _phase("extracting")
            _log("  extracting...")
            _extract_tar_zst(tarball, root)
            os.remove(tarball)                    # ~0.7 GiB back before the merge

            manifest_path = os.path.join(root, MANIFEST_NAME)
            if not os.path.exists(manifest_path):
                raise R2Error(f"the snapshot has no {MANIFEST_NAME}; refusing to merge "
                              f"a tarball of unknown provenance.")
            with open(manifest_path, "r", encoding="utf-8") as f:
                _check_manifest(json.load(f))

            _check_no_legacy(os.path.join(root, "semantics.lmdb"))
            # Re-check: minutes passed downloading, and a collection run may have
            # started since. Nothing has been written yet, so refusing here is free.
            if require_idle:
                _require_idle(force)
            _phase("merging")
            _log("  merging...")
            _mark_pull_incomplete()          # a crash mid-merge leaves this behind
            _merge_snapshot(root)
            _clear_pull_incomplete()         # merge finished -> DB is whole again
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        _write_marker(etag=head.etag, pulled_at=time.time(),
                      last_checked_at=time.time())
        _log(f"Pulled and merged (ETag {head.etag}).")
        return True


# ---------------------------------------------------------------------------
# status / the automatic path
# ---------------------------------------------------------------------------

def semantic_db_is_empty() -> bool:
    """True when there is no local semantic database worth speaking of: the
    `semantics.lmdb` store is absent, or present but holds zero records.  Cheap --
    an O(1) LMDB stat, never a scan -- so the AoA hook can call it every `by aoa`.
    """
    if not os.path.isdir(os.path.join(CACHE_DIR, "semantics.lmdb")):
        return True
    from .semantics import Semantic_DB
    return Semantic_DB._ensure_env().stat()["entries"] == 0


def semantic_db_record_count() -> int:
    """How many records the local `semantics.lmdb` holds (0 if absent)."""
    if not os.path.isdir(os.path.join(CACHE_DIR, "semantics.lmdb")):
        return 0
    from .semantics import Semantic_DB
    return Semantic_DB._ensure_env().stat()["entries"]


def _mark_pull_incomplete() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    open(INCOMPLETE_PATH, "w").close()


def _clear_pull_incomplete() -> None:
    try:
        os.remove(INCOMPLETE_PATH)
    except FileNotFoundError:
        pass


def pull_was_interrupted() -> bool:
    """True if a previous merge died partway (the sentinel outlived it), so the
    local DB may be non-empty yet incomplete and should be re-pulled."""
    return os.path.exists(INCOMPLETE_PATH)


def status() -> None:
    """Compare the local database with the remote.  Needs no credentials."""
    s = settings()
    marker = read_marker()
    _log(f"config      : {config_path()}")
    _log(f"read from   : {s.public_object_url if s.public_url else '(no public URL)'}")
    _log(f"push to     : s3://{s.bucket}/{s.object_key}  ({s.endpoint})")
    _log(f"auto_check  : {s.auto_check}, every {s.check_interval_hours}h "
         f"(checks and warns; never merges)")
    _log("")

    for label, ts in (("last pulled", marker.get("pulled_at")),
                      ("last pushed", marker.get("pushed_at")),
                      ("last checked", marker.get("last_checked_at"))):
        when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "never"
        _log(f"{label:<13}: {when}")
    _log(f"local ETag   : {marker.get('etag') or '—'}")

    head = remote_head(s)
    _write_marker(last_checked_at=time.time())
    if head is None:
        _log("remote       : (no object yet)")
        return
    _log(f"remote ETag  : {head.etag}")
    _log(f"remote size  : {_human(head.size)}, uploaded "
         f"{head.last_modified:%Y-%m-%d %H:%M}")
    _log("")
    _log("Up to date." if head.etag == marker.get("etag")
         else f"A newer Semantic-Embedding DB is available. "
              f"Run: {manage_script()} pull")


_checked_this_process = False


def check_update(log: 'Callable[[str], None] | None' = None) -> None:
    """Probe the remote at most once per `check_interval_hours` and, if it holds a
    newer database, say so.  Anonymous: no credentials.  Never downloads, never
    merges, never raises.

    Blocking, on purpose: it is one synchronous HTTPS HEAD.  A caller on an event
    loop — the Isabelle RPC host runs several coroutines on one — should know it
    stalls that loop for the request, and for `_HTTP_TIMEOUT` when unreachable.

    `log` defaults to printing, which is right for a terminal and useless inside
    the RPC host: `fork_and_launch__` (Isabelle_RPC's rpc.py) daemonizes and
    dup2's stdout onto /dev/null.  Pass `connection.server.logger.warning` there.

    Suppresses every exception.  It is a courtesy check running inside somebody
    else's process — a headless AoA batch, a collection run — where a traceback
    would take down the run and nobody would ever see it.
    """
    global _checked_this_process
    log = log or _log
    try:
        # The weekly marker throttles the network call; this throttles the file
        # read, because the AoA hook fires once per `by aoa`, not once per process.
        if _checked_this_process:
            return
        _checked_this_process = True

        s = settings()
        if not s.auto_check:
            return
        marker = read_marker()
        last = marker.get("last_checked_at") or 0
        if time.time() - last < s.check_interval_hours * 3600:
            return
        try:
            head = remote_head(s)
        finally:
            # Advance the clock even on failure. Otherwise a machine whose R2
            # egress is blackholed re-runs this slow path at every startup.
            _write_marker(last_checked_at=time.time())
        if head is None or head.etag == marker.get("etag"):
            return

        synced = marker.get("pulled_at") or marker.get("pushed_at")
        since = (datetime.fromtimestamp(synced).strftime("%Y-%m-%d") if synced
                 else "never")
        log(f"[semantic-db] A newer Semantic-Embedding DB is available "
            f"({head.last_modified:%Y-%m-%d}, {_human(head.size)}), "
            f"last synced here {since}. Run: {manage_script()} pull")
    except Exception as e:                        # noqa: BLE001 — see the docstring
        log(f"[semantic-db] update check skipped: {e}")