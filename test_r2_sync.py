"""Tests for the R2 sync's pure logic: the merge rules, settings resolution, and
the gates that reject an incompatible snapshot.

Nothing here touches the network or the real database.  `merge_env` is exercised
on throwaway LMDB environments, which is the whole of the merge: `pull` adds only
the download, the extraction, and the index rebuild around it.

The merge is where a bug is expensive.  It is half-reversible (the pre-merge
backup is the only way back), and its one non-obvious rule -- a theory finished
locally must not be knocked back to WIP by a remote that has not finished it --
guards against silently re-spending money on re-interpretation and re-embedding.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime

import lmdb
import msgpack
import pytest
import zstandard as zstd

from Isabelle_Semantic_Embedding import r2_sync
from Isabelle_Semantic_Embedding._user_config import env_bool


def _env(tmp_path, name: str) -> lmdb.Environment:
    path = tmp_path / name
    path.mkdir()
    return lmdb.open(str(path), map_size=1 << 20)


def _fill(env: lmdb.Environment, records: dict[bytes, bytes]) -> None:
    with env.begin(write=True) as txn:
        for k, v in records.items():
            txn.put(k, v)


def _read(env: lmdb.Environment) -> dict[bytes, bytes]:
    with env.begin() as txn:
        return {bytes(k): bytes(v) for k, v in txn.cursor()}


def _thy(finished: bool, cost: float) -> bytes:
    return msgpack.packb({b"finished": finished, b"cost_usd": cost})


THY_A = b"\xAA" * 16
THY_B = b"\xBB" * 16
ENTITY = b"\xCC" * 16 + b"\x01" + b"\x00" * 15


# ---------------------------------------------------------------------------
# merge_env
# ---------------------------------------------------------------------------

def test_remote_wins_on_entity_records(tmp_path):
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    new_key = b"\xDD" * 32
    _fill(dst, {ENTITY: b"local"})
    _fill(src, {ENTITY: b"remote", new_key: b"fresh"})

    stats = r2_sync.merge_env(src, dst)

    assert _read(dst) == {ENTITY: b"remote", new_key: b"fresh"}
    assert (stats.added, stats.overwritten, stats.thy_kept_local) == (1, 1, 0)


def test_an_identical_record_is_neither_added_nor_overwritten(tmp_path):
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    _fill(dst, {ENTITY: b"same"})
    _fill(src, {ENTITY: b"same"})

    stats = r2_sync.merge_env(src, dst)

    assert (stats.added, stats.overwritten, stats.thy_kept_local) == (0, 0, 0)


def test_a_theory_finished_locally_is_not_knocked_back_to_wip(tmp_path):
    """The one rule that is not "remote wins".

    Overwriting blindly would mark a locally-finished theory unfinished, and the
    next collection run would re-interpret and re-embed it -- pure API spend for
    a result already on disk.  `finished` is therefore a logical OR, and the
    other fields follow the more-finished side.
    """
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    _fill(dst, {THY_A: _thy(True, 5.0), THY_B: _thy(False, 1.0)})
    _fill(src, {THY_A: _thy(False, 0.5), THY_B: _thy(True, 9.0)})

    stats = r2_sync.merge_env(src, dst)

    got = _read(dst)
    assert msgpack.unpackb(got[THY_A]) == {b"finished": True, b"cost_usd": 5.0}, \
        "local `finished` survived a remote WIP"
    assert msgpack.unpackb(got[THY_B]) == {b"finished": True, b"cost_usd": 9.0}, \
        "remote `finished` was adopted over a local WIP"
    assert (stats.overwritten, stats.thy_kept_local) == (1, 1)


def test_when_both_sides_finished_the_remote_metadata_wins(tmp_path):
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    _fill(dst, {THY_A: _thy(True, 5.0)})
    _fill(src, {THY_A: _thy(True, 9.0)})

    r2_sync.merge_env(src, dst)

    assert msgpack.unpackb(_read(dst)[THY_A])[b"cost_usd"] == 9.0


def test_a_theory_absent_locally_is_taken_verbatim(tmp_path):
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    _fill(src, {THY_A: _thy(False, 1.0)})

    stats = r2_sync.merge_env(src, dst)

    assert msgpack.unpackb(_read(dst)[THY_A]) == {b"finished": False, b"cost_usd": 1.0}
    assert (stats.added, stats.thy_kept_local) == (1, 0)


def test_the_merge_spans_batches_and_is_idempotent(tmp_path):
    """merge_env commits in batches -- one write txn could not hold a whole
    vector store's dirty pages.  A batch boundary must not drop or re-count a key.
    """
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    records = {i.to_bytes(32, "big"): f"v{i}".encode() for i in range(25)}
    _fill(src, records)

    first = r2_sync.merge_env(src, dst, batch=4)     # 25 keys, 7 transactions
    assert _read(dst) == records
    assert (first.added, first.overwritten) == (25, 0)

    again = r2_sync.merge_env(src, dst, batch=4)
    assert _read(dst) == records
    assert (again.added, again.overwritten, again.thy_kept_local) == (0, 0, 0)


def test_merging_an_empty_snapshot_changes_nothing(tmp_path):
    src, dst = _env(tmp_path, "src"), _env(tmp_path, "dst")
    _fill(dst, {ENTITY: b"local"})

    assert r2_sync.merge_env(src, dst) == (0, 0, 0)
    assert _read(dst) == {ENTITY: b"local"}


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Point the loader at a throwaway config.yaml and drop the process cache."""
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("SEMANTIC_EMBEDDING_CONFIG_PATH", str(path))
    for var in ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ENDPOINT", "R2_OBJECT_KEY",
                "R2_PUBLIC_URL", "R2_AUTO_CHECK", "R2_CHECK_INTERVAL_HOURS"):
        monkeypatch.delenv(var, raising=False)

    def write(text: str) -> None:
        path.write_text(text)
        r2_sync._CONFIG.load(force_reload=True)
    write("")                                        # seeded-but-empty by default
    return write


def test_defaults_come_from_code_not_from_the_template(cfg):
    """Seeding only ever creates a *missing* file, so a key added to the template
    later never reaches an existing user.  An empty config must still resolve."""
    s = r2_sync.settings()
    assert s.account_id == r2_sync.DEFAULT_ACCOUNT_ID
    assert s.bucket == r2_sync.DEFAULT_BUCKET
    assert s.object_key == r2_sync.DEFAULT_OBJECT_KEY
    assert s.endpoint == f"https://{r2_sync.DEFAULT_ACCOUNT_ID}.r2.cloudflarestorage.com"
    assert s.public_url == r2_sync.DEFAULT_PUBLIC_URL
    assert s.auto_check is True
    assert s.check_interval_hours == r2_sync.DEFAULT_CHECK_INTERVAL_HOURS


def test_there_is_no_automatic_merge(cfg):
    """`check_update` warns; only the CLI merges.  If an `auto_pull` switch ever
    comes back, it must come back with its guardrails, not by accident."""
    assert not hasattr(r2_sync.settings(), "auto_pull")
    assert not hasattr(r2_sync, "maybe_auto_pull")


def test_pull_and_status_need_no_credentials(cfg, monkeypatch):
    """The whole point of the public origin.  `_client` is the only thing that
    demands keys, and only `push` may reach it."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    s = r2_sync.settings()

    assert s.public_object_url == \
        f"{r2_sync.DEFAULT_PUBLIC_URL}/{r2_sync.DEFAULT_OBJECT_KEY}"
    with pytest.raises(r2_sync.R2Error, match="only .?push.? does"):
        r2_sync._client(s)          # push's path, and only push's


def test_a_trailing_slash_on_the_public_url_does_not_double_up(cfg):
    cfg("r2:\n  public_url: https://example.test/\n  object_key: snap.tar.zst\n")
    assert r2_sync.settings().public_object_url == "https://example.test/snap.tar.zst"


def test_an_empty_public_url_forces_reads_through_s3(cfg):
    """`pick` treats "" as unset; this key must not, or a user who deliberately
    disables the public origin would silently keep using it."""
    cfg('r2:\n  public_url: ""\n')
    assert r2_sync.settings().public_url == ""


def test_the_endpoint_follows_a_configured_account_id(cfg):
    cfg("r2:\n  account_id: abc123\n")
    assert r2_sync.settings().endpoint == "https://abc123.r2.cloudflarestorage.com"


def test_env_beats_the_config_file(cfg, monkeypatch):
    cfg("r2:\n  bucket: from_file\n")
    assert r2_sync.settings().bucket == "from_file"
    monkeypatch.setenv("R2_BUCKET", "from_env")
    assert r2_sync.settings().bucket == "from_env"


def test_r2_auto_check_0_disables_it(cfg, monkeypatch):
    """The trap this exists to prevent: the package's one prior env-boolean idiom
    is `os.getenv(x, "") != ""`, under which `R2_AUTO_CHECK=0` reads as True --
    turning a switch ON for a user who set it to turn it off.
    """
    cfg("r2:\n  auto_check: true\n")
    assert r2_sync.settings().auto_check is True
    monkeypatch.setenv("R2_AUTO_CHECK", "0")
    assert r2_sync.settings().auto_check is False


def test_a_nonsense_boolean_is_an_error_not_a_guess(cfg, monkeypatch):
    monkeypatch.setenv("R2_AUTO_CHECK", "sure")
    with pytest.raises(ValueError, match="not a boolean"):
        r2_sync.settings()


# ---------------------------------------------------------------------------
# check_update — the one automatic path, and it only ever logs
# ---------------------------------------------------------------------------

@pytest.fixture
def check(cfg, monkeypatch, tmp_path):
    """Isolate the marker file and reset the once-per-process latch."""
    monkeypatch.setattr(r2_sync, "MARKER_PATH", str(tmp_path / "marker.json"))
    monkeypatch.setattr(r2_sync, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(r2_sync, "_checked_this_process", False)
    lines: list[str] = []

    def run(**marker):
        if marker:
            r2_sync._write_marker(**marker)
        monkeypatch.setattr(r2_sync, "_checked_this_process", False)
        r2_sync.check_update(log=lines.append)
        return lines
    return run


def _head(etag="new"):
    return r2_sync.Remote_Head(etag=etag, size=750 * 1024 ** 2,
                               last_modified=datetime(2026, 7, 9, 12, 0))


def test_check_update_names_the_command_to_run(check, monkeypatch):
    monkeypatch.setattr(r2_sync, "remote_head", lambda s: _head())
    [line] = check(etag="old", pulled_at=datetime(2026, 6, 1).timestamp())

    assert "A newer Semantic-Embedding DB is available" in line
    assert "2026-07-09" in line and "last synced here 2026-06-01" in line
    assert f"Run: {r2_sync.manage_script()} pull" in line
    assert os.path.isabs(r2_sync.manage_script()), "the command must be runnable as printed"


def test_check_update_is_silent_when_current(check, monkeypatch):
    monkeypatch.setattr(r2_sync, "remote_head", lambda s: _head(etag="same"))
    assert check(etag="same") == []


def test_check_update_is_silent_when_the_remote_is_empty(check, monkeypatch):
    monkeypatch.setattr(r2_sync, "remote_head", lambda s: None)
    assert check(etag="old") == []


def test_check_update_respects_the_interval(check, monkeypatch):
    called = []
    monkeypatch.setattr(r2_sync, "remote_head",
                        lambda s: called.append(1) or _head())
    assert check(etag="old", last_checked_at=time.time()) == []
    assert called == [], "probed the network inside the throttle window"


def test_check_update_probes_once_per_process(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(r2_sync, "MARKER_PATH", str(tmp_path / "m.json"))
    monkeypatch.setattr(r2_sync, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(r2_sync, "_checked_this_process", False)
    called = []
    monkeypatch.setattr(r2_sync, "remote_head", lambda s: called.append(1) or None)

    for _ in range(5):                    # the AoA hook fires once per `by aoa`
        r2_sync.check_update(log=lambda _: None)
    assert len(called) == 1


def test_check_update_advances_the_clock_even_when_r2_is_unreachable(check, monkeypatch):
    """Otherwise a machine whose R2 egress is blackholed re-runs the slow path at
    every single startup, paying the full connect+read timeout each time."""
    def boom(s):
        raise OSError("network is unreachable")
    monkeypatch.setattr(r2_sync, "remote_head", boom)

    [line] = check(etag="old")

    assert "update check skipped" in line and "unreachable" in line
    assert r2_sync.read_marker().get("last_checked_at"), "the clock did not advance"


def test_check_update_never_raises(check, monkeypatch):
    """It runs inside somebody else's process; a traceback would take down a
    headless AoA batch and nobody would ever see it."""
    def boom(s):
        raise RuntimeError("anything at all")
    monkeypatch.setattr(r2_sync, "remote_head", boom)
    assert len(check(etag="old")) == 1        # logged, not raised


def test_env_bool_distinguishes_unset_from_empty():
    assert env_bool("_R2_TEST_ABSENT") is None       # fall through to the config
    os.environ["_R2_TEST_EMPTY"] = ""
    try:
        assert env_bool("_R2_TEST_EMPTY") is False   # an explicit "off"
    finally:
        del os.environ["_R2_TEST_EMPTY"]


# ---------------------------------------------------------------------------
# the gates.  All of them read the unpacked snapshot: what a snapshot claims
# about itself is written by whoever pushed it, but its bytes are not.  There is
# deliberately no pre-download gate -- that would need metadata the anonymous
# endpoint does not serve, to save a download that is free and takes ten seconds.
# ---------------------------------------------------------------------------

def test_a_manifest_from_a_future_client_is_refused():
    with pytest.raises(r2_sync.R2Error, match="schema_version"):
        r2_sync._check_manifest({"schema_version": "99"})
    with pytest.raises(r2_sync.R2Error, match="vector_format"):
        r2_sync._check_manifest({"schema_version": r2_sync.SCHEMA_VERSION,
                                 "vector_format": "float32"})
    r2_sync._check_manifest({"schema_version": r2_sync.SCHEMA_VERSION})


def test_a_corrupt_download_never_reaches_the_merge(tmp_path):
    """`pull` verifies nothing by hand: the zstd frame carries an XXH64 content
    checksum (we write it with write_checksum=True), so a flipped byte fails to
    extract and no merge is attempted.  This pins that property, which is what let
    the sha256 pass go."""
    payload = tmp_path / "d" / "blob"
    payload.parent.mkdir()
    payload.write_bytes(bytes(range(256)) * 4096)
    tar = tmp_path / "snap.tar.zst"
    r2_sync._write_tar_zst(str(tar), str(tmp_path), ["d"])

    raw = bytearray(tar.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    tar.write_bytes(raw)

    with pytest.raises(zstd.ZstdError):     # the frame checksum fails on a flipped byte
        r2_sync._extract_tar_zst(str(tar), str(tmp_path / "out"))


def test_pack_and_extract_round_trip_excludes_and_interoperates_with_the_cli(tmp_path):
    """The pure-Python .tar.zst is byte-for-byte a normal one: `exclude` drops a
    named subtree, and the CLI `tar --zstd` can still read what we wrote (so the
    already-published snapshot, made by the CLI, stays pullable and vice versa)."""
    src = tmp_path / "src" / "sub"
    src.mkdir(parents=True)
    (tmp_path / "src" / "a.txt").write_text("alpha")
    (src / "b.txt").write_text("beta")
    (tmp_path / "src" / "embed_cache").mkdir()
    (tmp_path / "src" / "embed_cache" / "junk").write_text("drop me")

    tar = tmp_path / "s.tar.zst"
    r2_sync._write_tar_zst(str(tar), str(tmp_path), ["src"], exclude="embed_cache")

    out = tmp_path / "out"
    r2_sync._extract_tar_zst(str(tar), str(out))
    assert (out / "src" / "a.txt").read_text() == "alpha"
    assert (out / "src" / "sub" / "b.txt").read_text() == "beta"
    assert not (out / "src" / "embed_cache").exists(), "exclude did not drop the subtree"

    # ours -> CLI
    cli = tmp_path / "cli"
    cli.mkdir()
    rc = subprocess.run(["tar", "--zstd", "-xf", str(tar), "-C", str(cli)],
                        capture_output=True, text=True)
    assert rc.returncode == 0, f"CLI tar could not read our archive: {rc.stderr}"
    assert (cli / "src" / "a.txt").read_text() == "alpha"

    # CLI -> ours (the deployment-critical direction: the object currently on R2 was
    # written by `tar --zstd`, and every new client reads it with _extract_tar_zst)
    cli_tar = tmp_path / "cli.tar.zst"
    subprocess.run(["tar", "--zstd", "-cf", str(cli_tar), "-C", str(tmp_path), "src"],
                   check=True)
    from_cli = tmp_path / "from_cli"
    r2_sync._extract_tar_zst(str(cli_tar), str(from_cli))
    assert (from_cli / "src" / "a.txt").read_text() == "alpha"
    assert (from_cli / "src" / "sub" / "b.txt").read_text() == "beta"


# ---------------------------------------------------------------------------
# the pull lock, the empty-DB probe, and the auto-pull idle-gate bypass
# ---------------------------------------------------------------------------

def test_the_pull_lock_serialises_pulls_and_raises_r2busy(monkeypatch, tmp_path):
    """filelock replaces the hand-rolled flock; a second acquirer (any process,
    or even another coroutine in this one) fails fast rather than queueing."""
    monkeypatch.setattr(r2_sync, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(r2_sync, "LOCK_PATH", str(tmp_path / ".r2_pull.lock"))
    with r2_sync._pull_lock():
        with pytest.raises(r2_sync.R2Busy):
            with r2_sync._pull_lock():
                pass
    with r2_sync._pull_lock():          # released after the block -> reacquirable
        pass


def test_r2busy_is_an_r2error_so_the_manual_cli_still_catches_it():
    assert issubclass(r2_sync.R2Busy, r2_sync.R2Error)


def test_the_pull_incomplete_sentinel_round_trips(monkeypatch, tmp_path):
    """pull_snapshot marks this before the merge and clears it after; if the merge
    crashes it stays set, so the next run re-pulls a half-written DB."""
    monkeypatch.setattr(r2_sync, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(r2_sync, "INCOMPLETE_PATH", str(tmp_path / ".pull_incomplete"))
    assert r2_sync.pull_was_interrupted() is False
    r2_sync._mark_pull_incomplete()
    assert r2_sync.pull_was_interrupted() is True      # a crash here would leave it set
    r2_sync._clear_pull_incomplete()
    assert r2_sync.pull_was_interrupted() is False
    r2_sync._clear_pull_incomplete()                   # idempotent when already absent


def test_semantic_db_is_empty_when_the_store_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(r2_sync, "CACHE_DIR", str(tmp_path))   # no semantics.lmdb here
    assert r2_sync.semantic_db_is_empty() is True
    assert r2_sync.semantic_db_record_count() == 0


def _stub_pull_until_download(monkeypatch, tmp_path, idle):
    """Stub pull_snapshot's heavy steps and stop it the instant the download would
    begin, so a test can observe the idle gate and the first on_phase call."""
    monkeypatch.setattr(r2_sync, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(r2_sync, "LOCK_PATH", str(tmp_path / ".r2_pull.lock"))
    monkeypatch.setattr(r2_sync, "remote_head",
                        lambda s, client=None: r2_sync.Remote_Head(
                            "new", 10, datetime(2026, 7, 9, 12, 0)))
    monkeypatch.setattr(r2_sync, "read_marker", lambda: {})
    monkeypatch.setattr(r2_sync, "_require_disk", lambda *a: None)
    monkeypatch.setattr(r2_sync, "_require_idle", lambda force: idle.append(1))

    class _Stop(RuntimeError):
        pass

    def stop_download(url, dest, expected):
        raise _Stop()
    monkeypatch.setattr(r2_sync, "_download", stop_download)
    return _Stop


def test_auto_pull_skips_the_idle_gate_and_announces_the_download_phase(cfg, monkeypatch, tmp_path):
    idle, phases = [], []
    Stop = _stub_pull_until_download(monkeypatch, tmp_path, idle)
    with pytest.raises(Stop):
        r2_sync.pull_snapshot(require_idle=False, backup=False, on_phase=phases.append)
    assert idle == [], "require_idle=False must skip the lsof gate for the auto path"
    assert phases == ["downloading"], "on_phase should fire as the download begins"


def test_a_manual_pull_keeps_the_idle_gate(cfg, monkeypatch, tmp_path):
    idle = []
    Stop = _stub_pull_until_download(monkeypatch, tmp_path, idle)
    with pytest.raises(Stop):
        r2_sync.pull_snapshot(require_idle=True, backup=False)
    assert idle == [1], "the default path must still run the idle gate"
