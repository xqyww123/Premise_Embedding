"""Tests for the two compiled artifacts: the Q1.15 SIMD kernel and its gather glue.

Run against whatever ``Isabelle_Semantic_Embedding`` is importable -- a source
checkout or an installed wheel. The wheel is the interesting case: it is how CI
learns that the extension module it just built can actually be imported by the
interpreter the wheel claims to serve, and that the kernel beside it loads and
computes. Nothing here needs Isabelle, a vector store, or the network.

The package's ``__init__`` pulls in transformers and Isabelle_RPC_Host, none
of which these artifacts depend on, so the submodules are loaded directly under a
stub parent. That stub is not a convenience: making ``from . import _vecgather``
resolve through the real package directory is precisely the step that fails on an
Intel Mac if the extension module was built arm64-only.

The file lives under tests/ rather than beside the package, because pytest prepends
the test file's own directory to sys.path -- from the repository root that would put
the source tree's Isabelle_Semantic_Embedding/ ahead of the installed one, and the
suite would quietly test the checkout while believing it tested the wheel. Set
VECARITH_REQUIRE_INSTALLED=1 to make that mistake an error rather than a silence.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import pathlib
import sys
import threading
import time
import types

import numpy as np
import pytest

PACKAGE = "Isabelle_Semantic_Embedding"


def _load_vecarith():
    spec = importlib.util.find_spec(PACKAGE)  # does not execute the package
    if spec is None or not spec.submodule_search_locations:
        pytest.skip(f"{PACKAGE} is not importable")
    pkg_dir = pathlib.Path(list(spec.submodule_search_locations)[0])

    installed = "site-packages" in pkg_dir.parts or "dist-packages" in pkg_dir.parts
    if os.environ.get("VECARITH_REQUIRE_INSTALLED") and not installed:
        raise AssertionError(
            f"VECARITH_REQUIRE_INSTALLED is set but {PACKAGE} resolved to {pkg_dir}, "
            "which is a source checkout. Run pytest from a directory that does not "
            "contain the package, and use the `pytest` entry point rather than "
            "`python -m pytest` (which prepends the cwd to sys.path)."
        )

    stub = types.ModuleType(PACKAGE)
    stub.__path__ = [str(pkg_dir)]
    sys.modules[PACKAGE] = stub

    sub = importlib.util.spec_from_file_location(f"{PACKAGE}._vecarith", pkg_dir / "_vecarith.py")
    mod = importlib.util.module_from_spec(sub)
    sys.modules[sub.name] = mod
    sub.loader.exec_module(mod)  # imports _vecgather; arch mismatch raises here
    return mod, pkg_dir, installed


V, PKG_DIR, INSTALLED = _load_vecarith()
D = 512  # small enough to keep the suite quick, wide enough to exercise every lane


def reference_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """What MulFixedPoint15 computes: round((a*b) / 32768), summed in int32.

    The kernel accumulates in int16, but the Cauchy-Schwarz bound in _vecarith's
    docstring says no partial sum can leave int16's range, so a wider reference is
    the right oracle -- if they disagree, the kernel overflowed.
    """
    prod = (a.astype(np.int32) * b.astype(np.int32) + 16384) >> 15
    return prod.sum(axis=-1)


def random_q15(rng, *shape) -> np.ndarray:
    return np.ascontiguousarray(V.encode_q15(rng.standard_normal(shape)))


def as_buffers(domain: np.ndarray, offset: int = 0) -> tuple[list, bytearray, int]:
    """Pack rows into one allocation at a chosen byte offset, as LMDB pages do.

    LMDB puts a large value at page_base + 16, so the addresses handed to the kernel
    are never 64-byte aligned; the SIMD path must therefore use unaligned loads.
    Returning the backing bytearray keeps the memoryviews (and the addresses they
    yield) alive for as long as the caller holds it.
    """
    n, d = domain.shape
    row = d * 2
    buf = bytearray(offset + n * row)
    raw = domain.tobytes()
    buf[offset:] = raw
    mv = memoryview(buf)
    return [mv[offset + i * row: offset + (i + 1) * row] for i in range(n)], buf, row


# --------------------------------------------------------------- the two artifacts

def test_kernel_and_glue_are_separate_files():
    """The whole point of the split: no CPython symbol may live in the kernel.

    Isabelle/ML loads that file into a process with no interpreter. Here we can only
    check the consequence -- gather_addrs is not in it -- but that is the symbol a
    regression would drag CPython back in through.
    """
    kernel = pathlib.Path(V.library_path())
    glue = pathlib.Path(V._vecgather.__file__)
    assert kernel != glue
    assert glue.parent == PKG_DIR
    if INSTALLED:
        # In a wheel the kernel is package data beside the glue. In a checkout it
        # stays in Tools/Vector_Arith/build, which _candidate_paths() prefers.
        assert kernel.parent == PKG_DIR

    lib = ctypes.CDLL(str(kernel))
    assert hasattr(lib, "dot_q15") and hasattr(lib, "top_k_q15_gather")
    assert not hasattr(lib, "gather_addrs"), "the glue leaked back into the kernel"


def test_extension_is_abi3_and_floor_is_respected():
    assert sys.version_info >= (3, 11), "Py_buffer entered the limited API in 3.11"
    name = pathlib.Path(V._vecgather.__file__).name
    assert name.endswith(".pyd") or ".abi3." in name, name


# ------------------------------------------------------------------ the Q1.15 contract

def test_encode_q15_hits_the_target_norm():
    rng = np.random.default_rng(0)
    v = random_q15(rng, 64, D)
    norms = np.linalg.norm(v.astype(np.float64) / V.Q15_SCALE, axis=1)
    assert np.allclose(norms, V.TARGET_NORM, atol=1e-3)
    assert v.dtype == np.dtype("<i2")


def test_dot_matches_the_rounding_reference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = random_q15(rng, D), random_q15(rng, D)
        assert V.dot_q15(a, b) == reference_dot(a, b)


def test_accumulator_cannot_overflow_at_the_worst_case():
    """A vector against itself is the largest a dot product can be.

    The design bound is |s| <= 32768 * TARGET_NORM**2 = 29573 at norm 0.95, leaving
    1.11x headroom under int16's 32767. Add wraps rather than saturates, so a
    violation would surface as a sign flip, not a clamp -- hence the hard assert.

    The soft bound is exceeded, slightly and legitimately: quantizing each component
    and rounding each product both perturb the sum, so s overshoots 29573 by a few
    counts (measured max 29593 at D=512, i.e. 7e-4 relative). The headroom absorbs
    it with more than 3000 counts to spare. What matters is that nothing approaches
    32767.
    """
    rng = np.random.default_rng(2)
    bound = V.Q15_SCALE * V.TARGET_NORM**2
    worst = 0
    for _ in range(50):
        a = random_q15(rng, D)
        s = V.dot_q15(a, a)
        assert 0 < s < 32767, f"int16 accumulator overflowed: {s}"
        assert s <= bound * 1.001, (s, bound)
        worst = max(worst, s)
    assert 32767 - worst > 2000, f"headroom shrank to {32767 - worst}"


def test_recovered_cosines_stay_in_range():
    """recover_cos clips, so callers never have to.

    The bare scale lands slightly outside [-1, 1] -- quantization and per-term
    rounding both perturb the sum, and a self-match reaches 1.0007 at D=512. Every
    such excess is noise (a cosine of real vectors cannot exceed 1), so clipping
    projects onto the feasible set: error can only shrink, and monotonicity means
    nothing reorders.
    """
    rng = np.random.default_rng(3)
    a, b = random_q15(rng, 256, D), random_q15(rng, D)
    cos = V.recover_cos(reference_dot(a, b), D)
    assert np.all(np.abs(cos) <= 1.0)

    raw = V.dot_q15(b, b)
    unclipped = raw / (V.Q15_SCALE * V.TARGET_NORM**2)
    assert unclipped > 1.0, "the overshoot this test exists for has disappeared"
    assert V.recover_cos(np.array([raw]), D)[0] == 1.0

    # Clipping must not disturb anything already inside the range.
    inside = np.array([-29000, -1, 0, 1, 29000])
    assert np.allclose(V.recover_cos(inside, D),
                       inside / (V.Q15_SCALE * V.TARGET_NORM**2))


def test_quantization_error_against_float64_truth():
    """Q1.15 must not merely be self-consistent; it has to approximate the cosine.

    Both bounds are empirical, measured on the real 110k-vector store during the
    float32 -> Q1.15 migration (recall@10 0.9900, |dcos| mean 0.00052). They are set
    well clear of those numbers so that only a real regression trips them.
    """
    rng = np.random.default_rng(4)
    n, k = 400, 10
    fdomain = rng.standard_normal((n, D))
    fquery = rng.standard_normal(D)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    truth = unit(fdomain) @ unit(fquery)
    top_true = set(np.argsort(-truth)[:k].tolist())

    domain, query = V.encode_q15(fdomain), V.encode_q15(fquery)
    bufs, _keep_alive, _ = as_buffers(domain)
    addrs, *_ = V.gather_addrs(bufs, D * 2)
    idx, cos = V.top_k_q15_gather(addrs, query, D, k)

    recall = len(top_true & set(idx.tolist())) / k
    assert recall >= 0.9, f"recall@{k} = {recall}"
    assert np.abs(cos - truth[idx]).mean() < 0.01


# ------------------------------------------------------------------------ top-k

@pytest.mark.parametrize("offset", [0, 1, 3, 7, 16])
def test_topk_is_exact_at_any_buffer_alignment(offset):
    """LMDB hands out addresses at page_base + 16; nothing is 64-byte aligned.

    A regression from LoadU to an aligned Load would fault or, worse, read shifted
    data. Odd offsets are not merely unaligned, they are unaligned for int16 too.
    """
    rng = np.random.default_rng(5 + offset)
    n, k = 300, 10
    domain, query = random_q15(rng, n, D), random_q15(rng, D)
    bufs, _keep_alive, _ = as_buffers(domain, offset)

    addrs, keep, missing, skipped = V.gather_addrs(bufs, D * 2)
    assert (addrs.size, missing.size, skipped) == (n, 0, 0)
    idx, cos = V.top_k_q15_gather(addrs, query, D, k)

    ref = reference_dot(domain, query)
    expect = np.argsort(-ref, kind="stable")[:k]
    assert idx.tolist() == expect.tolist()
    assert np.allclose(cos, V.recover_cos(ref[expect], D))
    assert np.all(np.diff(cos) <= 0), "scores must come back descending"


def test_no_excluded_candidate_beats_the_kth():
    rng = np.random.default_rng(6)
    n, k = 500, 7
    domain, query = random_q15(rng, n, D), random_q15(rng, D)
    bufs, _keep_alive, _ = as_buffers(domain)
    addrs, *_ = V.gather_addrs(bufs, D * 2)
    idx, cos = V.top_k_q15_gather(addrs, query, D, k)

    ref = reference_dot(domain, query)
    kth = ref[idx[-1]]
    outside = np.setdiff1d(np.arange(n), idx)
    assert ref[outside].max() <= kth


def test_k_is_clamped_and_degenerate_cases_are_empty():
    rng = np.random.default_rng(7)
    n = 4
    domain, query = random_q15(rng, n, D), random_q15(rng, D)
    bufs, _keep_alive, _ = as_buffers(domain)
    addrs, *_ = V.gather_addrs(bufs, D * 2)

    idx, cos = V.top_k_q15_gather(addrs, query, D, 99)  # k > N
    assert idx.size == cos.size == n

    for bad_k in (0, -1):
        idx, cos = V.top_k_q15_gather(addrs, query, D, bad_k)
        assert idx.size == cos.size == 0

    empty = np.empty(0, dtype=np.uintp)
    idx, cos = V.top_k_q15_gather(empty, query, D, 5)  # N == 0
    assert idx.size == cos.size == 0


def test_topk_rejects_mistyped_arguments():
    rng = np.random.default_rng(8)
    domain, query = random_q15(rng, 4, D), random_q15(rng, D)
    bufs, _keep_alive, _ = as_buffers(domain)
    addrs, *_ = V.gather_addrs(bufs, D * 2)

    with pytest.raises(ValueError):
        V.top_k_q15_gather(addrs.astype(np.int32), query, D, 2)  # wrong dtype
    with pytest.raises(ValueError):
        V.top_k_q15_gather(addrs, query.astype(np.int32), D, 2)
    with pytest.raises(ValueError):
        V.top_k_q15_gather(addrs, query[:-1], D, 2)  # query shorter than D


# ----------------------------------------------------------------------- gather

def test_gather_reports_the_real_addresses():
    rng = np.random.default_rng(9)
    n = 16
    domain = random_q15(rng, n, D)
    bufs, buf, row = as_buffers(domain, offset=16)
    base = np.frombuffer(buf, dtype=np.uint8).ctypes.data
    addrs, keep, _missing, _skipped = V.gather_addrs(bufs, D * 2)
    expected = [base + 16 + i * row for i in range(n)]
    assert addrs.tolist() == expected
    assert keep.tolist() == list(range(n))


def test_gather_classifies_missing_and_stale_records():
    """None means "key absent"; a D*4 value is a float32 record left by an old store.

    Trusting the latter would read half a vector as a whole one, silently. The
    kernel cannot tell -- it is handed a bare address -- so this is the only place
    the length is ever checked.
    """
    rng = np.random.default_rng(10)
    n = 12
    domain = random_q15(rng, n, D)
    bufs, _keep_alive, _ = as_buffers(domain)
    stale = memoryview(np.zeros(D, dtype="<f4").tobytes())  # D*4 bytes
    bufs[3] = stale
    bufs[7] = None
    bufs[9] = memoryview(b"")

    addrs, keep, missing, skipped = V.gather_addrs(bufs, D * 2)
    assert addrs.size == n - 3
    assert missing.tolist() == [7]
    assert skipped == 2  # the float32 record and the empty one
    assert 3 not in keep.tolist() and 9 not in keep.tolist()


def test_gather_handles_the_empty_and_all_missing_cases():
    addrs, keep, missing, skipped = V.gather_addrs([], D * 2)
    assert addrs.size == keep.size == missing.size == 0 and skipped == 0

    addrs, keep, missing, skipped = V.gather_addrs([None] * 5, D * 2)
    assert addrs.size == 0 and missing.tolist() == list(range(5)) and skipped == 0


def test_gather_rejects_a_non_list():
    rng = np.random.default_rng(11)
    bufs, _keep_alive, _ = as_buffers(random_q15(rng, 3, D))
    with pytest.raises(TypeError):
        V.gather_addrs(tuple(bufs), D * 2)


# -------------------------------------------------------------------------- GIL

def _longest_main_thread_stall(scan, calls: int) -> float:
    """Run ``scan`` on a worker; return how long the main thread was starved worst.

    Counting how *often* the main thread ran does not distinguish the two GIL
    disciplines: even under PyDLL it runs freely between successive foreign calls,
    and a busy loop racks up hundreds of thousands of iterations in those gaps
    (measured: 216k under PyDLL against 987k under CDLL -- a factor of five, not the
    difference between "stalled" and "not stalled"). The stall length does
    distinguish them, and it is also the quantity the event loop actually cares about.
    """
    done = threading.Event()

    def worker():
        for _ in range(calls):
            scan()
        done.set()

    old = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)  # so a GIL-releasing kernel yields promptly
    try:
        t = threading.Thread(target=worker)
        t.start()
        last, worst = time.perf_counter(), 0.0
        while not done.is_set():
            now = time.perf_counter()
            worst = max(worst, now - last)
            last = now
        t.join()
    finally:
        sys.setswitchinterval(old)
    return worst


def test_the_kernel_releases_the_gil():
    """The reason topk can run under asyncio.to_thread without stalling the loop.

    ctypes.CDLL drops the GIL for the duration of a foreign call; ctypes.PyDLL holds
    it. Nothing in the library's *results* distinguishes them, so a change to PyDLL
    would silently reintroduce the 17-second event-loop freeze this design exists to
    remove. Under CDLL the main thread's worst stall is bounded by the switch
    interval; under PyDLL it is a whole foreign call.

    Measured on the reference machine: stall/call = 0.04 with CDLL, 2.28 with PyDLL.
    The threshold sits an order of magnitude from each. CI runners share their cores,
    so an unlucky preemption can inflate a single measurement; noise can only make
    the stall longer, never shorter, so the best of three rounds is taken.
    """
    rng = np.random.default_rng(12)
    n = 60000  # a call long enough that the threshold clears scheduler noise
    domain, query = random_q15(rng, n, D), random_q15(rng, D)
    bufs, _keep_alive, _ = as_buffers(domain)
    addrs, *_ = V.gather_addrs(bufs, D * 2)

    scan = lambda: V.top_k_q15_gather(addrs, query, D, 10)
    t0 = time.perf_counter()
    scan()
    call_time = time.perf_counter() - t0

    stall = min(_longest_main_thread_stall(scan, calls=8) for _ in range(3))
    assert stall < 0.5 * call_time, (
        f"main thread stalled {stall * 1e3:.2f} ms against a {call_time * 1e3:.2f} ms "
        "call: the kernel is holding the GIL (ctypes.PyDLL instead of CDLL?)"
    )
