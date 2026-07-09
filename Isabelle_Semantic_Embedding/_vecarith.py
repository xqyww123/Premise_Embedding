"""Q1.15 fixed-point vector arithmetic: the Highway SIMD kernel plus the encoding
convention that both the semantic vector store and premise_selection share.

This module is the single owner of the Q1.15 contract:

  * encoding      : ``rint(target_norm * v/|v| * 32768)`` as little-endian int16
  * dot product   : Highway ``MulFixedPoint15`` == round((a*b) / 32768), in int16
  * score recovery: ``cos = clip(s / (32768 * target_norm**2), -1, 1)``

The kernel deliberately uses a *rounding* fixed-point multiply rather than
``MulHigh`` (a floor). ``x - floor(x)`` lies in [0,1) for negative products just as
for positive ones, so a floor's per-term residue never cancels; it biases the sum
low by a data-dependent amount — roughly 0.5 per dimension for near-orthogonal
vectors, but far less when the two vectors are highly correlated (most products
then being small and positive). No constant correction can fix both regimes: a
``+D/2`` term overshot by +0.023 at cos->1 and produced "cosines" above 1.0.
Rounding is unbiased, cuts the error ~10x, and costs nothing (VPMULHRSW and
VPMULHW have identical throughput; measured 25.8ms vs 25.7ms). It does not make
the overshoot vanish -- a self-match still reaches 1.0007 at D=512, forty times
smaller than the floor's 1.0243 -- so ``recover_cos`` clips.

``target_norm`` is below 1.0 for two independent reasons: a unit vector may have a
component of exactly 1.0, which is unrepresentable in Q1.15's [-1, 1); and it keeps
every partial sum inside the int16 accumulator. With the rounding multiply each
term carries scale 32768, so ``|s| <= 32768 * target_norm**2``; at 0.95 that is
29573, leaving 1.11x headroom under 32767. Cauchy-Schwarz bounds any subset of
dimensions (hence every per-lane and reduction-tree intermediate) by the same
quantity, so nothing can overflow along the way.

The kernel is reached through ``ctypes.CDLL``, which releases the GIL for the
duration of the foreign call. That is what allows the KNN to run under
``asyncio.to_thread`` without starving the event loop; a pure-Python gather loop
would hold the GIL throughout.

Address gathering is a separate matter: it lives in the ``_vecgather`` extension
module, whose entry point holds the GIL by construction because it calls the
CPython C API. That costs nothing -- it is ~10ms against the kernel's ~150ms, and
the event-loop stall is set by the part that does release the GIL. The kernel
itself contains no Python at all, so Isabelle/ML can load the very same shared
object into a process that has no interpreter.
"""
from __future__ import annotations

import ctypes
import os
import pathlib
import sys

import numpy as np

from . import _vecgather

# ---------------------------------------------------------------- Q1.15 contract

Q15_SCALE = 32768.0
"""Q1.15 has 15 fractional bits: stored_int = value * 32768, range [-1, 1-2**-15]."""

TARGET_NORM = 0.95
"""L2 norm every stored/queried vector is scaled to before quantization.

Chosen so the int16 accumulator keeps 1.11x headroom: |s| <= 32768*0.95**2 = 29573.
"""

_DOT_SCALE = 32768.0
"""MulFixedPoint15 yields Q1.15 per term: round((a*b)/32768) represents a_i*b_i."""


def encode_q15(vectors: np.ndarray, target_norm: float = TARGET_NORM) -> np.ndarray:
    """L2-normalize to ``target_norm`` and quantize to little-endian int16 Q1.15.

    Accepts a single vector (1-D) or a batch (2-D, one vector per row).
    Uses ``rint`` rather than ``astype``'s truncate-toward-zero, which would add a
    directional bias of its own on top of the kernel's rounding.
    """
    v = np.asarray(vectors, dtype=np.float64)
    axis = v.ndim - 1
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    unit = v / (norm + 1e-12)
    return np.rint(target_norm * unit * Q15_SCALE).astype("<i2")


def recover_cos(scores: np.ndarray, D: int, target_norm: float = TARGET_NORM) -> np.ndarray:
    """Turn raw int16 dot products back into cosine similarities.

    Clipped to [-1, 1]. The scale alone lands slightly outside it -- a vector against
    itself scores up to 29593 where 32768*0.95**2 = 29573 was predicted, a relative
    7e-4, because quantizing each component and rounding each product both perturb
    the sum. A cosine of real vectors cannot exceed 1, so every excess is noise, and
    clipping projects the result back onto the feasible set: it can only reduce the
    error, never increase it, and being monotone it cannot reorder anything. Callers
    therefore never have to clamp on this function's behalf.

    ``D`` is accepted for symmetry with the caller's shape checks; the rounding
    multiply needs no dimension-dependent correction.
    """
    del D  # unused: the rounding multiply is unbiased
    s = np.asarray(scores, dtype=np.float64)
    return np.clip(s / (_DOT_SCALE * target_norm * target_norm), -1.0, 1.0)


# ------------------------------------------------------------------ shared object

_ENV_VAR = "ISABELLE_VECTOR_SO"
_REQUIRED_SYMBOL = "top_k_q15_gather"
_LIB_NAME = {
    "linux": "libisabelle_vector.so",
    "darwin": "libisabelle_vector.dylib",
    "win32": "isabelle_vector.dll",
}.get(sys.platform, "libisabelle_vector.so")
"""Tools/simd_vector.ML picks the same name from ML_System.platform_is_windows/macos."""


def _candidate_paths() -> list[pathlib.Path]:
    """Where the shared object may live, most authoritative first.

    A source checkout builds it under Tools/Vector_Arith/build and that copy wins:
    a developer who just rebuilt expects to be running what they built. A wheel
    ships it as package data next to this module, which is all an installed
    deployment has. Tools/simd_vector.ML asks library_path() rather than guessing,
    so Isabelle/ML always dlopens the same file.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return [pathlib.Path(override)]
    here = pathlib.Path(__file__).resolve().parent          # the package directory
    checkout = here.parent                                   # contrib/Semantic_Embedding
    return [
        checkout / "Tools" / "Vector_Arith" / "build" / _LIB_NAME,
        here / _LIB_NAME,
    ]


def library_path() -> str:
    """Absolute path of the shared object, after checking it is the right one.

    Loading it is what validates it, so this reports a library that actually
    exports the kernel rather than merely a file that exists. Isabelle/ML calls
    this out of Tools/simd_vector.ML: a wheel install puts the library under
    site-packages, which no hard-coded checkout path can find.
    """
    _lib()
    assert _lib_path is not None
    return str(_lib_path)


def _load() -> ctypes.CDLL:
    """Bind the kernel through ``CDLL``, which releases the GIL around the call.

    That is the whole point: the event loop keeps running while the scan proceeds
    in a worker thread. Nothing in this library touches the Python API, so there
    is no GIL to hold -- address gathering lives in the ``_vecgather`` extension
    module instead.
    """
    tried: list[str] = []
    for path in _candidate_paths():
        if not path.exists():
            tried.append(f"{path} (missing)")
            continue
        lib = ctypes.CDLL(str(path))
        if not hasattr(lib, _REQUIRED_SYMBOL):
            # A pre-gather build is still on disk (its dot_q15 also used aligned
            # Load and would fault on LMDB's page+16 addresses). Fail loudly.
            tried.append(f"{path} (stale: no {_REQUIRED_SYMBOL})")
            continue
        lib.dot_q15.restype = ctypes.c_int16
        lib.dot_q15.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.top_k_q15_gather.restype = ctypes.c_int
        lib.top_k_q15_gather.argtypes = [
            ctypes.c_void_p,  # const uintptr_t* vec_addrs
            ctypes.c_void_p,  # const int16_t* query
            ctypes.c_int32,   # D
            ctypes.c_int32,   # N
            ctypes.c_int32,   # k
            ctypes.c_void_p,  # int32_t* out_idx
            ctypes.c_void_p,  # int16_t* out_scores
        ]
        global _lib_path
        _lib_path = path
        return lib
    raise RuntimeError(
        "%s with %s not found. Tried:\n  %s\n"
        "Rebuild with:  cmake -S contrib/Semantic_Embedding/Tools/Vector_Arith "
        "-B contrib/Semantic_Embedding/Tools/Vector_Arith/build_new && "
        "cmake --build contrib/Semantic_Embedding/Tools/Vector_Arith/build_new\n"
        "Or set %s to an explicit path."
        % (_LIB_NAME, _REQUIRED_SYMBOL, "\n  ".join(tried), _ENV_VAR)
    )


_lib_cache: ctypes.CDLL | None = None
_lib_path: pathlib.Path | None = None


def _lib() -> ctypes.CDLL:
    """Load the shared object on first use, GIL-releasing handle.

    Deliberately lazy: ``encode_q15`` / ``recover_cos`` are pure numpy, so writing
    vectors and running the migration must not require a built kernel. Only the
    KNN path does.
    """
    global _lib_cache
    if _lib_cache is None:
        _lib_cache = _load()
    return _lib_cache


def dot_q15(a: np.ndarray, b: np.ndarray) -> int:
    """Raw int16 dot product of two Q1.15 vectors (no cosine recovery)."""
    if a.dtype != np.int16 or b.dtype != np.int16 or a.size != b.size:
        raise ValueError("dot_q15 expects two int16 arrays of equal length")
    return int(_lib().dot_q15(a.ctypes.data, b.ctypes.data, a.size))


def gather_addrs(buffers: list, expected: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pull the raw addresses out of a list of buffer objects, in C.

    ``buffers`` comes straight from ``[txn.get(k) for k in keys]``: memoryviews for
    keys that exist, None for those that don't. Asking Python for each address
    instead — ``np.frombuffer(mv, uint8).ctypes.data`` — costs ~2.9us per key,
    which over a 10^5 domain is more time than the SIMD scan it feeds.

    Buffers whose length is not ``expected`` are skipped, not trusted: a stale
    float32 record is twice as long and would otherwise be read as a truncated
    vector.

    Returns ``(addrs, keep, missing, n_skipped)`` where ``keep[j]`` is the index in
    ``buffers`` of the j-th address, and ``missing`` indexes the Nones. The caller
    must keep ``buffers`` alive, and the transaction open, while using ``addrs``.
    """
    n = len(buffers)
    addrs = np.empty(n, dtype=np.uintp)
    keep = np.empty(n, dtype=np.int32)
    missing = np.empty(n, dtype=np.int32)
    counts = np.zeros(3, dtype=np.int32)
    # An extension module, so the GIL is held for the duration -- which the CPython
    # calls inside require. It raises TypeError itself when ``buffers`` is not a list.
    _vecgather.gather_addrs(buffers, expected, addrs.ctypes.data, keep.ctypes.data,
                            missing.ctypes.data, counts.ctypes.data)
    kept, n_missing, skipped = (int(c) for c in counts)
    return addrs[:kept], keep[:kept], missing[:n_missing], skipped


def top_k_q15_gather(
    addrs: np.ndarray,
    query: np.ndarray,
    D: int,
    k: int,
    *,
    target_norm: float = TARGET_NORM,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k over candidate vectors read *in place* from the addresses in ``addrs``.

    ``addrs`` holds one raw pointer per candidate (dtype ``uintp``), each pointing
    at exactly ``D`` int16 values. Nothing is copied: the kernel runs unaligned
    SIMD loads straight over that memory.

    Two obligations fall on the caller, neither of which the kernel can check:
      * every address must have exactly ``D*2`` readable bytes — verify
        ``len(value) == D*2`` when gathering, since a stale float32 record is
        ``D*4`` and would be silently mis-read as a truncated vector;
      * when the addresses point into an LMDB mmap, its read transaction must stay
        open for the whole call.

    Returns ``(indices, cosines)`` sorted by descending score, where ``indices``
    index into ``addrs``.
    """
    if addrs.dtype != np.uintp or not addrs.flags["C_CONTIGUOUS"]:
        raise ValueError("addrs must be a C-contiguous uintp array")
    if query.dtype != np.int16 or query.size != D:
        raise ValueError(f"query must be an int16 array of length D={D}")
    N = int(addrs.size)
    k = min(k, N)
    if k <= 0 or N <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    out_idx = np.empty(k, dtype=np.int32)
    out_scores = np.empty(k, dtype=np.int16)
    rc = _lib().top_k_q15_gather(
        addrs.ctypes.data, query.ctypes.data, D, N, k,
        out_idx.ctypes.data, out_scores.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"top_k_q15_gather failed with rc={rc}")
    return out_idx, recover_cos(out_scores, D, target_norm)
