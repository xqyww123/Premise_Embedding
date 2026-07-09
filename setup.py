"""Build the platform wheel that carries the Highway SIMD kernel.

The distribution ships two compiled artifacts, and they are deliberately separate:

* ``libisabelle_vector.so`` -- the SIMD kernel, built by CMake out of
  ``Tools/Vector_Arith`` and staged beside ``_vecarith.py`` as package data (the
  second, and without a source checkout the only, entry in
  ``_vecarith._candidate_paths()``). It contains **no Python**. Two consumers load
  it: Python through ``ctypes.CDLL``, which releases the GIL for the scan, and
  Isabelle/ML through ``Foreign.loadLibrary`` into a process with no interpreter,
  asking ``_vecarith.library_path()`` over RPC rather than guessing a path.
* ``_vecgather.abi3.so`` -- the address-gathering glue, a CPython extension module
  built by ``build_ext``. It is imported, never dlopened.

Keeping them apart is what makes the kernel loadable by Isabelle/ML on Windows,
where ``LoadLibrary`` resolves the entire import table eagerly and would not
tolerate a ``python3.dll`` import; it also retires the ``ctypes.PyDLL`` handle the
glue used to need, and the segfault that followed from reaching for ``CDLL``.

Why the wheel is ``cp311-abi3-<platform>``:

* ``cp311``/``abi3`` -- the extension module is compiled against the limited API,
  so one build serves CPython 3.11 through 3.14 and there is no per-interpreter
  wheel. 3.11 is the floor because ``Py_buffer`` and ``PyObject_GetBuffer`` entered
  the limited API there and not earlier (``pybuffer.h``).
* ``<platform>``   -- both artifacts are pinned to one architecture. Highway's
  runtime dispatch chooses a SIMD target *within* an architecture; it does not
  cross between them. An ``any`` wheel would happily install an x86-64 ``.so`` on
  aarch64 and fail at first dlopen. The tag is computed from the kernel's own
  headers (``_platform_tag``) rather than hard-coded, and an explicitly passed
  ``--plat-name`` is checked against it.

Build with::

    python -m build --wheel

Plain ``python -m build`` would produce an sdist first and then build the wheel
from inside it. The sdist has no ``Tools/`` tree -- it is not a Python package --
so that path cannot compile the kernel. Build the wheel directly, from a checkout.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py

try:  # setuptools >= 70.1 vendors it; older installs need the `wheel` package
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

HERE = Path(__file__).parent.resolve()
PACKAGE = "Isabelle_Semantic_Embedding"
LIB_NAME = "libisabelle_vector.so"
KERNEL_SRC = HERE / "Tools" / "Vector_Arith"
KERNEL_BUILD = KERNEL_SRC / "build"

# The manylinux floor we advertise. The tag is a *requirement on the host*, so
# naming a glibc newer than we need is safe, while naming an older one is not. Our
# kernel bottoms out at GLIBC_2.4, but manylinux_2_17 (== manylinux2014) is the
# oldest tag every pip in circulation understands without a compatibility dance,
# and no realistic target runs glibc older than 2.17 (released 2012).
MIN_GLIBC = (2, 17)

# Libraries a manylinux wheel may leave to the host (PEP 600's core allowlist,
# minus the X11/graphics entries we could never need). Anything else would have to
# be vendored into the wheel -- which is exactly what we avoid by linking Highway
# statically and never touching libstdc++.
ALLOWED_NEEDED = {
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "libgcc_s.so.1", "libstdc++.so.6", "libnsl.so.1", "libutil.so.1",
    "libresolv.so.2", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
}

# ELF e_machine -> the architecture spelling used in platform tags.
ELF_MACHINES = {0x03: "i686", 0x28: "armv7l", 0x3E: "x86_64", 0xB7: "aarch64", 0xF3: "riscv64"}


# --------------------------------------------------------------- building the .so

def _library() -> Path:
    """Return the shared object to ship, compiling it if the checkout has none.

    ``ISABELLE_VECTOR_SO`` lets a cross-build or a CI matrix drop in an artifact
    produced elsewhere -- the same variable ``_vecarith`` and ``simd_vector.ML``
    honour at run time, so one override redirects the whole system.
    """
    override = os.environ.get("ISABELLE_VECTOR_SO")
    if override:
        so = Path(override).resolve()
        if not so.is_file():
            raise SystemExit(f"ISABELLE_VECTOR_SO={override} does not exist")
        return so

    if not KERNEL_SRC.is_dir():
        raise SystemExit(
            f"{KERNEL_SRC} is missing, so the kernel cannot be compiled.\n"
            "Build the wheel from a source checkout (not from an sdist), or point\n"
            "ISABELLE_VECTOR_SO at a prebuilt libisabelle_vector.so."
        )

    cmake = shutil.which("cmake")
    so = KERNEL_BUILD / LIB_NAME
    if cmake is None:
        if so.is_file():
            print(f"warning: cmake not found; shipping the existing {so} as is", file=sys.stderr)
            return so
        raise SystemExit("cmake not found and no prebuilt library to fall back on")

    # Incremental: a configured tree with an up-to-date library reruns in ~1s. The
    # kernel target only -- the test executables are not part of the distribution.
    subprocess.run([cmake, "-S", str(KERNEL_SRC), "-B", str(KERNEL_BUILD)], check=True)
    subprocess.run([cmake, "--build", str(KERNEL_BUILD), "--target", "isabelle_vector",
                    "--parallel"], check=True)
    if not so.is_file():
        raise SystemExit(f"cmake reported success but {so} was not produced")
    return so


# ------------------------------------------------------------ inspecting the .so

def _elf_arch(so: Path) -> str:
    """Architecture of an ELF object, read straight out of its header.

    ``e_machine`` sits at offset 0x12 in both ELF32 and ELF64, so this needs no
    parser and no binutils.
    """
    head = so.open("rb").read(20)
    if head[:4] != b"\x7fELF":
        raise SystemExit(f"{so} is not an ELF object")
    endian = "little" if head[5] == 1 else "big"
    machine = int.from_bytes(head[18:20], endian)
    if machine not in ELF_MACHINES:
        raise SystemExit(f"{so}: unhandled ELF machine 0x{machine:02x}")
    return ELF_MACHINES[machine]


def _dynamic_info(so: Path) -> tuple[set[str], set[tuple[int, ...]]]:
    """The library's DT_NEEDED entries and the glibc symbol versions it requires."""
    for cmd in (["objdump", "-p"], ["readelf", "-d", "-V"]):
        exe = shutil.which(cmd[0])
        if exe:
            out = subprocess.run([exe, *cmd[1:], str(so)],
                                 check=True, capture_output=True, text=True).stdout
            break
    else:
        raise SystemExit(
            "neither objdump nor readelf is available, so the manylinux claim "
            "cannot be verified; install binutils or pass an explicit --plat-name"
        )
    needed = set(re.findall(r"NEEDED\s+(?:Shared library: \[)?([^\s\]]+)", out))
    versions = {tuple(int(n) for n in v.split("."))
                for v in re.findall(r"GLIBC_(\d+(?:\.\d+)*)", out)}
    return needed, versions


def _platform_tag(so: Path) -> str:
    """The most permissive manylinux tag this exact binary honestly satisfies."""
    arch = _elf_arch(so)
    needed, versions = _dynamic_info(so)
    external = needed - ALLOWED_NEEDED
    if external:
        raise SystemExit(
            f"{so} needs libraries outside the manylinux allowlist: {sorted(external)}\n"
            "They would have to be vendored into the wheel. Link them statically instead."
        )
    glibc = max(versions | {MIN_GLIBC})[:2]
    return f"manylinux_{glibc[0]}_{glibc[1]}_{arch}"


_LEGACY_MANYLINUX = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}


def _check_supplied_tag(so: Path, plat: str) -> None:
    """Reject a --plat-name the binary cannot back up.

    A wrong tag is not a build failure but a runtime one, on someone else's
    machine, at the first dlopen -- so it is worth catching here.
    """
    arch = _elf_arch(so)
    if not plat.endswith("_" + arch) and not plat.endswith("-" + arch):
        raise SystemExit(f"--plat-name {plat} does not match the library's architecture ({arch})")
    if match := re.fullmatch(r"manylinux_(\d+)_(\d+)_.+", plat):
        claimed = (int(match[1]), int(match[2]))
    elif match := re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_.+", plat):
        claimed = _LEGACY_MANYLINUX[match[1]]
    else:
        return  # a plain linux_* tag promises nothing, so there is nothing to check
    needed, versions = _dynamic_info(so)
    if external := needed - ALLOWED_NEEDED:
        raise SystemExit(f"{plat} forbids depending on {sorted(external)}")
    if required := max(versions, default=(0,)):
        if required[:2] > claimed:
            raise SystemExit(
                f"--plat-name {plat} promises glibc {claimed[0]}.{claimed[1]}, but the "
                f"library requires GLIBC_{'.'.join(map(str, required))}"
            )


# ------------------------------------------------------------------- setuptools

# The gather glue, as a real extension module rather than a ctypes target. Its
# entry point holds the GIL by construction, which is what the CPython calls
# inside it require; the ctypes.PyDLL handle that used to provide that -- and the
# segfault that followed from reaching for CDLL instead -- are gone.
#
# Py_LIMITED_API 0x030B0000 because Py_buffer and PyObject_GetBuffer entered the
# limited API in 3.11 and not before (pybuffer.h). One abi3 build then serves
# 3.11 through 3.14, so this stays one wheel per platform, not one per interpreter.
_vecgather = Extension(
    f"{PACKAGE}._vecgather",
    sources=[f"{PACKAGE}/_vecgather.c"],
    define_macros=[("Py_LIMITED_API", "0x030B0000")],
    py_limited_api=True,
)


class build_py(_build_py):
    """Stage the CMake-built kernel next to the package's Python sources.

    build_py runs before build_ext (distutils' sub_commands order), and the two
    write different filenames into build_lib, so neither clobbers the other.
    """

    def run(self) -> None:
        so = _library()
        super().run()
        if getattr(self, "editable_mode", False):
            # An editable install still finds the checkout's build/ directory,
            # which _candidate_paths() prefers anyway. Nothing to stage.
            return
        target = Path(self.build_lib) / PACKAGE
        target.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(so), str(target / LIB_NAME))


class bdist_wheel(_bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        # (cp311, abi3) comes from the py_limited_api option below; root_is_pure is
        # already False because ext_modules is non-empty. Only the platform half is
        # ours to compute, and it comes from the kernel's own headers rather than
        # from the host: a wrong tag fails on someone else's machine, not this one.
        impl, abi, _ = super().get_tag()
        so = Path(self.bdist_dir) / PACKAGE / LIB_NAME if self.bdist_dir else None
        if so is None or not so.is_file():
            so = _library()  # get_tag can run before the staging copy exists
        if self.plat_name_supplied:
            plat = self.plat_name.replace("-", "_").replace(".", "_")
            _check_supplied_tag(so, plat)
        else:
            plat = _platform_tag(so)
        return impl, abi, plat


setup(
    ext_modules=[_vecgather],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
    cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel},
)
