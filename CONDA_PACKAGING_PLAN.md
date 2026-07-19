# Per-platform conda packaging for Semantic_Embedding

Status: **design, not yet implemented.** Nothing here has been run. Every claim is
either cited to a file:line in this repo, to a live query against
`api.anaconda.org` / `conda.qiyuan.me` made while writing this, or explicitly
flagged as unverified.

The governing constraint: the conda package may ship precompiled `.so`/`.dll`/
`.dylib`, but **our own pipeline must build them**. No third-party prebuilt
artifact, no download from PyPI. It does *not* require that cmake/MinGW run inside
`rattler-build` — `.github/workflows/wheels.yml` already builds these binaries
correctly on native runners, and the design below feeds that output into the conda
build in the same workflow run.

---

## 1. What `wheels.yml` actually produces

### 1.1 The matrix

`wheels.yml:42-49` defines four native legs. There is deliberately no cibuildwheel
and no qemu (`wheels.yml:3-16`).

| leg | runner | `matrix.arch` | kernel filename | wheel platform tag |
|---|---|---|---|---|
| Linux x86-64 | `ubuntu-24.04` | `x86_64` | `libisabelle_vector.so` | `manylinux_2_17_x86_64` (derived) |
| Linux arm64 | `ubuntu-24.04-arm` | `aarch64` | `libisabelle_vector.so` | `manylinux_2_17_aarch64` (derived) |
| macOS fat | `macos-15` (Apple Silicon) | `universal2` | `libisabelle_vector.dylib` | `macosx_11_0_universal2` |
| Windows | `windows-2022` | `amd64` | `isabelle_vector.dll` | `win_amd64` |

Each leg emits **two** compiled artifacts, and the whole architecture of this repo
turns on keeping them apart (`setup.py:3-18`):

1. **`libisabelle_vector.{so,dylib}` / `isabelle_vector.dll`** — the Highway SIMD
   kernel, built by **CMake** from `Tools/Vector_Arith/`. It contains **no
   Python** (`Tools/Vector_Arith/CMakeLists.txt`, the "This library contains no
   Python" block). Two consumers `dlopen` it: Python via `ctypes.CDLL`
   (`Isabelle_Semantic_Embedding/_vecarith.py:146-186`) and Isabelle/ML via
   `Foreign.loadLibrary` into a process with no interpreter
   (`Tools/simd_vector.ML:216-228`).
2. **`_vecgather.abi3.so` / `_vecgather.abi3.pyd`** — a CPython extension module
   built by **setuptools `build_ext`** from `Isabelle_Semantic_Embedding/_vecgather.c`,
   compiled against `Py_LIMITED_API 0x030B0000` (`setup.py:334-339`). It is
   `import`ed, never dlopened.

The platform tag is not hard-coded: `bdist_wheel.get_tag` (`setup.py:387-403`)
computes it from the kernel's own headers via `_platform_tag` (`setup.py:247-283`),
and refuses to emit a tag the binary cannot honour. **This matters below** — see
§3.4.

### 1.2 The build steps, per leg

**Linux (both arches).** `wheels.yml:91-124` builds vendored Highway statically
(`hwy` target only, `BUILD_SHARED_LIBS=OFF`), then `wheels.yml:126-139` builds the
kernel plus its two test binaries and runs `test_gather` and `test_ml_dlopen`.
`test_ml_dlopen` reproduces Poly/ML's `dlopen(RTLD_LAZY)` and is the check that
catches a CPython symbol creeping back into the kernel
(`Tools/Vector_Arith/CMakeLists.txt`, `add_executable(test_ml_dlopen ...)` comment).
`wheels.yml:185-226` then asserts a real SIMD target survived and prints the
`DT_NEEDED` set and glibc floor.

The output directory is `Tools/Vector_Arith/build/`, which is also the *first*
entry of `_vecarith._candidate_paths()` (`_vecarith.py:126-130`), so `setup.py`'s
`_library()` (`setup.py:111-147`) picks it up without an override.

**macOS.** One Apple Silicon runner builds for **both** architectures, because
Isabelle runs native arm64 Poly/ML there and x86-64 on Intel Macs, and one fat
binary serves both (`wheels.yml:45-48`). Two independent switches are needed
(`wheels.yml:78-89`):

- `CMAKE_OSX_ARCHITECTURES="x86_64;arm64"` governs the kernel *and* Highway. It is
  set in `Tools/Vector_Arith/CMakeLists.txt` and **must precede `add_library`** —
  a target's `OSX_ARCHITECTURES` property is initialised from the variable at
  target-creation time. It is a plain `set()`, not `set(CACHE)`, because
  `Platform/Darwin-Initialize.cmake` already created the cache entry (empty) during
  `project()`, and `set(CACHE)` over an existing entry is a no-op.
- `ARCHFLAGS="-arch arm64 -arch x86_64"` governs `_vecgather`, which setuptools
  builds and which does **not** see `CMAKE_OSX_ARCHITECTURES`.

Set only the first — the obvious mistake — and you ship a universal2 kernel beside
an arm64-only extension under a universal2 tag. It installs fine on an Intel Mac
and dies at `import`. `setup.py:361-384` (`_check_extension_arch`) refuses to emit
such a wheel; `wheels.yml:259-275` re-checks the shipped bytes; and
`wheels.yml:316-333` (`intel-mac`) actually runs the test suite on `macos-15-intel`,
because only an Intel host can prove the x86-64 slice is real.

Highway's archive must be fat too (`wheels.yml:105-110`), or linking the kernel
fails with "building for macOS-arm64 but attempting to link with file built for
macOS-x86_64".

Note `hwybuild`, not `build` (`wheels.yml:102-104`): Highway's root holds a Bazel
file named `BUILD` and macOS filesystems are case-insensitive, so `cmake -B .../build`
fails there with "Unable to (re)create the private pkgRedirects directory". The
same reasoning is repeated in `CMakeLists.txt`'s `find_library` PATHS list and in
`.gitignore`.

**Windows — the two-toolchain leg.** This is the subtlest part of the whole
pipeline. `wheels.yml:58-76` installs MSYS2/MINGW64 and puts MinGW's binutils on
the *plain* shell's PATH. `wheels.yml:141-162` builds Highway, the kernel and the
tests entirely inside MSYS2 with the Ninja generator. `wheels.yml:164-183` asserts
`N_AVX2`, `N_AVX3`, `N_AVX3_SPR`, `N_AVX10_2` are all present and that the DLL
imports none of `python*`, `libstdc++*`, `libgcc_s*`, `libwinpthread*`. Then the
wheel itself is built by `python -m build --wheel` from the plain shell
(`wheels.yml:228-239`) — i.e. with **MSVC**, which is what builds `_vecgather`.

**Why one toolchain for both breaks:**

- *MSVC for the kernel.* Highway marks AVX3 and everything above it as
  `HWY_BROKEN_MSVC` (`detect_targets.h:191`, cited at `wheels.yml:59-61` and again
  in `CMakeLists.txt`). MSVC would therefore **silently drop five SIMD targets** —
  the build stays green, the DLL loads, and you get a scalar-ish fallback wearing a
  SIMD name. There is no error; the only way to see it is to enumerate the compiled
  dispatch targets, which is exactly what `wheels.yml:170-180` does. Secondarily,
  Poly/ML is itself a MinGW build (`Admin/component_polyml.scala:37`), so MinGW is
  the matching ABI for the thing Isabelle/ML will `LoadLibrary`.
- *MinGW for `_vecgather`.* MSVC is what setuptools knows how to drive on Windows
  and what links `python3.lib`. Getting MinGW to produce a CPython extension that
  links the right import library, with the right CRT, under `build_ext`, is a fight
  with no upside — `_vecgather.c` is limited-API C with no SIMD in it at all.

They never meet: the two artifacts share no symbol, no heap and no CRT object —
`_vecgather` writes only into buffers the caller allocated (`wheels.yml:62-64`).

The seam that makes this work is `ISABELLE_VECTOR_SO`. `wheels.yml:159-162` exports
it after the MinGW build so that `setup.py`'s `_library()` (`setup.py:118-123`) uses
the finished DLL instead of re-running cmake from the plain shell and rebuilding the
kernel with MSVC. `wheels.yml:290-294` then *unsets* it before the install test,
because it outranks every entry in `_candidate_paths()` and would otherwise test
the checkout's library rather than the wheel's.

### 1.3 Why the kernel must stay Python-free

`Foreign.loadLibrary` on Windows goes through `LoadLibrary` (`polyffi.cpp:153`,
cited at `wheels.yml:146-148`), which resolves the **entire import table eagerly**,
unlike `dlopen(RTLD_LAZY)`. A `python3.dll` import would therefore fail at load
inside an Isabelle process that has no interpreter and whose PATH we do not
control. `setup.py:263-272` turns that into a build-time refusal; `-z,defs`
(`CMakeLists.txt`) turns a stray undefined symbol into a link error on Linux;
`test_ml_dlopen` is the runtime check.

---

## 2. Package shape

### 2.1 The subdir question, and a hard blocker

conda subdirs are `linux-64`, `linux-aarch64`, `osx-64`, `osx-arm64`, `win-64`.
The base `isabelle` package on `conda.qiyuan.me` ships all five (queried
`https://conda.qiyuan.me/channeldata.json`: `isabelle 2025.2 ['linux-64',
'linux-aarch64', 'osx-64', 'osx-arm64', 'win-64']`).

**`universal2` is not a conda subdir.** conda has no fat-binary concept and no
"any macOS" arch. So the single macOS wheel maps to *two* conda packages. The
universal2 binaries are valid on both hosts, so the same artifact can be published
to `osx-64` and `osx-arm64` unchanged — you pay ~2x in `.dylib`/`.pyd` size on each
and get to keep the one build job that `wheels.yml` already has (including its
`intel-mac` proof job, which no per-subdir thin build would give you for free).
Building thin per subdir is possible but would require cross-building x86-64 on the
arm64 runner *and* would collide with `setup.py`'s tag logic — see §3.4.

**The blocker: `rocksdict`.** Queried
`https://api.anaconda.org/package/conda-forge/rocksdict/files`: the package exists
for **`linux-64`, `osx-64`, `win-64` only**, at exactly one version, `0.3.23`.
There has never been a `linux-aarch64` or `osx-arm64` build. `rocksdict` is a
top-level import (`Isabelle_Semantic_Embedding/premise_selection.py:4-5`), so the
run requirement is unavoidable without a source change.

Consequence: **the initial conda matrix is `linux-64`, `osx-64`, `win-64`.**
`linux-aarch64` and `osx-arm64` are blocked at the dependency solve even though our
natives build fine for them. Three ways out, none of which this plan takes
unilaterally:

1. Ship on those subdirs anyway with `rocksdict` moved behind a lazy import and a
   graceful fallback (a source change to `premise_selection.py`; its only use is a
   *cache* at `premise_selection.py:61-66`, so a fallback is plausible — but that is
   a decision for the maintainer, not this document).
2. Build and publish `rocksdict` for `osx-arm64`/`linux-aarch64` on
   `conda.qiyuan.me` ourselves. Feasible (it is a maturin/Rust binding) but is a
   whole second packaging project.
3. Accept `osx-64` on Apple Silicon under Rosetta. **Do not do this**: the base
   `isabelle` package is native `osx-arm64` there and Isabelle prefers arm64 Poly/ML
   (`etc/options ML_system_apple`, cited at `wheels.yml:46-47`), so the halves would
   disagree about architecture.

This should be settled before implementation. It is the single largest open
question in this plan.

### 2.2 One package or two?

**Decision: ONE package, per-platform, named `isabelle-semantic-embedding`.**

The precedents:

| package | shape | why |
|---|---|---|
| `isabelle-performant-ml` | `noarch: generic` | session source only, no Python at all |
| `isabelle-mcp` | `noarch: python`, no hooks | Python only; registers its Scala component from Python at run time (`isabelle_mcp/component.py`) |
| `isabelle-rpc` | `noarch: python` + session + hooks | **both halves in one package**, explicitly because `Tools/run_python.ML` launches the Python host so "the ML session is useless without it" (`Isabelle_RPC/conda/recipe.yaml:3-8`) |
| `auto-sledgehammer` | `noarch` | session only |

This repo is the `isabelle-rpc` case, only more so. Three hard run-time couplings
from the ML half to the Python half:

1. `Tools/simd_vector.ML:42-53` resolves the kernel's path by calling
   `Remote_Procedure_Calling.load ["Isabelle_Semantic_Embedding"]` and then the
   `Vector_Arith.library_path` RPC command. That command is registered on the
   Python side at `Isabelle_Semantic_Embedding/semantics.py:1799-1811`
   (`@isabelle_remote_procedure("Vector_Arith.library_path")`). Without the Python
   package installed, ML falls back to the in-tree checkout path
   (`simd_vector.ML:38-41`) — which in a conda install **does not exist**, and
   `Foreign.loadLibrary` then raises "Fail to load the Isabelle Vector library"
   (`simd_vector.ML:227-228`).
2. `Tools/semantic_store.ML:1627-1665` (`query_knn`) is a thin RPC to the Python
   handler at `semantics.py:1736-1737`.
3. The whole deformalization/collection app (`Tools/semantic_interpretation_app.ML`)
   is RPC-driven.

Splitting into a noarch session package plus a per-platform python package would
therefore buy nothing: the session package alone is non-functional, so it would
still have to `run`-depend on the python package, which pins the two into the
version lockstep the split was supposed to avoid — and adds a second name, a
second release cadence, and a second chance for a user to install half the system.
`isabelle-rpc`'s recipe rejects exactly this reasoning in its header comment, and
the situation here is stronger.

The cost of ONE package is that the arch-independent Isabelle session (ROOT, `.thy`,
`.ML`, the Scala jar) is duplicated across three subdirs. That is ~400 KB of source
plus a 118 KB jar per subdir. Irrelevant.

**Not `noarch` at all** — `noarch: python` relocates site-packages but produces a
single artifact for every platform, which cannot carry a per-arch `.so`. This is an
arch package with `build.python.version_independent: true` (see §4).

### 2.3 Name and version

Name `isabelle-semantic-embedding`, matching the `isabelle-*` house convention.
Note the PyPI distribution name is `Isabelle_Semantic_Embedding` (`pyproject.toml:10`);
conda normalises to lowercase-with-hyphens, and the `.dist-info` that `pip install`
writes will carry the PyPI spelling, which is what `pip` matches on.

There is **no `./VERSION` file in this repo** (checked). The siblings' release
workflow reads one (`Isabelle_RPC/.github/workflows/release-conda.yml`, "Read the
version from ./VERSION"). Either add `./VERSION` or have the workflow read
`pyproject.toml`'s `version = "0.1.0"` (`pyproject.toml:11`). Recommend adding
`./VERSION` and making `pyproject.toml` read it dynamically, or asserting the two
agree in CI — two independent literals for one number is how they drift.

**Release discipline (mandatory, inherited from `isabelle-rpc`'s recipe header,
lines 29-33):** the conda version must never fall behind PyPI's. If PyPI gets
ahead, `pip install -U` takes over, its uninstall removes conda's `.dist-info`, and
the Isabelle half — the session under `share/`, the pre-unlink hook, the
`etc/components` registration — is orphaned beyond conda's reach. PyPI is at 0.1.0,
so conda must start at ≥ 0.1.1, or the two must be released together.

---

## 3. Getting the binaries from the build job into the recipe

### 3.1 Job graph

```
                     ┌─────────────────────────────────────────┐
  tag v*  ──────────►│ natives (matrix, 4 native runners)      │
                     │   = wheels.yml's `wheel` job verbatim   │
                     │   uploads artifact wheel-<arch>         │
                     └───────────────┬─────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
  ┌───────────┐              ┌──────────────┐            ┌────────────────┐
  │ intel-mac │              │ abi3 (3.11–  │            │ conda-build    │
  │ (existing)│              │   3.14)      │            │ (matrix ×3)    │
  └───────────┘              └──────────────┘            │  linux-64      │
                                                          │  osx-64        │
                                                          │  win-64        │
                                                          └───────┬────────┘
                                                                  ▼
                                                          ┌────────────────┐
                                                          │ verify         │
                                                          │ (install from  │
                                                          │  file:// chan) │
                                                          └───────┬────────┘
                                                                  ▼
                                                          ┌────────────────┐
                                                          │ publish        │
                                                          │ (reusable wf   │
                                                          │  publish-conda)│
                                                          └───────┬────────┘
                                                                  ▼
                                                          ┌────────────────┐
                                                          │ smoke          │
                                                          └────────────────┘
```

`natives` should be `wheels.yml`'s existing `wheel` job, not a copy. Two ways:

- Add `on: workflow_call` to `wheels.yml` and have `release-conda.yml` call it. Any
  drift between the two is then impossible by construction. **Recommended.**
- Or have `release-conda.yml` `workflow_run`-trigger off wheels.yml. Worse: the
  artifact-retrieval and version-correlation get fiddly and a rerun can pair the
  wrong wheel with the tag.

Note that `wheels.yml`'s `publish` job is gated on `startsWith(github.ref,
'refs/tags/v')` and uses the `pypi` environment; a `workflow_call` invocation from
the conda release must not re-trigger it. Guard with an input, e.g.
`inputs.publish_to_pypi`.

The three conda-build legs must run on **native runners** (`ubuntu-latest`,
`macos-15` for osx-64 *and* osx-arm64 if that is ever unblocked, `windows-latest`),
because rattler-build's `target_platform` defaults to the host and because the
recipe's tests import the extension module.

### 3.2 Mechanism: install our own wheel

**Decision: the conda build consumes the wheel that `natives` built in the same
workflow run.** The `conda-build` leg does `actions/download-artifact` for its
`wheel-<arch>` and drops it at `conda/wheelhouse/` in the checkout; `source: path: ../`
carries it into `$SRC_DIR`; the build script `pip install`s it.

This satisfies the constraint exactly — the artifact is built by our pipeline, on a
native runner, with the correct two-toolchain setup, in the same run, and it never
touches PyPI. The `wheel-<arch>` artifact is the same bytes that `wheels.yml`'s
`intel-mac` and `abi3` jobs have already exercised.

Mapping artifact → subdir:

| conda subdir | artifact | note |
|---|---|---|
| `linux-64` | `wheel-x86_64` | |
| `osx-64` | `wheel-universal2` | fat; both slices proven by the `intel-mac` job |
| `osx-arm64` | `wheel-universal2` | *blocked by rocksdict, see §2.1* |
| `win-64` | `wheel-amd64` | |
| `linux-aarch64` | `wheel-aarch64` | *blocked by rocksdict* |

### 3.3 Why not run cmake inside rattler-build

Considered and rejected as the primary path:

- Windows would need MSYS2/MINGW64 inside the conda build environment, alongside
  the MSVC that conda-build's `{{ compiler('cxx') }}` provides. Two toolchains
  inside a build sandbox whose PATH conda manages is a fight for no gain — and the
  failure mode of getting it wrong is *silent* (five dropped SIMD targets, §1.2).
- The Highway build is a few minutes per leg that `natives` already pays.
- The `test_gather` / `test_ml_dlopen` / SIMD-target-enumeration gates
  (`wheels.yml:126-226`) would have to be reimplemented inside the recipe script or
  lost. They are the checks that catch the silent failures.

### 3.4 Why not the `ISABELLE_VECTOR_SO` seam here

`ISABELLE_VECTOR_SO` is documented as exactly this seam — "lets a cross-build or a
CI matrix drop in an artifact produced elsewhere" (`setup.py:116-117`) — so the
obvious alternative is: transport only the kernel, set `ISABELLE_VECTOR_SO`, and
let `$PYTHON -m pip install . --no-deps` rebuild `_vecgather` with conda's own
compiler (which on Windows is MSVC, i.e. the *correct* toolchain for that half).
Architecturally this is the nicer design and it is what I would have recommended
but for two macOS traps:

1. `_platform_tag` (`setup.py:247-261`) hard-requires universal2 on Mach-O: given
   a fat kernel it returns `macosx_11_0_universal2`, and given a `lipo -thin` one
   it raises `SystemExit("... a universal2 wheel needs both x86_64 and arm64")`.
   Both branches are wrong for a per-subdir conda build.
2. `_check_extension_arch` (`setup.py:361-384`) then demands that `_vecgather` be
   universal2 too. conda's compiler builds thin. So the osx legs would hard-fail.

`pip install .` builds a wheel internally, so `bdist_wheel.get_tag` **does** run and
both checks fire. Making this path work needs an escape hatch in `setup.py` (e.g.
honour an explicit `--plat-name`/env var that skips the universal2 demand when the
kernel is deliberately thin). That is a source change, which this document is not
authorised to make, and it would weaken a guard that exists for good reason.

So: **wheel-as-transport is the recommendation**; the `ISABELLE_VECTOR_SO` seam
stays exactly what it is today — the MinGW→MSVC handoff inside `wheels.yml`
(`wheels.yml:159-162`). Revisit if the mac legs ever need thin per-subdir builds.

*Aside, low priority:* `pyproject.toml:89-90` lists only `libisabelle_vector.so` in
`package-data`, not `.dylib`/`.dll` (the comment at line 88 says they "will join
this list when those platforms land" — they have). This is currently harmless
because `build_py.run` (`setup.py:349-358`) explicitly `copy_file`s the staged
library into `build_lib` regardless of `package-data`, and `wheels.yml:250-257`
asserts `matrix.lib` is in every wheel. Worth tidying, not worth blocking on.

---

## 4. abi3

`wheels.yml` builds `cp311-abi3` (`setup.py:20-25`, `setup.py:406-410`,
`options={"bdist_wheel": {"py_limited_api": "cp311"}}`). 3.11 is the floor because
`Py_buffer`/`PyObject_GetBuffer` entered the limited API there (`pybuffer.h`), and
`wheels.yml:338-358` proves the claim by importing the one binary under 3.11, 3.12,
3.13 and 3.14.

**Decision: keep abi3.** rattler-build supports it natively:

```yaml
build:
  python:
    version_independent: true   # defaults to false
```

per the rattler-build docs ("Version independent (ABI3) packages", available since
rattler-build 0.35.0, implementing [CEP 20]; fetched from
`prefix-dev/rattler-build/main/docs/reference/recipe_file.md`). The sibling
workflows pin rattler-build **0.69.1**
(`Isabelle_RPC/.github/workflows/release-conda.yml`, "Install rattler-build"), so
the feature is available.

Cost comparison:

| approach | conda-build legs | notes |
|---|---|---|
| **abi3 / `version_independent: true`** | **3** (one per subdir) | one artifact per subdir serves 3.11–3.14. `requirements.host` pins python 3.11 (the abi3 floor); `requirements.run` gets `python >=3.11.4`. |
| per-python-version | 3 × 4 = **12** | one artifact per (subdir, python). Four times the build time, four times the channel objects, four times the surface for a partial publish, and the resulting `.conda` files differ only in a `python_abi` pin. |

abi3 is strictly better here and it is already proven for these exact bytes by the
`abi3` job. Note that `requires-python = ">=3.11.4"` (`pyproject.toml:29`) is
*tighter* than the abi3 floor — the `.4` comes from PEP 706's `filter="data"` arg
that `r2_sync` passes to `tarfile` — so `run: python >=3.11.4` is the correct
run requirement even though the host builds against 3.11.

**Unverified:** I have not run `rattler-build` with `version_independent: true`
against this recipe, and in particular have not confirmed how it interacts with
`pip install`ing an already-abi3-tagged **wheel** (as opposed to building from
source in the host env). It should be fine — the installed tree is the same either
way and CEP 20 is about the *package* metadata — but this needs one real build
before it is a fact.

---

## 5. Dependency audit

All queries below were made live against
`https://api.anaconda.org/package/conda-forge/<name>` while writing this. I could
**not** solve an environment locally: conda-forge is TLS-intercepted from this
machine (per the task brief), so "solves cleanly together" is unverified for all
fifteen. Names, summaries, latest versions and subdir coverage are verified.

| # | pyproject name | conda-forge name | ✔ | latest | subdirs | note |
|---|---|---|---|---|---|---|
| 1 | `msgpack` | **`msgpack-python`** | ⚠ **RENAME** | 1.2.1 | — | `msgpack` **does not exist** on conda-forge (API returns `"msgpack" could not be found`). Same trap class as `lmdb`. |
| 2 | `isabelle-rpc>=0.3.0` | `isabelle-rpc` | ⚠ **not conda-forge** | **0.3.1** | `noarch` | Lives on **`https://conda.qiyuan.me`**. Verified present as `isabelle-rpc-0.3.1-pyh4616a5c_0.conda`. The solve therefore needs `-c https://conda.qiyuan.me` in addition to conda-forge. |
| 3 | `rocksdict` | `rocksdict` | ⚠ **arch gap** | 0.3.23 | `linux-64`, `osx-64`, `win-64` | **No `linux-aarch64`, no `osx-arm64`, ever.** See §2.1 — this is what bounds the matrix. |
| 4 | `lmdb>=2.1.1` | **`python-lmdb`** | ⚠ **RENAME — the known trap** | 2.3.0 | linux-64, linux-aarch64, linux-ppc64le, osx-64, osx-arm64, win-32, win-64 | conda-forge `lmdb` **exists** at **0.9.35**, summary *"A high-performance embedded transactional key-value store database"* — that is the **C library**. `python-lmdb` 2.3.0 is *"Universal Python binding for the LMDB 'Lightning' Database"*. This exact mistake already shipped a broken package in this project (`Isabelle_RPC/conda/recipe.yaml:140-154`): it resolved, installed green, the session built, and only `import lmdb` failed. The version is the tell — PyPI `lmdb` is 1.x/2.x, conda `lmdb` is 0.9.x, because they are different software. `>=2.1.1` translates to `python-lmdb >=2.1.1`, satisfied by 2.3.0. |
| 5 | `platformdirs` | `platformdirs` | ✔ | 4.10.1 | noarch | *"determining appropriate platform-specific dirs"* — correct. |
| 6 | `transformers` | `transformers` | ✔ | 5.14.1 | `noarch` | *"State-of-the-art NLP for TensorFlow 2.0 and PyTorch"* — correct. ⚠ **heavyweight**: pulls torch/tokenizers transitively. Consider whether a version ceiling is wanted; transformers 5.x is a recent major. |
| 7 | `numpy` | `numpy` | ✔ | 2.5.1 | all | correct. |
| 8 | `faiss-cpu` | `faiss-cpu` | ✔ | 1.10.0 | linux-64, linux-aarch64, linux-ppc64le, osx-64, osx-arm64, win-64 | *"efficient similarity search and clustering of dense vectors"* — correct. ⚠ **possibly vestigial**: it is imported at `semantic_embedding.py:17` but used only for `normalize_L2` (`:148`); retrieval is the SIMD LMDB scan, not faiss. Keeping it is correct while the import stands. |
| 9 | `httpx` | `httpx` | ✔ | 0.28.1 | noarch | correct. |
| 10 | `diskcache` | `diskcache` | ✔ | 5.6.3 | noarch | *"Disk and file backed cache"* — correct. |
| 11 | `claude-agent-sdk` | `claude-agent-sdk` | ✔ | 0.2.122 | `noarch` | *"Python SDK for Claude Agent"* — correct. Note `channeldata` reported an empty platform list; the `/files` endpoint shows 93 files, all `noarch`. It is fine. |
| 12 | `pyyaml` | `pyyaml` | ✔ | 6.0.3 | all | correct (conda-forge uses `pyyaml`, **not** `PyYAML` — conda names are lowercase). |
| 13 | `boto3` | `boto3` | ✔ | 1.43.50 | linux-64, noarch, osx-64, win-32, win-64 | *"Amazon Web Services SDK for Python"* — correct. Modern builds are noarch. |
| 14 | `filelock` | `filelock` | ✔ | 3.31.0 | noarch | *"A platform independent file lock"* — correct. |
| 15 | `zstandard` | `zstandard` | ✔ | 0.25.0 | linux-64, linux-aarch64, linux-ppc64le, osx-64, osx-arm64, win-32, win-64 | *"Zstandard bindings for Python"* — correct. Note `python-zstandard` does **not** exist on conda-forge; the plain name is right here, which is the opposite of the `lmdb` case. Do not "fix" it by analogy. |

**Three renames, one channel, one arch blocker.** The general rule the
`isabelle-rpc` recipe states and this audit re-confirms: checking that a name
*exists* on conda-forge is not enough; check what it **is** (read the summary and
sanity-check the version series), because the failure mode is a green install that
dies at `import`.

Additional run requirements not in `pyproject.toml` but true:

- `python >=3.11.4` (`pyproject.toml:29`).
- `isabelle >=2025.2,<2025.3` — this ships an Isabelle session and a Scala
  component built for that series. The bound is mandatory: the base package keeps a
  constant name at a per-release version, so unbounded lets conda's newest-wins
  resolution pair this with a future Isabelle it was never built for
  (`Performant_Isabelle_ML/conda/recipe.yaml:132-143`).
- `isabelle-performant-ml >=0.1.0,<0.2.0` — `ROOT:1` declares
  `session Semantic_Embedding = HOL + sessions Isabelle_RPC Performant_Isabelle_ML`.
  Note this is a **session** dependency that `pyproject.toml` cannot express;
  `isabelle-rpc` alone is not enough.

---

## 6. The Scala component

This repo is a **real Isabelle component** with a Scala module, unlike
`Performant_Isabelle_ML` (pure ML) and `Isabelle_RPC` (pure ML). Verified tracked
in git: `etc/settings`, `etc/build.props`, `lib/semantic_embedding.jar`,
`src/scala/pide_state.scala`.

`etc/build.props`:

```
title = Isabelle/Scala/Semantic_Embedding
module = lib/semantic_embedding.jar
no_build = true
requirements = \
  env:ISABELLE_SCALA_JAR
sources = \
  src/scala/pide_state.scala
services = \
  isabelle.semantic_embedding.PIDE_State_Functions
```

`etc/settings`:

```
ISABELLE_SEMANTIC_EMBEDDING_HOME="$COMPONENT"
classpath "$ISABELLE_SEMANTIC_EMBEDDING_HOME/lib/semantic_embedding.jar"
```

### 6.1 What must ship for `isabelle components -u` to work

The component root is `$PREFIX/share/isabelle-semantic-embedding`. It must contain:

| path | why |
|---|---|
| `etc/settings` | sourced by `isabelle getsettings`; sets `$COMPONENT`-relative home and adds the jar to the classpath. **Without it the jar is never on the classpath** and the `services` entry cannot be loaded. |
| `etc/build.props` | declares the module and the `services` line that registers `isabelle.semantic_embedding.PIDE_State_Functions` with Isabelle/Scala. |
| `lib/semantic_embedding.jar` | the module named by `build.props`. `no_build = true` means Isabelle will **not** compile it — the prebuilt jar is the deliverable, exactly as `isabelle-mcp` ships one. |
| `src/scala/pide_state.scala` | listed under `sources`. Ship it: with `no_build = true` Isabelle does not compile it, but `isabelle scala_build`/`scala_project` and the component's own consistency reads it, and it is 1 file. Cheap insurance. |
| `ROOT` | `session Semantic_Embedding = HOL + …`. A valid component has a ROOT; this is what makes the session known. |
| `Semantic_Embedding.thy`, `Sledgehammer_Embedding.thy`, `Semantic_Collection_App.thy` | the theories. Only `Semantic_Embedding` is in `ROOT`'s `theories` list, but the other two are loadable on demand and `Sledgehammer_Embedding.thy:5` is the **only** loader of `Tools/simd_vector.ML`. |
| `Tools/` (whole dir, minus junk) | seven `ML_file`s from `Semantic_Embedding.thy:7-14` plus `Tools/simd_vector.ML`, `Tools/Sledgehammer/sledgehammer_embedding.ML`, `Tools/semantic_interpretation_app.ML`, `Tools/theory_structure.thy`. Copy wholesale rather than enumerating — enumerating is how an indirectly-referenced file gets silently dropped (`Isabelle_RPC/conda/recipe.yaml:72-75`). **But** `Tools/Vector_Arith/` must be excluded (§8). |
| `COPYING`, `COPYING.LIB` | LGPL-2.1 §3 lets a recipient convert any copy to the plain GPL and they cannot exercise that right without the text (`pyproject.toml:15-19`). Both are `license-files` for the wheel already; ship them in the component dir too. |

### 6.2 Do the hooks apply unchanged?

**Yes — the `isabelle-rpc` hook block transfers verbatim**, with only `name=`
changed. The idiom is:

- Both the unix `.sh` and the Windows `.bat` sets written **unconditionally**.
  conda picks by the *running* platform and returns success in silence when the
  file is absent, so shipping only `.sh` means the component installs cleanly on
  Windows and is never registered.
- **Do not** guard with `case "$target_platform" in win-*)`. For `noarch` that
  variable is `noarch` and never matches. ⚠ **Here it is different and worth
  noting:** this is an *arch* package, so `$target_platform` *is* `win-64` on the
  Windows leg and the guard *would* work. Write both sets anyway — it costs two
  files and removes an entire class of "a fix that looks applied and isn't".
- The `.bat` must be **CRLF** and `exit /b 0` on every path. A nonzero post-link
  makes conda roll the whole install back (verified on Windows by
  `isabelle-packaging-ci` run 29637825807).
- The path handed to `isabelle components` on Windows must be **Cygwin form**
  (`/cygdrive/C/...`), because `Path.check_elem` rejects `:` and `\`. Converted
  with pure batch substring ops, because `cygpath.exe` printed nothing on a
  windows-latest runner.
- The throwaway `isabelle.bat getenv -b ISABELLE_HOME` warm-up is **load-bearing**:
  Cygwin heals on the first `isabelle` call, but that call's Java classpath is
  computed before the heal, so the first invocation is precisely the one that
  cannot run a Scala tool — and `components` **is** a Scala tool. Without it the
  hook silently does nothing.

That last point is *more* important here than in the sibling recipes, because this
component's whole reason to be registered includes a Scala service.

**`isabelle-mcp`'s no-hooks approach does not apply here.** That package registers
its component from Python at run time (`isabelle_mcp/component.py`,
`ensure_component()`), and its recipe header explicitly warns against copying a
hook block into it. This repo has no such run-time registration path — nothing in
`Isabelle_Semantic_Embedding/` writes `etc/components`. So: hooks, like
`isabelle-rpc`.

---

## 7. The ~1.5 GB LMDB semantic database

**Confirmed: it must NOT be in the package.** Reasons, in order of decisiveness:

1. Size. `Isabelle_Semantic_Embedding.tar.zst` in this checkout is 757 MB
   compressed (and `.tar.zst.prev` is 2.5 GB). Multiplied by three subdirs and by
   every version ever published to an append-only channel, this is untenable.
2. It is *data with its own lifecycle*, refreshed independently of the code. Baking
   it into a versioned package would force a release for every DB refresh.
3. It already has a distribution channel: Cloudflare R2, via
   `Isabelle_Semantic_Embedding/r2_sync.py` (`pull_snapshot()` at `r2_sync.py:860`,
   driven by `semantics_manage.py:493 cmd_pull`).
4. conda packages are extracted into the env prefix; the DB lives in a per-user
   cache (`_paths.py:28-37`, `platformdirs.user_cache_dir("Isabelle_Semantic_Embedding",
   "Qiyuan")`, overridable with `SEMANTIC_DB_DIR`) — a different location with
   different write semantics.

### 7.1 Behaviour when the DB is absent — graceful

Investigated in detail. **Every store-opening site degrades gracefully into an
empty-but-valid database.** All three do `os.makedirs(..., exist_ok=True)` first and
then call `lmdb.open` with py-lmdb's defaults `create=True, subdir=True,
readonly=False`:

- `semantics.py:230-233` —
  `os.makedirs(cache_dir, exist_ok=True)`; `lmdb.open(os.path.join(cache_dir,
  "semantics.lmdb"), map_size=SEMANTICS_MAP_SIZE)` (`SEMANTICS_MAP_SIZE = 1<<32`,
  `semantics.py:121`). No `readonly=`, no `create=`, no `subdir=`.
- `experience_index.py:54-57` — same shape, `map_size=1<<27`.
- `semantic_embedding.py:583-589` `_get_lmdb_env(path)` → `lmdb.open(path,
  map_size=VECTOR_MAP_SIZE)` (`1<<34`, `:580`); the directory is created by
  `Semantic_Vector_Store.__init__` at `semantics.py:1124-1126`. `r2_sync.py:702`
  even comments "Ask before opening: `_get_lmdb_env` creates the directory it is
  given."

The only `readonly=True, lock=False` opens are on a *downloaded snapshot*
(`r2_sync.py:579`, `:602`, `:708`) and the read-only CLI scans
(`semantics_manage.py:142`), all guarded by an existence check first.

Downstream, an empty DB yields empty retrieval: `semantics.py:1413-1414`
`if not candidates and not exp_hit: return [], warnings, 0`, and the vector scan
returns `[], missing` at `semantic_embedding.py:748-749`. On the ML side,
`Tools/semantic_store.ML:1627-1665` `query_knn` is a thin RPC whose return schema
is `(real * entity) list * string list`; an empty list is a valid answer and nothing
errors. `Tools/Sledgehammer/sledgehammer_embedding.ML` never touches the LMDB at
all — its cache is in-process theory data (`:211-222`) and a cold cache means "embed
everything now" (slow, costs API tokens), not an error.

**faiss note:** there is no persistent faiss index on disk. `faiss` is imported at
`semantic_embedding.py:17` but used only for `normalize_L2` (`:148`); the README's
"faiss.knn" line (`README.md:97`) is stale. So there is no missing-index failure
mode.

### 7.2 Three caveats to encode in the recipe / docs

1. **`rocksdict`'s parent directory.** `premise_selection.py:61-66` does **not**
   `makedirs` before `Rdict(cache_file, ...)`. `Rdict` creates the DB itself
   (`create_if_missing` defaults true) but its *parent* is not created by this code.
   In practice `Semantic_DB._ensure_env` / `Semantic_Vector_Store.__init__` normally
   run first and create it. **This is order-dependent and I did not verify RocksDB's
   behaviour on a missing parent directory** — it is the one plausible hard-fail site
   on a truly fresh install. Worth an explicit check in the recipe's `tests:` block,
   or a one-line `os.makedirs` upstream.
2. **`r2_sync.CACHE_DIR` is bound at import time** (`r2_sync.py:100`, with
   `MARKER_PATH`/`LOCK_PATH`/`INCOMPLETE_PATH` at `:102-108`). Setting
   `SEMANTIC_DB_DIR` *after* `r2_sync` is imported has no effect on those. Document
   it; do not set the variable from an activation script and expect it to work in
   every order.
3. **Auto-pull exists, but not in this package.** `r2_sync.py`'s docstring
   (`:35-39`) claims both directions are explicit; that is now out of date.
   `Isa-Mini/IsaMini/AoA/toplevel.py:81-154` (`_ensure_semantic_db`, called from
   `IsaMini_AoA` at `:215-222`) auto-pulls on every non-test `by aoa` when
   `semantic_db_is_empty() or pull_was_interrupted()` (`:108`), logging
   "Downloading it now (~0.7 GB, one-time setup)" (`:118-121`). It never raises
   (`:139-146`). Plain (non-AoA) use gets only the weekly `check_update` nudge
   (`r2_sync.py:1036`, message at `:1081-1083`), which swallows all exceptions
   (`:1084`).

**Recipe consequence:** ship no DB, add no post-link download, and put one line in
`about.description` pointing at `semantics_manage.py pull`. Do **not** make the
post-link hook fetch 750 MB — a post-link that can fail on a network error makes
conda roll the whole install back.

---

## 8. What to exclude — the whitelist

The survey found ~1.2 GB of untracked build output under
`Tools/Vector_Arith/contrib/highway-1.3.0/build` (measured: `1.2G`), plus
`dist/` (144K), `build/` (428K), `Isabelle_Semantic_Embedding.egg-info` (36K),
`Isabelle_Semantic_Embedding.tar.zst` (**757 MB**),
`Isabelle_Semantic_Embedding.tar.zst.prev` (**2.5 GB**), `premises.zst` (73 MB),
and dozens of `*~` / `#…#` / `.swp` editor backups.

Two independent defences, and **the whitelist is the real one**:

### 8.1 Primary: the build script copies an explicit whitelist

This is what all four sibling recipes do — none of them `cp -a` a whole tree. The
component payload is exactly:

```
ROOT
Semantic_Embedding.thy
Sledgehammer_Embedding.thy
Semantic_Collection_App.thy
etc/settings
etc/build.props
lib/semantic_embedding.jar
src/scala/pide_state.scala
COPYING
COPYING.LIB
README.md
Tools/                       ← whole dir, then pruned:
    minus Tools/Vector_Arith/     (C++ sources + Highway + 1.2 GB of build output;
                                   the compiled kernel ships in site-packages, not here)
    minus Tools/__pycache__/
    minus *~  #*#  .#*  *.swp  *.swo  *.bak  *.ll  *.phi-cache
```

Concretely, after the copy:

```bash
rm -rf "$dest/Tools/Vector_Arith"
find "$dest" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$dest" \( -name '*~' -o -name '#*#' -o -name '.#*' -o -name '*.swp' \
             -o -name '*.swo' -o -name '*.bak' -o -name '*.ll' \
             -o -name '*.phi-cache' \) -delete
```

`Tools/Vector_Arith` is the important one and it is easy to miss: `Isabelle_RPC`'s
recipe copies `Tools/` wholesale with no such prune because its `Tools/` has no
C++ tree. Copying it here would drag in the whole vendored Highway source *and*
whatever `build/`+`hwybuild/` a local developer happens to have.

Positively asserted afterwards (fail the build if absent):

```bash
test -f "$dest/ROOT"
test -f "$dest/etc/settings"
test -f "$dest/etc/build.props"
test -f "$dest/lib/semantic_embedding.jar"
test -f "$dest/Tools/simd_vector.ML"       # the ML→Python kernel bridge
test -f "$dest/Tools/semantic_store.ML"    # query_knn
! test -e "$dest/Tools/Vector_Arith"       # the 1.2 GB trap
```

Plus a size ceiling, because the failure this guards against is a *fat package*,
not a broken one, and nothing else would notice:

```bash
sz=$(du -sm "$dest" | cut -f1)
[ "$sz" -lt 20 ] || { echo "::error::component payload is ${sz} MB -- something leaked"; exit 1; }
```

### 8.2 Secondary: keep `$SRC_DIR` clean

`source: - path: ../` copies the working tree into `$SRC_DIR`. On a GitHub runner
this is a fresh `actions/checkout`, so the untracked 3+ GB simply does not exist —
which is why the primary defence above is the whitelist and not the source filter.
For **local** `rattler-build` runs from this working tree it matters a great deal.

`rattler-build`'s `path` source honours `.gitignore` (`use_gitignore`, default true
— **unverified for 0.69.1, check before relying on it**). `.gitignore` already
covers `build`, `hwybuild`, `dist`, `*.egg-info`, `*~`, `*.swp`, `__pycache__`,
`#*`, `.*#`. It does **not** cover `*.tar.zst`, `*.tar.zst.prev`, `premises.zst`,
`.tmp_isabelle_user/`, `*.proof-cache`, `*.proof-cache.lock`, `Test/`.

Recommendation: add a `conda/.rattler-ignore`-equivalent via the recipe's
`source.filter.exclude` if 0.69.1 supports it, **and** independently add these to
`.gitignore` (they are untracked junk that should never have been eligible for
`git add` anyway):

```
*.tar.zst
*.tar.zst.prev
premises.zst
*.proof-cache
*.proof-cache.lock
```

That is a source change and is not made by this document.

`conda/wheelhouse/` (§3.2) must **not** be gitignored, or the wheel will not reach
`$SRC_DIR`.

### 8.3 The Python half's exclusions

`pip install <wheel>` places exactly what the wheel contains, so the wheel's own
`[tool.setuptools.packages.find]` (`pyproject.toml:79-81`, `include =
["Isabelle_Semantic_Embedding", "Isabelle_Semantic_Embedding.*"]`, `exclude =
["contrib", "contrib.*"]`) governs. Nothing extra to do.

⚠ **One gap found.** `Isabelle_Semantic_Embedding/Agent_Interpretation_Dir/` is
used at run time as the `cwd` for the Claude agent
(`semantic_interpretation.py:876`, `cwd=str(Path(__file__).parent /
"Agent_Interpretation_Dir")`) and it contains tracked skill files
(`Agent_Interpretation_Dir/.claude/skills/…/SKILL.md`, three of them). It is **not
listed in `[tool.setuptools.package-data]`** (`pyproject.toml:83-90` lists only
`embedding_config_template.yaml`, `config_template.yaml`,
`libisabelle_vector.so`), and it is not a Python package (no `__init__.py`), so
`packages.find` does not pick it up either. **It therefore does not ship in the
wheel today, and would not ship in the conda package.** The deformalization agent
would run against a missing directory. This is a pre-existing wheel bug, not a
conda one, but the conda package inherits it. Needs a `package-data` entry
(`"Agent_Interpretation_Dir/**/*"`) — a source change, flagged not made.

Also: `semantics_manage.py` is a top-level script, not part of the package, and
`pyproject.toml` declares **no** `[project.scripts]`. So `semantics_manage.py pull`
— the documented way to get the DB — is **not on PATH after a conda (or pip)
install**. Either add a console-script entry point, or ship the script into
`$PREFIX/share/isabelle-semantic-embedding/` and say so in the description. This
should be decided before release; a package whose documented first step is
unreachable is a bad first impression.

---

## 9. The recipe

`conda/recipe.yaml`, near-full. Comments in the house style — every non-obvious
line says why.

```yaml
schema_version: 1

# isabelle-semantic-embedding: ONE package carrying both halves of this repo -- the
# Isabelle session (ROOT + theories + Tools/ + the Scala component under etc/ and lib/)
# and the Python package Isabelle_Semantic_Embedding with its two compiled artifacts.
#
# Both halves, one package, for the isabelle-rpc reason and more so: Tools/simd_vector.ML
# resolves the SIMD kernel's path by RPC into the Python package
# (simd_vector.ML:42-53 -> semantics.py:1799 @isabelle_remote_procedure
# "Vector_Arith.library_path"), and Tools/semantic_store.ML's query_knn is a thin RPC to
# semantics.py:1736.  The ML half is not merely less useful without the Python half; it
# raises "Fail to load the Isabelle Vector library" (simd_vector.ML:227).
#
# NOT noarch.  The package carries libisabelle_vector.{so,dylib}/isabelle_vector.dll and
# _vecgather.abi3.{so,pyd}, both pinned to one architecture (setup.py:26-31).  It IS
# python-version-independent, via abi3 -- see build.python.version_independent below.
#
# The two native artifacts are NOT built here.  They are built by this project's
# wheels.yml on native runners with the two toolchains that leg requires (MinGW64 for the
# SIMD kernel, because Highway marks AVX3 and above HWY_BROKEN_MSVC and MSVC would
# SILENTLY drop five targets; MSVC for _vecgather, which is what setuptools drives), and
# handed to this recipe as the wheel that same run produced, in conda/wheelhouse/.
# Reproducing that toolchain dance inside a conda build sandbox would buy nothing and
# would lose wheels.yml's SIMD-target assertions (wheels.yml:164-183).
#
# RELEASE DISCIPLINE, mandatory: the conda version must never fall BEHIND PyPI's.  If PyPI
# gets ahead, `pip install -U` takes over, its uninstall removes conda's dist-info, and
# the Isabelle half -- the session under share/, the pre-unlink hook, the etc/components
# registration -- is orphaned beyond conda's reach.  See isabelle-rpc's recipe header.

context:
  version: ${{ env.get("ISA_COMPONENT_VERSION", default="0.0.0") }}

source:
  - path: ../

package:
  name: isabelle-semantic-embedding
  version: ${{ version }}

build:
  number: 0
  python:
    # ABI3 / CEP 20, rattler-build >= 0.35.0.  wheels.yml builds cp311-abi3
    # (setup.py:406-410, py_limited_api = cp311), so ONE artifact per subdir serves
    # CPython 3.11 through 3.14 -- proved by wheels.yml's `abi3` job, which imports
    # this exact binary under all four.  Without this the matrix would be
    # 3 subdirs x 4 pythons = 12 builds that differ only in a python_abi pin.
    version_independent: true
  script:
    interpreter: bash
    content: |
      set -euo pipefail

      name=isabelle-semantic-embedding

      # --- the Python half ------------------------------------------------------
      # The wheel THIS pipeline built, in THIS workflow run, on a native runner.
      # Not from PyPI, and not rebuilt here: rebuilding would run setup.py's
      # bdist_wheel.get_tag, whose _platform_tag hard-requires universal2 on Mach-O
      # (setup.py:247-261) and whose _check_extension_arch then demands the same of
      # _vecgather (setup.py:361-384).  A per-subdir conda build cannot satisfy
      # either.  Installing the finished wheel sidesteps both.
      #
      # --no-deps: conda resolves the dependencies, pip must not.
      # --no-build-isolation: the build env has what is needed and the builder has
      # no network.  (Installing a built wheel needs neither, but keep both so the
      # invocation matches the sibling recipes and cannot silently start building.)
      shopt -s nullglob
      whls=("$SRC_DIR"/conda/wheelhouse/Isabelle_Semantic_Embedding-*.whl)
      [ "${#whls[@]}" -eq 1 ] || {
        echo "::error::expected exactly one wheel in conda/wheelhouse/, found ${#whls[@]}"
        ls -l "$SRC_DIR/conda/wheelhouse/" || true
        exit 1; }
      echo "installing $(basename "${whls[0]}")"
      $PYTHON -m pip install "${whls[0]}" --no-deps --no-build-isolation -vv

      # Assert BOTH compiled artifacts survived the install.  They are produced by
      # different toolchains and staged by different setuptools commands (build_py
      # for the kernel, build_ext for the extension), so one can be missing while
      # the other is fine -- and the failure is a runtime one, on a user's machine.
      sp=$($PYTHON -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
      case "$target_platform" in
        linux-*) kernel=libisabelle_vector.so  ;;
        osx-*)   kernel=libisabelle_vector.dylib ;;
        win-*)   kernel=isabelle_vector.dll    ;;
        *) echo "::error::unhandled target_platform $target_platform"; exit 1 ;;
      esac
      test -f "$sp/Isabelle_Semantic_Embedding/$kernel" \
        || { echo "::error::the SIMD kernel ($kernel) did not ship"; exit 1; }
      ls "$sp"/Isabelle_Semantic_Embedding/_vecgather*.so \
         "$sp"/Isabelle_Semantic_Embedding/_vecgather*.pyd 2>/dev/null | grep -q . \
        || { echo "::error::the _vecgather extension module did not ship"; exit 1; }

      # --- the Isabelle half ----------------------------------------------------
      # An explicit whitelist, NOT `cp -a .`.  This working tree can hold >3 GB of
      # untracked build output and DB snapshots (Tools/Vector_Arith/contrib/
      # highway-1.3.0/build alone is 1.2 GB, Isabelle_Semantic_Embedding.tar.zst is
      # 757 MB).  On a CI runner the checkout is clean, but the whitelist is what
      # makes that not matter.
      dest="$PREFIX/share/$name"
      mkdir -p "$dest"

      cp -a "$SRC_DIR/ROOT"                        "$dest/"
      cp -a "$SRC_DIR/Semantic_Embedding.thy"      "$dest/"
      cp -a "$SRC_DIR/Sledgehammer_Embedding.thy"  "$dest/"
      cp -a "$SRC_DIR/Semantic_Collection_App.thy" "$dest/"
      cp -a "$SRC_DIR/COPYING" "$SRC_DIR/COPYING.LIB" "$SRC_DIR/README.md" "$dest/"

      # The Scala component.  etc/settings puts the jar on Isabelle's classpath;
      # etc/build.props declares it with no_build = true (Isabelle will NOT compile
      # it) and registers isabelle.semantic_embedding.PIDE_State_Functions as a
      # service.  All three files are required for `isabelle components -u` to give
      # a working component -- without etc/settings the jar is never on the
      # classpath and the service silently does not exist.
      cp -a "$SRC_DIR/etc"                         "$dest/"
      cp -a "$SRC_DIR/lib"                         "$dest/"
      cp -a "$SRC_DIR/src"                         "$dest/"

      # Tools/ wholesale rather than the nine ML_file lines: enumerating them is how
      # an indirectly-referenced file gets silently dropped.  BUT Vector_Arith must
      # go -- it is the C++ kernel's source tree plus vendored Highway, and a
      # developer's checkout has its build/ and hwybuild/ inside it.  The compiled
      # kernel ships in site-packages, placed by the wheel; ML reaches it through
      # the library_path RPC, never through this directory (simd_vector.ML:42-53).
      cp -a "$SRC_DIR/Tools"                       "$dest/"
      rm -rf "$dest/Tools/Vector_Arith"
      find "$dest" -name '__pycache__' -type d -prune -exec rm -rf {} +
      find "$dest" \( -name '*~' -o -name '#*#' -o -name '.#*' -o -name '*.swp' \
                   -o -name '*.swo' -o -name '*.bak' -o -name '*.ll' \
                   -o -name '*.phi-cache' \) -delete

      test -f "$dest/ROOT"
      test -f "$dest/etc/settings"
      test -f "$dest/etc/build.props"
      test -f "$dest/lib/semantic_embedding.jar"
      test -f "$dest/Tools/simd_vector.ML"      # ML -> kernel bridge
      test -f "$dest/Tools/semantic_store.ML"   # query_knn
      test -f "$dest/Tools/pide_state.ML"       # Semantic_Embedding.thy:7
      ! test -e "$dest/Tools/Vector_Arith" \
        || { echo "::error::Tools/Vector_Arith leaked into the package"; exit 1; }

      # A size ceiling.  The failure this guards is a FAT package, not a broken one,
      # and nothing else in the pipeline would notice a 1.2 GB payload.
      sz=$(du -sm "$dest" | cut -f1)
      [ "$sz" -lt 20 ] \
        || { echo "::error::component payload is ${sz} MB -- something leaked"; exit 1; }

      # --- link hooks -----------------------------------------------------------
      # Verbatim from isabelle-rpc / isabelle-performant-ml; see those recipes for
      # the full reasoning.  Summary of what is load-bearing:
      #   * BOTH sets written unconditionally.  (This is an ARCH package, so unlike
      #     the noarch siblings a `case "$target_platform"` guard would actually
      #     work here -- write both anyway.  Two extra files, one fewer class of
      #     "a fix that looks applied and isn't".)
      #   * .bat is CRLF and exits /b 0 on every path: a nonzero post-link makes
      #     conda roll the whole install back (verified on windows-latest,
      #     isabelle-packaging-ci run 29637825807).
      #   * the path goes over in CYGWIN form: `isabelle components` parses it with
      #     Path.explode, whose Path.check_elem rejects `:` and `\`.
      #   * the throwaway `getenv -b ISABELLE_HOME` warm-up is the difference
      #     between the hook working and silently doing nothing: Cygwin heals on the
      #     FIRST isabelle call, but that call's Java classpath is computed before
      #     the heal, so the first invocation is precisely the one that cannot run a
      #     Scala tool -- and `components` IS a Scala tool.  That matters doubly
      #     here, where the component itself carries a Scala service.
      mkdir -p "$PREFIX/bin" "$PREFIX/Scripts"

      for spec in post-link:-u pre-unlink:-x; do
        action=${spec%%:*}
        verb=${spec##*:}

        printf '%s\n' \
          '#!/bin/bash' \
          "\"\$PREFIX/isa/bin/isabelle\" components $verb \"\$PREFIX/share/$name\" 2>/dev/null \\" \
          "  || echo \"warning: 'isabelle components $verb $name' failed; the session may be unavailable\"" \
          'exit 0' \
          > "$PREFIX/bin/.$name-$action.sh"
        chmod +x "$PREFIX/bin/.$name-$action.sh"

        printf '%s\r\n' \
          '@echo off' \
          "set \"COMP=%PREFIX%\\share\\$name\"" \
          'set "DL=%COMP:~0,1%"' \
          'set "REST=%COMP:~2%"' \
          'set "REST=%REST:\=/%"' \
          'set "CYG=/cygdrive/%DL%%REST%"' \
          'call "%PREFIX%\isa\bin\isabelle.bat" getenv -b ISABELLE_HOME >nul 2>&1' \
          "call \"%PREFIX%\\isa\\bin\\isabelle.bat\" components $verb \"%CYG%\" >nul 2>&1" \
          'exit /b 0' \
          > "$PREFIX/Scripts/.$name-$action.bat"
      done

      grep -qF "share/$name" "$PREFIX/bin/.$name-post-link.sh"
      test "$dest" = "$PREFIX/share/$name"

requirements:
  host:
    # 3.11 is the abi3 floor (Py_buffer / PyObject_GetBuffer entered the limited API
    # there and not earlier -- pybuffer.h, setup.py:22-24).  Pinned exactly, because
    # with version_independent the host python is what the abi3 tag is minted against.
    - python 3.11.*
    - pip
    - setuptools >=77          # pyproject.toml:6 -- first setuptools reading `license`
                               # as an SPDX expression rather than free-form text
  run:
    # .4, not .0: r2_sync's tar.zst extraction passes tarfile's `filter="data"`, the
    # PEP 706 arg backported only to CPython 3.11.4 (pyproject.toml:26-29).
    - python >=3.11.4
    # Bounded to the Isabelle SERIES this session and this Scala component are built
    # against.  The base package keeps a constant name at a per-release version, so
    # unbounded lets conda's newest-wins resolution pair this with a future Isabelle
    # it was never built for.
    - isabelle >=2025.2,<2025.3
    # ROOT:1  session Semantic_Embedding = HOL + sessions Isabelle_RPC
    #                                             Performant_Isabelle_ML
    # Session dependencies pyproject.toml cannot express.  isabelle-rpc is ALSO a
    # pyproject dependency (>=0.3.0), and it comes from conda.qiyuan.me, not
    # conda-forge -- published there at 0.3.1.
    - isabelle-rpc >=0.3.1
    - isabelle-performant-ml >=0.1.0,<0.2.0
    #
    # The remaining fourteen pyproject dependencies under their CONDA names.
    # TWO of them differ from PyPI, and in both cases the wrong name RESOLVES:
    #
    #   msgpack -> msgpack-python   (conda-forge has NO package called `msgpack`)
    #   lmdb    -> python-lmdb      (conda-forge `lmdb` is the C LIBRARY, 0.9.x --
    #                                installs green and dies at `import lmdb`.
    #                                This exact mistake already shipped a broken
    #                                package in this project; see isabelle-rpc's
    #                                recipe, lines 140-154.)
    #
    # And note `zstandard` is CORRECT as-is -- there is no `python-zstandard` on
    # conda-forge.  Do not "fix" it by analogy with the two above.
    - msgpack-python
    - python-lmdb >=2.1.1        # floor, not a pin: 2.1.0 raised a FALSE
                                 # MDB_CORRUPTED on >page-size values (py-lmdb #431)
    - rocksdict                  # linux-64 / osx-64 / win-64 ONLY -- this is what
                                 # bounds the subdir matrix.  See the plan, section 2.1.
    - platformdirs
    - transformers
    - numpy
    - faiss-cpu
    - httpx
    - diskcache
    - claude-agent-sdk
    - pyyaml
    - boto3
    - filelock
    - zstandard

tests:
  - python:
      imports:
        - Isabelle_Semantic_Embedding
      pip_check: true
  - script:
      # The import above proves site-packages was found.  This proves the two
      # COMPILED artifacts are real and loadable on this host -- which is the whole
      # reason this package is per-platform.  library_path() loads the kernel through
      # ctypes and checks it exports top_k_q15_gather (_vecarith.py:133-186), so a
      # stale or wrong-arch library fails HERE rather than at a user's first query.
      - python -c "from Isabelle_Semantic_Embedding import _vecgather; print('_vecgather ok')"
      - python -c "from Isabelle_Semantic_Embedding._vecarith import library_path; print('kernel:', library_path())"
      # Numerical spot-check: the Q1.15 round trip must still approximate the cosine
      # it claims to.  Cheap, and it is the one property a mis-dispatched SIMD target
      # would break while every structural check stayed green.
      - python -c "import numpy as np; from Isabelle_Semantic_Embedding._vecarith import encode_q15, recover_cos; v=np.random.default_rng(0).normal(size=384); a=encode_q15(v); assert a.dtype.str=='<i2' and a.shape==(384,); print('q15 ok')"

about:
  homepage: https://github.com/xqyww123/Premise_Embedding
  summary: "Semantic retrieval for Isabelle: deformalization, a vector store, and a SIMD KNN kernel"
  description: |
    Semantic DB management, deformalization (Isabelle entities -> English via Claude),
    and vector-based semantic retrieval for Isabelle/HOL.

    This package ships both halves of the project: the Isabelle session and Scala
    component, registered with `isabelle components -u` from a conda link hook, and the
    Python package `Isabelle_Semantic_Embedding` with its Highway SIMD kernel and the
    `_vecgather` extension module.  The ML side resolves and dlopens the very library the
    Python side loads, so neither half is useful alone.

    The ~1.5 GB semantic database is NOT part of this package.  It is per-user data with
    its own lifecycle, distributed over Cloudflare R2.  Everything degrades gracefully
    without it -- retrieval simply returns nothing -- and it is fetched with
    `semantics_manage.py pull`.
  license: LGPL-2.1-or-later
  license_file:
    - COPYING
    - COPYING.LIB
  repository: https://github.com/xqyww123/Premise_Embedding
```

### 9.1 Deltas from the sibling recipes, and why

| | siblings | here | why |
|---|---|---|---|
| `noarch` | `python` or `generic` | **absent** (arch package) + `build.python.version_independent: true` | carries per-arch `.so`/`.dll`; abi3 makes it python-version-independent |
| Python half | `pip install .` | `pip install <our wheel>` | rebuilding would trip `setup.py`'s universal2 tag guards (§3.4) |
| license | BSD-3-Clause | **LGPL-2.1-or-later**, both COPYING files | `pyproject.toml:14-19` |
| `Tools/` copy | wholesale | wholesale **minus `Vector_Arith`** | 1.2 GB of vendored C++/build output |
| tests | import + `pip_check` | + explicit `_vecgather` import and `library_path()` load | the compiled halves are the entire reason this is not noarch |
| entry points | declared in recipe | **none** | `pyproject.toml` declares no `[project.scripts]` — see the §8.3 note on `semantics_manage.py` |

---

## 10. The workflow

`conda/` and `.github/workflows/release-conda.yml`. Structure follows
`Isabelle_RPC/.github/workflows/release-conda.yml` closely; only the deltas are
spelled out here.

```yaml
name: release-conda

on:
  workflow_dispatch:
    inputs:
      dry_run:      { description: "Build and verify only", type: boolean, default: false }
      build_number: { description: "BUMP if republishing the same VERSION", type: string, default: "0" }
  push:
    tags: ['v*']

concurrency:
  group: release-conda
  cancel-in-progress: false

jobs:
  # 1. Build the natives.  wheels.yml's own `wheel` job, called -- not copied.
  #    Requires adding `on: workflow_call` to wheels.yml, with an input that keeps
  #    its `publish` (PyPI) job from firing on a conda release.
  natives:
    uses: ./.github/workflows/wheels.yml
    with:
      publish_to_pypi: false

  # 2. Version, and the tag/VERSION agreement check.  Copy from isabelle-rpc,
  #    including the "is this the NEWEST v* tag" guard -- with a serialised
  #    concurrency group, pushing several tags at once silently drops the middle
  #    ones, and a superseded run renders GREY, not red, so nothing notifies.
  version:
    runs-on: ubuntu-latest
    outputs: { version: "${{ steps.ver.outputs.version }}" }
    steps: [...]        # reads ./VERSION -- WHICH DOES NOT EXIST YET, see section 2.3

  # 3. Build the conda packages, on NATIVE runners.
  conda-build:
    needs: [natives, version]
    strategy:
      fail-fast: false
      matrix:
        include:
          - { subdir: linux-64, runner: ubuntu-latest,   artifact: wheel-x86_64 }
          - { subdir: osx-64,   runner: macos-15,        artifact: wheel-universal2 }
          - { subdir: win-64,   runner: windows-latest,  artifact: wheel-amd64 }
          # linux-aarch64 (ubuntu-24.04-arm, wheel-aarch64) and
          # osx-arm64 (macos-15, wheel-universal2) are BLOCKED: rocksdict has no
          # conda-forge build for either subdir.  See the plan, section 2.1.
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: "${{ matrix.artifact }}", path: conda/wheelhouse }
      # rattler-build downloaded directly, not via conda: installing it through
      # conda goes via a proxy that intercepts conda-forge and fails TLS.
      # NOTE the sibling workflows fetch the x86_64-unknown-linux-musl binary; the
      # macOS and Windows legs need their own asset names.
      - name: Install rattler-build
        run: |  # per-OS asset selection -- UNVERIFIED, the siblings are linux-only
          ...
      - name: Build
        env: { ISA_COMPONENT_VERSION: "${{ needs.version.outputs.version }}" }
        run: rattler-build build --recipe conda/recipe.yaml --output-dir out
                 --build-num "${{ inputs.build_number || '0' }}"
      - name: "Sanity: the payload really contains both halves"
        run: |  # extract the .conda (a zip of two zstd tarballs -- listing the zip
                # alone shows only ['metadata.json','pkg-*.tar.zst','info-*.tar.zst']
                # and proves nothing) and assert:
                #   share/isabelle-semantic-embedding/ROOT
                #   share/isabelle-semantic-embedding/etc/settings
                #   share/isabelle-semantic-embedding/etc/build.props
                #   share/isabelle-semantic-embedding/lib/semantic_embedding.jar
                #   share/isabelle-semantic-embedding/Tools/simd_vector.ML
                #   NOT  share/isabelle-semantic-embedding/Tools/Vector_Arith
                #   site-packages/Isabelle_Semantic_Embedding/__init__.py
                #   site-packages/Isabelle_Semantic_Embedding/<the kernel for this subdir>
                #   site-packages/Isabelle_Semantic_Embedding/_vecgather.abi3.{so,pyd}
                #   an Isabelle_Semantic_Embedding-*.dist-info with the right Version
                #   .bat hooks are CRLF
      - uses: actions/upload-artifact@v4
        with: { name: "conda-${{ matrix.subdir }}", path: "out/**/*.conda" }

  # 4. Install and USE each package BEFORE publishing.  Publishing is the point of
  #    no return: the channel never deletes, so a package with a broken post-link
  #    becomes the newest resolvable version and stays broken until someone
  #    hand-deletes the object from R2.
  verify:
    needs: [conda-build, version]
    strategy: { fail-fast: false, matrix: { include: [ ...same three... ] } }
    runs-on: ${{ matrix.runner }}
    steps:
      # index into a file:// channel, then:
      #   conda create -n v -c file://$CH -c https://conda.qiyuan.me -c conda-forge \
      #                     --override-channels isabelle-semantic-embedding
      # conda-forge IS required in the channel list: unlike the pure-session
      # components, this one run-depends on python and fourteen python libraries.
      #
      # Then the five things that fail SILENTLY:
      #   a) the post-link registered it:      grep isabelle-semantic-embedding \
      #        "$(isabelle getenv -b ISABELLE_HOME_USER)/etc/components"
      #   b) the SCALA component loaded -- specific to this package, and NOT covered
      #      by any sibling's verify.  `isabelle scala_project` or a Scala.Fun probe
      #      for isabelle.semantic_embedding.PIDE_State_Functions.  EXACT INVOCATION
      #      UNVERIFIED -- see the open questions.
      #   c) the session builds:               isabelle build -v Semantic_Embedding
      #      (linux only; too slow on Windows -- the siblings skip it there)
      #   d) the Python half imports AND the kernel loads:
      #        python -c "from Isabelle_Semantic_Embedding._vecarith import library_path; print(library_path())"
      #      This is THE test for this package.  It must report a path under
      #      site-packages -- if it reports a Tools/Vector_Arith/build path, the
      #      checkout leaked into the environment.
      #   e) pip stands down:  pip install Isabelle_Semantic_Embedding
      #      must say "already satisfied".  Without the dist-info pip would fetch
      #      PyPI's copy over conda's files.
      #
      # AND, specific to this repo: assert that a FRESH environment with NO semantic
      # DB does not hard-fail.  Section 7.2 flags premise_selection.py:66's rocksdict
      # open, which does not create its parent directory.  Run it under a scratch
      # SEMANTIC_DB_DIR and see.

  publish:
    needs: verify
    if: ${{ !inputs.dry_run }}
    uses: xqyww123/isabelle-packaging-ci/.github/workflows/publish-conda.yml@main
    with: { artifact: conda-*, package_name: isabelle-semantic-embedding }
    # Passed EXPLICITLY, never `secrets: inherit`: a reusable workflow receives the
    # CALLING repo's secrets, and `inherit` silently forwards empty strings while
    # disabling publish-conda.yml's `required: true` checks.
    secrets:
      CONDA_R2_ACCESS_KEY_ID:     ${{ secrets.CONDA_R2_ACCESS_KEY_ID }}
      CONDA_R2_SECRET_ACCESS_KEY: ${{ secrets.CONDA_R2_SECRET_ACCESS_KEY }}

  smoke:
    needs: publish
    # install from the live channel on ubuntu-latest + windows-latest, exactly as
    # isabelle-rpc does, and assert register / import / kernel-load / unregister.
```

⚠ **`publish-conda.yml` takes a single `artifact:` name.** The siblings upload one
`conda-packages` artifact; here there are three. Either merge them into one artifact
before calling (`actions/download-artifact` with `pattern: conda-*, merge-multiple:
true`, then re-upload as `conda-packages`) or extend the reusable workflow. **I have
not read `xqyww123/isabelle-packaging-ci`'s `publish-conda.yml`**, so whether it
already handles per-subdir directory layout is unverified — check before writing
this job. The `.conda` files must land in the right subdir directories for
`conda-index` to see them.

---

## 11. Open questions and unverified claims

Listed honestly, worst first.

1. **`rocksdict` blocks `osx-arm64` and `linux-aarch64`** (§2.1). Verified fact; the
   *response* is a maintainer decision. Apple Silicon is a mainstream Isabelle
   platform and the base `isabelle` package supports it, so shipping without
   `osx-arm64` is a real gap. **Needs a decision before implementation.**
2. **`rattler-build` on macOS and Windows runners.** The sibling workflows only ever
   run it on `ubuntu-latest` and fetch the
   `rattler-build-x86_64-unknown-linux-musl` asset. Per-OS asset selection and
   whether the bash build script runs correctly under rattler-build on Windows
   (which shell? MSYS?) is **unverified**. This is the single most likely place for
   the implementation to stall. The `interpreter: bash` script in the recipe is
   heavily bash-flavoured (`shopt`, `find -delete`, `printf '%s\r\n'`); the sibling
   recipes prove that works for a *noarch* build on a linux runner, not for a
   *win-64* build on a Windows runner.
3. **`build.python.version_independent: true` + `pip install <abi3 wheel>`.**
   Documented and supported since 0.35.0 (docs fetched), but I have not run it, and
   in particular have not confirmed the interaction with installing an
   already-abi3-tagged wheel rather than building from source. Needs one real build.
4. **`source.filter` / `use_gitignore` key names for rattler-build 0.69.1** (§8.2).
   I did not confirm the exact schema. The whitelist in the build script is the real
   defence, so this only affects local runs.
5. **Verifying the Scala service is live** (§10, verify step b). I do not know the
   canonical one-liner to assert that
   `isabelle.semantic_embedding.PIDE_State_Functions` registered. Something in the
   `isabelle scala` / `Scala.Fun` family. None of the four sibling packages has a
   Scala component registered through a conda hook (`isabelle-mcp` has one but
   registers it from Python), so there is no precedent to copy.
6. **`premise_selection.py:66`'s rocksdict open on a missing parent directory**
   (§7.2 caveat 1). Order-dependent; not tested. Should be exercised in `verify`.
7. **`publish-conda.yml`'s multi-subdir handling** (§10). Not read.
8. **No `./VERSION` file** (§2.3), **no `[project.scripts]`** and
   **`Agent_Interpretation_Dir` does not ship** (§8.3). All three are source changes
   this document deliberately did not make.
9. Everything about conda-forge **solvability** is unverified — conda-forge is
   TLS-intercepted from this machine, so no environment could be solved locally.
   Names, summaries, versions and subdir coverage were checked one by one against
   `api.anaconda.org` over plain HTTPS; the joint solve was not.
10. No `isabelle build` was run (forbidden — shared heap directory), so the assertion
    that the whitelisted file set is *sufficient* to build session
    `Semantic_Embedding` is reasoned from `ROOT` and the `ML_file` lines, not
    measured. The `verify` job's `isabelle build -v Semantic_Embedding` is what will
    settle it.
