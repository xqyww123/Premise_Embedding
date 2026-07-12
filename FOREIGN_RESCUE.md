# Rescuing Poly/ML's FFI structures (`Foreign`, `RunCall`) without patching Isabelle

## TL;DR

Isabelle2025 / Isabelle2025-2 hide `Foreign`, `RunCall`, `CInterface` and `Signal`
from the ML name space, so any ML file touching the FFI fails to compile with

```
*** Structure (Foreign) has not been declared
```

They are **not gone from the heap** — only the *name bindings* were removed. Ten
lines of plain ML, loaded as an ordinary `ML_file` before any FFI code, put them
back:

```ml
(* rescue_ffi.ML *)
local
  val isabelle_ns = ML_Env.make_name_space ML_Env.Isabelle;
  val bootstrap_thy = Thy_Info.get_theory "ML_Bootstrap";
  fun from_bootstrap name =
    Context.setmp_generic_context (SOME (Context.Theory bootstrap_thy))
      (fn () => #lookupStruct isabelle_ns name) ();
in
  val _ =
    List.app
      (fn name =>
        case from_bootstrap name of
          SOME v => #enterStruct ML_Name_Space.global (name, v)
        | NONE => ())
      ["Foreign", "RunCall", "Signal"];
end;
```

```isabelle
ML_file ‹rescue_ffi.ML›
ML_file ‹my_ffi_code.ML›   (* Foreign.loadLibrary … now compiles *)
```

No distribution file is modified, no Pure heap rebuild is needed, and it works
under `isabelle build` as well as in interactive/PIDE sessions. This makes the
`expose_foreign` patch of `my_better_isabelle_prover` unnecessary.

## The problem

`src/Pure/ML/ml_name_space.ML:67` (Isabelle2025 and Isabelle2025-2):

```ml
val hidden_structures = ["CInterface", "Foreign", "RunCall", "Signal"];
```

`src/Pure/ML_Bootstrap.thy` then removes them from the global Poly/ML name
space while Pure bootstraps. The removal is saved into the `Pure` heap, so every
downstream session inherits it.

The existing workaround
(`my_better_isabelle_prover/patches/Isabelle2025-2/expose_foreign/`) edits
`ml_name_space.ML` to shrink the list to `["Signal"]`. That means patching the
distribution and rebuilding the Pure heap — painful on a shared checkout and on
every Isabelle upgrade.

## Why a rescue is possible

`ML_Bootstrap.thy` does two things, **in this order**:

1. **It copies each hidden structure into its own Isabelle ML environment first**
   (lines 11–15 — the file header even says "ML bootstrap environment — with
   access to low-level structures!"):

   ```ml
   #allStruct ML_Name_Space.global () |> List.app (fn (name, _) =>
     if member (op =) ML_Name_Space.hidden_structures name then
       ML (Input.string ("structure " ^ name ^ " = " ^ name))
     else ());
   ```

   `ML_Env`'s name-space tables are `Generic_Data`, i.e. **theory data**. So the
   `structureVal` for `Foreign` ends up inside theory `ML_Bootstrap`, and is
   therefore baked into every session heap that contains that theory (every
   HOL-based image does).

2. **Only then does it forget them from the global name space** (lines 23–25):

   ```ml
   Context.setmp_generic_context NONE
     ML ‹ List.app ML_Name_Space.forget_structure ML_Name_Space.hidden_structures; … ›
   ```

   `PolyML.Compiler.forgetStructure` removes a *binding* from a name space. It
   does not destroy the value, and the copy from step 1 is not affected.

The rescue therefore just reads the value back out of `ML_Bootstrap`'s ML
environment and re-enters it into the global name space with `#enterStruct`.
Because Isabelle's ML compiler falls back to the global name space whenever
`ML_read_global` is true (the default; see `ml_env.ML:255-262`), every ML unit
compiled *afterwards* — Isabelle `ML`/`ML_file` and raw Poly/ML alike — resolves
`Foreign` normally.

## Evidence

All of the following was run against a **pristine Isabelle2025** (unpacked from
the official tarball, `hidden_structures` = all four names, stock prebuilt HOL
heap). Nothing was patched.

**1. Injected ML (`ML_Process` `use_prelude` path: raw Poly/ML `--use` against
the heap's global name space).**

```
PRISTINE/before: Foreign=no  RunCall=no  CInterface=no  Signal=no
PRISTINE/rescue Foreign: recovered from ML_Bootstrap env
PRISTINE/rescue RunCall: recovered from ML_Bootstrap env
PRISTINE/rescue CInterface: NOT FOUND
PRISTINE/rescue Signal: recovered from ML_Bootstrap env
PRISTINE/after : Foreign=YES RunCall=YES CInterface=no  Signal=YES
PRISTINE/check: second compilation unit resolved Foreign / RunCall / Signal -- OK
```

**2. Isabelle's own ML compiler** (`isabelle process -l HOL -f rescue.ML -f check.ML`)
— the second unit reports the real types, so the structure is genuinely usable,
not merely present as a name-space entry:

```
val probe_load_library = fn: string -> Foreign.library
val probe_symbol = fn: Foreign.library -> string -> Foreign.symbol
```

**3. A real `isabelle build`** of a session on top of HOL, whose theory contains

```isabelle
ML_file ‹rescue.ML›
ML ‹ val lib = Foreign.loadLibrary "libm.so.6";
      val sym = Foreign.getSymbol lib "cos"; … ›
```

* with the rescue: `Finished Test_FFI (0:00:01 elapsed time)`
* **negative control**, same theory with `ML_file ‹rescue.ML›` removed:

  ```
  *** ML error (line 8 of "…/Test_FFI.thy"):
  *** Structure (Foreign) has not been declared
  ```

which is exactly the failure the `expose_foreign` patch exists to prevent.

## Notes and limits

* **The rescue must be its own compilation unit.** Name resolution happens when
  a unit is compiled, so `ML_file ‹rescue_ffi.ML›` has to precede — not share a
  unit with — the code that mentions `Foreign`.

* **Per ML process.** The re-entry happens at runtime in the global name space of
  the running Poly/ML process. Load it once, early (e.g. the first `ML_file` of
  your session's first theory). Nothing is written back to any heap on disk.

* **`CInterface` cannot be rescued and does not need to be.** It no longer exists
  in the Poly/ML 5.9 basis at all, so the entry in `hidden_structures` is dead.
  (This is why it shows up as `NOT FOUND` above, and as absent even on a
  distribution with the `expose_foreign` patch applied.)

* **`Signal`** is rescued by the same recipe. Note the `expose_foreign` patch
  deliberately leaves `Signal` hidden, so the rescue is strictly more capable
  here.

* **Requires theory `ML_Bootstrap` in the session image.** True for any HOL-based
  heap (verified). A rescue against a bare `Pure` image should be checked before
  relying on it.

* **In injected ML, use `ML_Name_Space.*`, never `PolyML.*`.** The very same
  `Context.setmp_generic_context NONE` block in `ML_Bootstrap.thy` also *replaces*
  the global `PolyML` structure with a four-item stub (`pointerEq`, `IntInf`,
  `context`, `pretty`). `ML_Name_Space.global` and
  `ML_Name_Space.forget_structure` still work because `ml_name_space.ML` bound
  them *before* that replacement.

## Alternative injection point (when you cannot add an `ML_file`)

`Isabelle_Process.start(…, use_prelude = List("/path/to/rescue_ffi.ML"))` is a
public Scala API. `ML_Process` places `--use` files after the heap is loaded and
`Options`/`Resources` are initialised, but **before** `Isabelle_Process.init ()`
starts the PIDE protocol loop (`ml_process.scala:106-109`). That is the right
hook for a server that spawns its own prover and wants the FFI available without
touching the theory graph. Such a file is compiled by the **raw Poly/ML
compiler**, so it must be plain SML: no antiquotations, no cartouches.

Note that `isabelle build` hard-codes its own `use_prelude`
(`build_job.scala:255`) and offers no user hook — which is precisely why the
`ML_file` form above matters: it is the one that covers heap builds.
