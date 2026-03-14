# Go-to-Definition from Session Export Database

## Overview

Isabelle's Ctrl-Click "go to definition" navigation works in editors (jEdit, VSCode) via
PIDE markup generated during theory processing. However, for **already-compiled session
theories** (loaded from the heap), no live markup is generated, so Ctrl-Click does not work.

**Key finding**: The full PIDE markup tree — including all entity reference→definition
position mappings — is already persisted to the session export SQLite database during
standard `isabelle build`. No Isabelle source code modifications are needed to extract
this data.

This document describes the complete pipeline, the data format, and how to extract
go-to-definition information from existing build artifacts.

---

## 1. How Ctrl-Click Works in Isabelle Editors

### 1.1 ML-Side Markup Generation

During theory evaluation, the ML backend generates `Markup.ENTITY` elements for every
entity reference and definition. The key function is in `Pure/General/position.ML`:

```sml
fun make_entity_markup {def} serial kind (name, pos) =
  let val props =
    if def then (Markup.defN, Value.print_int serial) :: properties_of pos
    else (Markup.refN, Value.print_int serial) :: def_properties_of pos;
  in Markup.entity kind name |> Markup.properties props end;
```

- **Definition** (`def=true`): `def=<serial>` + the definition's own position properties
  (`file`, `offset`, `end_offset`, `line`, `id`)
- **Reference** (`def=false`): `ref=<serial>` + the definition's position as `def_*`
  properties (`def_file`, `def_offset`, `def_end_offset`, `def_line`, `def_id`)

### 1.2 Two Report Paths

Entity reports flow through two distinct paths in `Pure/context_position.ML`:

1. **`reports_generic`** — used by outer syntax (commands, keywords, theory imports):
   ```sml
   fun reports_generic context reps =
     if reports_enabled_generic context then Position.reports reps else ();
   ```

2. **`reports_text`** — used by inner syntax (terms, types, via `Syntax_Phases.ML`):
   ```sml
   fun reports_text ctxt reps =
     if reports_enabled ctxt then Position.reports_text reps else ();
   ```

Both are gated by `reports_enabled`, which checks:
```sml
fun reports_enabled0 () = Options.default_bool "pide_reports";  (* defaults to true *)
fun reports_enabled_generic context = reports_enabled0 () andalso is_visible_generic context;
```

The `pide_reports` option defaults to `true` (defined in `etc/options`), so reports are
generated during `isabelle build`.

### 1.3 Name Resolution

`Name_Space.check_reports` resolves short names to fully-qualified names and generates
reference markup:

```sml
fun check_reports context (Table (space, tab)) (xname, ps) =
  let val name = intern space xname in
    (case Change_Table.lookup tab name of
      SOME x =>
        let val reports =
          filter (Context_Position.is_reported_generic context) ps
          |> map (fn pos => (pos, markup space name));
        in ((name, reports), x) end
      ...)
  end;
```

This is why we need the markup from build time: short name → fully-qualified name
resolution requires the full evaluation context (imports, locale context, etc.).

### 1.4 PIDE Protocol Transport

In PIDE mode (`isabelle_process.ML`), `Output.report_fn` is set to send reports via
the protocol:

```sml
fun report_message ss =
  if Context_Position.reports_enabled0 ()
  then standard_message [] Markup.reportN ss else ();
```

The reports are accumulated into a `Markup_Tree` on the Scala side, forming the
`Document.Snapshot` that editors query.

### 1.5 Editor-Side Hyperlink Extraction

Both jEdit and VSCode extract hyperlinks using `snapshot.cumulate()`:

```scala
// VSCode (vscode_rendering.scala:285-315)
def hyperlinks(range: Text.Range): List[Line.Node_Range] = {
  snapshot.cumulate[List[Line.Node_Range]](
    range, Nil, VSCode_Rendering.hyperlink_elements, _ => {
      case (links, Text.Info(_, XML.Elem(Markup(Markup.ENTITY, props), _))) =>
        hyperlink_def_position(props).map(_ :: links)
      case (links, Text.Info(_, XML.Elem(Markup(Markup.POSITION, props), _))) =>
        hyperlink_position(props).map(_ :: links)
      case _ => None
    }) match { case Text.Info(_, links) :: _ => links.reverse case _ => Nil }
}
```

`hyperlink_def_position` reads `def_file`, `def_offset`, `def_end_offset`, `def_line`
from the ENTITY element's properties.

---

## 2. Session Export Database

### 2.1 Location

Session databases are stored at:
```
~/.isabelle/Isabelle2024/heaps/<ML_ID>/log/<session_name>.db
```

Where `<ML_ID>` is the Poly/ML identifier (e.g., `polyml-5.9.1_x86_64_32-linux`).

### 2.2 Schema

Standard SQLite database with table:
```sql
CREATE TABLE isabelle_exports (
  session_name TEXT NOT NULL,
  theory_name TEXT NOT NULL,
  name TEXT NOT NULL,
  executable BOOLEAN NOT NULL,
  compressed BOOLEAN NOT NULL,
  body BLOB NOT NULL
);
```

Bodies are typically **zstd-compressed** (`compressed=1`). Content is encoded as
**YXML** (see Section 3) or plain text, depending on the export type.

### 2.3 Complete List of Export Entries

Each theory in the database has some subset of these exports. The following table
is based on the HOL session (114 theories).

#### Universal exports (all 114 theories):

| Export name | Description | Body format |
|-------------|-------------|-------------|
| `PIDE/document_id` | Command execution ID (small integer) | Plain text |
| `PIDE/files` | Source file paths for the theory, one per line. First line is always the `.thy` file. Subsequent lines are `.ML` files loaded by the theory. | Plain text, newline-separated |
| `PIDE/markup` | **Full PIDE markup tree** — the complete semantic annotation of the source text, including all entity references, syntax highlighting, command spans, etc. For go-to-definition, this is the critical export. | Zstd-compressed YXML |
| `theory/parents` | Parent theory names (fully qualified) | Plain text, newline-separated |
| `theory/other_kinds` | List of "other" entity kinds present in this theory (e.g., `oracle`, `fact`, `bundle`, `attribute`, `method`) | Plain text, newline-separated |
| `document/latex` | LaTeX presentation output for the theory document | YXML with LaTeX content |

#### PIDE markup chunks:

| Export name | Count | Description |
|-------------|-------|-------------|
| `PIDE/markup1` through `PIDE/markup34` | 1–51 | For large theories, the markup tree is split across numbered chunks. `PIDE/markup` is the main theory file's markup; `PIDE/markupN` corresponds to the Nth auxiliary file (`.ML` files listed in `PIDE/files`). The maximum chunk count observed is 34 (for `HOL.SMT`). |

#### Theory entity definition exports (YXML, flat list of `<entity>` elements):

These exports record the **definitions** introduced by this theory (not inherited
from parents). Each `<entity>` element has properties: `name` (fully qualified),
`xname` (short name), `offset`, `end_offset`, `file`, `id`, `serial`, and
optionally `label`.

| Export name | Count | Description |
|-------------|-------|-------------|
| `theory/consts` | 100 | Constant definitions (functions, operators) |
| `theory/types` | 33 | Type constructor definitions |
| `theory/axioms` | 99 | Axiom declarations |
| `theory/thms` | 108 | Named theorem/lemma declarations |
| `theory/classes` | 32 | Type class definitions |
| `theory/locales` | 47 | Locale definitions |
| `theory/other/fact` | 109 | Named facts (overlaps with `theory/thms`) |
| `theory/other/method` | 19 | Proof method definitions |
| `theory/other/attribute` | 40 | Attribute definitions |
| `theory/other/bundle` | 10 | Bundle definitions |
| `theory/other/oracle` | 3 | Oracle declarations |

**Example** (`theory/consts` for `HOL.Nat`, showing the entity for `Suc_Rep`):
```
⟨XY⟩entity⟨Y⟩name=Nat.Suc_Rep⟨Y⟩xname=Suc_Rep⟨Y⟩offset=283⟨Y⟩end_offset=290
  ⟨Y⟩file=~~/src/HOL/Nat.thy⟨Y⟩id=44⟨Y⟩serial=654182⟨X⟩⟨XY⟩⟨X⟩
```

(where `⟨X⟩` = `\x05`, `⟨Y⟩` = `\x06`)

#### Messages:

| Export name | Count | Description |
|-------------|-------|-------------|
| `PIDE/messages` | 109 | Build-time output messages (writeln, warnings, errors) with source positions. Each message is a YXML element like `<writeln_message serial=... line=... offset=... file=...>text</writeln_message>`. |

#### Distinction: `PIDE/markup` vs `theory/*` exports

- **`PIDE/markup`** contains the **full markup tree** over the source text, including
  entity **references** (with `ref=<serial>` and `def_file`/`def_offset` properties),
  syntax highlighting (`keyword1`, `operator`, `literal`, etc.), command spans, typing
  information, language regions, and more. This is what editors consume for navigation,
  hover, and highlighting.

- **`theory/consts`**, **`theory/types`**, etc. contain only entity **definitions**
  introduced by this theory. They are flat lists used by `Document_Info.scala` for
  cross-session presentation (e.g., HTML browser info), not for editor navigation.

**For go-to-definition, `PIDE/markup` is the only export needed.**

### 2.4 How `PIDE/markup` is Exported

During `isabelle build`, `build_job.scala` (lines 288-319) exports the full markup tree:

```scala
session.finished_theories += Session.Consumer[Document.Snapshot]("finished_theories") {
  case snapshot =>
    if (!progress.stopped) {
      // ...
      export_(Export.MARKUP, snapshot.xml_markup())   // <-- ALL markup persisted here
      export_(Export.MESSAGES, snapshot.messages.map(_._1))
    }
}
```

### 2.5 Accessing from Python

```python
import sqlite3
import zstandard

db_path = "~/.isabelle/Isabelle2024/heaps/<ML_ID>/log/HOL.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
dctx = zstandard.ZstdDecompressor()

# List all theories
cursor.execute("SELECT DISTINCT theory_name FROM isabelle_exports WHERE name='PIDE/markup'")
theories = [row[0] for row in cursor.fetchall()]

# Read markup for a theory
cursor.execute(
    "SELECT compressed, body FROM isabelle_exports WHERE theory_name=? AND name='PIDE/markup'",
    (theory_name,))
compressed, body = cursor.fetchone()

if compressed:
    text = dctx.decompress(body).decode('utf-8')
else:
    text = body.decode('utf-8')

# Get source file path from PIDE/files
cursor.execute(
    "SELECT compressed, body FROM isabelle_exports WHERE theory_name=? AND name='PIDE/files'",
    (theory_name,))
compressed, body = cursor.fetchone()
files_text = dctx.decompress(body).decode('utf-8') if compressed else body.decode('utf-8')
thy_file = files_text.strip().split('\n')[0]  # first line = .thy file path
```

---

## 3. YXML Format

Isabelle uses YXML (a custom XML encoding) with two control characters instead of
angle brackets:

| Symbol | Byte | Name |
|--------|------|------|
| X      | `\x05` | Unit separator |
| Y      | `\x06` | Record separator |

### 3.1 Encoding Rules

- **Element start**: `XY` + element_name + (`Y` + key `=` value)* + `X`
- **Element end**: `XYX`
- **Text**: literal characters (no escaping needed since `\x05`/`\x06` don't appear in normal text)

Example YXML for `<entity kind="constant" name="Nat.Suc" ref="42" def_file="/path/Nat.thy" def_offset="123">Suc</entity>`:
```
\x05\x06entity\x06kind=constant\x06name=Nat.Suc\x06ref=42\x06def_file=/path/Nat.thy\x06def_offset=123\x05Suc\x05\x06\x05
```

### 3.2 Markup Tree Structure

The YXML encodes a `Markup_Tree` — nested XML elements where each element's
**text range** corresponds to the source text it annotates. The text content between
markup elements represents the source text, and character positions are tracked by
counting non-markup characters.

For entity references, the structure is:
```xml
<entity kind="constant" name="Nat.Suc" ref="42"
        def_file="/path/Nat.thy" def_offset="123" def_end_offset="126" def_line="15">
  Suc
  <xml_elem typing>   <!-- optional: hover/tooltip information -->
    <fixed name="Suc"/>
    ...type markup...
  </xml_elem>
</entity>
```

The reference position (where "Suc" appears in the source) is determined by the text
offset range surrounding the entity element's text content.

### 3.3 Entity Element Children: Hover Information

Entity elements may contain child elements providing **hover tooltip information**
(the popup shown when hovering over an identifier in the editor):

- `<xml_elem typing>` — wraps the type annotation display
- Contains sub-elements like `<fixed>`, `<free>`, `<tfree>`, `<type_name>` for
  syntax-highlighted type display

This is separate from the navigation data. For go-to-definition, only the element's
**properties** (`def_file`, `def_offset`, etc.) and the **text range** (reference
position) matter.

---

## 4. Entity Reference Properties

Each `Markup.ENTITY` reference element contains these properties:

| Property | Description | Coverage |
|----------|-------------|----------|
| `kind` | Entity kind: `constant`, `type_name`, `fact`, `command`, `method`, `class`, `attribute`, `locale`, `fixed`, `bound`, etc. | 100% |
| `name` | Fully-qualified entity name (e.g., `Nat.Suc`) | 100% |
| `ref` | Serial number linking to the corresponding definition | 100% (by definition — only refs have this) |
| `def_file` | Absolute path to the file containing the definition | ~99.8% |
| `def_line` | Line number of the definition | ~99.8% |
| `def_offset` | Character offset of the definition start | ~89% |
| `def_end_offset` | Character offset of the definition end | ~89% |

Some entity kinds (e.g., `command`, `method`, `attribute`) are defined in ML code
rather than theory files, so they may lack `def_offset` but still have `def_file`
and `def_line`.

### 4.1 Entity Kinds and Counts (HOL session)

From 114 theories in the HOL session, 882,533 entity references were extracted:

| Kind | Count | Notes |
|------|-------|-------|
| `type_name` | ~390K | Most frequent — types are referenced pervasively |
| `constant` | ~188K | Constants, functions, operators |
| `command` | ~95K | Isar commands (lemma, definition, etc.) |
| `fact` | ~47K | Named theorems and lemmas |
| `class` | ~40K | Type classes |
| `method` | ~30K | Proof methods (auto, simp, etc.) |
| `attribute` | ~25K | Attributes ([simp], [intro], etc.) |
| `fixed` | ~20K | Fixed variables in locale/context |
| `locale` | ~15K | Locale references |
| Others | ~32K | `sort`, `bound`, `axiom`, `type_syntax`, etc. |

---

## 5. Python Entity Reference Extractor

### 5.1 YXML Parser

```python
X = '\x05'
Y = '\x06'

def parse_yxml_entity_refs(text):
    """Parse YXML text and extract entity reference elements with their positions.

    Returns a list of dicts, each containing:
      - ref_offset: int — character offset of the reference start in source text
      - ref_end_offset: int — character offset of the reference end
      - kind: str — entity kind (constant, type_name, fact, etc.)
      - name: str — fully-qualified entity name
      - ref: str — serial number
      - def_file: str — path to definition file (if present)
      - def_line: str — line number of definition (if present)
      - def_offset: str — character offset of definition (if present)
      - def_end_offset: str — character offset of definition end (if present)
    """
    refs = []
    text_offset = 0
    stack = []
    i = 0
    n = len(text)

    while i < n:
        if i < n - 1 and text[i] == X and text[i + 1] == Y:
            # Start of markup: XY...X or end of element: XYX
            i += 2
            if i >= n or text[i] == X:
                # End of element: XYX
                if stack:
                    elem_name, props, start_offset = stack.pop()
                    end_offset = text_offset
                    if elem_name == 'entity' and 'ref' in props:
                        refs.append({
                            'ref_offset': start_offset,
                            'ref_end_offset': end_offset,
                            **props
                        })
                if i < n and text[i] == X:
                    i += 1
                continue

            # Start of element: XY <name> (Y <key>=<value>)* X
            attr_end = text.find(X, i)
            if attr_end == -1:
                break
            attr_str = text[i:attr_end]
            parts = attr_str.split(Y)
            elem_name = parts[0] if parts else ''
            props = {}
            for p in parts[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    props[k] = v
            stack.append((elem_name, props, text_offset))
            i = attr_end + 1
        elif text[i] == X:
            # Stray X (shouldn't happen in well-formed YXML)
            i += 1
        else:
            # Regular text character — advances the source text offset
            text_offset += 1
            i += 1

    return refs
```

### 5.2 Full Extraction Pipeline

```python
import sqlite3
import zstandard

def extract_entity_refs_from_session(db_path):
    """Extract all entity reference→definition pairs from a session database.

    Args:
        db_path: Path to the session SQLite database
                 (e.g., ~/.isabelle/Isabelle2024/heaps/<ML_ID>/log/HOL.db)

    Returns:
        Dict mapping theory_name → list of entity reference dicts
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    dctx = zstandard.ZstdDecompressor()

    # Get the source file path for each theory from PIDE/files export
    theory_files = {}
    cursor.execute(
        "SELECT theory_name, compressed, body FROM isabelle_exports WHERE name='PIDE/files'")
    for theory_name, compressed, body in cursor.fetchall():
        if compressed:
            text = dctx.decompress(body).decode('utf-8')
        else:
            text = body.decode('utf-8')
        # First line of PIDE/files is the .thy file path
        lines = text.strip().split('\n')
        if lines:
            theory_files[theory_name] = lines[0]

    # Extract entity references from PIDE/markup
    all_refs = {}
    cursor.execute(
        "SELECT theory_name, compressed, body FROM isabelle_exports WHERE name='PIDE/markup'")
    for theory_name, compressed, body in cursor.fetchall():
        if compressed:
            text = dctx.decompress(body).decode('utf-8')
        else:
            text = body.decode('utf-8')

        refs = parse_yxml_entity_refs(text)

        # Add ref_file from PIDE/files
        ref_file = theory_files.get(theory_name, '')
        for r in refs:
            r['ref_file'] = ref_file

        all_refs[theory_name] = refs

    conn.close()
    return all_refs
```

### 5.3 Building a Lookup Index

For efficient cursor-position lookup, build a sorted index:

```python
import bisect

def build_lookup_index(all_refs):
    """Build a sorted index for O(log n) cursor-position lookup.

    Returns:
        Dict mapping ref_file → sorted list of (ref_offset, ref_end_offset, def_info)
    """
    index = {}
    for theory_name, refs in all_refs.items():
        for r in refs:
            ref_file = r.get('ref_file', '')
            if not ref_file:
                continue
            entry = (
                int(r['ref_offset']),
                int(r['ref_end_offset']),
                {
                    'def_file': r.get('def_file', ''),
                    'def_offset': r.get('def_offset', ''),
                    'def_end_offset': r.get('def_end_offset', ''),
                    'def_line': r.get('def_line', ''),
                    'kind': r.get('kind', ''),
                    'name': r.get('name', ''),
                }
            )
            index.setdefault(ref_file, []).append(entry)

    # Sort each file's entries by offset
    for f in index:
        index[f].sort(key=lambda x: x[0])

    return index


def goto_definition(index, file_path, cursor_offset):
    """Look up the definition position for a cursor position.

    Args:
        index: The lookup index from build_lookup_index()
        file_path: Absolute path to the source file
        cursor_offset: Character offset of the cursor in the file

    Returns:
        Definition info dict, or None if no entity at cursor position
    """
    entries = index.get(file_path, [])
    if not entries:
        return None

    # Binary search for the entry containing the cursor offset
    offsets = [e[0] for e in entries]
    pos = bisect.bisect_right(offsets, cursor_offset) - 1

    if pos >= 0:
        start, end, def_info = entries[pos]
        if start <= cursor_offset < end:
            return def_info

    return None
```

---

## 6. Key Source Files

| File | Role |
|------|------|
| `src/Pure/General/position.ML:237-271` | `make_entity_markup` — generates ref/def properties |
| `src/Pure/General/name_space.ML:611-629` | `check_reports` — resolves names, generates reports |
| `src/Pure/context_position.ML:67-96` | `reports_enabled`, `reports_text`, `reports_generic` |
| `src/Pure/General/output.ML:103,153` | `report_fn` ref, `report` function |
| `src/Pure/System/isabelle_process.ML:120-128` | PIDE mode report handler |
| `src/Pure/Syntax/syntax_phases.ML:155-330` | Inner syntax (term) report accumulation |
| `src/Pure/Build/build_job.scala:288-319` | Exports `snapshot.xml_markup()` during build |
| `src/Pure/Build/build.scala:719-784` | `Build.read_theory()` — reconstructs snapshot |
| `src/Pure/Build/export.scala:20` | `Export.MARKUP = "PIDE/markup"` |
| `src/Pure/Build/export.ML` | ML-side export API |
| `src/Pure/Build/export_theory.ML:183-210` | Entity definition exports |
| `src/Pure/Build/export_theory.scala:180-213` | `Entity[A]` case class |
| `src/Pure/PIDE/document_info.scala:1-184` | Entity position data from exports |
| `src/Pure/PIDE/markup.scala:105-123` | `Markup.ENTITY`, `Entity.Ref/Def` |
| `src/Tools/jEdit/src/jedit_rendering.scala:244-286` | jEdit hyperlink extraction |
| `src/Tools/VSCode/src/vscode_rendering.scala:285-315` | VSCode hyperlink extraction |
| `src/Tools/jEdit/src/jedit_editor.scala:263-310` | `hyperlink_def_position` resolution |

---

## 7. Reconstruction via Scala (Alternative)

Instead of parsing YXML directly in Python, one can use the existing Scala API:

```scala
// Using Build.read_theory() to reconstruct a Document.Snapshot
using(Export.open_database_context(store)) { database_context =>
  val session_context = database_context.open_session(
    Sessions.background(options, session_name).base)

  for (theory_name <- session_context.theory_names()) {
    val theory_context = session_context.theory(theory_name)
    Build.read_theory(theory_context) match {
      case Some(snapshot) =>
        // Use snapshot.cumulate() exactly like the editors do
        snapshot.cumulate[List[Entity_Ref]](
          Text.Range.full, Nil,
          Markup.Elements(Markup.ENTITY),
          _ => {
            case (acc, Text.Info(range, XML.Elem(Markup(Markup.ENTITY, props), _))) =>
              props.collectFirst { case (Markup.REF, _) => () } match {
                case Some(_) =>
                  val ref = Entity_Ref(
                    ref_offset = range.start,
                    ref_end_offset = range.stop,
                    def_file = Properties.get(props, Markup.DEF_FILE),
                    def_offset = Properties.get_int(props, Markup.DEF_OFFSET),
                    def_end_offset = Properties.get_int(props, Markup.DEF_END_OFFSET),
                    def_line = Properties.get_int(props, Markup.DEF_LINE),
                    kind = Properties.get(props, Markup.KIND),
                    name = Properties.get(props, Markup.NAME))
                  Some(ref :: acc)
                case None => None
              }
          }).flatMap(_.info)
      case None => // skip
    }
  }
}
```

This approach leverages existing infrastructure but requires the Scala/JVM environment.

---

## 8. Limitations and Notes

1. **Offset semantics**: The `ref_offset` and `ref_end_offset` from markup are
   **character offsets** in the source text (1-based in Isabelle's convention, but the
   YXML text offset counter is 0-based). Verify against the actual source text.

2. **Cross-session references**: Entity references to definitions in parent sessions
   (e.g., HOL referencing Pure entities) include `def_file` paths pointing to the
   parent session's source files. The lookup index should cover the full session
   ancestry.

3. **ML-defined entities**: Some entities (commands, methods, attributes) are defined
   in ML source files. Their `def_file` points to `.ML` files, and they typically
   have `def_line` but may lack `def_offset`.

4. **Incremental updates**: The session database is rebuilt on each `isabelle build`.
   The extraction should be re-run after rebuilding sessions.

5. **Multiple databases and session ancestry**: See Section 9.

---

## 9. Session Database Ancestry

### 9.1 One Database Per Session

Each session database (`<session_name>.db`) contains exports **only for theories
built in that session**, not for theories inherited from parent sessions. For example:

| Database | Contains | Does NOT contain |
|----------|----------|------------------|
| `Pure.db` | `Pure`, `ML_Bootstrap`, `Pure.Sessions` | — |
| `HOL.db` | `HOL.HOL`, `HOL.Nat`, ..., `Complex_Main` (114 theories) | `Pure`, `ML_Bootstrap` |
| `HOL-Library.db` | `HOL-Library.AList`, ..., (145 theories) | Any theory from HOL or Pure |

This means: to get the full entity reference data for a session like HOL-Library,
you must load the databases for the **entire session ancestry chain**:
`Pure.db` + `HOL.db` + `HOL-Library.db`.

### 9.2 Heap Files Mirror the Database Chain

Each session database corresponds to a heap file in the same directory:

```
~/.isabelle/Isabelle2024/heaps/<ML_ID>/
├── Pure              # heap
├── HOL               # heap (includes Pure)
├── HOL-Library       # heap (includes Pure + HOL)
├── log/
│   ├── Pure.db       # exports for Pure theories only
│   ├── HOL.db        # exports for HOL theories only
│   └── HOL-Library.db  # exports for HOL-Library theories only
```

At runtime, Isabelle loads heaps cumulatively (each heap includes its ancestors),
but the export databases are **not cumulative** — each only stores its own session's
data.

### 9.3 Session Ancestry Resolution

Isabelle resolves the ancestry chain through the session structure defined in ROOT
files. The relevant code is in:

- **`Sessions.Info`** (`sessions.scala`): Each session has a `parent: Option[String]`
  and `imports: List[String]`. The `deps` method returns `parent.toList ::: imports`.

- **`sessions_structure.build_requirements()`** (`sessions.scala`): Computes the
  transitive closure of all ancestor sessions via `build_graph.all_preds_rev()`.
  For example, `build_requirements(["HOL-Library"])` returns `["Pure", "HOL", "HOL-Library"]`.

- **`Store.get_session()`** (`store.scala`): Resolves session names to heap/database
  file paths by searching `input_dirs` (user output dir, then system output dir).

### 9.4 Determining the Ancestry Chain Without Isabelle

If you don't want to invoke the Isabelle Scala infrastructure, you can determine the
ancestry from ROOT files. The `session` declaration in a ROOT file specifies the
parent:

```
session "HOL-Library" (main) in "Library" = HOL +
  description "..."
  sessions ...
  theories ...
```

Here `= HOL` means HOL is the parent session. Recursively, HOL's ROOT declares
`= Pure`, and Pure has no parent.

Alternatively, the `theory/parents` export in each database lists the parent
**theories** (not sessions), and you can trace from a theory to its session by
looking at the `session_name` column. For cross-session parent references
(e.g., `HOL.Archimedean_Field` lists parent `Main` which is in a different DB),
you know you need the parent session's database.

### 9.5 Python: Loading the Full Ancestry

```python
import sqlite3
import os
import zstandard

def load_session_ancestry(session_name, heaps_dir):
    """Load entity references from a session and all its ancestors.

    Args:
        session_name: Target session (e.g., "HOL-Library")
        heaps_dir: Path to ~/.isabelle/Isabelle2024/heaps/<ML_ID>

    Returns:
        Combined dict of theory_name → entity reference list
    """
    log_dir = os.path.join(heaps_dir, "log")

    # Determine ancestry by reading theory/parents and tracing cross-session refs
    # Simple approach: just load all available ancestor DBs
    # For a known chain like HOL-Library, this is: Pure, HOL, HOL-Library
    ancestry = resolve_session_ancestry(session_name, log_dir)

    all_refs = {}
    for ancestor in ancestry:
        db_path = os.path.join(log_dir, f"{ancestor}.db")
        if os.path.exists(db_path):
            refs = extract_entity_refs_from_session(db_path)
            all_refs.update(refs)

    return all_refs


def resolve_session_ancestry(session_name, log_dir):
    """Resolve the session ancestry chain from the databases themselves.

    Heuristic: a theory T in session S has parents in session S or an ancestor.
    If any parent theory is not found in S.db, we need the parent session.

    Returns list of session names in topological order (ancestors first).
    """
    known_sessions = set()
    queue = [session_name]
    order = []

    while queue:
        s = queue.pop(0)
        if s in known_sessions:
            continue
        known_sessions.add(s)
        order.append(s)

        db_path = os.path.join(log_dir, f"{s}.db")
        if not os.path.exists(db_path):
            continue

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        dctx = zstandard.ZstdDecompressor()

        # Get all theory names in this session
        cur.execute("SELECT DISTINCT theory_name FROM isabelle_exports")
        local_theories = {r[0] for r in cur.fetchall()}

        # Get all parent theory names
        cur.execute("SELECT theory_name, compressed, body FROM isabelle_exports WHERE name='theory/parents'")
        for theory_name, compressed, body in cur.fetchall():
            text = dctx.decompress(body).decode() if compressed else body.decode()
            for parent in text.strip().split('\n'):
                parent = parent.strip()
                if parent and parent not in local_theories:
                    # This parent is from an ancestor session — guess session name
                    # Convention: theory "HOL.Nat" belongs to session "HOL"
                    # But "Main" (no dot) could be session "Main" or a theory in the parent
                    # Best heuristic: try the part before the first dot
                    if '.' in parent:
                        parent_session = parent.split('.')[0]
                    else:
                        parent_session = parent
                    if parent_session not in known_sessions:
                        queue.append(parent_session)

        conn.close()

    # Reverse to get ancestors first
    order.reverse()
    return order
```

Note: The heuristic of deriving session names from theory qualifiers (`HOL.Nat` →
session `HOL`) works for standard Isabelle sessions but may need adjustment for
sessions with non-standard naming conventions. For robust resolution, parse the
ROOT files or use Isabelle's `Sessions.structure()` Scala API.
