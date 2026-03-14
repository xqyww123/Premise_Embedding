"""
Go-to-definition and hover message for Isabelle entities.

Each function first tries the live PIDE runtime (via ML→Scala callback),
then falls back to session export databases for compiled theories.
"""

import bisect
import os
import sqlite3
from functools import lru_cache
from typing import Optional

from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from Isabelle_RPC_Host.position import IsabellePosition


# ---------------------------------------------------------------------------
# RPC procedures (called from Isabelle/ML)
# ---------------------------------------------------------------------------

@isabelle_remote_procedure("pide_state.goto_definition_from_db")
def _goto_definition_from_db_rpc(arg: tuple, connection: Connection):
    """RPC wrapper: given (file_path, offset), return (def_file, def_line, def_offset, def_end_offset) or None."""
    file_path, offset = arg
    if isinstance(file_path, bytes):
        file_path = file_path.decode('utf-8')
    pos = IsabellePosition(0, offset, file_path)
    return _goto_definition_from_db(pos)


@isabelle_remote_procedure("pide_state.hover_message_from_db")
def _hover_message_from_db_rpc(arg: tuple, connection: Connection):
    """RPC wrapper: given (file_path, offset), return message string or None."""
    file_path, offset = arg
    if isinstance(file_path, bytes):
        file_path = file_path.decode('utf-8')
    pos = IsabellePosition(0, offset, file_path)
    return _hover_message_from_db(pos)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def goto_definition(
    pos: IsabellePosition,
    connection: Optional[Connection] = None,
) -> Optional[tuple[str, int, int, int]]:
    """Given a cursor position, return the definition position.

    Returns (def_file, def_line, def_offset, def_end_offset) or None.
    First tries the live PIDE runtime via ML callback.
    Falls back to session export databases for compiled theories.
    """
    if connection is not None:
        try:
            result = connection.callback(
                "pide_state.goto_definition", (pos.file, pos.raw_offset))
            if result is not None:
                def_file, def_line, def_offset, def_end_offset = result
                return (_resolve_path(def_file), def_line, def_offset, def_end_offset)
        except Exception:
            pass

    return _goto_definition_from_db(pos)


def hover_message(
    pos: IsabellePosition,
    connection: Optional[Connection] = None,
) -> Optional[str]:
    """Given a cursor position, return the hover tooltip text.

    First tries the live PIDE runtime via ML callback.
    Falls back to session export databases for compiled theories.
    """
    if connection is not None:
        try:
            result = connection.callback(
                "pide_state.hover_message", (pos.file, pos.raw_offset))
            if result is not None and result != "":
                return result
        except Exception as e:
            pass

    return _hover_message_from_db(pos)


# ---------------------------------------------------------------------------
# YXML parser
# ---------------------------------------------------------------------------

_X = '\x05'
_Y = '\x06'


def _parse_yxml_markup(text: str) -> list[dict]:
    """Parse YXML text and extract entity and typing markup with text offsets.

    Returns a list of dicts with keys:
      - elem: element name (e.g., 'entity', 'xml_elem')
      - props: dict of properties
      - start: text offset of element start (0-based)
      - end: text offset of element end
      - body_text: text content of the element

    Typing info is stored as ``<xml_elem xml_name=typing><xml_body>...text...</xml_body></xml_elem>``
    in the serialized Markup_Tree. We extract the text content from xml_body.
    """
    results = []
    text_offset = 0
    # stack entry: [elem_name, props, start_offset, body_parts]
    stack: list[list] = []
    i = 0
    n = len(text)

    while i < n:
        if i < n - 1 and text[i] == _X and text[i + 1] == _Y:
            i += 2
            if i >= n or text[i] == _X:
                # End of element: XYX
                if stack:
                    entry = stack.pop()
                    elem_name = entry[0]
                    props = entry[1]
                    start_offset = entry[2]
                    body_parts = entry[3]
                    end_offset = text_offset
                    if elem_name == 'entity' and 'ref' in props:
                        results.append({
                            'elem': 'entity',
                            'props': props,
                            'start': start_offset,
                            'end': end_offset,
                            'body_text': ''.join(body_parts),
                        })
                    elif elem_name == 'xml_elem':
                        xml_name = props.get('xml_name', '')
                        if xml_name in ('typing', 'sorting'):
                            results.append({
                                'elem': xml_name,
                                'props': props,
                                'start': start_offset,
                                'end': end_offset,
                                'body_text': ''.join(body_parts),
                            })
                    elif elem_name == 'xml_body':
                        # Pass body text up to parent xml_elem
                        if stack and stack[-1][0] == 'xml_elem':
                            stack[-1][3].extend(body_parts)
                if i < n and text[i] == _X:
                    i += 1
                continue

            # Start of element: XY <name> (Y <key>=<value>)* X
            attr_end = text.find(_X, i)
            if attr_end == -1:
                break
            attr_str = text[i:attr_end]
            parts = attr_str.split(_Y)
            elem_name = parts[0] if parts else ''
            props: dict[str, str] = {}
            for p in parts[1:]:
                eq = p.find('=')
                if eq >= 0:
                    props[p[:eq]] = p[eq + 1:]
            stack.append([elem_name, props, text_offset, []])
            i = attr_end + 1
        elif text[i] == _X:
            i += 1
        else:
            # Regular text character — collect into innermost xml_body or entity
            for entry in stack:
                if entry[0] in ('xml_body', 'entity'):
                    entry[3].append(text[i])
            text_offset += 1
            i += 1

    return results


# ---------------------------------------------------------------------------
# DB fallback: index building and lookup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _load_theory_markup(db_path: str, theory_name: str) -> tuple[
    list[tuple[int, int, dict]],   # entity refs sorted by start offset
    list[tuple[int, int, str]],    # typing/sorting sorted by start offset
]:
    """Load and parse markup for a theory, returning sorted lookup structures."""
    try:
        import zstandard
    except ImportError:
        return [], []

    conn = sqlite3.connect(db_path)
    try:
        # Read all markup chunks (PIDE/markup, PIDE/markup1, ...)
        cursor = conn.execute(
            "SELECT name, compressed, body FROM isabelle_exports "
            "WHERE theory_name = ? AND name LIKE 'PIDE/markup%' "
            "ORDER BY name",
            (theory_name,))
        dctx = zstandard.ZstdDecompressor()

        entity_refs: list[tuple[int, int, dict]] = []
        typing_info: list[tuple[int, int, str]] = []

        for name, compressed, body in cursor.fetchall():
            text = dctx.decompress(body).decode('utf-8') if compressed else body.decode('utf-8')
            markups = _parse_yxml_markup(text)

            for m in markups:
                if m['elem'] == 'entity' and 'ref' in m['props']:
                    entity_refs.append((m['start'], m['end'], m['props']))
                elif m['elem'] in ('typing', 'sorting'):
                    label = ':: ' if m['elem'] == 'typing' else ':: '
                    typing_info.append((m['start'], m['end'], label + m['body_text']))

        entity_refs.sort(key=lambda x: x[0])
        typing_info.sort(key=lambda x: x[0])
        return entity_refs, typing_info
    finally:
        conn.close()


def _find_entity_at_offset(
    entities: list[tuple[int, int, dict]], offset: int
) -> Optional[dict]:
    """Binary search for entity containing the given offset."""
    if not entities:
        return None
    starts = [e[0] for e in entities]
    idx = bisect.bisect_right(starts, offset) - 1
    if idx >= 0:
        start, end, props = entities[idx]
        if start <= offset < end:
            return props
    return None


def _find_info_at_offset(
    infos: list[tuple[int, int, str]], offset: int
) -> Optional[str]:
    """Binary search for typing/sorting info containing the given offset."""
    if not infos:
        return None
    starts = [e[0] for e in infos]
    idx = bisect.bisect_right(starts, offset) - 1
    if idx >= 0:
        start, end, text = infos[idx]
        if start <= offset < end:
            return text
    return None


# ---------------------------------------------------------------------------
# DB fallback: theory/file mapping
# ---------------------------------------------------------------------------

_file_to_theory_cache: dict[str, tuple[str, str]] = {}  # file_path -> (db_path, theory_name)


def _find_theory_for_file(file_path: str) -> Optional[tuple[str, str]]:
    """Find the session DB and theory name for a given file path.

    Returns (db_path, theory_name) or None.
    """
    normalized = _normalize_path(file_path)
    if normalized in _file_to_theory_cache:
        return _file_to_theory_cache[normalized]

    log_dirs = _find_log_dirs()
    try:
        import zstandard
        dctx = zstandard.ZstdDecompressor()
    except ImportError:
        return None

    for log_dir in log_dirs:
        for db_file in os.listdir(log_dir):
            if not db_file.endswith('.db'):
                continue
            db_path = os.path.join(log_dir, db_file)
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute(
                    "SELECT theory_name, compressed, body FROM isabelle_exports "
                    "WHERE name = 'PIDE/files'")
                for theory_name, compressed, body in cursor.fetchall():
                    text = (dctx.decompress(body).decode('utf-8')
                            if compressed else body.decode('utf-8'))
                    files = text.strip().split('\n')
                    for f in files:
                        f_normalized = _normalize_path(f.strip())
                        if f_normalized:
                            _file_to_theory_cache[f_normalized] = (db_path, theory_name)
                conn.close()
            except Exception:
                continue

    return _file_to_theory_cache.get(normalized)


@lru_cache(maxsize=1)
def _find_log_dirs() -> list[str]:
    """Find all Isabelle session log directories."""
    isabelle_home_user = os.environ.get(
        "ISABELLE_HOME_USER",
        os.path.expanduser("~/.isabelle"))
    # Try common Isabelle version directories
    dirs = []
    for entry in sorted(os.listdir(isabelle_home_user), reverse=True):
        heaps_dir = os.path.join(isabelle_home_user, entry, "heaps")
        if not os.path.isdir(heaps_dir):
            continue
        for ml_id in os.listdir(heaps_dir):
            log_dir = os.path.join(heaps_dir, ml_id, "log")
            if os.path.isdir(log_dir):
                dirs.append(log_dir)
    return dirs


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """Expand ~~ prefix to ISABELLE_HOME."""
    if path.startswith("~~"):
        isabelle_home = os.environ.get("ISABELLE_HOME", "")
        if isabelle_home:
            return isabelle_home + path[2:]
    return path


def _normalize_path(path: str) -> str:
    """Normalize an Isabelle file path for comparison."""
    path = _resolve_path(path.strip())
    if path:
        try:
            return os.path.realpath(path)
        except Exception:
            return path
    return path


# ---------------------------------------------------------------------------
# DB fallback: main functions
# ---------------------------------------------------------------------------

def _goto_definition_from_db(
    pos: IsabellePosition,
) -> Optional[tuple[str, int, int, int]]:
    """Extract go-to-definition from session export DB markup.

    Returns (def_file, def_line, def_offset, def_end_offset) or None.
    """
    result = _find_theory_for_file(pos.file)
    if result is None:
        return None
    db_path, theory_name = result

    entity_refs, _ = _load_theory_markup(db_path, theory_name)
    # Isabelle offsets are 1-based; YXML text offsets are 0-based
    props = _find_entity_at_offset(entity_refs, pos.raw_offset - 1)
    if props is None:
        return None

    def_file = props.get('def_file', '')
    if not def_file:
        return None

    def_line = int(props.get('def_line', '0'))
    def_offset = int(props.get('def_offset', '0'))
    def_end_offset = int(props.get('def_end_offset', '0'))

    return (_resolve_path(def_file), def_line, def_offset, def_end_offset)


def _hover_message_from_db(pos: IsabellePosition) -> Optional[str]:
    """Extract hover tooltip from session export DB markup."""
    result = _find_theory_for_file(pos.file)
    if result is None:
        return None
    db_path, theory_name = result

    entity_refs, typing_info = _load_theory_markup(db_path, theory_name)
    offset_0based = pos.raw_offset - 1

    tips: list[str] = []

    props = _find_entity_at_offset(entity_refs, offset_0based)
    if props is not None:
        kind = props.get('kind', '')
        name = props.get('name', '')
        if kind and name:
            kind_display = kind.replace('_', ' ')
            tips.append(f'{kind_display} "{name}"')
        elif kind:
            tips.append(kind.replace('_', ' '))

    typing = _find_info_at_offset(typing_info, offset_0based)
    if typing is not None:
        tips.append(typing)

    return '\n'.join(tips) if tips else None
