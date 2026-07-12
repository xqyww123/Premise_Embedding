"""Unit tests for the single "record -> embedding document text" authority.

These pin the invariant the EMBED_TEXT_LAYERING_REFACTOR exists to establish:

    the embedding document text is a PURE FUNCTION OF THE STORED RECORD

so the text write_memory embeds and the text a later re-embed (auto/offline)
reconstructs are byte-identical BY CONSTRUCTION -- the regression that must never
come back (the EXPERIENCE and entity conventions previously drifted apart).

Pure Python: no Isabelle session, no LMDB, no network.  (pretty_unicode does need the
Isabelle symbol table, i.e. ISABELLE_HOME or `isabelle` on PATH.)
"""

import json

import pytest

from Isabelle_RPC_Host.universal_key import EntityKind
from Isabelle_RPC_Host.unicode import pretty_unicode
from Isabelle_Semantic_Embedding.semantics import _Semantic_DB, SemanticRecord
from Isabelle_Semantic_Embedding.document_text import (
    document_text_of, entity_document_text, experience_document_text)


def _entity(name: str = "foo.bar", expr: str = "a = b",
            interp: 'str | None' = "the fact that a equals b"):
    return SemanticRecord(EntityKind.THEOREM, name, expr, interp)


def _experience(name: str = "exp1", pats: 'list[str] | None' = None,
                desc: 'str | None' = "When the goal is over a finite set"):
    pats = ["\\<forall>x. P x \\<longrightarrow> Q x", "finite S"] if pats is None else pats
    return SemanticRecord(EntityKind.EXPERIENCE, name, json.dumps(pats), desc,
                          None, [("Some_Theory", b"h" * 16)], "how to prove it")


# --- the two conventions -----------------------------------------------------

def test_entity_uses_the_entity_convention():
    rec = _entity()
    assert document_text_of(rec) == rec.pretty_print + "\n" + rec.interpretation
    assert entity_document_text(rec) == document_text_of(rec)


def test_experience_uses_the_framing_convention_with_unicode_patterns():
    pats = ["\\<forall>x. P x \\<longrightarrow> Q x", "finite S"]
    rec = _experience(pats=pats)
    # Patterns are STORED as ASCII (Isabelle's inner lexer needs it) but EMBEDDED as
    # the unicode "semantic form", reconstructed from rec.expr -- not from a separate
    # transient list that a re-embed could not see.
    assert document_text_of(rec) == experience_document_text(
        [pretty_unicode(p) for p in pats], rec.interpretation)
    assert "∀x. P x ⟶ Q x" in document_text_of(rec)


def test_the_two_conventions_differ():
    """The whole point: an experience must NOT be embedded with the entity template."""
    rec = _experience()
    assert document_text_of(rec) != entity_document_text(rec)


# --- the core invariant: pure function of the STORED record ------------------

@pytest.mark.parametrize("rec", [_entity(), _experience()])
def test_text_survives_a_store_roundtrip(rec):
    """Write path and re-embed path must agree byte-for-byte.

    write_memory computes the text from the in-memory record it is about to store;
    _auto_embed / the offline embed compute it from the record read back out of LMDB.
    Encoding and decoding the record must therefore not change the text at all.
    """
    decoded = _Semantic_DB._decode(_Semantic_DB._encode(rec))
    assert document_text_of(decoded) == document_text_of(rec)


# --- not-embeddable records --------------------------------------------------

def test_no_interpretation_is_not_embeddable():
    assert document_text_of(_entity(interp=None)) is None
    assert document_text_of(_experience(desc=None)) is None


def test_corrupt_experience_expr_is_skipped_not_raised():
    """A malformed/legacy expr must not abort a whole embed batch (it is one record
    among thousands in the offline embed / migration)."""
    rec = SemanticRecord(EntityKind.EXPERIENCE, "bad", "not json{", "desc")
    assert document_text_of(rec) is None
