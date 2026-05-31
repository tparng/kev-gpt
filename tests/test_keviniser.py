"""Tests for the Keviniser.

The compression() tests are pure and always run. The transform tests need the
spaCy model and are skipped (not failed) when it is not installed, so the suite
is still useful on a fresh checkout before `spacy download`.

Transform assertions are written as properties, not exact-string matches,
because exact output depends on the spaCy version's tagging. We assert the
behaviours the doc actually promises.
"""

import pytest

from keviniser.keviniser import compression, keviniser

CANONICAL = "Why do we waste time saying a lot of words when a few words do the trick"


def _model_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(
    not _model_available(),
    reason="en_core_web_sm not installed (run: python -m spacy download en_core_web_sm)",
)


# --- pure, no spaCy ---------------------------------------------------------

def test_compression_basic():
    assert compression("a b c d", "b d") == (4, 2, 0.5)


def test_compression_empty_input():
    assert compression("", "") == (0, 0, 0.0)


def test_compression_reports_fraction_remaining():
    wb, wa, frac = compression("one two three four five", "two four")
    assert (wb, wa) == (5, 2)
    assert frac == pytest.approx(0.4)


# --- transform behaviour ----------------------------------------------------

@needs_model
def test_drops_determiners_and_prepositions():
    out = keviniser(CANONICAL).split()
    for fn in ("a", "the", "of"):
        assert fn not in out


@needs_model
def test_keeps_main_verb_do_drops_auxiliary_do():
    # Two `do` in the canonical line: the first is auxiliary (drop), the second
    # is the main verb (keep). Net: exactly one `do` survives. This is the whole
    # reason for POS tags over a stopword list.
    out = keviniser(CANONICAL).split()
    assert out.count("do") == 1


@needs_model
def test_keeps_quantifiers_few_and_lot():
    out = keviniser(CANONICAL).split()
    assert "few" in out
    assert "lot" in out


@needs_model
def test_lemmatise_flattens_inflection():
    out = keviniser(CANONICAL)
    assert "say" in out.split()      # saying -> say
    assert "word" in out.split()     # words -> word
    assert "saying" not in out.split()
    assert "words" not in out.split()


@needs_model
def test_objectify_turns_subject_pronoun_to_object():
    # "we" should land as "us" with objectify on (the default).
    out = keviniser(CANONICAL).split()
    assert "us" in out
    assert "we" not in out


@needs_model
def test_no_objectify_keeps_subject_form():
    out = keviniser("we play in the park", objectify=False).split()
    assert "we" in out
    assert "us" not in out


@needs_model
def test_output_is_lowercase():
    out = keviniser("The Little Dog Ran")
    assert out == out.lower()


@needs_model
def test_keep_punct_preserves_sentence_break():
    with_punct = keviniser("the dog ran. the cat slept.", keep_punct=True)
    without = keviniser("the dog ran. the cat slept.", keep_punct=False)
    assert "." in with_punct
    assert "." not in without


@needs_model
def test_no_lemmatise_preserves_surface_form():
    out = keviniser("the dogs were running", lemmatise=False).split()
    # function words still gone, but inflection kept on content words
    assert "dogs" in out
    assert "running" in out


# --- substitutions ----------------------------------------------------------

@needs_model
def test_default_subs_fix_shining_lemma():
    # spaCy lemmatises the verb "shining" to "shin"; SUBSTITUTIONS repairs it.
    out = keviniser("the sun was shining").split()
    assert "shine" in out
    assert "shin" not in out


@needs_model
def test_when_i_am_president_becomes_when_me_president():
    # objectify (i -> me) plus dropping the auxiliary "am".
    assert keviniser("when i am president") == "when me president"


@needs_model
def test_custom_subs_override_defaults():
    out = keviniser("watch the television", subs={"television": "tv"}).split()
    assert "tv" in out
    assert "television" not in out


@needs_model
def test_custom_subs_can_disable_a_default():
    # mapping shin->shin neutralises the built-in shin->shine fix
    out = keviniser("the sun was shining", subs={"shin": "shin"}).split()
    assert "shin" in out
    assert "shine" not in out
