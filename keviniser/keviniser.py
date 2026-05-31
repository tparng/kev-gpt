"""The Keviniser: a part-of-speech corpus preprocessor.

Strips function words and flattens inflection to turn ordinary English into
dense, telegraphic "Kevin-speak". See `1-keviniser.md` for the why.

Design rule that earns the spaCy dependency: stripping is done on part-of-speech
tags, not a flat stopword list, so auxiliary `do` (AUX) is dropped while
main-verb `do` (VERB) is kept. A stopword list cannot make that distinction.

The model is meant to be trained on the *output* of this tool. We strip the
training corpus, never the inference input, so generation comes out telegraphic
on its own.
"""

from __future__ import annotations

from functools import lru_cache

# Function-word categories to strip entirely.
DROP_POS = {"DET", "ADP", "AUX", "PART", "CCONJ"}

# Quantifiers and negation we want to survive even if the tagger calls them
# function words. "few" and "lot" carry the meaning in "few word do trick".
KEEP_LEMMAS = {"few", "lot", "no", "not", "many", "more", "less"}

# Specific filler to drop even when it is tagged as content.
DROP_LEMMAS = {"that"}

# Optional Kevin-ism: subject pronouns collapse to their object form, which is
# also what the lemmatiser does to object pronouns, so everything lands the same
# way. "when me president, they see".
SUBJ_TO_OBJ = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}

# Free-form word substitutions, applied as the very last step to every token we
# keep (after lemmatise + objectify + lowercasing), keyed on the output word.
# Two uses:
#   1. Patch spaCy's wrong verb lemmas. en_core_web_sm lemmatises the verb
#      "shining"/"shines" to "shin" (it strips -ing but drops the silent e); one
#      entry keyed on the bad lemma fixes every inflection at once.
#   2. Impose your own Kevin vocabulary, e.g. {"television": "tv"}.
# Caveat: keyed on the word, so a substitution fires regardless of part of
# speech. "shin" -> "shine" will also rewrite the (rare) body-part noun. For a
# toddler-grade corpus that trade is fine; drop the entry if it bites you.
# Pass extra/override entries via the `subs` argument or the harness --subs flag;
# they are merged over these defaults.
SUBSTITUTIONS = {
    "shin": "shine",   # verb "shining"/"shines" mis-lemmatised by spaCy
}

_DEFAULT_MODEL = "en_core_web_sm"


@lru_cache(maxsize=4)
def load_nlp(model: str = _DEFAULT_MODEL):
    """Load and cache the spaCy pipeline.

    NER is disabled (we never use entities). The dependency parser is kept on
    purpose, even though it costs ~20% throughput: en_core_web_sm tags
    *main-verb* "do" as AUX without it (so "few words do the trick" loses its
    "do"), and only the parser recovers it as VERB/ROOT on real English. Keeping
    main-verb "do" is the project's signature distinction, so we pay for it.
    The parser is unreliable on already-telegraphic input, but we only ever feed
    this well-formed English, so that case does not arise.
    """
    import spacy

    try:
        return spacy.load(model, disable=["ner"])
    except OSError as exc:  # model not downloaded
        raise SystemExit(
            f"spaCy model {model!r} is not installed. Run:\n"
            f"    python -m spacy download {model}"
        ) from exc


def _kevinise_doc(doc, lemmatise: bool, objectify: bool, keep_punct: bool, subs: dict) -> str:
    out = []
    for tok in doc:
        if tok.is_space:
            continue
        if tok.is_punct:
            if keep_punct:
                out.append(tok.text)
            continue

        lemma = tok.lemma_.lower()
        if lemma in DROP_LEMMAS:
            continue
        if tok.pos_ in DROP_POS and lemma not in KEEP_LEMMAS:
            continue

        word = tok.lemma_ if lemmatise else tok.text
        if objectify and tok.pos_ == "PRON":
            word = SUBJ_TO_OBJ.get(word.lower(), word)

        word = word.lower()
        word = subs.get(word, word)  # final pass: lemma fixes + custom vocabulary
        out.append(word)

    return " ".join(out)


def _resolve_subs(subs: dict | None) -> dict:
    """Merge caller-supplied substitutions over the module defaults."""
    if not subs:
        return SUBSTITUTIONS
    return {**SUBSTITUTIONS, **subs}


def keviniser(
    text: str,
    lemmatise: bool = True,
    objectify: bool = True,
    keep_punct: bool = False,
    subs: dict | None = None,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Kevinise a single string.

    Knobs:
      lemmatise  flatten inflection (say<-saying, word<-words). Most of the
                 telegraphic feel comes from this. On by default.
      objectify  full Kevin: subject pronouns become object pronouns. On by
                 default. Turn off for cleaner telegraphic text without the
                 childish grammar.
      keep_punct keep sentence-break punctuation so the model can still learn
                 where utterances end. Off by default (rawest word stream).
      subs       extra word substitutions merged over SUBSTITUTIONS, applied as
                 the final step to each kept token (lemma fixes, custom vocab).
    """
    nlp = load_nlp(model)
    return _kevinise_doc(nlp(text), lemmatise, objectify, keep_punct, _resolve_subs(subs))


def kevinise_stream(
    texts,
    lemmatise: bool = True,
    objectify: bool = True,
    keep_punct: bool = False,
    subs: dict | None = None,
    model: str = _DEFAULT_MODEL,
    batch_size: int = 256,
    n_process: int = 1,
):
    """Kevinise an iterable of strings using nlp.pipe for throughput.

    Yields one Kevinised string per input string, in order. Use this for
    corpora: feeding spaCy one line at a time stays under the per-doc length
    limit and pipelining batches is markedly faster than calling keviniser()
    in a loop.

    n_process > 1 fans the spaCy pipeline across processes (the parser is the
    expensive part, so this scales well). Order is still preserved.
    """
    nlp = load_nlp(model)
    resolved = _resolve_subs(subs)
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        yield _kevinise_doc(doc, lemmatise, objectify, keep_punct, resolved)


def compression(before: str, after: str):
    """Word-count compression: (words_before, words_after, fraction_remaining)."""
    wb = len(before.split())
    wa = len(after.split())
    return wb, wa, (wa / wb if wb else 0.0)


if __name__ == "__main__":
    # stdin filter, matching the doc usage:
    #   cat tinystories.txt | python keviniser.py > tinystories.kevin.txt
    # For corpus runs with full stats, use harness.py instead.
    import sys

    raw = sys.stdin.read()
    kev = keviniser(raw)
    wb, wa, frac = compression(raw, kev)
    sys.stderr.write(f"words {wb} -> {wa}  ({frac:.0%} of original)\n")
    print(kev)
