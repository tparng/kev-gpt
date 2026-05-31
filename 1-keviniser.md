# The Keviniser

A corpus preprocessor that strips function words to turn ordinary English into dense, telegraphic "Kevin-speak", the way Kevin Malone communicates when he decides that saying fewer words saves time.

This is the first of three docs. It is the data tool. The model that learns from its output lives in `2-llm-on-kria.md`, and the two come together in `3-kevin-on-kria.md`.

## Why this exists

Two reasons, and they happen to be the same reason wearing two hats.

The honest engineering reason. A model that fits entirely in on-chip BRAM/URAM is the only version of this project that beats the A53s by a meaningful margin (the bandwidth argument is in doc 2). Fewer tokens to express the same meaning is therefore not a gimmick, it is free headroom. Strip the function words and the same content costs fewer tokens, so a longer effective context and a bigger effective vocabulary fit on-chip, and decode throughput goes up for real. I can put a number on the compression and it feeds straight into the speed story.

The fun reason. A 2 to 4M parameter model is going to be dumb. Rather than pretend it writes good prose, I lean into the dumbness and give it a personality. A model trained on telegraphic text natively talks like Kevin. The limitation becomes the joke, and the joke turns out to be the thesis. Why waste BRAM say lot word when few word do trick.

So I train the model on Kevin-ised text. I do not strip at inference time, I strip the training corpus so generation comes out telegraphic on its own.

## Getting the texture right

The temptation is to grab an off the shelf stopword list and nuke everything in it. That over-strips. It drops "do" even when "do" is the main verb, drops every pronoun, and leaves inflection on the words it keeps, so you get clipped but still grammatically tidy text. That is not Kevin. Kevin keeps content words, keeps some subject pronouns, drops articles and the copula, and flattens inflection so verbs land in base form and nouns go singular.

Look at the canonical line. The full sentence is roughly "Why do we waste time saying a lot of words when a few words do the trick". The transform that produces the Kevin version is:

- drop the determiners (`a`, `the`)
- drop the auxiliaries (`do` as a helper, `is`/`are`/`will` and friends)
- drop the prepositions (`of`)
- flatten inflection (`saying` becomes `say`, `words` becomes `word`)
- keep the content words and the meaningful connectors (`why`, `when`, `few`, `lot`, `trick`)
- keep `do` when it is the actual verb

That last point is the one a stopword list gets wrong, and it is why I do this on part of speech tags rather than a flat word list. spaCy tags helper `do` as `AUX` and main verb `do` as `VERB`, so I can drop the first and keep the second for free.

## The implementation

Part of speech based, tunable, prints the compression ratio so the number is there from day one. Needs spaCy and the small English model:

```
pip install spacy
python -m spacy download en_core_web_sm
```

```python
import sys
import spacy

nlp = spacy.load("en_core_web_sm")

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


def keviniser(text, lemmatise=True, objectify=True, keep_punct=False):
    doc = nlp(text)
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

        out.append(word.lower())

    return " ".join(out)


def compression(before, after):
    wb = len(before.split())
    wa = len(after.split())
    return wb, wa, (wa / wb if wb else 0.0)


if __name__ == "__main__":
    raw = sys.stdin.read()
    kev = keviniser(raw)
    wb, wa, frac = compression(raw, kev)
    sys.stderr.write(f"words {wb} -> {wa}  ({frac:.0%} of original)\n")
    print(kev)
```

Usage:

```
cat tinystories.txt | python keviniser.py > tinystories.kevin.txt
```

The three knobs:

- `lemmatise` flattens inflection. On by default, it is most of what gives the telegraphic feel.
- `objectify` is full Kevin, subject pronouns become object pronouns. On by default because the whole point is to lean in. Turn it off if you want cleaner telegraphic text without the childish grammar.
- `keep_punct` keeps sentence breaks if you would rather the model still learn where utterances end. Off by default, which gives you the rawest possible word stream.

## The quick and dirty version

If you just want to eyeball the effect in thirty seconds with no model download, this is the crude one. It over-strips, no part of speech awareness and no lemmatisation, but it gets the idea across:

```python
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "am", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "that", "this", "these", "those", "with", "as",
    "by", "from", "it", "its",
}

def quick_keviniser(text):
    kept = [w for w in text.split() if w.strip(".,!?;:").lower() not in STOPWORDS]
    return " ".join(kept)
```

I would not train on this output. It drops main-verb `do`, drops every `it`, and leaves plurals and tenses intact, so it reads more like clipped notes than Kevin. Fine for a gut check, not for the corpus.

## The measurement

The compression ratio printed to stderr is the headline. On simple narrative text I expect to land somewhere around 55 to 70 percent of the original word count, more aggressive if `objectify` and `lemmatise` are both on. Two numbers worth logging properly when I run the real corpus:

- word count before and after, the raw compression
- a tokeniser count before and after, since that is what actually decides how much fits on-chip, and the two will not move by the same amount

This is the number that lets me say "few word do trick" and mean it, rather than just quoting Kevin. It also feeds doc 3, where the claim is that the compression is part of why the fabric version is fast.

## Dead ends and judgement calls

- Pronoun handling is genuinely a judgement call, because Kevin himself is inconsistent. "Why waste time" drops the subject entirely, but "when me president, they see" keeps two pronouns. I chose to keep pronouns and objectify them, which matches the second example and reads funnier. If I wanted the first example's flavour I would drop subject pronouns too. There is no clean rule, it is a taste dial.
- Lemmatising pronouns is why `objectify` works out so neatly. spaCy turns object pronouns back into subject lemmas (`us` becomes `we`), and the objectify map turns them straight back (`we` becomes `us`), so every pronoun ends up in object form whichever way it came in. That is a happy accident, not a plan, but it is consistent so I kept it.
- Casing is not preserved. Everything goes lowercase. For a telegraphic toddler register that is correct, Kevin does not capitalise carefully either, and it shrinks the vocabulary slightly which helps on-chip. If I later want sentence structure back I would revisit this.
- Over-stripping kills trainability. If the corpus gets too sparse the model has nothing to predict and learns noise. The quick version above is past that line. The part of speech version with default knobs is the sane starting point, and I will tune from there based on what the model actually produces.

## Next step

Run it on the candidate corpus (TinyStories is the lead, it is the only public dataset where a model this small stays coherent), log the real compression numbers, and hand the stripped corpus to the training setup in doc 2.
