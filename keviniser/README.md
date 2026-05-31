# Keviniser

Telegraphic corpus preprocessor for the Kevin-on-Kria project. Strips function
words and flattens inflection to compress ordinary English into dense
"Kevin-speak". The design rationale is in [`../1-keviniser.md`](../1-keviniser.md);
this is the runnable implementation of it.

We strip the **training corpus**, never the inference input — the model learns
the compressed distribution and generates telegraphic text on its own.

## Setup

Platform-independent (developed on M1 mac, runs the same on the Windows/Vivado
box later). From the repo root:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run it

```
# Corpus run with logged stats (the headline numbers):
python -m keviniser.harness samples/canonical.txt -o out.kevin.txt --stats stats.json

# As a stdin/stdout filter (matches the doc usage):
cat corpus.txt | python -m keviniser.harness > corpus.kevin.txt
```

The harness logs both compression numbers doc 1 asks for: **word count** and
**token count** (the token count uses a GPT-2 BPE proxy via `tiktoken`, since the
real model's tokeniser does not exist yet — it is for relative comparison, not
the on-chip byte budget). On the bundled sample: words → 57.6%, tokens → 52.8%
of original, squarely in the doc's predicted 55–70% range.

### Knobs (all default to "full Kevin")

| flag | effect | default |
|---|---|---|
| `--no-lemmatise` | keep original inflection instead of flattening it | lemmatise on |
| `--no-objectify` | leave subject pronouns alone instead of `we`→`us` | objectify on |
| `--keep-punct` | keep sentence-break punctuation | off |
| `--subs PATH` | JSON map of extra word substitutions, merged over the defaults | — |

## Word substitutions

Two kinds of word-swapping happen, both editable:

- **Pronoun objectifying** (`SUBJ_TO_OBJ` in `keviniser.py`): `i`→`me`, `we`→`us`,
  etc. This, plus dropping the auxiliary, is what turns *"when i am president"*
  into *"when me president"*. Controlled by the `objectify` knob.
- **Free-form substitutions** (`SUBSTITUTIONS`): a `{word: replacement}` map
  applied as the final step to every kept token. Use it to impose your own
  vocabulary (`{"television": "tv"}`) or to patch spaCy's wrong verb lemmas.

The shipped default fixes one such lemma: spaCy lemmatises the verb *shining* to
*shin* (it strips `-ing` but drops the silent `e`), so `SUBSTITUTIONS` maps
`"shin" → "shine"`. That's why *"the sun was shining"* now Kevinises to
*"sun shine"* and not *"sun shin"*. Other verbs hit the same bug (`hoping`→`hop`,
`riding`→`rid`); add them to the map if your corpus needs them — they're left out
by default because `hop` and `rid` are also valid verbs, so the fix is a
judgement call.

Add or override entries without touching code via a JSON file:

```
echo '{"television":"tv","hop":"hope"}' > subs.json
python -m keviniser.harness corpus.txt --subs subs.json -o out.kevin.txt
```

Caveat: substitutions are keyed on the word, not its part of speech, so
`"shin"→"shine"` also rewrites the rare body-part noun. Map a word back to itself
(`{"shin":"shin"}`) to disable a default.

## Library use

```python
from keviniser import keviniser, kevinise_stream, compression

keviniser("Why do we waste time saying a lot of words when a few words do the trick")
# -> "why us waste time say lot word when few word do trick"
```

Use `kevinise_stream(iterable_of_lines)` for corpora — it pipelines batches
through `nlp.pipe` and is much faster than calling `keviniser()` in a loop.

## Test

```
python -m pytest tests/
```

The transform tests skip (rather than fail) if the spaCy model isn't installed,
so the pure `compression()` tests still run on a bare checkout.

## Implementation note: the "do" distinction

Doc 1's signature example keeps **main-verb** "do" (in "few word do trick") while
dropping **auxiliary** "do" (in "why do we waste time"). In practice
`en_core_web_sm` tags *both* as `AUX` unless the **dependency parser** is enabled
— so we keep the parser on (NER off), at ~20% throughput cost, because that
distinction is the whole reason this is POS-based rather than a stopword list.
The parser only recovers main-verb "do" on well-formed English (it misreads
already-telegraphic input), which is fine: we only ever feed it real English.
