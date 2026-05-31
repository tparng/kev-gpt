# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **design-document repository**, not a software project. It holds the planning notes for "Kevin on Kria" — a project to run a tiny telegraphic language model entirely inside the FPGA fabric of a Xilinx Kria KV260, with weights baked into on-chip BRAM/URAM so they never touch DDR. There is no build system, no test suite, and (so far) no committed source tree. The only code in the repo is a Python reference implementation embedded inside `1-keviniser.md`; it is not extracted into a runnable file.

When asked to "work on the code," clarify scope: most tasks here are editing prose, keeping the cross-document argument consistent, or pulling the embedded Python out into a real file. Do not invent commands, directories, or files that the docs only describe as future work.

## Document structure and reading order

The five `.md` files form one argument. Read in numeric order; `0-master.md` is the through-line and the four numbered docs hold the detail.

- `0-master.md` — the single entry point. The whole thesis, the four-piece map, the staged build order (stages 0–4), and the "what is actually true" honesty section.
- `1-keviniser.md` — **the data tool.** A spaCy part-of-speech preprocessor that strips function words and flattens inflection to compress English into telegraphic "Kevin-speak." Contains the canonical Python implementation.
- `2-llm-on-kria.md` — **the platform.** The hardware build plan (HLS/RTL systolic GEMV, on-chip weights, fabric-native softmax/RMSNorm). Stands alone, independent of the Kevin angle.
- `3-kevin-on-kria.md` — **the fusion.** Joins docs 1 and 2: train the doc-2 model on a doc-1 corpus. The framing doc.
- `4-live-chatbot.md` — **the public stress test.** Web front end on the board, served behind a Cloudflare Tunnel, linked on Hacker News to measure concurrency.

## The core thesis (keep edits consistent with it)

Two facts the entire project hangs on. Any edit must not contradict them:

1. **The bandwidth wall.** Single-stream autoregressive decode is memory-bandwidth bound, not compute bound. On the KV260 the A53 cores and the PL fabric share the same ~20 GB/s DDR controller, so a DDR-resident model gets no speedup from the fabric. The *only* source of uplift is keeping the whole model (≈1–2 MB, INT4, a few million params) on-chip in BRAM/URAM, where bandwidth is hundreds of GB/s to TB/s. The ~3 MB on-chip budget is a hard ceiling.
2. **The fusion (the joke is the thesis).** Telegraphic "few word do trick" output means fewer tokens → less weight streamed and smaller KV cache → more fits on-chip → faster. The model's dumbness and its speed are the same property. This is the load-bearing idea — the comedy and the optimisation must always be presented as one thing, not a bolt-on gag.

Corollary the docs repeat: the Keviniser strips the **training corpus, not the inference input**. The model learns the compressed distribution and generates telegraphic text on its own.

## The Keviniser implementation (doc 1)

The reference code uses spaCy with the small English model:

```
pip install spacy
python -m spacy download en_core_web_sm
```

Intended usage pattern (the file `keviniser.py` does not yet exist — it lives only in the doc):

```
cat tinystories.txt | python keviniser.py > tinystories.kevin.txt
```

Key design decisions to preserve if you touch this code:
- It strips on **part-of-speech tags, not a flat stopword list** — specifically so it drops auxiliary `do` (`AUX`) but keeps main-verb `do` (`VERB`). A stopword list gets this wrong; that is the whole reason for spaCy.
- Three knobs: `lemmatise` (flatten inflection), `objectify` (subject→object pronouns, full Kevin), `keep_punct`. Defaults are `lemmatise=True, objectify=True, keep_punct=False`.
- `KEEP_LEMMAS` rescues quantifiers/negation (`few`, `lot`, `not`…) the tagger would otherwise strip. `DROP_LEMMAS` removes filler (`that`) even when tagged as content.
- It prints the compression ratio to **stderr** — that number is the headline metric and feeds the speed story in docs 3/4. Don't silence it.
- TinyStories is the lead corpus (the public dataset where a model this small stays coherent).

## Writing conventions (these are project rules, not style preferences)

These are stated explicitly across the docs and should be honored when editing prose:

- **Mischief in the title, dry in the body.** Playful framing is fine in headlines; the body is measurement, tradeoffs, and the roofline crossover. Never pretend the prose output is good — the deliberate dumbness is the point and the speed numbers carry the post alone.
- **Honest-first.** Every doc states where the approach *loses* (DDR-resident models, long context once KV spills, high dev effort) out loud. The "crossover" — the model size where the fabric stops winning — is treated as a result, not a caveat. Keep this discipline; don't let edits drift toward overclaiming.
- **Bit-honest before fast.** Fabric output is validated against the `goformer` golden reference to cosine > 0.9999 before any speed number is trusted.
- Numbers are given as honest ranges (e.g. "55–70% of original tokens," "tens of x, not 100x"). Preserve the hedging; don't sharpen estimates into false precision.

## The build order (referenced across all docs)

Stages 0–4, each ships something demonstrable: (0) A53 baseline proving it's bandwidth bound, (1) raw on-chip matmul throughput, (2) the measured heterogeneous ping-pong tax, (3) full zero-DRAM uplift + crossover plot, (4) live chatbot under load with two ceilings (inference vs. serving). The data-side track runs alongside: Keviniser on TinyStories → train (Brevitas INT4 QAT) → validate vs. goformer → wire into stage 3.
