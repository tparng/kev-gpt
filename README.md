# Kevin on Kria

> *Why waste BRAM say lot word when few word do trick.*

A tiny language model that runs entirely inside the FPGA fabric of a Xilinx Kria
KV260 — weights baked into on-chip BRAM/URAM so they never touch DDR — trained on
telegraphic text so it talks like Kevin Malone. The joke is the thesis: a model
small enough to live on-chip is necessarily dumb, and on-chip residency is the
*only* thing that beats the board's Arm cores. **Being dumb and being fast are
the same property.**

This repo holds both the **design notes** (the argument) and the **code** (the
data tool and the model) for that project.

## The two facts it hangs on

1. **The bandwidth wall.** Single-stream decode is memory-bandwidth bound. On the
   KV260 the A53 cores and the PL fabric share one ~20 GB/s DDR controller, so a
   DDR-resident model gets no uplift from the fabric. The only escape is keeping
   the whole model (~1–2 MB, INT4) in on-chip memory at hundreds of GB/s to TB/s.
2. **The fusion.** Telegraphic output = fewer tokens = less weight streamed and a
   smaller KV cache = more fits on-chip = faster. The compression is the comedy
   *and* the optimisation.

## Design docs (read in order)

| doc | what |
|---|---|
| [`0-master.md`](0-master.md) | the through-line, the four-piece map, the build order |
| [`1-keviniser.md`](1-keviniser.md) | **the data tool** — POS-based telegraphic preprocessor |
| [`2-llm-on-kria.md`](2-llm-on-kria.md) | **the platform** — on-chip systolic GEMV, fabric-native softmax/RMSNorm |
| [`3-kevin-on-kria.md`](3-kevin-on-kria.md) | **the fusion** — train doc 2's model on doc 1's corpus |
| [`4-live-chatbot.md`](4-live-chatbot.md) | **the stress test** — serve it behind a Cloudflare Tunnel, link on HN |

## Code

| dir | what |
|---|---|
| [`keviniser/`](keviniser/) | the Keviniser preprocessor + corpus harness + TinyStories fetcher ([README](keviniser/README.md)) |
| [`model/`](model/) | a 2–4M-param nanoGPT-scale model, trainer, sampler, evolution renderer ([README](model/README.md)) |
| [`tests/`](tests/) | pytest suite for the Keviniser |

## What works today

- **Keviniser**: runs on TinyStories, ~70% word / ~61% token compression (the
  headline metric), POS-based so it keeps main-verb "do" and drops auxiliary
  "do". Parallel (`--nproc`) for the full corpus.
- **Proof-of-life model**: a 3.16M-param char-level GPT trained on the Kevinised
  validation set climbs from random characters to coherent telegraphic Kevin in
  ~35 min on an M1 (see `data/evolution.md` after a run).

Not yet built: INT4 QAT, goformer validation, the HLS/RTL fabric engine.

## Quickstart

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Kevinise the bundled sample
python -m keviniser.harness samples/canonical.txt
#  -> why us waste time say lot word when few word do trick

# Or the real thing: fetch TinyStories, Kevinise it, train a tiny model
python -m keviniser.fetch_tinystories                       # validation split (~20 MB)
python -m keviniser.harness data/TinyStories-valid.txt \
    -o data/TinyStories-valid.kevin.txt --marker "<|endoftext|>"
python -m model.train --max-iters 4000                      # ~37 min on M1
python -m model.sample data/ckpt.pt --prompt "once upon time"
```

Training the **full** corpus belongs on a CUDA GPU — see
[`model/SETUP-DELL.md`](model/SETUP-DELL.md).

## Conventions (project rules, honored in the docs)

- **Mischief in the title, dry in the body.** The output prose is deliberately
  bad; the speed numbers carry the story.
- **Honest-first.** Every doc states where the approach loses. The roofline
  crossover (where the fabric stops winning) is a result, not a caveat.
- **Bit-honest before fast.** Fabric output is validated against goformer to
  cosine > 0.9999 before any speed number is trusted.

## Hardware split

The Keviniser is CPU work (spaCy); training is GPU work (the 3050 Ti, where INT4
QAT also lives); inference is the FPGA fabric. They don't compete — different
machines for different stages.
