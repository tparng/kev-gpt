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

- **Keviniser**: POS-based so it keeps main-verb "do" and drops auxiliary "do".
  The **full TinyStories train split is processed**: 2,119,718 stories,
  371.7M → 260.5M words (**70.1%**), ~67% tokens (gpt2 proxy) — the headline
  compression metric, holding constant from the validation slice to the full
  corpus. ~2.9 h on the M1 with `--nproc 7`. Output is the ~1.3 GB Kevin training
  corpus the GPU run consumes.
- **Proof-of-life model**: a 3.16M-param char-level GPT trained on the Kevinised
  validation set climbs from random characters to coherent telegraphic Kevin in
  ~35 min on an M1, ~4 min on an RTX 3050 Ti (see `data/evolution.md` after a run).
- **Brevitas INT4 QAT path** (`model/qgpt.py`): the same architecture with every
  `nn.Linear` swapped for INT4-weight / INT8-activation `QuantLinear`. Warm-starts
  from an FP checkpoint via `--init-from`; inherits FP val loss within ~0.01 nats
  on the smoke test, which is the bit-honesty signal that the remap is faithful.

- **Fabric path (PL prep, no board needed)**: a roofline/crossover model (3.15M
  model = 1.5 MB INT4, fits the ~3 MB budget; crossover ~6.3M params); INT4
  weight export to `$readmemh` images + per-channel scales; a numpy integer
  golden ("goformer"); a **Stage 1 SystemVerilog systolic GEMV** verified
  bit-exact (`maxabserr=0`) against the golden on all 17 layers in iverilog and
  synthesised in Vivado for the KV260 part; a cosine validator showing the
  exported integer datapath reconstructs the Brevitas QAT forward to **cosine
  1.000** (gate > 0.9999); and a **Stage 0** A53 weight-streaming baseline (the
  bandwidth-wall proof).

Not yet built: Stage 2 (heterogeneous ping-pong tax) and Stage 3 (fabric-native
softmax/RMSNorm + the zero-DRAM headline); the per-lane banked, fully-pipelined
GEMV for peak throughput; on-board bring-up. All need the Kria connected.

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
python -m model.train --max-iters 4000                      # ~37 min M1, ~4 min RTX 3050 Ti
python -m model.sample data/ckpt.pt --prompt "once upon time"

# Optional: INT4 QAT fine-tune off the FP checkpoint (requires `pip install brevitas`)
python -m model.train --qat --init-from data/ckpt.pt --max-iters 2000 \
    --out data/ckpt.qat.pt
```

Training the **full** corpus belongs on a CUDA GPU — see
[`model/SETUP-DELL.md`](model/SETUP-DELL.md).

## Data & checkpoints (GitHub as Dropbox)

This project spans two machines — the **M1** runs the Keviniser (CPU/spaCy) and
the **XPS 15 / RTX 3050 Ti** does the training (CUDA). The handoff is a single
~1.3 GB corpus file, and the big artifacts (corpora, checkpoints) are gitignored
and *not* in the repo. So we abuse **GitHub Releases as an artifact store** — a
free Dropbox that lives next to the code:

- A direct `git commit` is the wrong tool: GitHub **hard-rejects files > 100 MB**,
  and a committed binary bloats clone history *forever*.
- A **Release asset** allows **up to 2 GB per file**, lives *outside* git history
  (zero repo bloat), and is one command each way.

**Current artifacts:**

| release | asset | what |
|---|---|---|
| [`corpus-v1`](https://github.com/michaelayles/kev-gpt/releases/tag/corpus-v1) | `TinyStories-train.kevin.txt.gz` (394 MB) | the full Kevinised train corpus, gzipped |

**Pull an artifact (e.g. on the Dell):**

```
gh release download corpus-v1 -R michaelayles/kev-gpt -D data/
# verify against the sha256 in the release notes, then gunzip
```

**Publish a new artifact (e.g. a trained checkpoint):**

```
gzip -k data/ckpt.pt
shasum -a 256 data/ckpt.pt.gz          # put the hash in the notes
gh release create ckpt-v1 data/ckpt.pt.gz --title "..." --notes "sha256: ..."
```

The repo is **private**, so release assets are private too — `gh` auth (or a
token) is required to download. Code and docs travel through normal git; only the
multi-hundred-MB blobs go through Releases.

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
