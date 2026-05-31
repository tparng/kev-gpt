# Kevin GPT — training

A tiny nanoGPT-lineage transformer trained on the Kevinised corpus. Deliberately
2–4M params with short context so it can later be INT4-quantised and baked into
the KV260's on-chip memory (see [`../2-llm-on-kria.md`](../2-llm-on-kria.md)).
This is the FP proof-of-life; INT4 QAT (Brevitas) and goformer validation come
after.

## Pipeline

```
data/TinyStories-valid.kevin.txt        # Kevinised corpus (from the harness)
        │  model.data  (char-level tokenise, <|endoftext|> -> story boundary)
        ▼
data/char/{train,val}.bin + meta.json
        │  model.train  (device auto: CUDA > MPS > CPU, mixed precision)
        ▼
data/ckpt.pt            best-val checkpoint  (automatic anti-overfit rollback)
data/ckpts/ckpt_*.pt    per-eval snapshots   (manual rollback by taste)
data/states.jsonl       per-eval losses + fixed-seed sample
        │  model.evolution
        ▼
data/evolution.md       the garbage -> text progression, readable
```

## Commands

```
# 1. measure throughput on this machine, get a time estimate, then stop
python -m model.train --smoke 50

# 2. train (defaults: 3.16M params, 4 layers, d=256, ctx=256, 4000 iters)
python -m model.train --max-iters 4000 --eval-interval 500

# 3. watch it learn
python -m model.evolution data/states.jsonl -o data/evolution.md

# 4. audition any checkpoint (pick a rollback point by taste, not just val loss)
python -m model.sample data/ckpts/ckpt_02000.pt -n 300
python -m model.sample data/ckpt.pt --prompt "once upon time"
```

## Hardware notes

- **3050 Ti (XPS) is the intended trainer** — CUDA + tensor cores + Brevitas QAT.
  4GB VRAM is ample at this size. Add `--compile` there for another ~1.3–2×.
- **M1 (MPS) works** as a dev/proof box at ~30k tok/s (fp16 autocast). A 4000-iter
  run is ~37 min — under an hour, so the proof-of-life lives here.
- The model is small enough that a run is minutes-to-an-afternoon either way; the
  real work is the pipeline (tokenise → train → validate vs goformer → INT4 QAT),
  not GPU throughput.

## Rollback / overfitting

`ckpt.pt` always holds the lowest-val model, so it is already the automatic
rollback if val turns up. For manual control, every eval also drops a snapshot in
`data/ckpts/`; compare them with `model.sample` and copy the one you like over
`ckpt.pt`. Lower `--eval-interval` for finer snapshot resolution near a suspected
overfit point.

## Tokeniser choice

Char-level here on purpose: zero deps, no OOV, tiny vocab (~57), and it proves
the pipeline. The Kevinised corpus is lowercase with punctuation stripped, so the
char set is small. The eventual FPGA model may switch to a small BPE sized to the
URAM budget — that is a vocabulary-vs-on-chip-memory tradeoff to make then, not
now.
