# Training on the Dell XPS 15 (RTX 3050 Ti, 4 GB)

The M1 did the proof-of-life and the corpus prep (the Keviniser is CPU/spaCy
work — no GPU benefit). The real training run belongs on the 3050 Ti: CUDA +
tensor cores for mixed precision, and it is where the INT4 QAT (Brevitas) path
lives. The handoff between machines is a single text file.

## 1. Get the corpus across

The only thing you need from the Mac is the Kevinised corpus (a plain text
file). Everything else is in this repo.

```
# on the Mac, once the Keviniser finishes:
data/TinyStories-train.kevin.txt        # ~1.3 GB
data/tinystories-train.stats.json       # the compression numbers
```

Copy `TinyStories-train.kevin.txt` to `data/` on the Dell (scp, USB, cloud —
whatever). You do NOT need to copy the raw 1.9 GB `TinyStories-train.txt` or the
`.venv`.

## 2. Environment (Windows)

```
git clone https://github.com/MichaelAyles/kev-gpt
cd kev-gpt
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then install the **CUDA** build of PyTorch (the default wheel is CPU-only).
Match the index URL to your driver's CUDA version — `nvidia-smi` shows it top
right; cu124 is a safe modern default:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

You do not need spaCy on the Dell unless you want to re-run the Keviniser there
(you don't — the corpus is already prepared).

## 3. Train

```
# tokenise the Kevin corpus to data/char/ (one-off, fast)
python -m model.data data/TinyStories-train.kevin.txt data/char

# train on the GPU. CUDA auto-selects bf16 AMP; --compile adds ~1.3-2x.
python -m model.train --device cuda --compile ^
    --corpus data/TinyStories-train.kevin.txt ^
    --max-iters 20000 --eval-interval 1000 ^
    --batch-size 96 --n-embd 256 --n-layer 4
```

Notes for the 4 GB card:
- **Batch size**: 96 is safe at ctx 256 for this 3M-param model. If you hit
  `CUDA out of memory`, drop to 64; if you have headroom, 128+ is fine — the
  model is tiny, activations are the only real consumer.
- **More iters than the M1 run**: the full corpus is ~100x the proof-of-life
  data, so it will not overfit at 4k. 20k+ iters is reasonable; watch the val
  curve and the per-eval snapshots in `data/ckpts/`.
- **`--compile`** only helps on CUDA (skip on the Mac). First eval is slow while
  it compiles, then it speeds up.
- Everything else (snapshots, `states.jsonl`, `evolution.md`) works identically.

Watch it learn:
```
python -m model.evolution data/states.jsonl -o data/evolution.md
python -m model.sample data/ckpt.pt --prompt "once upon time"
```

## 4. Next: the FPGA path (not built yet)

This trains an FP model. The KV260 needs INT4. The remaining steps, per
[`../2-llm-on-kria.md`](../2-llm-on-kria.md):

1. **Brevitas INT4 QAT** — re-train (or fine-tune) the model with `brevitas`
   quantised layers so the weights are INT4 and activations are low-bit. This is
   CUDA-friendly and a natural fit for the 3050 Ti.
2. **Validate against goformer** to cosine > 0.9999 before trusting any speed
   number — bit-honest before fast.
3. **Vocabulary decision**: char-level here keeps the embedding tiny and proves
   the pipeline. For the FPGA model, size a small BPE against the ~3 MB on-chip
   budget — fewer, denser tokens vs a bigger embedding table is the tradeoff.
4. Bake the INT4 weights into URAM (the HLS/RTL work in doc 2).

QAT is its own milestone; the FP model from step 3 above is the prerequisite and
the thing to get coherent first.
