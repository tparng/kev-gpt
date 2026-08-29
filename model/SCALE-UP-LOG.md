# Word-vocab scale-up: does a bigger model fix story quality, and what breaks when you try

## Motivation

The deployed Genesys2 word-vocab checkpoint (D=128, NLAYER=12, NHEAD=2,
VOCAB=1900, val=2.022 — see `fabric/genesys2/PORT-NOTES.md`, "Word-level
vocabulary") produces real-hardware sampled chat that is rougher and more
repetitive than the char-level build's own sampled output ("to help him to
to help him get to help him get"). The open question: is that a model-
capacity problem (fixable by making the model bigger), or something else
(sampling, data, tokenizer)? This log is a pure PyTorch/GPU investigation —
no RTL, no synthesis, no Genesys2 BRAM constraints — meant to answer that
before spending any more real-hardware effort.

**Bottom line up front**: a bigger model (D=384, VOCAB=4096, ~8.7x the
deployed model's params) genuinely produces better prose when it trains
well — but this codebase's default training recipe (AdamW) has a real,
reproducible instability at this width that caps effective training to
roughly the first 3500-4500 iterations of any AdamW run, surviving lower
LR, longer warmup, GPT-2 residual-projection init scaling, excluding
LayerNorm gains from weight decay, logit-scale regularization at two
strengths, and even a completely frozen floor LR held for 10,000
iterations. **Root cause, confirmed in Attempt 7: it's specific to Adam's
optimizer dynamics, not the model, data, or width** — the identical setup
trained with plain SGD+momentum instead shows no divergence at all, just
a (much worse) stable plateau. The key finding for "what does it take to
get acceptable stories": **training stability at width is the actual
bottleneck right now, and it's fixable in principle by tuning Adam's own
hyperparameters (`eps`, `beta2`) rather than needing a different
optimizer or architecture — just not yet tried.**

## Environment / how to reproduce

```bash
cd kev-gpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All runs below used an NVIDIA RTX 4060 Laptop GPU (8GB), `torch.cuda`
available, bf16 autocast (`model.train.amp_dtype` auto-selects bf16 on
CUDA when supported). Throughput at the D=384/VOCAB=4096 shape: ~12.3
it/s, ~100k tokens/s (`--smoke 50` on `model.train`). A 24000-iter run
takes ~48 minutes at this shape; a 10000-iter diagnostic probe ~19-21
minutes; a 6000-iter probe ~12 minutes.

**Corpus**: `data/TinyStories-train.filtered.txt` — the same filtered/
lowercased full TinyStories train split the deployed models use (see
`fabric/genesys2/PORT-NOTES.md`'s NLAYER=12 section for how this was
built). Not checked into the repo (`data/` is gitignored); regenerate via
`keviniser.fetch_tinystories --split train` + the filtering step described
there if starting from scratch.

**Baseline being compared against**: the deployed
D=128/NLAYER=12/NHEAD=2/VOCAB=1900 checkpoint, val=2.022, trained via:
```bash
python -m model.train --max-iters 24000 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 \
  --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16.pt --states data/states_word16.jsonl
```

## Step 0: data prep at the bigger vocab

```bash
python -m model.word_data data/TinyStories-train.filtered.txt data/word_big4096 4096
```
```
vocab_size=4096  train_tokens=414,307,096  val_tokens=4,184,920  unk_frac=0.0087
```
Real, useful data point on its own: `unk_frac` (fraction of tokens that
fall outside the vocab and get mapped to `<unk>`) drops from **0.0394 at
VOCAB=1900** to **0.0087 at VOCAB=4096** — under 1% out-of-vocabulary vs.
~4%. `train_tokens`/`val_tokens` are identical to the VOCAB=1900 prep
(same corpus, same tokenizer regex — only the vocab cutoff changes which
words get their own id vs. collapse to `<unk>`).

## Experiment 1: naive combined scale-up (D=384, NLAYER=12, NHEAD=6, VOCAB=4096)

Chosen deliberately as ONE big combined jump (not a careful one-axis-at-a-
time ablation) to answer "is this fixable by scale at all" fastest before
spending runs isolating which axis matters — `NHEAD=6` keeps head_dim=64,
matching this project's own established convention.

```bash
python -m model.train --max-iters 24000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 \
  --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096.pt --states data/states_word_big4096.jsonl
```

`params=22.82M` (~8.7x the deployed 2.6M model — computed directly by
`GPT.num_params()`, printed at startup).

| iter | train | val | lr |
|---|---|---|---|
| 500 | 2.829 | 2.815 | 1.0e-03 |
| 1000 | 2.495 | 2.481 | 1.0e-03 |
| 1500 | 2.370 | 2.358 | 9.9e-04 |
| 2000 | 2.327 | 2.312 | 9.9e-04 |
| **2500** | **2.322** | **2.309 (best)** | 9.8e-04 |
| 3000 | 2.343 | 2.330 | 9.7e-04 |
| 5000 | 2.523 | 2.510 | 9.1e-04 |
| 10000 | 3.262 | 3.239 | 6.7e-04 |
| 15000 | 4.162 | 4.143 | 3.8e-04 |
| 20000 | 4.934 | 4.911 | 1.6e-04 |
| 24000 | 5.634 | 5.608 | 1.0e-04 |

Train and val loss climb **together** for the entire back 21,500 iterations
of a 24000-iter run — not the "train keeps improving, val worsens"
signature of overfitting, but genuine instability (the model gets worse at
everything). `model.train`'s own auto-rollback correctly saved the
iter-2500 checkpoint (`data/ckpt_word_big4096.pt`, val=2.309) — nothing is
lost, but that checkpoint only saw ~10% of the intended schedule.

**Context**: this exact signature (train+val climbing together after an
early bottom) was already seen once before this investigation, at
D=256/NLAYER=12 **char-level** (see `fabric/genesys2/PORT-NOTES.md`'s
"NLAYER=16 and D=256" section) — where a lower LR was tried once and made
it *worse*. Two independent configs (different width, different vocab
scheme, different corpus tokenization) hitting the same signature is what
motivated treating this as a real, general phenomenon worth root-causing
rather than a one-off fluke.

## Diagnosing the instability

### Attempt 1: lower peak LR + longer warmup

Hypothesis: default `--lr 1e-3 --warmup 100` is too aggressive for a wider
model (a well-known phenomenon without width-aware LR scaling / muP).

**Short probe first (6000 iters)** — looked like it worked:
```bash
python -m model.train --max-iters 6000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_lowlr.pt --states data/states_word_big4096_lowlr.jsonl
```
Best val **2.291 @ iter 4000**, only a small uptick by iter 6000 (2.343).
Looked stable — **this was a false positive**, caught only by re-running
the full schedule (see below). Lesson embedded in every subsequent
diagnostic in this log: **a short probe is not sufficient evidence of
stability** — always extend to at least iter 10000 before trusting a fix.

**Same recipe, full 24000 iters**:
```bash
python -m model.train --max-iters 24000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_stable.pt --states data/states_word_big4096_stable.jsonl
```
Best val **2.332 @ iter 4000**, then the SAME climbing shape all the way to
**val=5.624 @ iter 24000**. The fix only delayed the onset (iter 2500 with
the default recipe -> iter 4000 here) without changing its character.
**Verdict: LR/warmup alone does not fix it.**

### Attempt 2: GPT-2 residual-projection init scaling

Hypothesis: this codebase's `GPT._init()` (`model/gpt.py`) applies a flat
`std=0.02` to every `nn.Linear`, including the two matrices that write
DIRECTLY into the residual stream (`attn.proj`, the second MLP `Linear`).
The original GPT-2 paper scales those specifically by
`1/sqrt(2*n_layer)` to keep residual-stream variance from growing
unboundedly with depth — this codebase never had that. Plausible root
cause since residual-variance blowup compounds with BOTH depth and width.

**Code change** (`model/gpt.py`, in `GPT.__init__`, after `self.apply(self._init)`):
```python
for pn, p in self.named_parameters():
    if pn.endswith("attn.proj.weight") or pn.endswith("mlp.2.weight"):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
```
This is now a permanent, unconditional part of `GPT.__init__` — it affects
every future training run (no flag to disable), since it's a strict
correctness improvement to the init scheme regardless of this
investigation's outcome. Existing already-trained checkpoints are
unaffected (init only matters at construction time).

**Short probe (6000 iters, same LR=3e-4/warmup=2000)**:
```bash
python -m model.train --max-iters 6000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_initfix.pt --states data/states_word_big4096_initfix.jsonl
```
Best val **2.235 @ iter 4500** — slightly better than the non-init-fix
probe, but per the lesson above, not trusted on its own.

**Full 24000 iters**:
```bash
python -m model.train --max-iters 24000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_initfix_full.pt --states data/states_word_big4096_initfix_full.jsonl
```
Best val **2.271 @ iter 3500**, then the same climb to **val=5.531 @ iter
24000**. **Verdict: the init fix does not resolve the instability either**
(kept as a permanent improvement regardless — it's still correct practice
— but it isn't the fix).

**Weight-norm comparison** (best checkpoint, iter 3500, vs. the diverged
end, iter 24000 — both pulled from `data/ckpts/ckpt_03500.pt` and
`data/ckpts/ckpt_24000.pt`, the per-eval snapshot dir; **note**: this
snapshot dir is shared across ALL runs in this log since none of the
commands above override `--snapshot-dir`, so files get overwritten by
whichever run last wrote that iter number — only trust these snapshots
immediately after the run you care about, or pass a distinct
`--snapshot-dir` per run if repeating this):
```python
import torch
for f in ["data/ckpts/ckpt_03500.pt", "data/ckpts/ckpt_24000.pt"]:
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    print(f, "iter", ckpt["iter"], "val", ckpt["val"])
    print("  tok_emb norm", sd["tok_emb.weight"].norm().item())
    print("  block0 qkv norm", sd["blocks.0.attn.qkv.weight"].norm().item())
    print("  block11 mlp.2 norm", sd["blocks.11.mlp.2.weight"].norm().item())
    print("  ln_f weight mean/std", sd["ln_f.weight"].mean().item(), sd["ln_f.weight"].std().item())
```
| | iter 3500 (best) | iter 24000 (diverged) |
|---|---|---|
| val | 2.271 | 5.531 |
| tok_emb norm | 48.79 | 59.03 (+21%) |
| block0 qkv norm | 15.44 | 24.93 (+61%) |
| block11 mlp.2 norm | 9.97 | 25.93 (+160%) |
| ln_f weight mean | 1.45 | 2.81 (+93%) |

Weight norms, especially the final LayerNorm's own gain, grow
substantially over the divergent period — the seed for the logit-scale
hypothesis tested next.

**Qualitative check**: samples from the iter-3500 checkpoint (val=2.271,
"worse" by raw val loss than the deployed model's val=2.022) are
noticeably MORE coherent than the deployed model's real-hardware sampled
output:
```python
import torch
from model.gpt import GPT, GPTConfig
from model.train import sample, pick_device

device = pick_device("auto")
ck = torch.load("data/ckpts/ckpt_03500.pt", map_location=device, weights_only=False)
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"]); model.eval()
for p in ["once upon a time", "the dog ran", "a little girl"]:
    print(p, "->", sample(model, ck["meta"], device, n_tokens=80, prompt=p, seed=42))
```
```
"once upon a time" -> ", there was a girl named lily. she loved to play outside and pick
  flowers. one day, she found a box that was very dirty. she wanted to find a way to make it
  look nice. but, she had an idea. she saw a big jar of orange, a shiny <unk>. it was yellow,
  and her mommy gave her a bottle. lily was excited and started to help."
"the dog ran"       -> " to the tree and gave it a hug. the squirrel said, 'you are very
  brave and strong. you can help me and the dog.' they both got a band-aid from the tree and
  helped the dog. the dog felt proud and happy. tim and his friend played with the dog every
  day. they were happy and cozy."
"a little girl"     -> " named lily. one day, lily and her mommy went to the park. there,
  the park and the swings. it was so pretty! there, they saw many kids playing with a ball."
```
Coherent, grammatical, far less repetitive than the real-hardware
"to help him to to help him get to help him get" pattern — despite a
nominally worse val loss (not directly comparable across different vocab
sizes: cross-entropy over 4096 classes isn't the same scale as over 1900)
and despite far fewer total training iterations. **This is the strongest
evidence that capacity genuinely helps quality when the model actually
gets to train** — the practical blocker is training length, not
architecture capacity.

### Attempt 3: exclude LayerNorm gains/biases from weight decay + add logit-scale instrumentation

Hypothesis refinement: `AdamW(model.parameters(), ..., weight_decay=0.1)`
applies decay uniformly to EVERY parameter, including 1-D LayerNorm gains
and biases — standard GPT-2/nanoGPT practice explicitly excludes those
(decay only >=2D weight matrices). Also added direct instrumentation
(logit std/max, `ln_f` gain norm) to every eval step instead of inferring
weight-scale drift from snapshot diffs after the fact.

**Code change** (`model/train.py`, optimizer construction):
```python
decay, no_decay = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95))
```
Also permanent/unconditional — a strict correctness improvement kept
regardless of outcome.

**Code change** (`model/train.py`, eval block — logit-scale diagnostic):
```python
model.eval()
with torch.no_grad():
    xs, _ = get_batch(splits["val"], block, batch, device)
    with ctx:
        logits_dbg, _ = model(xs, xs)
    logit_std = logits_dbg.float().std().item()
    logit_max = logits_dbg.float().abs().max().item()
lnf_norm = model.ln_f.weight.norm().item()
model.train()
```
Printed alongside the usual `iter/train/val/lr` line as
`logit_std {..} logit_max {..} lnf_norm {..}`.

**Probe, this time 10000 iters** (lesson from the false-positive 6000-iter
probes earlier — 10000 gives enough runway past the ~4000-iter onset
point to actually see whether a fix changes the trajectory):
```bash
python -m model.train --max-iters 10000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_wdfix.pt --states data/states_word_big4096_wdfix.jsonl
```
| iter | val | logit_std | logit_max | lnf_norm |
|---|---|---|---|---|
| 0 | 8.400 | 0.39 | 2.5 | 19.60 |
| 2000 | 2.529 | 1.68 | 15.9 | 24.61 |
| **3500** | **2.262 (best)** | 1.87 | 19.6 | 29.98 |
| 5000 | 2.342 | 2.03 | 21.2 | 34.20 |
| 10000 | 3.221 | 2.39 | 31.8 | 41.24 |

Confirms the mechanism directly: `logit_std`, `logit_max`, and `lnf_norm`
all grow **smoothly and continuously from iteration 0** — they do NOT
spike specifically at the point training turns bad. Growing logit scale
is actually *helping* early (sharper, more confident correct predictions
lower loss) and only becomes harmful once the underlying ranking accuracy
stops improving, at which point the same growing confidence starts
amplifying wrong predictions instead. **Verdict: real, measurable
mechanism, but not itself the root trigger** — something else causes the
ranking-accuracy plateau/reversal around iter 3500-4000; unbounded logit
growth (nothing in this architecture caps it) then makes the consequences
worse than they'd otherwise be. Val at iter 10000 here (3.221) is close
to, not clearly better than, the earlier un-decayed-LN-gain runs (3.066
"stable", 3.085 "initfix_full") — the weight-decay fix alone made no
clear difference either, kept as a correctness improvement regardless.

### Attempt 4: z-loss (logit-scale penalty), weak then strong

Directly targets the confirmed-real mechanism from Attempt 3: penalize
`logsumexp(logits)^2` (the log-normalizer cross-entropy already computes
internally) so logit scale is capped by the loss function itself instead
of hoping weight decay controls it indirectly. Standard technique from
PaLM/ST-MoE, coefficient ~1e-4 there.

**Code change** (`model/gpt.py`, `GPTConfig`):
```python
z_loss_coef: float = 0.0     # 0.0 = off (unchanged default behavior)
```
**Code change** (`model/gpt.py`, `GPT.forward()`):
```python
flat_logits = logits.view(-1, logits.size(-1))
loss = F.cross_entropy(flat_logits, targets.view(-1))
if self.cfg.z_loss_coef > 0:
    z = torch.logsumexp(flat_logits, dim=-1)
    loss = loss + self.cfg.z_loss_coef * z.pow(2).mean()
```
**Code change** (`model/train.py`, new CLI flag `--z-loss-coef`, default
`0.0`, threaded into `GPTConfig`). **Also**: `estimate_loss()` was patched
to temporarily zero `model.cfg.z_loss_coef` during eval so reported `val`
stays pure cross-entropy and comparable across z-loss on/off runs (z-loss
is a training regularizer, not part of the tracked quality metric):
```python
z_loss_coef = model.cfg.z_loss_coef
model.cfg.z_loss_coef = 0.0
# ... compute val loss as before ...
model.cfg.z_loss_coef = z_loss_coef
```

**Weak (1e-4, the PaLM default), 10000 iters**:
```bash
python -m model.train --max-iters 10000 --lr 3e-4 --warmup 2000 --z-loss-coef 1e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_zloss.pt --states data/states_word_big4096_zloss.jsonl
```
Best val **2.261 @ iter 3500**, then the same climb to val=3.167 @ iter
10000 (logit_max=30.1, lnf_norm=41.14) — **nearly identical to the
no-z-loss run** (val=3.221, logit_max=31.8, lnf_norm 41.24) at every
tracked iter. At this scale (logit_max~30), the z-loss gradient
contribution is roughly `2 * 1e-4 * 30 ≈ 0.006` per logit — tiny next to
cross-entropy's own gradient (order 0.1-1) — too weak to move anything
measurably.

**Strong (1e-2, 100x), 10000 iters**:
```bash
python -m model.train --max-iters 10000 --lr 3e-4 --warmup 2000 --z-loss-coef 1e-2 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_zloss2.pt --states data/states_word_big4096_zloss2.jsonl
```
| iter | val | logit_std | logit_max | lnf_norm |
|---|---|---|---|---|
| **4000** | **2.244 (best)** | 2.86 | 27.8 | 31.69 |
| 10000 | 2.931 | 4.05 | 37.2 | 40.91 |

Counterintuitively, `logit_max` grew to **37.2** — HIGHER than either
weaker run (30.1-31.8) — yet the val-loss trajectory shows the exact same
shape and timing as every other run (best at iter 4000, climbing
afterward). **Verdict: z-loss, at 100x the standard strength, changed
logit growth (made it worse, if anything) but had zero measurable effect
on when or how severely the divergence happens.** This rules out
logit-scale growth as causal — it's a correlated symptom, not the
trigger.

### Attempt 5: per-layer gradient-norm logging

The next step flagged in the original write-up: instead of another
loss-function-level guess, instrument WHERE in the network the gradients
change character, to see if one specific layer/component is the trigger.

**Code change** (`model/train.py`): a new `layer_grad_norms(model)`
helper groups every parameter's gradient L2 norm by `blockN.{ln1,ln2,
qkv,proj,mlp_fc,mlp_proj}`, plus `tok_emb` (tied with `head`), `pos_emb`,
and `ln_f` — called right after `scaler.unscale_(opt)` and before
`clip_grad_norm_` (clipping is a single global rescale, so it doesn't
change the RELATIVE proportions between layers, but pre-clip keeps the
raw numbers meaningful on their own). New CLI flags: `--layer-grad-log
PATH` (JSONL output, one line per logged step) and `--layer-grad-interval
N` (default 100).

```bash
python -m model.train --max-iters 8000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --layer-grad-log data/gradlog_word_big4096.jsonl --layer-grad-interval 50 \
  --out data/ckpt_word_big4096_gradlog.pt --states data/states_word_big4096_gradlog.jsonl
```
Same val trajectory as every other run at this recipe (best val=2.257 @
iter 3500, climbing to 2.720 @ iter 8000) — confirms this run is
representative before trusting the gradient data.

**Per-block totals** (sum of that block's 6 component norms), sampled
every 500 iters:
```
iter   tok_emb  pos_emb  ln_f    blk0   blk1   blk2   blk3  ...  blk9  blk10  blk11
   0     1.58    0.55   0.03    7.64   7.67   7.00   6.39  ...  4.48   4.49   4.26
2500     0.39    0.22   0.01    1.04   0.57   0.64   0.66  ...  0.30   0.31   0.33
3500     0.39    0.23   0.01    1.06   0.50   0.62   0.64  ...  0.35   0.35   0.37
5000     0.41    0.30   0.01    1.38   0.65   0.78   0.85  ...  0.53   0.49   0.46
7500     0.47    0.39   0.01    1.90   0.94   1.24   1.31  ...  0.84   0.76   0.65
```
(full file: `data/gradlog_word_big4096.jsonl`, gitignored, one JSON object
per logged iter with all 75 per-component keys — regenerate via the
command above.)

**Finding: EVERY block bottoms out and re-grows in lockstep, uniformly
across depth and across attention/MLP sub-components** — this is a
network-wide, synchronized phenomenon, not one layer misbehaving:

| group | grad-norm minimum | @ iter | value at iter 7950 | relative growth |
|---|---|---|---|---|
| tok_emb (tied w/ head) | 0.356 | 2100 | 0.459 | **+29%** |
| block0 (total) | 0.942 | 2900 | 1.954 | +107% |
| block5 (total) | 0.446 | 2950 | 1.121 | +151% |
| block11 (total) | 0.305 | 2350 | 0.654 | +115% |

Every transformer block's gradient norm bottoms out at almost exactly the
same iteration (2350-2950) that val loss bottoms (3500) and grows by a
similar, large relative amount (107-151%) regardless of depth — checked
down to the qkv/proj/mlp_fc/mlp_proj component level (block0 vs block11
compared directly), same synchronized bottom-then-grow shape everywhere.
**tok_emb/head is the one clear outlier: its gradient norm stays far more
stable (+29% vs +107-151% for the transformer stack).**

**Interpretation**: this rules out "one specific layer is the trigger" —
whatever is happening, it's not localized to a particular depth or to
attention vs. MLP specifically. The one real asymmetry (embedding/head
comparatively stable, the full 12-layer transformer stack uniformly
unstable) also weakens a "big VOCAB=4096 tied embedding/head is the
problem" hypothesis, since that's exactly the layer that stays *most*
stable. The synchronized, network-wide growth is consistent with a
genuine optimization feedback loop rather than a localized bug: cross-
entropy gradients scale with prediction error, so once the model drifts
even slightly worse from its iter-3500 optimum (for whatever underlying
reason — plausibly just normal SGD noise failing to stay inside a narrow
minimum at this width, with LR still only partially decayed at that
point), every block's gradients grow together, reinforcing the drift
further. This is a plausible mechanism, not a proven one — it wasn't
tested directly (e.g., by comparing gradient-noise magnitude/variance
across widths, or checking if a much longer, gentler LR decay avoids ever
leaving the iter-3500 basin in the first place). **Root cause is still
open**; this rules out where it ISN'T (a single bad layer, the embedding/
head) more than it pins down where it IS.

### Attempt 6: longer/gentler LR decay — get to a low LR fast, then hold it there

The next step flagged above: instead of stretching decay across the WHOLE
run (so LR is still ~80-90% of peak right when the model usually leaves
its optimum, iter ~3500-4000), decouple the decay horizon from total
training length — ramp down to the LR floor FAST, then just keep training
at that tiny, fixed LR for a long time. Tests directly: does the model
stay near its optimum once it's off the high-LR plateau, or does it drift
away regardless of LR?

**Code change** (`model/train.py`): new `--lr-decay-iters` flag,
decoupled from `--max-iters` — `cosine_lr`'s own `total` parameter (which
governs when LR reaches its floor and holds there, via its existing
`if it > total: return lr * min_lr_frac` branch) now defaults to
`--max-iters` (old behavior, unchanged) but can be set independently.

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 5000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_gentledecay.pt --states data/states_word_big4096_gentledecay.jsonl
```
LR ramps over 0-2000 (warmup), decays over 2000-5000 (reaching floor
`0.1*3e-4=3e-5` right at iter 5000), then **holds flat at 3.0e-05 for all
10,000 remaining iters (5000-15000)**.

| iter | train | val | lr |
|---|---|---|---|
| 3500 | 2.254 | 2.231 | 1.6e-04 |
| 4000 | 2.231 | 2.209 | 9.8e-05 |
| **4500** | **2.225** | **2.203 (best of the ENTIRE investigation)** | 4.8e-05 |
| 5000 | 2.242 | 2.219 | 3.0e-05 (floor reached) |
| 6000 | 2.304 | 2.282 | 3.0e-05 (holding) |
| 8000 | 2.496 | 2.475 | 3.0e-05 (holding) |
| 10000 | 2.814 | 2.792 | 3.0e-05 (holding) |
| 12500 | 3.365 | 3.344 | 3.0e-05 (holding) |
| 15000 | 3.598 | 3.570 | 3.0e-05 (holding) |

**This is a decisive negative result: the model diverges even at a
completely frozen, near-zero learning rate held for two-thirds of the
run.** Train and val loss both climb continuously and substantially
(val +61% from iter 5000 to 15000) with LR literally unchanged the whole
time. **This conclusively rules out the LR-schedule hypothesis
entirely** — every variant tried (lower peak, longer warmup, faster
decay, and now a fully pinned floor LR) shows the same divergence, so it
was never really about LR magnitude or shape.

**What this points to instead**: Adam's per-parameter update is
variance-normalized (`lr * m_hat / (sqrt(v_hat) + eps)`), not simply
proportional to raw gradient magnitude — so even at `lr=3e-5`, the
*normalized* step size per parameter doesn't shrink to zero, and
compounded over 10,000 steps that's still a meaningful cumulative drift
independent of the nominal LR value. This would explain why NO amount of
LR tuning fixes it: the mechanism producing the drift isn't LR-gated at
all. Not tested directly in this pass (candidates for whoever picks this
up next): plain SGD instead of AdamW (removes the per-parameter variance
normalization entirely, a clean way to test this theory); an even more
extreme floor LR (e.g. 1e-6) to see if the drift ever actually stops;
`weight_decay=0` to rule out decay-driven erosion; or replaying a FIXED
batch sequence across runs to rule out data-order effects instead of
optimizer dynamics.

### Attempt 7: plain SGD instead of AdamW — the direct test, and the answer

The most direct test of Attempt 6's leading hypothesis: SGD's update is
directly proportional to the raw gradient (with momentum), with no
per-parameter variance normalization at all. If the divergence still
happens with SGD, Adam's normalization is cleared. If it doesn't, that's
the root cause.

**Code change** (`model/train.py`): new `--optimizer {adamw,sgd}` flag
(default `adamw`, unchanged behavior). SGD uses `momentum=0.9`, the same
decay/no_decay parameter grouping as AdamW (weight decay only on >=2D
matrices).

**LR calibration first** (SGD needs a much higher LR than Adam — no
adaptive per-parameter scaling to compensate for raw gradient magnitude).
Three short 1000-iter probes at `--warmup 200`:

| lr | val @ iter 1000 |
|---|---|
| 0.03 | 4.549 |
| 0.1 | 4.753 (worse) |
| 0.3 | 5.743 (worse, visibly noisy: train loss went UP from iter 200→600) |

`0.03` was the best of the three (higher values were actively worse, not
just slower) — used for the real run.

**Real run**, same ramp/decay/hold structure as Attempt 6 (swap only the
optimizer), given a longer decay horizon since SGD converges slower:
```bash
python -m model.train --max-iters 15000 --optimizer sgd --lr 0.03 --warmup 2000 --lr-decay-iters 6000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_sgd.pt --states data/states_word_big4096_sgd.jsonl
```
| iter | train | val | lr | logit_max | lnf_norm |
|---|---|---|---|---|---|
| 2000 (peak lr) | 4.752 | 4.743 | 3.0e-02 | 9.6 | 22.74 |
| 4000 | 4.517 | 4.501 | 1.7e-02 | 11.0 | 24.43 |
| **6000 (floor reached)** | 4.421 | **4.409 (best)** | 3.0e-03 | 10.8 | 24.99 |
| 8000 | 4.427 | 4.415 | 3.0e-03 (holding) | 10.8 | 25.20 |
| 10000 | 4.426 | 4.414 | 3.0e-03 (holding) | 10.8 | 25.39 |
| 12500 | 4.418 | 4.406 | 3.0e-03 (holding) | 10.9 | 25.63 |
| 15000 | 4.414 | **4.401** | 3.0e-03 (holding) | 10.9 | 25.85 |

**No divergence.** Val loss settles at iter ~6000 (right as LR reaches
its floor) and then stays completely flat — fluctuating in a narrow
4.401-4.416 band — for the remaining 9000 iterations, the exact same
hold-period length where AdamW (Attempt 6, identical model/data/LR-decay
structure) diverged by +61%. `logit_max` and `lnf_norm` plateau too
(logit_max barely moves at all, 10.8→10.9 across 9000 iters, vs AdamW's
continuous climb).

**This is conclusive: the instability is specific to Adam's optimizer
dynamics, not the model, the data, or the width itself.** The obvious
caveat: SGD's absolute loss (4.401) is far worse than AdamW's best
(2.203) — SGD is much less sample-efficient for transformers at this
scale, well-documented in the broader literature and confirmed here, not
a surprise. This isn't a usable fix on its own (nobody wants a model
stuck at val=4.4 when val=2.2 is reachable), but it decisively answers
the ROOT CAUSE question and points at a much narrower, more promising
next step than anything tried before: **target Adam's specific
mechanism directly rather than abandoning Adam** — most directly, try a
much larger `eps` in `AdamW(..., eps=...)` (default `1e-8`; something
like `1e-3` or `1e-2` would make the `sqrt(v_hat)+eps` denominator less
aggressive at small gradient-variance estimates, closer to un-normalized
SGD-like behavior while keeping Adam's momentum/adaptivity elsewhere).
Also worth trying: lower `beta2` (default `0.95` here; a value like
`0.9` or lower makes the variance estimate track recent gradients more
responsively instead of averaging over a longer window that may be
slow to "notice" it should shrink the effective step). Neither was
tried in this pass — this is the concrete next step, not yet taken.

## Summary: every intervention tried, and what it did

| Intervention | Best val | Divergence onset | Changed the core instability? |
|---|---|---|---|
| Default (LR=1e-3, warmup=100) | 2.309 @ iter 2500 | ~iter 2500 | baseline |
| LR=3e-4, warmup=2000 | 2.332 @ iter 4000 | ~iter 4000 | delayed onset only |
| + GPT-2 residual-projection init | 2.271 @ iter 3500 | ~iter 3500 | no |
| + weight decay excludes LN gains/biases | 2.262 @ iter 3500 | ~iter 3500 | no |
| + z-loss @ 1e-4 | 2.261 @ iter 3500 | ~iter 3500 | no (too weak to engage) |
| + z-loss @ 1e-2 (100x) | 2.244 @ iter 4000 | ~iter 4000 | no (engaged, still no effect) |
| per-layer gradient logging (no fix, pure diagnostic) | 2.257 @ iter 3500 | ~iter 3500 | n/a — localizes WHERE, not a fix: uniform across all 12 blocks, embedding/head comparatively stable |
| decay-iters=5000, held flat past that (fully pinned floor LR) | **2.203 @ iter 4500 (best overall)** | ~iter 4500, THEN CONTINUES for 10,000 more iters at frozen LR | **no — rules out LR entirely** |
| plain SGD+momentum (same ramp/decay/hold structure) | 4.409 @ iter 6000 | ~iter 6000, then FLAT for 9,000 more iters (4.401-4.416 band) | **N/A — no divergence at all; identifies Adam as the cause** |
| AdamW, `adam-beta2=0.9` (lower variance-EMA decay) | 2.218 @ iter 4500 | ~iter 4500, tracks the AdamW-default divergence almost exactly | **no — beta2 isn't the mechanism** |
| AdamW, `adam-eps=1e-3` | **2.743 @ iter 11500-13500 (genuinely flat, not a snapshot)** | none — holds flat for the entire 10,000-iter window | **yes — fixes it, efficiency mostly retained** |
| AdamW, `adam-eps=1e-2` | 3.919 @ iter 15000 (still slowly falling) | none | **yes — fixes it, but efficiency mostly lost (~SGD territory)** |
| AdamW, `adam-eps=3e-4` + gentler decay (decay-iters=8000) | 2.365 @ iter 7500 | ~iter 7500-8000, then climbs 3-4x slower than default AdamW | **partial — dampens but doesn't eliminate; best peak of any config, but not stable** |
| AdamW, `adam-eps=3e-4` at the *original* decay length (decay-iters=5000, filling Attempt 8's gap) | 2.426 @ iter 7500 | ~iter 8000, drifts +0.111 to iter 15000 | **partial — meaningfully better peak than eps=1e-3 at the same decay length, some residual drift** |
| AdamW, `adam-eps=1e-3` + gentler decay (decay-iters=10000) | **2.606 @ iter 9500** | ~iter 9500-10000, then drifts +0.052 over 5000 iters (~4x gentler than the eps=3e-4 combo) | **yes, close to fully — best practical tradeoff of the sweep** |
| AdamW, `adam-eps=1e-3` + even gentler decay (decay-iters=13000) | 2.607 @ iter 10500 (plateau, no improvement over decay-iters=10000) | ~iter 11000 (starts before the floor), drifts +0.058 over 4500 iters (same magnitude as decay-iters=10000) | **plateau — recipe already found its ceiling at decay-iters~10000** |
| Same recipe (`eps=1e-3`, decay@10000) ported to the deployment shape (D=128/NLAYER=12/VOCAB=1900) | 2.672 @ iter 23000, still improving | none — this run didn't diverge (see below: not guaranteed) | **N/A — negative transfer: worse than that shape's own existing recipe (val 2.022 @ iter 23000)** |
| Reproducing the deployment shape's *original* recipe (lr=5e-4, warmup=100, full cosine, default eps) | 2.208 @ iter 7500, then climbs to 2.398 @ iter 24000 | ~iter 7500-8000, continues the whole rest of the run — same shape of divergence as the wide model, milder | **N/A — a second run of the exact original recipe diverges; the clean 2.022 result wasn't guaranteed by the recipe alone** |
| Same original recipe + mild `adam-eps=1e-5` margin | 2.208 @ iter 7500 (same peak), 2.394 @ iter 24000 | same as unstabilized — no measurable change | **no — dose too small to matter at this width** |
| Same original recipe + mild `adam-eps=1e-4` margin | 2.213 @ iter 10000-10500 (~same peak, negligible cost), 2.310 @ iter 24000 | still drifts, but ~half the rate (+0.097 vs +0.186-0.190) and starting later | **yes — meaningfully reduces divergence risk at negligible peak-quality cost** |
| Same original recipe + `adam-eps=3e-4` margin | **2.203 @ iter 19500-20000 (best of all deployment-shape runs tried), 2.211 @ iter 24000** | none — genuinely flat (+0.008 over the last ~4500 iters) | **yes, best result — beats the unstabilized peak and is stable, superseding the eps=1e-4 recommendation** |

Every AdamW configuration bottoms at essentially the same *relative*
point — roughly 1500-2500 iterations past wherever warmup ends —
regardless of peak LR, warmup length, init scaling, weight-decay
grouping, or logit-scale regularization strength. Attempt 6 went further
and pinned LR completely flat at its floor for two-thirds of an entire
run, and the model STILL diverged just as much, ruling out the entire
LR-schedule axis. **Attempt 7 then closed the loop**: the exact same
model/data/ramp-decay-hold structure, with plain SGD instead of AdamW,
shows NO divergence at all — it settles and stays flat. Combined with
Attempt 5's finding (growth synchronized across the whole network, not
localized to one layer), **the root cause is now identified as Adam's
optimizer dynamics specifically** (most likely the variance-normalized
per-parameter step size not actually shrinking at low nominal LR), not
the model, the data, the width, or the LR schedule. **Attempt 8 then
found the actual fix**: a much larger `adam-eps` (`1e-3` or `1e-2`)
eliminates the divergence entirely, while a lower `adam-beta2` (`0.9`)
does not — pinning the mechanism specifically to the `sqrt(v_hat)+eps`
denominator floor, not the variance EMA's window length.

### Attempt 8: target Adam's mechanism directly — `eps` fixes it, `beta2` doesn't

Attempt 7 narrowed the fix to two candidates on Adam's own denominator
(`lr * m_hat / (sqrt(v_hat) + eps)`): a much larger `eps`, or a lower
`beta2`. This attempt tests both, isolated, against the exact same
ramp/decay/hold structure as Attempts 6 and 7 (`--lr 3e-4 --warmup 2000
--lr-decay-iters 5000 --max-iters 15000`, floor `3.0e-05` held from
iter 5000 to 15000).

**Code change** (`model/train.py`): new `--adam-eps` (default `1e-8`,
torch's own default) and `--adam-beta2` (default `0.95`, this codebase's
prior default) flags, both wired into the `AdamW(...)` constructor.
`--optimizer sgd` is unaffected (still fixed `momentum=0.9`).

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 5000 --adam-eps 1e-2 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps1e2.pt --states data/states_word_big4096_eps1e2.jsonl
# repeat with --adam-eps 1e-3 --out ..._eps1e3.pt --states ..._eps1e3.jsonl
# repeat with --adam-beta2 0.9 (eps left at default) --out ..._beta2_0.9.pt --states ..._beta2_0.9.jsonl
```

| iter | eps=1e-2 val | eps=1e-3 val | eps=3e-4 val | beta2=0.9 val | baseline (Attempt 6) val |
|---|---|---|---|---|---|
| 4000 | 4.095 | 2.876 | 2.509 | 2.225 | 2.209 |
| **4500** | 4.069 | 2.830 | 2.461 | **2.218 (best)** | **2.203 (best)** |
| 5000 (floor reached) | 4.056 | 2.808 | 2.442 | 2.236 | 2.219 |
| 6000 | 4.038 | 2.791 | 2.432 | 2.298 | 2.282 |
| **7500** | 4.014 | 2.772 | **2.426 (best)** | 2.429 | (not logged at this iter) |
| 8000 | 4.006 | 2.766 | 2.427 | 2.486 | 2.475 |
| 10000 | 3.978 | 2.751 | 2.437 | 2.802 | 2.792 |
| 12500 | 3.947 | **2.743 (best, flat 11500-13500)** | 2.473 | 3.237 | 3.344 |
| 15000 | **3.919 (best, still slowly falling)** | 2.746 | 2.537 | 3.527 | 3.570 |

(`eps=3e-4`'s own command: identical flags to `eps=1e-3`'s above except
`--adam-eps 3e-4 --out data/ckpt_word_big4096_eps3e4_decay5000.pt
--states data/states_word_big4096_eps3e4_decay5000.jsonl` — run
separately, later, to fill in this exact gap; see the addendum after
Attempt 9 below.)

**`--adam-beta2 0.9` does not fix it.** Its whole trajectory tracks the
AdamW-default baseline almost exactly, iter for iter (2.218 vs 2.203
best, 2.298 vs 2.282 at iter 6000, 3.527 vs 3.570 at iter 15000) —
lowering beta2 alone leaves the instability fully intact. This rules
out the "variance EMA window too slow to react" half of Attempt 7's
hypothesis.

**Both `--adam-eps` values fix it — no divergence, in either case, for
the entire 10,000-iteration hold period.** `eps=1e-2` is stable but
pays a steep sample-efficiency cost, settling near val=4.0 (closer to
SGD's 4.4 than to AdamW's 2.2 — a large `eps` mostly cancels out Adam's
variance normalization, which is exactly the mechanism responsible, so
this is the expected trade). `eps=1e-3` is the interesting result: it
converges smoothly to **val=2.743 and then genuinely holds flat**
(2.743-2.746 across iters 11500-15000, not still falling and not
climbing) — a real converged optimum, not a lucky early snapshot ahead
of divergence. That number (2.743) is numerically worse than the
default-AdamW/beta2=0.9 *best-checkpoint* value (2.203/2.218), but that
comparison is misleading: 2.203 was never a stable point the model
settled at — it was the single best moment glimpsed on the way to a run
that eventually reached val=3.57. `eps=1e-3` gives up some of that peak
but is a genuine floor, reachable without needing the auto-rollback to
gamble-catch it.

**This closes the loop Attempt 7 opened**: the fix is `eps`, not
`beta2`. `eps=1e-3` is the practical choice — meaningfully more
efficient than `1e-2` and, unlike default `1e-8`/`beta2=0.9`, converges
to a value it actually stays at rather than diverging away from. Not
yet tried: whether combining a moderate `eps` increase (e.g. `3e-4` to
`1e-3`) with a longer/gentler decay recovers closer to the ~2.2 peak
while keeping the flat-hold stability — the natural next calibration
step for anyone extending this further.

### Attempt 9: `eps=3e-4` + a gentler decay — dampens the divergence, doesn't eliminate it

Attempt 8's closing question: does a moderate `eps` (between the
default `1e-8` and the fix-strength `1e-3`) combined with a
longer/gentler LR decay recover closer to the ~2.2 peak while keeping
`eps=1e-3`'s flat-hold stability?

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 8000 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps3e4_decay8000.pt --states data/states_word_big4096_eps3e4_decay8000.jsonl
```
`lr-decay-iters=8000` (vs 5000 in every prior attempt) stretches the
decay from 2000-8000 instead of 2000-5000, reaching the same floor
(`3.0e-05`) later and holding it for 7000 iters instead of 10,000.

| iter | train | val | lr |
|---|---|---|---|
| 4000 | 2.541 | 2.523 | 2.3e-04 |
| 6000 | 2.409 | 2.390 | 9.8e-05 |
| **7500** | **2.384** | **2.365 (best)** | 3.5e-05 |
| 8000 (floor reached) | 2.389 | 2.369 | 3.0e-05 |
| 10000 | 2.427 | 2.408 | 3.0e-05 (holding) |
| 12500 | 2.497 | 2.479 | 3.0e-05 (holding) |
| 15000 | 2.597 | 2.580 | 3.0e-05 (holding) |

**Partial result: this beats `eps=1e-3`'s peak (2.365 vs 2.743) but
does not reproduce its flat-hold stability.** Val bottoms right around
where LR reaches its floor (iter 7500-8000, matching the pattern of
every earlier AdamW run) and then climbs steadily for the remaining
7000 iters — a real divergence, not noise, but a much gentler slope
than any other AdamW configuration tried (2.365→2.580 over 7000 iters
here, vs default AdamW's 2.203→3.570 over 10,000 iters in Attempt 6, or
`beta2=0.9`'s 2.218→3.527 in Attempt 8 — roughly 3-4x slower per-iter
growth). So `eps=3e-4` measurably dampens the instability relative to
default `eps=1e-8`, but doesn't cross the threshold `eps=1e-3` and
`eps=1e-2` do into genuine flatness — this is a continuous effect (more
`eps` = less divergence), not a hard on/off switch, and `3e-4` sits
partway across it. The best-val checkpoint (2.365 @ iter 7500) is
still, like the pre-Attempt-8 runs, a lucky early snapshot the
auto-rollback has to catch rather than a point the model settles at and
stays.

**Answer to Attempt 8's open question**: no, this combo doesn't get
both the ~2.2-ish peak and flat-hold stability simultaneously — there's
an efficiency/stability tradeoff along the `eps` axis, and `1e-3` (not
`3e-4`) is the practical operating point already found in Attempt 8 if
genuine stability (not just a better rollback target) is the goal.
Not yet tried: an even gentler decay (e.g. `lr-decay-iters` close to
`max-iters`, more like a standard single cosine sweep) at `eps=1e-3`,
which might let the model spend more time at higher effective LR before
eps's stabilizing effect has to do its work at the floor.

**Addendum, filling a gap in Attempt 8's own sweep**: the table above
pairs `eps=3e-4` with the *gentler* decay (`lr-decay-iters=8000`) — it
was never tested at Attempt 8's original `lr-decay-iters=5000`
structure, the same one `eps=1e-3`/`1e-2` were tested against, so the
two weren't cleanly comparable on decay length alone. Filling that gap:

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 5000 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps3e4_decay5000.pt --states data/states_word_big4096_eps3e4_decay5000.jsonl
```
Result: best val **2.426 @ iter 7500**, final 2.537 @ iter 15000, drift
**+0.111**. This slots cleanly between the two extremes already in
Attempt 8's table, at the *same* decay length: much better peak than
`eps=1e-3` (2.426 vs 2.743) at the cost of real but modest drift (+0.111
vs `eps=1e-3`'s ~0), and vastly better than default `eps=1e-8`'s peak
(2.203, but +1.367 drift to full divergence). Confirms `eps` behaves as
a genuinely continuous dial on this shape, independent of the decay-
length axis: `1e-8` (divergent) → `3e-4` (much dampened, some drift) →
`1e-3` (flat, worse peak) → `1e-2` (flat, much worse peak) — the same
efficiency/stability tradeoff curve Attempt 9's decay-8000 pairing
showed, just cleanly isolated from the decay-length variable this time.
Practically: `eps=3e-4` at this decay length is a genuinely attractive
middle option if a small residual drift (auto-rollback still catches
the 2.426 peak regardless) is acceptable in exchange for meaningfully
better quality than `eps=1e-3`'s full-flatness guarantee.

### Attempt 10: `eps=1e-3` + a gentler decay — the best combo so far

Direct test of Attempt 9's closing question, on the *other* eps value:
same `eps=1e-3` that gave genuine flatness at `lr-decay-iters=5000`, now
with a gentler decay (`lr-decay-iters=10000` instead of 5000 — decay
stretches 2000-10000, floor holds only 10000-15000).

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 10000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps1e3_decay10000.pt --states data/states_word_big4096_eps1e3_decay10000.jsonl
```

| iter | train | val | lr |
|---|---|---|---|
| 4000 | 2.889 | 2.868 | 2.6e-04 |
| 6000 | 2.706 | 2.686 | 1.6e-04 |
| 8000 | 2.642 | 2.622 | 7.0e-05 |
| **9500** | **2.625** | **2.606 (best)** | 3.3e-05 |
| 10000 (floor reached) | 2.627 | 2.607 | 3.0e-05 |
| 12500 | 2.648 | 2.629 | 3.0e-05 (holding) |
| 15000 | 2.677 | 2.659 | 3.0e-05 (holding) |

**This is the best result of the whole `eps` sweep.** Best val 2.606 —
beats `eps=1e-3`/decay-5000's flat floor (2.743) by a clear margin, and
close to `eps=3e-4`/decay-8000's peak (2.365, still the single best) but
without that config's real divergence: the post-floor drift here is
tiny (2.607→2.659 over 5000 iters, +0.052) compared to `eps=3e-4`'s
2.369→2.580 over roughly the same span (+0.211) — about 4x gentler. Not
perfectly flat like `eps=1e-3`/decay-5000 (which had ~0 net drift over
its whole hold), but close enough that it reads as "slowly settling",
not "diverging" — a genuinely better practical tradeoff point than
either Attempt 9 config.

**Confirms the tradeoff is continuous along two axes together** (`eps`
magnitude and decay gentleness), not a single knob: at fixed `eps`,
stretching the decay recovers peak quality at a small, controlled cost
to hold-flatness, rather than either being free or reproducing the
uncontrolled divergence a small `eps` shows under the same gentler
decay (Attempt 9). Since `model.train.py`'s auto-rollback always saves
the best-val checkpoint regardless of what happens afterward, this
config's practical deployment value is closer to its peak (2.606) than
its end-of-run number (2.659) — the residual drift matters for trusting
*how far past the peak* it's safe to keep training, not for the
checkpoint itself. Not yet tried: pushing `lr-decay-iters` even closer
to `max-iters` (e.g. 13000-14000) to see whether the drift keeps
shrinking toward zero, or whether it plateaus once eps's stabilizing
margin is fully substituted for a specific decay length.

### Attempt 11: `eps=1e-3` + `lr-decay-iters=13000` — the plateau

Direct test of Attempt 10's closing question: does stretching the decay
even further than 10000 keep buying back peak quality / shrinking the
residual drift, or does it plateau?

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 13000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps1e3_decay13000.pt --states data/states_word_big4096_eps1e3_decay13000.jsonl
```
Only a 2000-iter hold this time (13000-15000) instead of the prior
attempts' 5000-10000-iter holds, since decay itself now eats most of the
run.

| iter | train | val | lr |
|---|---|---|---|
| 4000 | 2.887 | 2.866 | 2.8e-04 |
| 8000 | 2.647 | 2.629 | 1.5e-04 |
| **10500** | **2.626** | **2.607 (best)** | 6.3e-05 |
| 13000 (floor reached) | 2.647 | 2.629 | 3.0e-05 |
| 15000 | 2.683 | 2.665 | 3.0e-05 (holding) |

**Plateau confirmed — no further improvement.** Best val 2.607 is
statistically identical to `lr-decay-iters=10000`'s 2.606 (Attempt 10):
stretching the decay another 3000 iters bought nothing. The end-of-run
drift (2.607→2.665, +0.058 over 4500 iters) is also essentially the
same magnitude as Attempt 10's (+0.052 over 5000 iters) — not smaller.
One new wrinkle: here the climb starts noticeably *before* the LR floor
is reached (val already rising from iter 11000, ~2000 iters before
`lr-decay-iters=13000`'s floor at iter 13000), where in every prior
attempt the climb began right at or after the floor. This is consistent
with the divergence being tied to an *absolute* LR threshold being
crossed (however low, still nonzero this early) rather than strictly
"whenever LR stops moving" — worth keeping in mind for anyone tuning
further.

**Answer to Attempt 10's open question: it plateaus, doesn't keep
improving.** `lr-decay-iters=10000` was already at (or past) the
efficient point on this curve — `eps=1e-3` combined with *some*
long-ish decay recovers about as much peak quality as this axis alone
is going to give (~2.6), with a small, roughly constant residual drift
regardless of exactly how much further the decay is stretched past
~10000. This closes out the eps/decay-length exploration for this model
size: **`eps=1e-3`, `lr-decay-iters≈10000` (of 15000) is the
recommended practical recipe** from this whole investigation — best
achievable peak without genuine, unbounded divergence.

### Attempt 12: porting the recipe to the deployment shape — a negative transfer result

Every attempt above trained the wide D=384/NLAYER=12/VOCAB=4096
diagnostic shape used to find and reproduce the instability. This
attempt asks the practical question: does the fix recipe (`eps=1e-3`,
`lr-decay-iters≈10000`) help the model actually deployed on real
hardware — D=128/NLAYER=12/NHEAD=2/VOCAB=1900 (`data/ckpt_word16.pt`,
`fabric/genesys2/PORT-NOTES.md`'s "Word-level vocabulary (current)"
section, best val 2.022 @ iter 23000 on real Genesys2 silicon)?

```bash
python -m model.train --max-iters 23000 --lr 3e-4 --warmup 2000 --lr-decay-iters 10000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_epsrecipe_23k.pt --states data/states_word16_epsrecipe_23k.jsonl
```
(First run stopped at `--max-iters 15000`, val 2.719 @ iter 15000, still
improving with no divergence; re-run to `--max-iters 23000` — this
architecture trains fast, 2.61M params, ~9 min total on this GPU — for a
same-iter-count comparison against the deployed checkpoint.)

| | this recipe (eps=1e-3, decay@10000) | original deployment run (`ckpt_word16.pt`) |
|---|---|---|
| peak LR | 3e-4 | ~5e-4 |
| warmup | 2000 | ~100 (old default) |
| LR schedule | fast decay to floor by iter 10000, held flat for the remaining 13,000 | standard single cosine sweep across the full 23000-24000 iters |
| `adam-eps` | 1e-3 | 1e-8 (default; predates the `--adam-eps` flag) |
| val @ iter 23000 | **2.672** | **2.022** |

**Clear negative transfer: the recipe is worse here, not neutral.**
Never diverges (still slowly improving through iter 23000, matching the
port notes' own finding that this narrower shape "trained clean and
stable... no instability, unlike D=256"), but it converges to a
meaningfully worse val loss than the deployment checkpoint's own
existing recipe at the identical iteration count. Two compounding
reasons, not one: (1) `eps=1e-3` was tuned specifically to counteract a
failure mode this shape doesn't have — at a width that never needed
denominator-floor damping, raising `eps` just gives up Adam efficiency
for a stability benefit that was never at risk, the same tradeoff
Attempts 8-9 already documented at the wide shape; (2) the fast-decay-
then-hold schedule (`lr-decay-iters=10000` of 23000) spends 13,000 of
the run's 23,000 iterations at a frozen `3.0e-05` floor, where the
original recipe spent that whole span still slowly decaying from a much
higher peak (5e-4 vs 3e-4) — strictly less effective high-LR training
time, independent of `eps` at all.

**Takeaway: this fix is scale-specific, not a universal drop-in
improvement.** It was diagnosed on, and should stay scoped to, model
shapes wide enough (D>=256-384 at this depth) to actually exhibit the
Adam-driven divergence documented in Attempts 1-7. The currently
deployed D=128/NLAYER=12/VOCAB=1900 checkpoint's existing training
recipe (higher peak LR, standard full-length cosine decay, default
`adam-eps`) remains the better choice for that shape and should not be
replaced by this investigation's findings. Not yet tried: whether the
deployment shape's own recipe (5e-4 peak, full cosine, default eps)
could be improved further on its own terms — a separate question from
this attempt, which only tested whether the *wide-model* fix transfers
(it doesn't).

**Correction, from Attempt 13 below: this attempt's framing — "this
shape never diverges" — turned out to be wrong.** It was based on a
single existing run (`ckpt_word16.pt`'s own clean history) plus this
attempt's own recipe-ported runs, which also happened not to diverge.
Attempt 13 re-ran the *original* recipe itself and got a real
divergence on the identical hyperparameters/architecture/data. The
negative-transfer verdict above (the wide-model fix is the wrong tool
for this shape, and costs quality here) still stands — but "this shape
never diverges" does not; see Attempt 13 for the corrected picture.

### Attempt 13: reproducing the original recipe — it's not a fluke, it's variance

Attempt 12 assumed the deployment shape (D=128/NLAYER=12/VOCAB=1900)
simply doesn't have the instability problem, based on `ckpt_word16.pt`'s
own clean training history (`data/states_word16.jsonl`: monotonic val
improvement the entire 24000-iter run, no divergence). This attempt
re-runs that *exact* original recipe (not the wide-model fix) to check
whether that clean result reproduces, or was itself a lucky run.

**Recipe reverse-engineered from `data/states_word16.jsonl`'s own LR
trace** (this checkpoint predates `--lr-decay-iters`/`--adam-eps`, so
there's no command string to copy — the schedule was inferred by
matching `cosine_lr()`'s output against the logged `lr` values): peak
`lr=5e-4`, `warmup=100` (the old default), `max-iters=24000`, no
`--lr-decay-iters` (single cosine sweep across the whole run, old
behavior), default `adam-eps=1e-8`.

```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_reproduce.pt --states data/states_word16_reproduce.jsonl
```
`model.train` calls `torch.manual_seed(1337)` unconditionally
(`model/train.py:236`) — every run uses the *same* seed, so init and
batch order are identical across runs, not a source of difference
between this reproduction and the original.

| iter | original (`ckpt_word16.pt`) val | reproduction val | reproduction lr |
|---|---|---|---|
| 4000 | (not logged at this iter) | 2.266 | 4.7e-04 |
| **7000-7500** | 2.150 (iter 7000, still improving) | **2.208 (best, iter 7500)** | 4.0e-04 |
| 10000 | 2.098 | 2.227 | 3.3e-04 |
| 13000 | 2.066 | 2.265 | 2.5e-04 |
| 18500 | (not logged at this iter) | 2.340 | 1.1e-04 |
| 23000 | **2.022 (best, real hardware checkpoint)** | 2.385 | 5.2e-05 |
| 24000 | 2.022 | **2.398 (final)** | 5.0e-05 |

**Real divergence, not noise.** The original run improves monotonically
end to end (2.150 → 2.022 over iters 7000-23000, never once regressing
in the logged evals). The reproduction peaks early (2.208 @ iter 7500)
and then climbs continuously for the remaining 16,500 iterations to
2.398 — the same *shape* of unbounded Adam-driven divergence documented
throughout this whole investigation (Attempts 1-7), just milder in
magnitude (+0.19 relative rise here vs. the wide model's +0.61-1.35) and
occurring under a schedule that never holds LR flat (LR keeps decaying
the entire time, unlike every wide-model attempt's floor-and-hold
structure) — so the divergence isn't even gated on LR going flat at
this shape, only on LR being low enough in absolute terms, consistent
with Attempt 11's "starts before the floor" observation.

**This means the deployment shape does carry real instability risk —
and, more surprisingly, it isn't even deterministic despite the fixed
seed.** Every AdamW configuration tried on the wide D=384 diagnostic
shape (9+ runs, every LR schedule/init/weight-decay variant in Attempts
1-6) diverged with 100% reliability. Here, *identical* hyperparameters,
*identical* data, and the *same* `torch.manual_seed(1337)` produced one
clean run (the original) and one diverging run (this reproduction) —
with no available source of run-to-run randomness to blame it on. The
likely mechanism is the same one Attempts 1-11 already characterize
(Adam's `sqrt(v_hat)+eps` step size not actually shrinking to zero at
low nominal LR), combined with ordinary GPU floating-point
non-determinism: CUDA/cuDNN matmul and reduction kernels don't
guarantee bit-identical results run to run even with a fixed seed (parallel
reduction order isn't fixed), so two runs' loss curves drift apart by a
tiny, invisible amount from iteration one — normally irrelevant, but
this shape sits close enough to a genuine edge-of-stability point that
which side of it a run lands on is sensitive to that drift. Not
"random" in the sense of a different seed or data order — genuinely the
same recipe landing differently due to sub-floating-point-epsilon
numerical noise compounding near an instability boundary.

**Revises Attempt 12's practical recommendation.** "Keep the original
recipe because this shape doesn't need stabilizing" is no longer
supportable — the original recipe produced the currently-deployed
checkpoint by getting a favorable run, not because that recipe is safe
by construction. The honest options going forward, not yet decided
between: (a) re-run the original high-LR recipe several more times and
keep the best (cheap here, ~9-10 min/run, but doesn't fix the
underlying risk, just plays the odds better); (b) add a *mild*
stabilizing margin (a smaller `adam-eps` bump than the wide model
needed, or a shorter floor-hold than Attempt 12's, tuned specifically
at this width) that trades little peak quality for real protection
against the divergence seen here; (c) accept the risk and always verify
any new deployment-shape training run against its own `states.jsonl`
curve before trusting a checkpoint, the way Conclusion 3 already
recommends project-wide. Not yet tried: any of the three.

### Attempt 14: a mild `adam-eps` margin at the deployment shape — `1e-4` works, `1e-5` doesn't

Tests option (b) from Attempt 13's closing list: keep the deployment
shape's own recipe exactly as-is (lr=5e-4, warmup=100, full cosine
decay across 24000 iters — Attempt 12 already showed changing the
schedule costs quality here) and add only a mild `adam-eps` bump on
top, well below the wide model's `1e-3` fix strength, to see whether a
small dose buys real protection without the peak-quality cost Attempt
12 paid.

```bash
# eps=1e-5
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 1e-5 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_epsmild1e5.pt --states data/states_word16_epsmild1e5.jsonl
# eps=1e-4 / eps=3e-4 (same flags otherwise, --out/--states renamed
# epsmild1e4 / epsmild3e4)
```

| iter | eps=1e-8 (Attempt 13 reproduction) | eps=1e-5 | eps=1e-4 | eps=3e-4 |
|---|---|---|---|---|
| 4000 | 2.266 | 2.277 | 2.372 | 2.475 |
| 13000 | 2.265 | 2.259 | 2.223 | 2.212 (still falling) |
| **best** | **2.208 @ iter 7500** | **2.208 @ iter 7500** | **2.213 @ iter 10000-10500** | **2.203 @ iter 19500-20000** |
| 18000 | 2.333 | 2.318 | 2.263 | 2.205 |
| 23000 | 2.385 | 2.381 | 2.301 | 2.208 |
| 24000 (final) | 2.398 | 2.394 | 2.310 | **2.211** |
| drift, best→final | +0.190 | +0.186 | +0.097 | **+0.008** |

**`eps=1e-5` does essentially nothing** — its whole trajectory tracks
the unstabilized reproduction almost exactly (2.208 vs 2.208 peak,
2.394 vs 2.398 final, same +0.186-0.190 drift) — too small a dose at
this width's gradient-variance scale to matter, the deployment-shape
analogue of the wide model's own finding that `eps` has a real
threshold below which it does nothing (Attempts 8-9).

**`eps=1e-4` genuinely helps.** Peak val is statistically the same as
the unstabilized runs (2.213 vs 2.208 — a negligible cost, unlike
Attempt 12's `eps=1e-3`+floor-hold recipe which gave up real peak
quality), but the post-peak drift roughly halves (+0.097 vs +0.186-
0.190) and the peak itself lands later (iter 10000-10500 vs iter 7500)
— more of the run is spent still improving before the divergence risk
sets in. Not a full fix like the wide model's `eps=1e-3` (which achieved
genuine flatness, Attempt 10) — this still drifts, just much more
slowly — but it's a real, nearly-free stability margin at this shape,
unlike Attempt 12's ported recipe which cost quality for no benefit.

**`eps=3e-4` does even better — genuine flatness, not just slower
drift.** Bottoms at 2.203 @ iter 19500-20000 (still falling at iter
13000, unlike `eps=1e-4` which had already peaked by iter 10500), then
holds essentially flat through the rest of the run (2.203→2.211 over
the remaining ~4500 iters, +0.008 — statistically noise, not a trend).
This both beats `eps=1e-4`'s peak (2.203 vs 2.213) *and* nearly
eliminates the drift (+0.008 vs +0.097). It mirrors the wide model's own
finding (Attempt 9 vs 10: a mid-strength `eps` dampens without fully
fixing, a stronger one achieves genuine flatness) — just scaled down for
this narrower D=128 shape, where `3e-4` (not the wide model's `1e-3`)
turns out to be the dose that actually crosses into stability rather
than merely slowing the climb.

**This is the answer to Attempt 13's open question, revised: option (b)
works, and `eps=3e-4` is the better dose — not `1e-4`.** On top of the
deployment shape's existing recipe (unchanged peak LR, unchanged
warmup, unchanged full-length cosine decay), `eps=3e-4` gives both a
peak at least as good as the unstabilized runs' best (2.203, versus
2.208 for the lucky unstabilized case) *and* genuine post-peak
stability, superseding the `eps=1e-4` recommendation above. Not yet
tried: whether `eps` values between `3e-4` and `1e-3` land even closer
to the original clean run's 2.022, or whether `3e-4` is itself already
near the efficiency/stability sweet spot the way the wide model's
`eps=1e-3`+`lr-decay-iters~10000` combo was found to be (Attempt 11's
plateau).

**Reproducibility check**: re-ran this exact `eps=1e-4` recipe a second
time (`data/ckpt_word16_v2.pt`, same command, no `--seed` flag needed
since `model.train` fixes `torch.manual_seed(1337)` — see the Attempt 13
correction above) and got an *exact* match with this attempt's own
numbers: best val 2.213 @ iter 10000-10500, final 2.310 @ iter 24000,
matching digit-for-digit through the whole run. One pair each isn't
conclusive, but it's a suggestive contrast with the *default*-eps recipe
(`ckpt_word16.pt` vs. Attempt 13's reproduction), which diverged from
itself between two nominally identical runs — consistent with the `eps`
margin also damping the floating-point-noise sensitivity Attempt 13
identified, not just the post-peak drift once divergence starts. Kept
as a separate checkpoint (`data/ckpt_word16_v2.pt`), not promoted over
the real, hardware-verified `data/ckpt_word16.pt` — that decision is
left to whoever picks this up next, given the downstream QAT/export/
real-hardware chain a swap would affect.

## Conclusions

1. **Capacity helps.** Every run's best-val checkpoint from this whole
   investigation lands in the same narrow val=2.20-2.31 range (the single
   lowest across all 9 runs is `data/ckpt_word_big4096_gentledecay.pt`,
   val=2.203 @ iter 4500 — but they're all close enough to be roughly
   interchangeable, since the instability caps every run at nearly the
   same point regardless of what else changed). The samples shown above
   (from the iter-3500 checkpoint, val=2.271) are clearly more coherent
   and less repetitive prose than the deployed D=128/VOCAB=1900
   real-hardware model, despite a "worse" raw val-loss number that isn't
   actually comparable across different vocab sizes.
2. **Training stability at width is the real bottleneck, not raw
   capacity.** This codebase's recipe (AdamW, cosine LR, the defaults
   used throughout this project) cannot currently sustain long, stable
   training at D>=256-384. Every model this size trained so far has had
   to rely on the automatic best-val rollback catching an early,
   under-trained checkpoint rather than genuine convergence.
3. **`model.train.py`'s auto-rollback is doing real, silent work.** None
   of these experiments lost anything to the instability — the saved
   `--out` checkpoint is always the best-val snapshot, not the final
   (diverged) one. Anyone repeating this doesn't need to babysit runs for
   divergence; just don't trust a checkpoint's `iter` field to mean "fully
   trained" without checking it against `states.jsonl`'s own val curve.
4. **Root cause identified: Adam's optimizer dynamics, not the model, the
   data, the width, or the LR schedule.** Ruled out as causal: LR
   magnitude, warmup length, missing GPT-2 residual-projection init
   scaling, weight decay applied to LayerNorm gains, unconstrained logit
   growth (at two very different strengths), and — decisively, in Attempt
   6 — the LR schedule itself: pinning LR completely flat at its floor for
   10,000 iterations still produced the same divergence (val +61% while LR
   never moved). Per-layer gradient-norm logging (Attempt 5) also rules
   out a single misbehaving layer: growth is synchronized across all 12
   transformer blocks and both attention/MLP sub-components, with the
   embedding/head layer comparatively the MOST stable part of the network.
   **Attempt 7 then pinned it down**: the identical model/data/ramp-decay-
   hold structure, run with plain SGD+momentum instead of AdamW, shows NO
   divergence at all — it settles at iter ~6000 and stays flat for the
   remaining 9,000 iterations, where AdamW diverged badly over the same
   window. All "fixes" tried along the way are kept in the codebase as
   permanent, unconditional improvements (correct practice regardless),
   even though none of them was *the* fix — the fix has to target Adam's
   mechanism directly. **Attempt 8 then found it**: raising AdamW's `eps`
   (default `1e-8`) to `1e-3` or `1e-2` eliminates the divergence
   entirely — the model holds flat for the full 10,000-iteration window
   instead of climbing — while lowering `beta2` (default `0.95`) to `0.9`
   does not fix it at all, tracking the default-AdamW divergence almost
   exactly. This isolates the mechanism specifically to the
   `sqrt(v_hat)+eps` denominator floor being too small at low
   gradient-variance estimates, not to the variance EMA's window length.
   `eps=1e-3` is the practical choice (converges to a genuine flat
   val=2.743, not a lucky snapshot) since `eps=1e-2` gives up most of
   Adam's efficiency advantage (floor ~3.9-4.0, close to SGD's own
   val=4.4). **Attempt 9 tested the obvious follow-up** — `eps=3e-4`
   (between the default and the fix strength) with a longer/gentler
   decay (`lr-decay-iters=8000` instead of 5000) — and found a
   continuous tradeoff, not a free lunch: best peak of any AdamW config
   (2.365 @ iter 7500, beating `eps=1e-3`'s 2.743), but still diverges
   afterward, just 3-4x slower than default AdamW. `eps` size trades
   peak quality against hold-stability along a spectrum; `1e-3` remains
   the practical choice if genuine flatness (not just a better
   rollback target) is the goal. **Attempt 10 then tried the same
   gentler-decay idea at `eps=1e-3` instead of `3e-4`**
   (`lr-decay-iters=10000`) and found the best practical config of the
   whole sweep: best val 2.606 @ iter 9500 — clearly better than
   `eps=1e-3`'s original flat floor (2.743) — with only a small residual
   drift afterward (+0.052 over 5000 iters, about 4x gentler than
   `eps=3e-4`'s drift under its own gentler decay). Confirms the
   tradeoff runs along two axes together (`eps` size and decay
   gentleness): stretching the decay at the already-stabilizing
   `eps=1e-3` buys back peak quality at a small, controlled cost to
   flatness, rather than reproducing `eps=3e-4`'s uncontrolled
   divergence. **Attempt 11 then found the ceiling**: pushing
   `lr-decay-iters` to 13000 gave no further improvement (best val 2.607,
   statistically identical to decay-iters=10000's 2.606) and the same
   residual drift magnitude (+0.058 vs +0.052 over a similar span) — a
   genuine plateau, not a curve still improving. One new detail: at
   decay-iters=13000 the val climb starts noticeably before the LR floor
   is reached, unlike every earlier attempt where it began right at the
   floor — consistent with the divergence tracking an absolute LR
   threshold rather than strictly "whenever LR goes flat." **Recommended
   practical recipe from this whole investigation: `eps=1e-3`,
   `lr-decay-iters≈10000` (of a 15000-iter run)** — best peak this axis
   reaches (~2.6) without unbounded divergence, and stretching the decay
   further buys nothing more.
5. **The recipe is scale-specific — it does not transfer to the actual
   deployed model, and makes it worse if ported blindly.** Attempt 12
   ported `eps=1e-3` + `lr-decay-iters≈10000` to the real D=128/NLAYER=12/
   VOCAB=1900 hardware shape (`data/ckpt_word16.pt`) and got a clearly
   worse result at the same iteration count (val 2.672 vs the deployed
   checkpoint's own 2.022 @ iter 23000), with the recipe's lower peak LR
   and its fast-decay-then-hold schedule spending most of the run at a
   frozen floor instead of the deployed recipe's full-length decay from a
   higher peak. This part of the finding stands: **the fix is scoped to
   model shapes wide enough to actually exhibit the Adam-driven
   divergence (D>=256-384 at this depth)** — porting it to a narrower
   shape just costs quality.
6. **But the deployment shape is NOT immune to the instability — Attempt
   12's "this shape never diverges" premise was wrong, corrected by
   Attempt 13.** Re-running the *original* deployment recipe itself
   (lr=5e-4, warmup=100, full cosine, default eps — not the wide-model
   fix) a second time, on identical hyperparameters/architecture/data,
   produced a real divergence: best val 2.208 @ iter 7500, then a
   continuous climb to 2.398 @ iter 24000 — the same *shape* of
   unbounded Adam-driven divergence as the wide model, just milder
   (+0.19 vs +0.61-1.35) and not gated on LR going flat (this run's LR
   never stopped decaying). The originally-deployed `ckpt_word16.pt`
   (val 2.022, monotonic improvement the whole run) is a *favorable*
   outcome of this recipe, not evidence the recipe is safe by
   construction — every AdamW config on the wide shape diverged with
   100% reliability across 9+ runs, but here identical settings (even
   the same fixed `torch.manual_seed(1337)` — there's no seed/batch-order
   difference available to blame) produced one clean run and one
   diverging run, i.e. the instability exists at this width too, just
   closer to a knife-edge — sensitive to ordinary GPU floating-point
   non-determinism (parallel reduction order isn't bit-fixed even under a
   fixed seed) rather than deterministic given the hyperparameters alone.
   **Practical implication: any
   future training run at this deployment shape should be checked
   against its own `states.jsonl` curve before trusting the result (see
   Conclusion 3) — a clean run isn't guaranteed just because the last one
   was clean.** Whether a mild stabilizing margin (smaller than the wide
   model needed) is worth the peak-quality cost at this shape is an open
   question, not resolved here.
7. **Resolved: a mild `adam-eps` margin is worth it — `3e-4` is the
   right dose, not `1e-4`, and `1e-5` does nothing.** Attempt 14 tested
   Conclusion 6's open question directly: kept the deployment shape's
   exact original recipe (peak LR, warmup, full cosine decay all
   unchanged) and added only an `adam-eps` bump, sweeping three values.
   `eps=1e-5` did essentially nothing (tracked the unstabilized
   trajectory almost exactly). `eps=1e-4` genuinely helped but only
   partially: same peak quality (2.213 vs 2.208) but still drifted
   afterward, just at roughly half the rate. `eps=3e-4` did best of all:
   peak 2.203 @ iter 19500-20000 — better than the unstabilized runs'
   own best (2.208) — and then genuinely flat (+0.008 drift over the
   last ~4500 iters, not a trend). Mirrors the wide model's own
   `eps`-dose-dependent transition from partial dampening to true
   flatness (Attempts 9 vs 10), just scaled down for this narrower
   width. Both `eps=1e-4` and `eps=3e-4` runs reproduced exactly on a
   second run each (`torch.manual_seed(1337)` is fixed — see Conclusion
   6's correction), consistent with the eps margin also damping the
   floating-point-noise sensitivity behind Attempt 13's run-to-run split,
   not just the drift once divergence starts. **Recommended addition for
   future deployment-shape training runs: keep the existing recipe
   (lr=5e-4, warmup=100, full cosine decay) and add `--adam-eps 3e-4`**
   — matches or beats the unstabilized recipe's own best-case peak, with
   genuine stability instead of a lucky-run gamble.

## Files touched

- `model/gpt.py`: `GPTConfig.z_loss_coef` (new field, default 0.0/off),
  `GPT.__init__`'s residual-projection init-scaling loop (new, always on),
  `GPT.forward()`'s optional z-loss term.
- `model/train.py`: `--z-loss-coef` CLI flag, `AdamW` param-grouping
  (decay/no_decay split, always on), `estimate_loss()`'s z-loss-neutral
  eval, the new `logit_std`/`logit_max`/`lnf_norm` print at every eval,
  the `layer_grad_norms()` helper plus `--layer-grad-log`/
  `--layer-grad-interval` CLI flags (opt-in, default off), `--lr-decay-iters`
  (decouples the cosine-decay horizon from `--max-iters`, default None =
  same as `--max-iters`, old behavior unchanged), `--optimizer {adamw,sgd}`
  (default `adamw`, unchanged behavior; `sgd` uses `momentum=0.9` with the
  same decay/no_decay param grouping), `--adam-eps` (default `1e-8`,
  torch's own default) and `--adam-beta2` (default `0.95`, this
  codebase's prior default), both wired into `AdamW(...)`.
- New data/checkpoints (all gitignored under `data/`, not checked in —
  regenerate via the commands above): `data/word_big4096/` (VOCAB=4096
  tokenizer + train/val bins), `data/ckpt_word_big4096*.pt` (16 checkpoints,
  one per run above), `data/states_word_big4096*.jsonl` (matching
  per-eval logs), `data/gradlog_word_big4096.jsonl` (Attempt 5's per-layer
  gradient-norm trace), `data/ckpts/ckpt_*.pt` (shared per-eval snapshot
  dir — see the overwrite caveat under Attempt 2). Attempt 12 reuses the
  existing `data/word_stream16/` tokenizer/bins (the deployment shape's
  own prepared data, not regenerated) and adds
  `data/ckpt_word16_epsrecipe.pt` / `data/ckpt_word16_epsrecipe_23k.pt`
  + matching `data/states_word16_epsrecipe*.jsonl` — deliberately
  separate filenames from the real deployed `data/ckpt_word16.pt`, which
  this attempt does not touch or replace. Attempt 13 adds
  `data/ckpt_word16_reproduce.pt` + `data/states_word16_reproduce.jsonl`
  (same deal — separate filename, doesn't touch the real deployed
  checkpoint). Attempt 14 adds `data/ckpt_word16_epsmild1e5.pt` /
  `data/ckpt_word16_epsmild1e4.pt` / `data/ckpt_word16_epsmild3e4.pt` +
  matching `data/states_word16_epsmild{1e5,1e4,3e4}.jsonl`, plus a
  reproducibility-check rerun `data/ckpt_word16_v2.pt` /
  `data/states_word16_v2.jsonl` — all standalone candidates, deliberately
  not promoted over the real deployed `data/ckpt_word16.pt`.
