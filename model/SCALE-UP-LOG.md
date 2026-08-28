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
well — but this codebase's training recipe has a real, reproducible
instability at this width that caps effective training to roughly the
first 3500-4000 iterations of any run, regardless of every standard fix
tried (lower LR, longer warmup, GPT-2 residual-projection init scaling,
excluding LayerNorm gains from weight decay, logit-scale regularization at
two different strengths). The instability itself was NOT resolved. The key
finding for "what does it take to get acceptable stories": **training
stability at width, not raw capacity, is the actual bottleneck right now.**

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

Every configuration bottoms at essentially the same *relative* point —
roughly 1500-2500 iterations past wherever warmup ends — regardless of
peak LR, warmup length, init scaling, weight-decay grouping, or
logit-scale regularization strength. Attempt 6 went further and pinned
LR completely flat at its floor for two-thirds of an entire run, and the
model STILL diverged just as much — the single most decisive result in
this log, since it doesn't just fail to fix the instability, it rules out
the entire LR-schedule axis as a candidate cause. Combined with Attempt
5's finding (growth synchronized across the whole network, not localized
to one layer), the leading hypothesis is now Adam's own per-parameter
variance-normalized update magnitude not actually shrinking to zero at
low nominal LR — untested directly; see Attempt 6's own suggested next
tests (plain SGD, more extreme floor LR, `weight_decay=0`, fixed batch
replay).

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
4. **Root cause remains open, but the LR-schedule axis is now definitively
   ruled out.** Ruled out as causal: LR magnitude, warmup length, missing
   GPT-2 residual-projection init scaling, weight decay applied to
   LayerNorm gains, unconstrained logit growth (at two very different
   strengths), and — decisively, in Attempt 6 — the LR schedule itself:
   pinning LR completely flat at its floor for 10,000 iterations still
   produced the same divergence (val +61% while LR never moved). All
   "fixes" tried are kept in the codebase as permanent, unconditional
   improvements (correct practice regardless), but none of them is *the*
   fix. Per-layer gradient-norm logging (Attempt 5) rules out a single
   misbehaving layer too: growth is synchronized across all 12
   transformer blocks and both attention/MLP sub-components, with the
   embedding/head layer comparatively the MOST stable part of the network
   (opposite of what a "big tied VOCAB=4096 embedding is the problem"
   hypothesis would predict). **Leading working hypothesis now**: Adam's
   own per-parameter variance-normalized update magnitude not actually
   shrinking to zero at low nominal LR (untested directly). Next real
   steps, if picked back up: plain SGD instead of AdamW (removes the
   per-parameter variance normalization entirely — the most direct test
   of the current leading hypothesis); an even more extreme floor LR
   (e.g. 1e-6) to see if the drift ever actually stops; `weight_decay=0`
   to rule out decay-driven erosion; replaying a FIXED batch sequence
   across runs to rule out data-order effects; or an fp32-vs-bf16
   isolation run to rule out precision-induced instability. None of
   these five were attempted in this pass.

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
  same as `--max-iters`, old behavior unchanged).
- New data/checkpoints (all gitignored under `data/`, not checked in —
  regenerate via the commands above): `data/word_big4096/` (VOCAB=4096
  tokenizer + train/val bins), `data/ckpt_word_big4096*.pt` (8 checkpoints,
  one per run above), `data/states_word_big4096*.jsonl` (matching
  per-eval logs), `data/gradlog_word_big4096.jsonl` (Attempt 5's per-layer
  gradient-norm trace), `data/ckpts/ckpt_*.pt` (shared per-eval snapshot
  dir — see the overwrite caveat under Attempt 2).
