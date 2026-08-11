"""Train the Kevin GPT on a Kevinised corpus.

Device auto-detects (CUDA > MPS > CPU) so the same script trains on the XPS
3050 Ti and the M1. Mixed precision is used where the device supports it.

Quick throughput probe before committing to a real run:
    python -m model.train --smoke 50

Real (proof-of-life) run on the validation corpus:
    python -m model.train --max-iters 4000 --eval-interval 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import load_split, prepare
from .gpt import GPT, GPTConfig


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def amp_dtype(device: str):
    """Pick an autocast dtype, or None to disable autocast on this device."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16  # bf16 autocast is flaky on MPS; fp16 is fine here
    return None  # cpu: plain fp32


def get_batch(data: np.ndarray, block: int, batch: int, device: str):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def cosine_lr(it: int, lr: float, warmup: int, total: int, min_lr_frac: float = 0.1):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > total:
        return lr * min_lr_frac
    ratio = (it - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return lr * (min_lr_frac + (1 - min_lr_frac) * coeff)


@torch.no_grad()
def estimate_loss(model, splits, block, batch, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block, batch, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def sample(model, meta, device, n_tokens=200, prompt="\n", seed=None):
    from .data import decode
    if seed is not None:
        torch.manual_seed(seed)  # fixed seed -> comparable samples across evals
    stoi, itos = meta["stoi"], meta["itos"]
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(ids, n_tokens, temperature=0.8, top_k=40)[0].tolist()
    return decode(out, itos).replace("\n", " | ")


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.train", description="Train the Kevin GPT.")
    p.add_argument("--arch", choices=["gpt", "mamba2"], default="gpt",
                   help="gpt = the nanoGPT-lineage transformer (default); "
                        "mamba2 = the SSD recurrent model (model.mamba2), "
                        "constant decode state instead of a KV cache.")
    p.add_argument("--corpus", default="data/TinyStories-valid.kevin.txt")
    p.add_argument("--data-dir", default="data/char")
    p.add_argument("--out", default="data/ckpt.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--states", default="data/states.jsonl",
                   help="append (iter, losses, sample) to this JSONL each eval; "
                        "render with model.evolution. Empty to disable.")
    p.add_argument("--snapshot-dir", default="data/ckpts",
                   help="save a checkpoint at every eval here (ckpt_<iter>.pt) so "
                        "you can roll back to any point. Empty to disable.")
    p.add_argument("--compile", action="store_true", help="torch.compile (CUDA).")
    p.add_argument("--no-amp", action="store_true",
                   help="train in plain fp32 (no autocast). bf16's 8-bit mantissa "
                        "is a suspect in the mamba2 late-run loss drift.")
    p.add_argument("--smoke", type=int, default=0,
                   help="run N iters, report tokens/sec and a time estimate, exit.")
    p.add_argument("--qat", action="store_true",
                   help="train the Brevitas INT4-weight / INT8-activation QAT model "
                        "(model.qgpt.QGPT) instead of the FP GPT. Warm-start from an FP "
                        "checkpoint with --init-from for the standard recipe.")
    p.add_argument("--init-from", default=None, metavar="CKPT",
                   help="load weights from this checkpoint before training. With --qat, "
                        "FP weights are remapped into the QAT module and Brevitas's scale "
                        "params initialise fresh (load is strict=False).")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    dtype = None if args.no_amp else amp_dtype(device)
    torch.manual_seed(1337)
    print(f"device={device}  amp={dtype}")

    meta = prepare(args.corpus, args.data_dir)
    print(f"vocab={meta['vocab_size']}  train_chars={meta['train_chars']:,}")
    splits = {"train": load_split(args.data_dir, "train"),
              "val": load_split(args.data_dir, "val")}

    if args.arch == "mamba2":
        if args.qat:
            raise SystemExit("--qat is transformer-only for now (no QMamba2 yet)")
        from .mamba2 import Mamba2, Mamba2Config
        cfg = Mamba2Config(
            block_size=args.block_size, vocab_size=meta["vocab_size"],
            n_layer=args.n_layer, d_model=args.n_embd, dropout=args.dropout,
        )
        model = Mamba2(cfg).to(device)
        print(f"params={model.num_params() / 1e6:.2f}M  "
              f"(mamba2: state/stream {model.state_bytes(1)/1024:.0f} KB int8, constant in T)")
        if args.init_from:
            ck = torch.load(args.init_from, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"warm-start from {args.init_from} (iter {ck.get('iter')})")
    elif args.qat:
        from .qgpt import QGPT, load_fp_into_qat
        cfg = GPTConfig(
            block_size=args.block_size, vocab_size=meta["vocab_size"],
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout,
        )
        # Build on CPU, warm-start, run one dry forward to materialise
        # Brevitas's lazy scale buffers, THEN move to device. .to() picks up
        # everything (params + buffers, lazy or not) only after they exist.
        model = QGPT(cfg)
        print(f"params={model.num_params() / 1e6:.2f}M  (QAT: INT4 weights, INT8 acts)")
        if args.init_from:
            ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
            missing, unexpected = load_fp_into_qat(model, ck["model"])
            n_scale_missing = sum(1 for k in missing if "input_quant" in k)
            n_real_missing = len(missing) - n_scale_missing
            print(f"warm-start from {args.init_from} (iter {ck.get('iter')}): "
                  f"transferred OK, {n_scale_missing} fresh quant-scale params, "
                  f"{n_real_missing} unmatched, {len(unexpected)} unexpected")
            if n_real_missing or unexpected:
                print(f"  unmatched: {[k for k in missing if 'input_quant' not in k]}")
                print(f"  unexpected: {unexpected}")
        with torch.no_grad():
            model(torch.zeros((1, 1), dtype=torch.long))
        model.to(device)
    else:
        cfg = GPTConfig(
            block_size=args.block_size, vocab_size=meta["vocab_size"],
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout,
        )
        model = GPT(cfg).to(device)
        print(f"params={model.num_params() / 1e6:.2f}M")
        if args.init_from:
            ck = torch.load(args.init_from, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"warm-start from {args.init_from} (iter {ck.get('iter')})")
    if args.compile and device == "cuda":
        model = torch.compile(model)

    # decay only matrix-shaped params: norms / biases / scalars (mamba2's
    # A_log, dt_bias, D) must not be pulled toward zero — decaying them is
    # what drove the rung-0 run's late loss drift (doc 8 §5 M1).
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    use_amp = dtype is not None
    ctx = (torch.autocast(device_type=device, dtype=dtype) if use_amp
           else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))

    block, batch = args.block_size, args.batch_size

    # --- smoke: measure throughput, extrapolate, and stop ---
    if args.smoke:
        model.train()
        for _ in range(3):  # warmup (kernels compile / lazy init)
            x, y = get_batch(splits["train"], block, batch, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.smoke):
            x, y = get_batch(splits["train"], block, batch, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        it_s = args.smoke / dt
        tok_s = it_s * batch * block
        print(f"\nthroughput: {it_s:.1f} it/s  |  {tok_s/1e3:.0f}k tokens/s")
        for target in (2000, 4000, 8000):
            print(f"  {target} iters ~= {target/it_s/60:.1f} min "
                  f"({target*batch*block/1e6:.0f}M tokens seen)")
        return

    # --- real training loop ---
    if args.states:
        os.makedirs(os.path.dirname(args.states) or ".", exist_ok=True)
        open(args.states, "w").close()  # truncate any previous run's states
    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    def save_ckpt(path, it, val):
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "meta": meta, "iter": it, "val": val, "qat": args.qat,
                    "arch": args.arch}, path)

    best_val = float("inf")
    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        lr = cosine_lr(it, args.lr, args.warmup, args.max_iters)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, block, batch, device, args.eval_iters)
            el = time.perf_counter() - t0
            # printed sample is free-running; logged sample is fixed-seed so the
            # garbage->text progression reflects the model, not the dice.
            shown = sample(model, meta, device, 180)
            logged = sample(model, meta, device, 180, seed=1234)
            print(f"iter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| lr {lr:.1e} | {el/60:.1f} min")
            print("   sample:", shown)
            if args.states:
                with open(args.states, "a") as f:
                    f.write(json.dumps({
                        "iter": it,
                        "train_loss": round(losses["train"], 4),
                        "val_loss": round(losses["val"], 4),
                        "minutes": round(el / 60, 2),
                        "lr": lr,
                        "sample": logged,
                    }) + "\n")
            # per-eval snapshot for manual rollback (e.g. if val turns up ~2000)
            if args.snapshot_dir:
                save_ckpt(os.path.join(args.snapshot_dir, f"ckpt_{it:05d}.pt"),
                          it, losses["val"])
            # best-val checkpoint = the automatic anti-overfit rollback
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_ckpt(args.out, it, best_val)
                print(f"   * new best val {best_val:.3f} -> {args.out}")

        if it == args.max_iters:
            break

        x, y = get_batch(splits["train"], block, batch, device)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    print(f"done. best val {best_val:.3f}, checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
