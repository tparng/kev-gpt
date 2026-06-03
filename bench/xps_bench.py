"""XPS15 laptop baseline: same goformer checkpoint, fastest single-stream decode.

Measures tok/s on CPU and CUDA, full-recompute vs KV-cached decode — the laptop
counterpart of the Kria fabric tok/s (single stream, B=1, greedy).

    python -m bench.xps_bench data/ckpt.fp.pt [--ctx 256] [-n 512] [--device cuda]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from model.sample import load


@torch.no_grad()
def full_decode(model, idx, n, ctx):
    for _ in range(n):
        logits, _ = model(idx[:, -ctx:])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


class KVBlockState:
    """Per-block K/V cache, [B, n_head, T, head_dim], preallocated."""

    def __init__(self, cfg, device, dtype):
        hd = cfg.n_embd // cfg.n_head
        s = (1, cfg.n_head, cfg.block_size, hd)
        self.k = torch.zeros(s, device=device, dtype=dtype)
        self.v = torch.zeros(s, device=device, dtype=dtype)


@torch.no_grad()
def kv_decode(model, idx, n, ctx):
    """Process only the new token each step (the fabric sequencer's algorithm)."""
    cfg = model.cfg
    dev, dtype = idx.device, model.tok_emb.weight.dtype
    nh, hd = cfg.n_head, cfg.n_embd // cfg.n_head
    caches = [KVBlockState(cfg, dev, dtype) for _ in model.blocks]

    out = idx
    T = 0
    tok = idx
    for step in range(idx.size(1) + n - 1):
        if step < idx.size(1):
            tok = idx[:, step:step + 1]
        else:
            tok = out[:, -1:]
        pos = min(T, cfg.block_size - 1)
        x = model.tok_emb(tok) + model.pos_emb(torch.tensor([pos], device=dev))
        for blk, kvc in zip(model.blocks, caches):
            h = blk.ln1(x)
            q, k, v = blk.attn.qkv(h).split(cfg.n_embd, dim=2)
            q = q.view(1, 1, nh, hd).transpose(1, 2)
            kvc.k[:, :, pos] = k.view(1, nh, hd)
            kvc.v[:, :, pos] = v.view(1, nh, hd)
            y = F.scaled_dot_product_attention(
                q, kvc.k[:, :, :pos + 1], kvc.v[:, :, :pos + 1])
            y = y.transpose(1, 2).reshape(1, 1, cfg.n_embd)
            x = x + blk.attn.proj(y)
            x = x + blk.mlp(blk.ln2(x))
        T += 1
        if step >= idx.size(1) - 1:
            logits = model.head(model.ln_f(x))
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    return out


def bench(fn, model, idx, n, ctx, device):
    fn(model, idx, 8, ctx)                       # warmup
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(model, idx, n, ctx)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return n / dt, out


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench.xps_bench")
    p.add_argument("ckpt", nargs="?", default="data/ckpt.fp.pt")
    p.add_argument("-n", type=int, default=512)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--devices", default="cpu,cuda")
    a = p.parse_args(argv)

    for device in a.devices.split(","):
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda unavailable; skipping")
            continue
        model, meta, _, _ = load(a.ckpt, device)
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        modes = [("full", full_decode), ("kv", kv_decode)]
        for name, fn in modes:
            r, out = bench(fn, model, prompt, a.n, a.ctx, device)
            print(f"XPS_BENCH device={device:4s} mode={name:4s} ctx={a.ctx} "
                  f"n={a.n} tok_s={r:,.1f}")


if __name__ == "__main__":
    main()
