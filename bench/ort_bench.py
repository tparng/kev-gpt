"""Off-the-shelf laptop baseline: goformer KV decode under ONNX Runtime.

Exports one KV-cached decode step (token + past K/V -> logits + appended K/V),
then runs greedy single-stream generation through ORT CPU / CUDA EPs.

    python -m bench.ort_bench data/ckpt.fp.pt -n 512 --providers cpu,cuda
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.sample import load


class StepModel(torch.nn.Module):
    """One KV-cached decode step over all blocks (the fabric sequencer in torch)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, tok, pos, past_k, past_v):
        cfg = self.m.cfg
        nh, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        x = self.m.tok_emb(tok) + self.m.pos_emb(pos)
        ks, vs = [], []
        for li, blk in enumerate(self.m.blocks):
            h = blk.ln1(x)
            q, k, v = blk.attn.qkv(h).split(cfg.n_embd, dim=2)
            q = q.view(1, 1, nh, hd).transpose(1, 2)
            k = k.view(1, 1, nh, hd).transpose(1, 2)
            v = v.view(1, 1, nh, hd).transpose(1, 2)
            k = torch.cat([past_k[li], k], dim=2)
            v = torch.cat([past_v[li], v], dim=2)
            ks.append(k)
            vs.append(v)
            y = F.scaled_dot_product_attention(q, k, v)
            y = y.transpose(1, 2).reshape(1, 1, cfg.n_embd)
            x = x + blk.attn.proj(y)
            x = x + blk.mlp(blk.ln2(x))
        logits = self.m.head(self.m.ln_f(x))
        return logits, torch.stack(ks), torch.stack(vs)


def export(ckpt, path):
    model, meta, _, _ = load(ckpt, "cpu")
    cfg = model.cfg
    nh, hd = cfg.n_head, cfg.n_embd // cfg.n_head
    step = StepModel(model).eval()
    tok = torch.zeros((1, 1), dtype=torch.long)
    pos = torch.zeros((1,), dtype=torch.long)
    pk = torch.zeros((cfg.n_layer, 1, nh, 1, hd))
    pv = torch.zeros_like(pk)
    torch.onnx.export(
        step, (tok, pos, pk, pv), path,
        input_names=["tok", "pos", "past_k", "past_v"],
        output_names=["logits", "out_k", "out_v"],
        dynamic_axes={"past_k": {3: "T"}, "past_v": {3: "T"},
                      "out_k": {3: "T1"}, "out_v": {3: "T1"}},
        opset_version=17)
    return cfg


def bench(path, cfg, provider, n, ctx, threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=[provider])
    nh, hd = cfg.n_head, cfg.n_embd // cfg.n_head

    def gen(n_tokens):
        tok = np.zeros((1, 1), dtype=np.int64)
        pk = np.zeros((cfg.n_layer, 1, nh, 0, hd), dtype=np.float32)
        pv = pk.copy()
        for i in range(n_tokens):
            pos = np.array([min(i, cfg.block_size - 1)], dtype=np.int64)
            logits, pk, pv = sess.run(None, {"tok": tok, "pos": pos,
                                             "past_k": pk[:, :, :, -(ctx - 1):],
                                             "past_v": pv[:, :, :, -(ctx - 1):]})
            tok = logits[0, -1].argmax()[None, None].astype(np.int64)

    gen(8)                                  # warmup
    t0 = time.perf_counter()
    gen(n)
    dt = time.perf_counter() - t0
    return n / dt


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench.ort_bench")
    p.add_argument("ckpt", nargs="?", default="data/ckpt.fp.pt")
    p.add_argument("-n", type=int, default=512)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--providers", default="cpu,cuda")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args(argv)

    path = os.path.join("data", "goformer_step.onnx")
    cfg = export(a.ckpt, path)
    prov = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
    for name in a.providers.split(","):
        r = bench(path, cfg, prov[name], a.n, a.ctx, a.threads)
        print(f"ORT_BENCH provider={name:4s} ctx={a.ctx} n={a.n} "
              f"threads={a.threads} tok_s={r:,.1f}")


if __name__ == "__main__":
    main()
