"""KVarN-style low-bit KV-cache quantization — go/no-go for on-chip KV at N=16+.

Context (fabric/stage3/research/KV-DDR-100K.md): past ~6-8 streams the per-stream
KV cache (64 KB at INT8) is what stops fitting on-chip, forcing the KV-DDR rung
whose DDR budget co-limits ~80k aggregate. KVarN (huawei-csl/KVarN, arXiv:2606.03458)
ships 4-bit-key / 2-bit-value KV quantization for vLLM: Hadamard rotation along
channels to spread outliers + variance-normalized asymmetric quantization
(per-channel key scales, per-token value scales). If that fidelity holds on OUR
3M-param model, KV drops to ~24 KB/stream and 16-24 streams keep KV on-chip —
deleting the KV-DDR project instead of feeding it.

This experiment fake-quantizes K/V AT WRITE TIME (each position's bits are fixed
when produced, exactly as stored RAM bits would be — no history re-quantization)
inside the bit-true KV decoder, on the TRAINED QAT export, and measures:

  1. teacher-forced: argmax agreement + logit cosine vs the unquantized decoder
     fed the same reference token stream (per-step fidelity, no error feedback);
  2. free-running: greedy generation, first-divergence position + stream match
     (what a precomputed speculative stream would actually blit).

Streaming-honest schemes only (everything below is implementable in fabric):
  - per-position scales: each new k/v vector gets its own asymmetric scale/zero
    per head (KVarN's "per-token value scales"; for K it is the streaming-native
    stand-in for their per-channel tile scales);
  - per-channel running scales for K: causal running min/max per channel — the
    scales a position is written with never see the future;
  - per-head Hadamard rotation (orthonormal, hd=64): k'=H k, q'=H q preserves
    q.k exactly in float; v'=H v with output unrotation. In fabric H is +-1
    adds/subtracts, no DSPs.

    python -m model.exp_kvarn                  # full ladder on the QAT export
    python -m model.exp_kvarn --gen 32 --prompts 4   # quicker look

Every number printed is MEASURED by this script on fabric/export/goformer.npz.
"""

from __future__ import annotations

import argparse

import numpy as np

from .goformer_full import load_params
from .goformer_kv import KVDecoder


# ---------------------------------------------------------------- quantizers
def fake_quant_asym(x: np.ndarray, bits: int, lo=None, hi=None) -> np.ndarray:
    """Asymmetric uniform fake-quant of x to `bits`, vectorised over leading axes.
    lo/hi default to per-call (per-position) min/max along the last axis."""
    if lo is None:
        lo = x.min(axis=-1, keepdims=True)
    if hi is None:
        hi = x.max(axis=-1, keepdims=True)
    qmax = (1 << bits) - 1
    scale = np.maximum((hi - lo) / qmax, 1e-12)
    q = np.clip(np.round((x - lo) / scale), 0, qmax)
    return q * scale + lo


def hadamard(n: int) -> np.ndarray:
    """Orthonormal Sylvester Hadamard (n a power of 2): H @ H.T == I."""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


# ------------------------------------------------------- quantizing decoder
class QuantKVDecoder(KVDecoder):
    """KVDecoder whose cache holds what low-bit storage would actually hold:
    each position is fake-quantized ONCE, at write, with only-causal scales."""

    def __init__(self, params, kbits=4, vbits=2, rotate=False, k_chan=False):
        super().__init__(params)
        self.kbits, self.vbits = kbits, vbits
        self.rotate, self.k_chan = rotate, k_chan
        d = params["tok_emb"].shape[1]
        self.hd = d // params["n_head"]
        self.H = hadamard(self.hd) if rotate else None
        nb = len(params["blocks"])
        # per-layer running per-channel K min/max (causal calibration)
        self.k_lo = [None] * nb
        self.k_hi = [None] * nb

    def _rot(self, x: np.ndarray) -> np.ndarray:
        """Per-head Hadamard: x (C,) -> (C,) rotated within each head."""
        nh = self.n_head
        return (x.reshape(nh, self.hd) @ self.H.T).reshape(-1)

    def _quant_k(self, k: np.ndarray, bi: int) -> np.ndarray:
        if self.kbits >= 16:
            return k
        if self.k_chan:
            # causal per-channel scales: update running min/max with THIS
            # position, then quantize it — bits written never depend on the future
            self.k_lo[bi] = k.copy() if self.k_lo[bi] is None else np.minimum(self.k_lo[bi], k)
            self.k_hi[bi] = k.copy() if self.k_hi[bi] is None else np.maximum(self.k_hi[bi], k)
            lo, hi = self.k_lo[bi], self.k_hi[bi]
            return fake_quant_asym(k, self.kbits, lo=lo, hi=np.maximum(hi, lo + 1e-9))
        return fake_quant_asym(k.reshape(self.n_head, self.hd), self.kbits).reshape(-1)

    def _quant_v(self, v: np.ndarray) -> np.ndarray:
        if self.vbits >= 16:
            return v
        return fake_quant_asym(v.reshape(self.n_head, self.hd), self.vbits).reshape(-1)

    def _attn_step(self, h_t, blk, bi):
        C = h_t.shape[0]
        nh, hd = self.n_head, C // self.n_head
        qkv = self.gf.qlin(h_t[None, :], blk["qkv"])[0]
        q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]

        if self.rotate:
            q, k, v = self._rot(q), self._rot(k), self._rot(v)
        k = self._quant_k(k, bi)
        v = self._quant_v(v)

        kc = k[None, :] if self.k_cache[bi] is None else np.vstack([self.k_cache[bi], k[None, :]])
        vc = v[None, :] if self.v_cache[bi] is None else np.vstack([self.v_cache[bi], v[None, :]])
        self.k_cache[bi], self.v_cache[bi] = kc, vc
        T = kc.shape[0]

        qh = q.reshape(1, nh, hd).transpose(1, 0, 2)
        kh = kc.reshape(T, nh, hd).transpose(1, 0, 2)
        vh = vc.reshape(T, nh, hd).transpose(1, 0, 2)
        scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(hd)      # q'.k' == q.k (H orthonormal)
        attn = self.gf._smax(scores) @ vh                        # rotated-V output
        y = attn.transpose(1, 0, 2).reshape(C)
        if self.rotate:                                          # unrotate: y = H^T y'
            y = (y.reshape(nh, hd) @ self.H).reshape(-1)
        return self.gf.qlin(y[None, :], blk["proj"])[0]


# ------------------------------------------------------------------- metrics
def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def kv_bytes_per_stream(kbits, vbits, d=256, n_layer=4, tmax=32, n_head=4):
    """Storage incl. fp16 scale+zero per head per position (K and V each)."""
    per_pos = d * (kbits + vbits) / 8 + 2 * n_head * 2 * 2
    return per_pos * n_layer * tmax


def run_variant(p, name, prompts, ref, n_gen, **cfg):
    tf_agree, tf_cos, fr_div = [], [], []
    for pi, prompt in enumerate(prompts):
        ref_stream, ref_logits = ref[pi]
        # teacher-forced: same tokens in, per-step logits out
        dec = QuantKVDecoder(p, **cfg)
        agree = 0
        for t, tok in enumerate(ref_stream[:-1]):
            lg = dec.step(int(tok))
            if t >= len(prompt) - 1:                 # generation region only
                gi = t - (len(prompt) - 1)
                agree += int(np.argmax(lg) == ref_stream[len(prompt) + gi])
                tf_cos.append(cosine(lg, ref_logits[gi]))
        tf_agree.append(agree / n_gen)
        # free-running greedy
        stream = QuantKVDecoder(p, **cfg).generate_greedy(list(prompt), n_gen, p["block_size"])
        gen_q, gen_r = stream[len(prompt):], ref_stream[len(prompt):]
        div = next((i for i, (a, b) in enumerate(zip(gen_q, gen_r)) if a != b), n_gen)
        fr_div.append(div)

    kb = kv_bytes_per_stream(cfg.get("kbits", 16), cfg.get("vbits", 16)) / 1024
    print(f"KVARN {name:34s} kv={kb:5.1f}KB/stream  "
          f"tf_argmax={np.mean(tf_agree):6.1%}  tf_cos={np.mean(tf_cos):.6f}  "
          f"freerun_div@={np.mean(fr_div):5.1f}/{n_gen} (min {min(fr_div)})")
    return np.mean(tf_agree), np.mean(tf_cos), np.mean(fr_div)


# natural telegraphic prompts (the training distribution): random-token prompts
# put the model out of distribution and 3/4 of its greedy generations collapse
# to one repeated token — trivially matchable, so they would inflate the result.
NATURAL_PROMPTS = [
    "once upon time there be little girl name",
    "one day big dog run to park and",
    "kevin go fast because few word do",
    "boy see bird in tree he want",
    "mum say no but tom take cake",
    "sun shine cat sleep on warm",
    "girl find shiny stone she show",
    "they walk to shop buy red",
]


def val_nll(p, text_ids, chunks, clen, **cfg):
    """Teacher-forced mean next-token NLL over `chunks` random chunks of clen."""
    rng = np.random.default_rng(7)
    nll, n = 0.0, 0
    for _ in range(chunks):
        s = int(rng.integers(0, len(text_ids) - clen - 1))
        ids = text_ids[s:s + clen + 1]
        dec = QuantKVDecoder(p, **cfg) if cfg else KVDecoder(p)
        for t in range(clen):
            lg = dec.step(int(ids[t]))
            lg = lg - lg.max()
            nll -= lg[ids[t + 1]] - np.log(np.exp(lg).sum())
            n += 1
    return nll / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="fabric/export/goformer.npz")
    ap.add_argument("--tok", default="fabric/export/tokenizer.json")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--val", default="data/sample.kevin.txt",
                    help="held-out text for the NLL-delta check ('' to skip)")
    ap.add_argument("--val-chunks", type=int, default=6)
    ap.add_argument("--val-len", type=int, default=96)
    args = ap.parse_args(argv)

    p = load_params(args.npz)
    import json
    itos = json.load(open(args.tok))["itos"]
    stoi = {c: i for i, c in enumerate(itos)}
    prompts = [[stoi.get(c, 0) for c in s] for s in NATURAL_PROMPTS[:args.prompts]]

    # unquantized reference: greedy stream + its generation-region logits.
    # Report each generation's token diversity so a degenerate (looping)
    # reference can't silently inflate the match numbers.
    ref = []
    for pi, prompt in enumerate(prompts):
        dec = KVDecoder(p)
        logits = None
        for tok in prompt:
            logits = dec.step(int(tok))
        stream, lgs = list(prompt), []
        for _ in range(args.gen):
            lgs.append(logits)
            nxt = int(np.argmax(logits))
            stream.append(nxt)
            logits = dec.step(nxt)
        ref.append((stream, lgs))
        gen = stream[len(prompt):]
        text = "".join(itos[t] for t in gen).replace("\n", " ")
        print(f"#   ref[{pi}] unique={len(set(gen)):2d}/{args.gen}: "
              f"{NATURAL_PROMPTS[pi][:24]!r} -> {text[:48]!r}")
    print(f"# exp_kvarn on {args.npz}: {args.prompts} natural prompts, "
          f"{args.gen} greedy tokens, INT8-KV baseline = 64.0KB/stream (research-note figure)")

    ladder = [
        ("K8/V8 per-pos",                dict(kbits=8, vbits=8)),
        ("K4/V4 per-pos",                dict(kbits=4, vbits=4)),
        ("K4/V2 per-pos (KVarN cfg)",    dict(kbits=4, vbits=2)),
        ("K4/V2 + Hadamard",             dict(kbits=4, vbits=2, rotate=True)),
        ("K4/V2 + Had + K chan-scales",  dict(kbits=4, vbits=2, rotate=True, k_chan=True)),
        ("K4/V4 + Hadamard",             dict(kbits=4, vbits=4, rotate=True)),
        ("K3/V2 + Hadamard",             dict(kbits=3, vbits=2, rotate=True)),
    ]
    results = {}
    for name, cfg in ladder:
        results[name] = run_variant(p, name, prompts, ref, args.gen, **cfg)

    # quality delta on held-out keviniserd text: NLL is the honest "did the
    # model get worse" measure — stream divergence alone conflates "different"
    # with "degraded".
    nlls = {}
    if args.val:
        with open(args.val, encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = np.array([stoi.get(c, 0) for c in text[:200_000]], dtype=np.int64)
        base = val_nll(p, ids, args.val_chunks, args.val_len)
        print(f"# val NLL ({args.val_chunks}x{args.val_len} tok of {args.val}): "
              f"unquantized = {base:.4f} nats/tok")
        for name, cfg in ladder:
            q = val_nll(p, ids, args.val_chunks, args.val_len, **cfg)
            nlls[name] = (q - base) / base
            print(f"KVARN_NLL {name:34s} nll={q:.4f}  delta={q - base:+.4f} "
                  f"({(q - base) / base:+.2%})")

    # verdict: (a) does KVarN's headline 4/2 config survive at 3M params?
    # (b) regardless, the smallest config with <=1% held-out NLL delta is the
    # quality-intact compression this model actually supports.
    if nlls:
        kvarn_ok = nlls["K4/V2 + Hadamard"] <= 0.01
        print(f"KVARN_CFG_AT_3M " + ("GO" if kvarn_ok else "NO_GO") +
              f" (K4/V2+Had NLL delta {nlls['K4/V2 + Hadamard']:+.2%}; plain K4/V2 "
              f"{nlls['K4/V2 per-pos (KVarN cfg)']:+.2%})")
        ok = [(kv_bytes_per_stream(c.get('kbits', 16), c.get('vbits', 16)), n)
              for n, c in ladder if nlls[n] <= 0.01]
        if ok:
            kb, best = min(ok)
            print(f"KVARN_VERDICT GO {best} — NLL delta {nlls[best]:+.2%}, "
                  f"{kb / 1024:.0f}KB/stream vs 64KB INT8 "
                  f"({64 / (kb / 1024):.2f}x KV-DDR traffic cut)")
        else:
            print("KVARN_VERDICT NO_GO — no config holds <=1% NLL delta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
