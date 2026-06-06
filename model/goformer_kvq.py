"""Fixed-point KV-quant decode reference — the bit-true contract for the kv_dma RTL.

Demo P1 (`5-demo-prd.md`): "K4/V4+Hadamard quant-at-write RTL gated against an
extended goformer_kv reference." `model/exp_kvarn.py` *measured* the scheme in FLOAT
(fake-quant: float->round->float) and landed K4/V4 + per-head Hadamard at **+0.72%
held-out NLL, 1.78x compression (MEASURED)**. That experiment proves the *quality*;
it is not the thing the RTL is gated against, because the RTL carries no floats.

This module is the bit-honest counterpart: it takes the pure-INTEGER sequencer
(`fabric/stage3/seq_ref.IntSequencer`, already gated token-identical to the float
goformer) and replaces ONLY the K/V cache write/read with the fixed-point quantizer
the fabric `kv_dma` brick will actually compute — integer Hadamard butterfly + shift,
asymmetric 4-bit pack, integer dequant — leaving every downstream score/softmax/ctx
op untouched and bit-true. So two gates bind the hardware:

    float goformer_kv  --(MEASURED +0.72% NLL, exp_kvarn)-->  the quality is real
    seq_ref IntSequencer (integer, token-identical to float)
        |  override K/V cache write/read with the integer quantizer below
        v
    goformer_kvq IntKVQSequencer  --(bit-exact .mem words)-->  rtl/kv_dma.sv

and the two key properties this file's gate proves:
  (a) FULL-PRECISION (kbits=vbits=16, no rotate): IntKVQSequencer is BIT-IDENTICAL
      to the plain IntSequencer over a multi-token decode (same logits, same stream)
      — i.e. the quant path is a transparent pass-through when disabled, so it cannot
      have changed the datapath;
  (b) K4/V4 + Hadamard: the held-out NLL delta vs the FP integer path matches the
      exp_kvarn float result (re-MEASURED here, reported as an honest number).

------------------------------------------------------------------------------
PINNED Q-FORMAT (the contract the kv_dma RTL is built to)
------------------------------------------------------------------------------
The integer sequencer already stores K/V as signed **Q.16** (`seq_ref.VFRAC=16`)
per channel — that is the value that today sits in the on-chip K/V scratch. The
quantizer operates on THAT word:

  * Hadamard (optional, per head, hd=64): orthonormal H = (1/8) * H_raw, H_raw the
    Sylvester +-1 matrix. In fabric this is a radix-2 butterfly (log2(64)=6 stages
    of adds/subtracts, NO DSPs) on the Q.16 vector, then a single >>3 (since
    sqrt(64)=8) to apply the 1/8 normalisation. Applied to q, k, v alike; q.k is
    preserved EXACTLY in integer arithmetic up to the butterfly's >>3 rounding, and
    V is un-rotated after the context sum with the same butterfly (H is symmetric and
    H^T = H, so unrotate == rotate). Rounding of each >>3 is round-half-away-from-zero
    (`seq_ref.rsh_round`) so it is bit-reproducible in RTL.

  * Asymmetric N-bit pack, per (head, position) — KVarN's "per-token" scale, the
    streaming-native choice (the bits a position is written with never depend on the
    future). Over the hd=64 channels of one head:
        lo = min(x), hi = max(x)         (signed Q.16 ints)
        qmax = (1<<bits) - 1
        scale = max( round_div( (hi-lo) , qmax ), 1 )    (Q.16 int, >=1 LSB)
        code  = clip( round_div( (x - lo) , scale ), 0, qmax )    (unsigned bits-wide)
    dequant at read:  x_hat = code * scale + lo   (back to Q.16 int, EXACT integer).
    All divides are round-half-away-from-zero integer divides (q_round_div) — the
    exact op the RTL divider does. `scale` and `lo` are the per-(head,pos) header.

------------------------------------------------------------------------------
DDR WORD LAYOUT (stream-major, position-row — KV-DDR-100K.md §2)
------------------------------------------------------------------------------
Chosen layout: **[stream][layer][K|V][position][header | packed-nibbles]**, i.e.
one contiguous DDR extent per (stream, layer, K|V) block, one ROW per position, so
a head's live window is one contiguous AXI INCR burst (the §2 burst-friendly access
the kv_dma FSM relies on). Within a position-row the bytes are head-major:

  row = [ head0_hdr | head0_codes | head1_hdr | ... | head{nh-1}_codes ]
  per head:  hdr  = lo (int16, signed Q.16 low 16b) || scale (uint16, Q.16)  -> 4 B
             codes= hd codes of `bits` each, packed LSB-first into bytes      -> hd*bits/8 B

For K4 (bits=4) one head = 4 hdr + 64*4/8 = 4 + 32 = 36 B; 4 heads = 144 B/position
for K, same for V at V4. (At full window TMAX=32 that is 32*144 = 4608 B per K block.)
The `ddr_words()` signal-exposure method emits exactly these per-position byte rows,
LSB-first nibble order, so the future `run_kv_*.py` gate can `$readmemh` a `.mem`
"DDR image" and compare word-for-word. `lo` is stored as the low 16 bits of the
signed Q.16 value (two's complement); the RTL sign-extends it back.

    python -m model.goformer_kvq                 # FP-identity + K4/V4+Hadamard NLL gate
    python -m model.goformer_kvq --gen 24        # quicker
"""

from __future__ import annotations

import argparse

import numpy as np

from fabric.stage3.seq_ref import (
    IntSequencer, build, rsh_round, q_round_div, sat, VFRAC, NHEAD, HEAD_DIM,
)
from model.goformer_full import load_params


# ============================================================ integer Hadamard
def _butterfly_hadamard(vec):
    """In-place radix-2 Walsh-Hadamard on a length-n (power of 2) integer vector.

    n=HEAD_DIM=64 -> 6 stages of pure +/- (the fabric butterfly, no DSPs). Returns
    H_raw @ vec where H_raw is the unnormalised Sylvester +-1 matrix. The 1/sqrt(n)
    normalisation is applied separately as a shift by the caller so the rounding is
    explicit and RTL-reproducible."""
    a = [int(v) for v in vec]
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a


# normalisation shift for the orthonormal H: H = H_raw / sqrt(n); sqrt(64)=8 -> >>3.
_HAD_SHIFT = int(round(np.log2(np.sqrt(HEAD_DIM))))   # = 3 for hd=64
assert (1 << (2 * _HAD_SHIFT)) == HEAD_DIM, "HEAD_DIM must be a perfect square power of two"


def hadamard_rotate_q16(x_q16, nh=NHEAD, hd=HEAD_DIM):
    """Per-head orthonormal Hadamard on a (nh*hd,) signed Q.16 integer vector.

    Butterfly (+/- only) then >>_HAD_SHIFT with round-half-away-from-zero. This is the
    EXACT integer op the fabric will do; q.k is preserved because the same H multiplies
    q and k (H^T H = I in float; in integer the only deviation is the per-stage >>3
    rounding, which the gate proves is harmless to the token stream)."""
    out = [0] * (nh * hd)
    for h in range(nh):
        base = h * hd
        raw = _butterfly_hadamard(x_q16[base:base + hd])
        for d in range(hd):
            out[base + d] = rsh_round(raw[d], _HAD_SHIFT)
    return out


# ============================================================ asym N-bit pack
def quant_head_asym(x_q16, bits):
    """Asymmetric per-head (per-token) quantiser of hd signed Q.16 ints to `bits`.

    Returns (codes, lo, scale): codes unsigned 0..(2^bits-1); lo signed Q.16 int;
    scale signed Q.16 int (>=1). All integer, round-half-away-from-zero — the RTL op.
    Dequant is `code*scale + lo` (exact integer)."""
    lo = int(min(x_q16))
    hi = int(max(x_q16))
    qmax = (1 << bits) - 1
    span = hi - lo
    scale = q_round_div(span, qmax) if span > 0 else 0
    if scale < 1:
        scale = 1                                  # >=1 LSB; degenerate constant head
    codes = [sat(q_round_div(int(v) - lo, scale), 0, qmax) for v in x_q16]
    return codes, lo, scale


def dequant_head(codes, lo, scale):
    """Inverse of quant_head_asym: codes -> signed Q.16 ints. Exact integer."""
    return [int(c) * int(scale) + int(lo) for c in codes]


# ============================================================ DDR byte packing
def _pack_codes_lsb(codes, bits):
    """Pack a list of `bits`-wide unsigned codes LSB-first into bytes (the .mem image
    byte order the kv_dma gate reads). Returns a list of uint8."""
    bitbuf = 0
    nbits = 0
    out = []
    for c in codes:
        bitbuf |= (int(c) & ((1 << bits) - 1)) << nbits
        nbits += bits
        while nbits >= 8:
            out.append(bitbuf & 0xFF)
            bitbuf >>= 8
            nbits -= 8
    if nbits:
        out.append(bitbuf & 0xFF)
    return out


def position_ddr_row(vec_q16, bits, nh=NHEAD, hd=HEAD_DIM):
    """Build the per-position DDR byte row for ONE K (or V) vector (nh*hd Q.16 ints,
    already rotated if Hadamard is on). Head-major: [lo16 | scale16 | packed codes]*nh.

    Returns (row_bytes, per_head) where per_head is a list of (codes, lo, scale) so the
    reference can dequant from EXACTLY the stored bits (no separate float path)."""
    row = []
    per_head = []
    for h in range(nh):
        base = h * hd
        codes, lo, scale = quant_head_asym(vec_q16[base:base + hd], bits)
        lo16 = lo & 0xFFFF                          # low 16b of signed Q.16 (RTL sign-extends)
        sc16 = int(scale) & 0xFFFF
        row += [lo16 & 0xFF, (lo16 >> 8) & 0xFF, sc16 & 0xFF, (sc16 >> 8) & 0xFF]
        row += _pack_codes_lsb(codes, bits)
        per_head.append((codes, lo, scale))
    return row, per_head


# ============================================================ the KVQ sequencer
class IntKVQSequencer(IntSequencer):
    """IntSequencer whose K/V cache holds what low-bit DDR storage would actually hold.

    Each new position's K and V Q.16 vector is (optionally) Hadamard-rotated, then
    asym-quantised to kbits/vbits ONCE at write (causal/per-token scales), and the
    cache stores the DEQUANTISED Q.16 vector — i.e. exactly the value a read from DDR
    reconstructs. Everything downstream (scores, softmax, ctx) is the parent's bit-true
    integer path; for the V un-rotation the parent's ctx sum is post-multiplied by the
    same Hadamard (H^T = H). kbits/vbits >= 16 disables that path entirely (transparent
    pass-through => bit-identical to IntSequencer)."""

    def __init__(self, params, cfg, kbits=4, vbits=4, rotate=True):
        self.kbits, self.vbits, self.rotate = kbits, vbits, rotate
        # per-position DDR byte rows, indexed [layer] -> list over positions of dict
        self.kv_ddr = None
        super().__init__(params, cfg)

    def reset(self):
        super().reset()
        nb = self.nblocks
        self.kv_ddr = [[] for _ in range(nb)]       # [layer][pos] -> {"k_row","v_row",...}

    def _kv_passthrough(self):
        return self.kbits >= 16 and self.vbits >= 16 and not self.rotate

    # --- override: quantise-at-write, store the dequantised Q.16 + the DDR row ----
    def _store_kv(self, k_q16, v_q16, bi):
        """k_q16/v_q16: lists of C signed Q.16 ints (the parent's qkv dequant output).
        Rotate (opt) + quantise + record the DDR row; return the (k,v) the cache holds."""
        if self._kv_passthrough():
            self.kv_ddr[bi].append({"k_row": None, "v_row": None})
            return k_q16, v_q16

        kr = hadamard_rotate_q16(k_q16) if self.rotate else list(k_q16)
        vr = hadamard_rotate_q16(v_q16) if self.rotate else list(v_q16)

        # full-precision (16-bit) channel keeps the rotated value verbatim (no pack)
        if self.kbits >= 16:
            k_row, k_heads = None, None
            k_stored = kr
        else:
            k_row, k_heads = position_ddr_row(kr, self.kbits)
            k_stored = []
            for h in range(NHEAD):
                k_stored += dequant_head(*k_heads[h])
        if self.vbits >= 16:
            v_row, v_heads = None, None
            v_stored = vr
        else:
            v_row, v_heads = position_ddr_row(vr, self.vbits)
            v_stored = []
            for h in range(NHEAD):
                v_stored += dequant_head(*v_heads[h])

        self.kv_ddr[bi].append({
            "k_row": k_row, "v_row": v_row,
            "k_rot_q16": kr, "v_rot_q16": vr,
            "k_deq_q16": k_stored, "v_deq_q16": v_stored,
        })
        return k_stored, v_stored

    # --- override attn step: identical to parent but with the KV-quant write hook --
    # and the V un-rotation folded into the ctx output.
    def _attn_step(self, x_q25, bi, sink=None):
        gamma = self.layers[(bi, "ln1")]
        y_q22 = self._ln(x_q25, gamma)
        ix = self._act_quant(y_q22, self.layers[(bi, "qkv")]["s_act"])
        y_int = self._gemv_int(self.layers[(bi, "qkv")], ix)
        qkv_q16 = self._dequant_to_q(self.layers[(bi, "qkv")], y_int, VFRAC)
        C = 256
        q = [int(v) for v in qkv_q16[:C]]
        k = [int(v) for v in qkv_q16[C:2 * C]]
        v = [int(v) for v in qkv_q16[2 * C:]]

        # Rotate q the same way as k so q.k is preserved (k is rotated inside _store_kv).
        if self.rotate and not self._kv_passthrough():
            q = hadamard_rotate_q16(q)

        k_stored, v_stored = self._store_kv(k, v, bi)
        self.k_cache[bi].append(k_stored)
        self.v_cache[bi].append(v_stored)

        if sink is not None:
            sink["ln1_out_q22"] = [int(z) for z in y_q22]
            sink["qkv_ix"] = [int(z) for z in ix]
            sink["qkv_yint"] = [int(z) for z in y_int]
            sink["qkv_q16"] = [int(z) for z in qkv_q16]
            sink["q_rot_q16"] = list(q)
            sink["k_stored_q16"] = list(k_stored)
            sink["v_stored_q16"] = list(v_stored)
            if self.kv_ddr[bi]:
                sink["kv_ddr_row"] = self.kv_ddr[bi][-1]

        T = len(self.k_cache[bi])
        nh, hd = NHEAD, HEAD_DIM
        from fabric.stage3.run_softmax import int_softmax_q
        from fabric.stage3.seq_ref import ISQRT, SCORE_FRAC, PROB_FRAC_A, RESID_FRAC
        ctx_q25 = np.zeros(C, dtype=object)
        for h in range(nh):
            base = h * hd
            sq = np.zeros(T, dtype=np.int64)
            for j in range(T):
                acc = 0
                for d in range(hd):
                    acc += q[base + d] * self.k_cache[bi][j][base + d]
                s_q88 = rsh_round(acc, 2 * VFRAC + ISQRT - SCORE_FRAC)
                sq[j] = sat(s_q88, -32768, 32767)
            prob_q20 = int_softmax_q(sq, np.ones(T, dtype=bool), self.exp_lut)
            for d in range(hd):
                acc = 0
                for j in range(T):
                    acc += int(prob_q20[j]) * self.v_cache[bi][j][base + d]
                ctx_q25[base + d] = rsh_round(acc, PROB_FRAC_A + VFRAC - RESID_FRAC)

        # un-rotate V output: ctx' = H ctx (H^T = H for the symmetric orthonormal H).
        if self.rotate and not self._kv_passthrough():
            ctx_list = [int(z) for z in ctx_q25]
            ctx_q25 = np.asarray(hadamard_rotate_q16(ctx_list), dtype=object)

        if sink is not None:
            sink["ctx_q25"] = [int(z) for z in ctx_q25]
        return self._proj_after_attn(ctx_q25, bi, sink=sink)

    # --- signal exposure for the kv_dma RTL gate --------------------------------
    def kv_ddr_words(self, prompt, n_steps=None):
        """Replay `prompt` (+ optional greedy extension) and return, per layer, the
        per-position DDR byte rows for K and V as they would sit in DDR. The future
        run_kv_*.py gate writes these to a .mem 'DDR image' and compares word-for-word
        against the kv_dma RTL. Each entry: {pos, k_row:[u8], v_row:[u8]}."""
        self.reset()
        toks = list(prompt)
        if n_steps is not None:
            logits, nxt = None, None
            for tok in toks:
                logits, nxt = self.step(int(tok))
            for _ in range(n_steps):
                toks.append(nxt)
                logits, nxt = self.step(nxt)
        else:
            for tok in toks:
                self.step(int(tok))
        out = []
        for bi in range(self.nblocks):
            rows = []
            for pos, e in enumerate(self.kv_ddr[bi]):
                rows.append({"pos": pos, "k_row": e["k_row"], "v_row": e["v_row"]})
            out.append(rows)
        return out


# ===================================================================== metrics
def val_nll_int(seq_factory, text_ids, chunks, clen, rng_seed=7):
    """Teacher-forced mean next-token NLL over `chunks` random chunks of clen, using a
    sequencer built fresh per chunk by seq_factory()."""
    rng = np.random.default_rng(rng_seed)
    nll, n = 0.0, 0
    for _ in range(chunks):
        s = int(rng.integers(0, len(text_ids) - clen - 1))
        ids = text_ids[s:s + clen + 1]
        dec = seq_factory()
        for t in range(clen):
            lg, _ = dec.step(int(ids[t]))
            lg = lg - lg.max()
            nll -= lg[ids[t + 1]] - np.log(np.exp(lg).sum())
            n += 1
    return nll / n


# ===================================================================== gate
def _validate(npz="fabric/export/goformer.npz", n_gen=24, prompt_len=8,
              val="data/sample.kevin.txt", val_chunks=6, val_len=96):
    import json
    import os

    p, cfg = build(npz)
    rng = np.random.default_rng(0)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=prompt_len))

    # ---- gate (a): FULL-PRECISION KVQ is BIT-IDENTICAL to the plain IntSequencer ----
    base = IntSequencer(p, cfg)
    fp = IntKVQSequencer(p, cfg, kbits=16, vbits=16, rotate=False)
    base.reset(); fp.reset()
    worst_logit_abs = 0.0
    toks = list(prompt)
    bl, bn, fl, fn = None, None, None, None
    for tok in toks:
        bl, bn = base.step(int(tok))
        fl, fn = fp.step(int(tok))
    stream_b, stream_f = list(prompt), list(prompt)
    for _ in range(n_gen):
        worst_logit_abs = max(worst_logit_abs, float(np.abs(bl - fl).max()))
        assert bn == fn, "FP-identity broke during greedy"
        stream_b.append(bn); stream_f.append(fn)
        bl, bn = base.step(bn)
        fl, fn = fp.step(fn)
    worst_logit_abs = max(worst_logit_abs, float(np.abs(bl - fl).max()))
    fp_identical = (stream_b == stream_f) and (worst_logit_abs == 0.0)
    print(f"src={npz} prompt_len={prompt_len} gen={n_gen}")
    print(f"(a) FP-identity: KVQ(kbits=vbits=16,no-rot) vs IntSequencer  "
          f"logit_maxabsdiff={worst_logit_abs:.3e} stream_identical={stream_b == stream_f}")

    # ---- gate (b): K4/V4 + Hadamard NLL delta vs the FP integer path ----
    nll_line = ""
    nll_ok = True
    if val and os.path.exists(val):
        itos = json.load(open("fabric/export/tokenizer.json"))["itos"]
        stoi = {c: i for i, c in enumerate(itos)}
        with open(val, encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = np.array([stoi.get(c, 0) for c in text[:200_000]], dtype=np.int64)
        base_nll = val_nll_int(lambda: IntSequencer(p, cfg), ids, val_chunks, val_len)
        q_nll = val_nll_int(lambda: IntKVQSequencer(p, cfg, kbits=4, vbits=4, rotate=True),
                            ids, val_chunks, val_len)
        delta = (q_nll - base_nll) / base_nll
        # honest range: the float exp_kvarn measured +0.72%; the integer datapath adds
        # its own rounding, so accept anything in a wide quality band (<=2% NLL).
        nll_ok = delta <= 0.02
        nll_line = (f"(b) K4/V4+Hadamard NLL: int-FP={base_nll:.4f} int-KVQ={q_nll:.4f} "
                    f"delta={q_nll - base_nll:+.4f} ({delta:+.2%}) nats/tok "
                    f"[{val_chunks}x{val_len} tok, exp_kvarn float ref +0.72%]")
        print(nll_line)
    else:
        print(f"(b) SKIPPED — held-out text {val!r} not found")

    # ---- DDR word layout sanity: a round-trip dequant matches the stored cache ----
    kvq = IntKVQSequencer(p, cfg, kbits=4, vbits=4, rotate=True)
    words = kvq.kv_ddr_words(prompt[:4])
    n_pos = len(words[0])
    k_bytes = len(words[0][0]["k_row"])
    v_bytes = len(words[0][0]["v_row"])
    print(f"DDR layout: stream-major [stream][layer][K|V][position][hdr|nibbles]; "
          f"per-position K={k_bytes}B V={v_bytes}B (nh={NHEAD} head-major, LSB-first), "
          f"{n_pos} positions/layer captured")

    ok = fp_identical and nll_ok
    print(f"KVQ_VERDICT fp_identical={fp_identical} nll_ok={nll_ok} pass={ok}")
    print("KVQ_" + ("PASS" if ok else "FAIL"))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="fabric/export/goformer.npz")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--val", default="data/sample.kevin.txt")
    ap.add_argument("--val-chunks", type=int, default=6)
    ap.add_argument("--val-len", type=int, default=96)
    args = ap.parse_args(argv)
    ok = _validate(args.npz, args.gen, args.prompt_len,
                   args.val, args.val_chunks, args.val_len)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
