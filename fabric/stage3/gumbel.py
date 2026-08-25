"""gumbel.py — the EXACT on-chip-sampling spec, shared by the RTL and the gate.

Sampling from softmax(logit/T) is identically argmax_i( logit_i + T*g_i ), with
g_i = -log(-log(u_i)), u_i ~ U(0,1)  (the Gumbel-max trick). So the fabric needs
NO softmax and NO host logit readback: it adds a precomputed noise value to each
head logit and the existing argmax winner IS the sample. The host writes ONE seed
per request instead of reading VOCAB logits per token.

This module is the single source of truth for BOTH:
  * the gumbel_lut.mem ROM the RTL loads, and
  * the bit-exact Python reference the gate checks the fabric against.

Fixed point: head logits are Q6.25 signed ints; noise is added on the same scale.
Determinism: a 32-bit xorshift seeded by the SEED register, advancing once per
logit and PERSISTING across a request's tokens (so every token gets fresh noise).
seed == 0 => greedy (noise forced to 0) => argmax, bit-exact to the old behaviour.
"""

from __future__ import annotations

import math

TEMP = 0.58          # baked into the LUT (the deployed host temperature).
                      # Recalibrated (was 0.85) after a real-hardware finding:
                      # 0.85 was tuned against an earlier checkpoint whose
                      # head-logit std was ~7.01 (Q6.25 real units, prompt
                      # "once upon a time"); the NLAYER=8 per-layer-streaming
                      # checkpoint (fabric/export_stream8) has a noticeably
                      # smaller std (~4.78) -- a real property of the bigger,
                      # differently-trained model, not a bug. A fixed-
                      # magnitude Gumbel noise at the old TEMP was then
                      # comparable to or larger than the actual logit signal,
                      # so on-chip sampling was noise-dominated (garbled real-
                      # hardware chat output) while greedy (zero noise) stayed
                      # bit-exact. Rescaled proportionally to preserve the
                      # earlier checkpoint's noise-to-signal ratio: 0.85 *
                      # (4.78/7.01) ~= 0.58. If a future checkpoint's logit
                      # scale drifts again, recheck this the same way (real
                      # head-logit std via seq_ref.build + IntKVQSequencer.
                      # step_head_q25, not guessed).
GLBITS = 10          # u resolution = 1/1024; LUT = 1024 x 32b = one BRAM tile
FRAC = 25            # logits are Q6.25
VOCAB = 193
MASK32 = 0xFFFFFFFF


def make_gumbel_lut(temp: float = TEMP, glbits: int = GLBITS,
                    frac: int = FRAC) -> list[int]:
    """gumbel_lut[idx] = round( temp * (-log(-log((idx+0.5)/2^glbits))) * 2^frac ),
    signed int. idx is the top `glbits` bits of an xorshift draw."""
    n = 1 << glbits
    out = []
    for i in range(n):
        u = (i + 0.5) / n                      # in (0,1), never exactly 0 or 1
        g = -math.log(-math.log(u))            # Gumbel(0,1) sample for that u
        out.append(int(round(temp * g * (1 << frac))))
    return out


def xorshift32(x: int) -> int:
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


class GumbelRng:
    """The persistent sampler state for one request. Construct from the seed the
    host wrote; call sample_token() once per generated token, in order."""

    def __init__(self, seed: int, lut: list[int] | None = None,
                 glbits: int = GLBITS, vocab: int = VOCAB):
        self.state = seed & MASK32
        self.enabled = (self.state != 0)       # seed 0 => greedy
        self.lut = lut if lut is not None else make_gumbel_lut(glbits=glbits)
        self.glbits = glbits
        self.vocab = vocab

    def sample_token(self, logits) -> int:
        """logits: sequence of Q6.25 ints (len >= vocab). Advance the rng once per
        logit (j = 0..vocab-1), perturb, return the argmax index (first wins ties).
        State persists for the next token."""
        best_v = None
        best_i = 0
        shift = 32 - self.glbits
        for j in range(self.vocab):
            self.state = xorshift32(self.state)
            v = int(logits[j])
            if self.enabled:
                v += self.lut[self.state >> shift]
            if best_v is None or v > best_v:    # strict > : first index wins ties
                best_v = v
                best_i = j
        return best_i


if __name__ == "__main__":
    # sanity: the Gumbel-max distribution must match softmax(logit/T) sampling
    import numpy as np
    rng = np.random.default_rng(0)
    logits_f = rng.normal(0, 4, VOCAB)                 # arbitrary logits
    lq = [int(round(x * (1 << FRAC))) for x in logits_f]
    lut = make_gumbel_lut()
    # empirical Gumbel-max histogram
    from collections import Counter
    c = Counter()
    g = GumbelRng(seed=0x12345678, lut=lut)
    for _ in range(20000):
        c[g.sample_token(lq)] += 1
    # reference softmax(logit/T)
    p = np.exp((logits_f - logits_f.max()) / TEMP); p /= p.sum()
    top = np.argsort(p)[::-1][:5]
    print("token  softmax_p   gumbel_freq")
    for t in top:
        print(f"  {t:3d}   {p[t]:.4f}     {c[t]/20000:.4f}")
    print(f"LUT[{1<<GLBITS}] range [{min(lut)}, {max(lut)}]  "
          f"(logit Q6.25 ~ {min(lq)}..{max(lq)})")
