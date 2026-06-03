"""Roundtrip test for the P-wide banked ROM emitter (run_sequencer.write_mems_banked).

Builds a small intseq from the real goformer.npz (same construction as
run_sequencer.run_multitoken: seq_ref.build -> IntSequencer), emits the banked ROMs
with P=8 into a temp dir under C:/kevbuild/, then reads back the P bank files for
dq_mant and gamma, DE-INTERLEAVES them (v[b::P] = bank_b), and asserts the
reconstruction equals the flat values write_mems would have written.

    python -m fabric.stage3.test_banked_mem
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_banked
from fabric.stage3.run_layernorm import G_FRAC as LN_GFRAC

P = 8
NLAYER = 4
LANES = 16
NPZ = "fabric/export/goformer.npz"


def _read_hex(path):
    with open(path) as f:
        return [int(l.strip(), 16) for l in f if l.strip()]


def _flat_gamma(p, nlayer):
    gam = []
    for bi in range(nlayer):
        gam += list(np.round(p["blocks"][bi]["ln1"] * (1 << LN_GFRAC)).astype(np.int64))
        gam += list(np.round(p["blocks"][bi]["ln2"] * (1 << LN_GFRAC)).astype(np.int64))
    gam += list(np.round(p["ln_f"] * (1 << LN_GFRAC)).astype(np.int64))
    return [int(v) & ((1 << 32) - 1) for v in gam]


def _flat_dq_mant(intseq, nlayer):
    mant_all = []
    for bi in range(nlayer):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            L = intseq.layers[(bi, nm)]
            mant_all += [int(m) for m in L["mant"]]
    mant_all += [int(m) for m in intseq.head["mant"]]
    return [int(v) & ((1 << 32) - 1) for v in mant_all]


def _deinterleave(prefix, P, length):
    v = [None] * length
    for b in range(P):
        bank = _read_hex(f"{prefix}.b{b}.mem")
        for j, val in enumerate(bank):
            idx = b + j * P
            if idx < length:        # ignore the trailing zero-pad beat (L not a multiple of P)
                v[idx] = val
    return v


def main():
    p, cfg = seq_ref.build(NPZ)
    intseq = seq_ref.IntSequencer(p, cfg)

    sim_dir = tempfile.mkdtemp(prefix="banked_", dir="C:/kevbuild")
    write_mems_banked(sim_dir, intseq, LANES, NLAYER, P, prompt=[0])

    # golden flat values (masked to the nibble width write_mems wrote: both 8 nib = 32-bit)
    gold_mant = _flat_dq_mant(intseq, NLAYER)
    gold_gamma = _flat_gamma(p, NLAYER)

    rec_mant = _deinterleave(os.path.join(sim_dir, "dq_mant"), P, len(gold_mant))
    rec_gamma = _deinterleave(os.path.join(sim_dir, "gamma"), P, len(gold_gamma))

    dq_mant_match = (rec_mant == gold_mant)
    gamma_match = (rec_gamma == gold_gamma)

    print(f"BANKED_MEM_OK p={P} dq_mant_match={dq_mant_match} gamma_match={gamma_match} "
          f"dq_mant_len={len(gold_mant)} gamma_len={len(gold_gamma)} dir={sim_dir}")
    return dq_mant_match and gamma_match


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
