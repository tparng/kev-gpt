"""Gate harness for conv_silu (doc 9 §7 rung 2, cosine gate — LUT phase).

Reference = mamba2_fixed's conv+silu at contract resolution: Q1.14 weights,
Q3.12 activations, the same 256-entry SiLU table the RTL loads. House rule:
cosine > 0.9999 for LUT/transcendental phases (bit-exactness is checked for
the MAC by also comparing the integer pre-activation path on a no-LUT run).

    python -m fabric.stage3.run_conv_silu --steps 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from fabric.stage3._simdir import kevbuild

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CH, K = 640, 4


def tool(name):
    t = shutil.which(name) or os.path.expanduser(f"~/.local/bin/{name}")
    if not os.path.exists(t):
        sys.exit(f"missing tool: {name}")
    return t


def silu_table():
    grid = (np.arange(256) - 128) / 32.0
    return np.round(grid / (1.0 + np.exp(-grid)) * 4096).astype(np.int64)


def ref_step(hist, x_q, w_q, b_q, lut):
    """Bit-true integer replica of the RTL datapath. x in Q6.9, y out Q4.11
    (widened from Q3.12: the dequantized xBC slice reaches ~33 and the silu
    output ~10, both of which Q3.12 clipped in the full engine)."""
    buf = np.concatenate([hist, x_q[:, None]], axis=1)          # (CH, K)
    hist[:] = buf[:, 1:]
    acc = (buf * w_q).sum(1) + (b_q.astype(np.int64) << 9)      # frac 23
    pre = (acc + (1 << 11)) >> 12                                # Q4.11
    pre = np.clip(pre, -32768, 32767)
    idx = np.clip((pre + 8192) >> 6, 0, 255)
    y = (lut[idx] + 1) >> 1                                      # Q3.12 -> Q4.11
    y = np.where(pre >= 8192, pre, y)                            # 4.0 = 8192 Q4.11
    y = np.where(pre < -8192, 0, y)
    return y


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_conv_silu")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    sim = args.dir or kevbuild("stage3_conv_silu")
    os.makedirs(sim, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    T = args.steps

    w_q = np.clip(np.round(rng.normal(0, 0.25, (CH, K)) * 16384), -32768, 32767).astype(np.int64)
    b_q = np.clip(np.round(rng.normal(0, 0.1, CH) * 16384), -32768, 32767).astype(np.int64)
    # x in Q6.9 (*512); wide sigma so the conv output exercises the >4 / >8
    # (>Q3.12) range the widening was for, plus the identity SiLU tail
    xs = np.clip(np.round(rng.normal(0, 6.0, (T, CH)) * 512), -32768, 32767).astype(np.int64)
    lut = silu_table()

    def hexdump(path, vals):
        with open(path, "w") as f:
            for v in np.asarray(vals).ravel():
                f.write(f"{int(v) & 0xFFFF:04x}\n")

    hexdump(f"{sim}/cs_w.mem", w_q)
    hexdump(f"{sim}/cs_b.mem", b_q)
    hexdump(f"{sim}/cs_lut.mem", lut)
    hexdump(f"{sim}/cs_x.mem", xs)
    with open(f"{sim}/cs_cfg.mem", "w") as f:
        f.write(f"{T:08x}\n")

    hist = np.zeros((CH, K - 1), dtype=np.int64)
    ref = np.stack([ref_step(hist, xs[t], w_q, b_q, lut) for t in range(T)])

    src = [f"{REPO}/fabric/stage3/rtl/conv_silu.sv",
           f"{REPO}/fabric/stage3/tb/tb_conv_silu.sv"]
    subprocess.run([tool("iverilog"), "-g2012", "-o", "cs.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "cs.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True)
    if "TB_CS_DONE" not in out.stdout:
        sys.exit(f"sim incomplete:\n{out.stdout[-400:]}")

    got = np.array([int(l, 16) for l in open(f"{sim}/cs_y.out")], dtype=np.int64)
    got = np.where(got >= 1 << 15, got - (1 << 16), got).reshape(T, CH)
    mism = int((got != ref).sum())
    first = np.argwhere(got != ref)
    where = f" first@t{first[0][0]},c{first[0][1]}" if len(first) else ""
    print(f"CONV_SILU_VERDICT: {'BIT-EXACT' if mism == 0 else 'FAIL'} "
          f"({T} tokens x {CH} ch, {mism} mismatches{where})  [{sim}]")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
