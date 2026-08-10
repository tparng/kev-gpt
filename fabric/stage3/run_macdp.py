"""Bit-exact gate for the double-pumped MAC bank (DOUBLE-PUMP-100K Stage 0, Phase A).

Drives BOTH the reference single-clock `mac_bank` (the real leaf, compiled in from
gemm_banked_resident_vec.sv) AND `mac_bank_dp` (clk2x, 2 K-steps/clk) from the SAME
random (weight-word, activation-byte) K-step sequence, for several seeds plus
adversarial corners, and asserts the final per-lane `acc` words are bit-identical.

    python -m fabric.stage3.run_macdp --lanes 128 --K 1024 \
        --dir C:/kevbuild/agent_dp0/macdp

Prints, per case, a comparison line, and at the end:
    MACDP_VERDICT bitexact=True/False cases=<n> mismatches=<m>

This is THE green light for the double-pump campaign: if the 2-per-cycle folding is
bit-exact here, the arithmetic risk is dead and Phase B (the >=400 MHz OOC) proceeds.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DP = os.path.join(HERE, "rtl", "mac_bank_dp.sv")
RTL_REF = os.path.join(HERE, "rtl", "gemm_banked_resident_vec.sv")
TB = os.path.join(HERE, "tb", "tb_macdp.sv")

NIB_MIN, NIB_MAX = -8, 7        # INT4 weight range
ACT_MIN, ACT_MAX = -128, 127    # INT8 activation range


def _nib_hex(v: int) -> str:
    """2's-complement INT4 -> single hex nibble."""
    return f"{v & 0xF:x}"


def write_case(sim_dir: str, lanes: int, K: int, mode: str, seed: int) -> None:
    """Write w.mem (K words of LANES INT4 nibbles) and x.mem (K INT8 bytes).

    mode: 'rand'      uniform random
          'neg'       all weights -8, all acts -128 (worst negative magnitude)
          'pos'       all weights  7, all acts  127 (worst positive magnitude)
          'altsign'   alternating-sign weights & acts (cancellation / ordering)
    """
    os.makedirs(sim_dir, exist_ok=True)
    rng = random.Random(seed)

    w_lines = []
    x_lines = []
    for k in range(K):
        if mode == "rand":
            nibs = [rng.randint(NIB_MIN, NIB_MAX) for _ in range(lanes)]
            act = rng.randint(ACT_MIN, ACT_MAX)
        elif mode == "neg":
            nibs = [NIB_MIN] * lanes
            act = ACT_MIN
        elif mode == "pos":
            nibs = [NIB_MAX] * lanes
            act = ACT_MAX
        elif mode == "altsign":
            s = 1 if (k % 2 == 0) else -1
            nibs = [(NIB_MAX if (l % 2 == 0) else NIB_MIN) for l in range(lanes)]
            act = (ACT_MAX if s > 0 else ACT_MIN)
        else:
            raise ValueError(mode)

        # weight word: lane 0 is the LOW nibble (bits [3:0]) -> emit MSB lane first.
        word = "".join(_nib_hex(nibs[l]) for l in reversed(range(lanes)))
        w_lines.append(word)
        x_lines.append(f"{act & 0xFF:02x}")

    with open(os.path.join(sim_dir, "w.mem"), "w") as f:
        f.write("\n".join(w_lines) + "\n")
    with open(os.path.join(sim_dir, "x.mem"), "w") as f:
        f.write("\n".join(x_lines) + "\n")


def check(sim_dir: str) -> tuple[bool, int]:
    """Bit-exact compare acc_ref.out vs acc_dp.out. Returns (ok, n_mismatch)."""
    with open(os.path.join(sim_dir, "acc_ref.out")) as f:
        ref = [ln.strip() for ln in f if ln.strip()]
    with open(os.path.join(sim_dir, "acc_dp.out")) as f:
        dp = [ln.strip() for ln in f if ln.strip()]
    if len(ref) != len(dp):
        print(f"  LENGTH MISMATCH ref={len(ref)} dp={len(dp)}")
        return False, max(len(ref), len(dp))
    mism = 0
    for i, (r, d) in enumerate(zip(ref, dp)):
        if int(r, 16) != int(d, 16):
            if mism < 8:
                print(f"  lane {i}: ref={r} dp={d}")
            mism += 1
    return mism == 0, mism


def run(lanes: int, K: int, abits: int, seeds: list[int], base_dir: str) -> bool:
    if K % 2 != 0:
        print("K must be even (2 K-steps/clk)")
        return False

    cases: list[tuple[str, int]] = []
    for s in seeds:
        cases.append(("rand", s))
    cases += [("neg", 0), ("pos", 0), ("altsign", 0)]

    total_mism = 0
    all_ok = True
    for idx, (mode, seed) in enumerate(cases):
        sim_dir = os.path.join(base_dir, f"case_{idx}_{mode}_{seed}")
        write_case(sim_dir, lanes, K, mode, seed)

        vvp = os.path.join(sim_dir, "sim.vvp")
        compile_cmd = [
            "iverilog", "-g2012", "-o", vvp,
            f"-DLANES={lanes}", f"-DABITS={abits}", f"-DKVAL={K}",
            TB, RTL_DP, RTL_REF,
        ]
        cp = subprocess.run(compile_cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            print("IVERILOG_COMPILE_FAIL")
            print(cp.stdout)
            print(cp.stderr)
            return False

        rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
        if rp.returncode != 0 or "TB_DONE" not in rp.stdout:
            print("VVP_RUN_FAIL")
            print(rp.stdout)
            print(rp.stderr)
            return False

        ok, mism = check(sim_dir)
        total_mism += mism
        all_ok = all_ok and ok
        tag = "OK " if ok else "FAIL"
        print(f"case[{idx}] mode={mode:8s} seed={seed} -> {tag} mismatches={mism}")

    print(f"MACDP_VERDICT bitexact={all_ok} cases={len(cases)} mismatches={total_mism}")
    return all_ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.run_macdp")
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--abits", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--dir", default=kevbuild("agent_dp0", "macdp"))
    a = p.parse_args(argv)
    ok = run(a.lanes, a.K, a.abits, a.seeds, a.dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
