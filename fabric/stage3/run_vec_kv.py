"""Multi-token KV-faithful gate for sequencer_vec (doc-7 R1): drives PLEN prompt
positions + NGEN greedy feedback positions through repeated GO pulses (kv_bank
persisting across passes) and demands the generated token stream equal the PINNED
golden reference IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True)
.generate_greedy — the R0 contract. Also reports per-pass cycles (the R1 number).

    python -m fabric.stage3.run_vec_kv --plen 4 --ngen 6 --dir C:/kevbuild/stage3_vec_kv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_wideword
from model.goformer_kvq import IntKVQSequencer

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_seq_vec_kv.sv")

RTL_FILES = ["sequencer_vec.sv", "kv_bank.sv", "vec_attn_w.sv", "layernorm_vec.sv",
             "vec_dequant.sv", "vec_gelu.sv", "gelu_lut.sv", "gelu_lut2.sv",
             "softmax_f.sv", "gemv_banked_resident_vec.sv", "weight_bank_tdp.sv"]


def _encode(meta_path, text):
    m = json.load(open(meta_path))
    stoi = m["stoi"]
    return [stoi.get(c, 0) for c in text]


def _decode(meta_path, ids):
    m = json.load(open(meta_path))
    itos = {int(k): v for k, v in m["itos"].items()}
    return "".join(itos.get(int(i), "?") for i in ids)


def run(sim_dir, prompt_ids, ngen, P=8, lanes=16, tmax=256,
        npz="fabric/export/goformer.npz", meta=None):
    os.makedirs(sim_dir, exist_ok=True)
    plen = len(prompt_ids)
    assert plen + ngen - 1 <= tmax, "prompt+gen must fit the KV window"

    p, cfg = seq_ref.build(npz)
    gold_seq = IntKVQSequencer(p, cfg, kbits=8, vbits=8, rotate=False, divfree=True)
    gold = gold_seq.generate_greedy(list(prompt_ids), ngen)[plen:]

    iseq = seq_ref.IntSequencer(p, cfg)
    write_mems_wideword(sim_dir, iseq, lanes, 4, P)
    # gelu_lut2 (the paired-lane core vec_gelu now uses) reads the even/odd split
    with open(os.path.join(sim_dir, "gelu_lut.mem")) as fh:
        lut = [ln.strip() for ln in fh if ln.strip()]
    with open(os.path.join(sim_dir, "gelu_lut_e.mem"), "w") as fh:
        fh.write("\n".join(lut[0::2]) + "\n")
    with open(os.path.join(sim_dir, "gelu_lut_o.mem"), "w") as fh:
        fh.write("\n".join(lut[1::2]) + "\n")
    # kv_bank R4a: inv = q_round_div(2^24, scale), split lo (25b) / hi (13b) ROMs
    from fabric.stage3.seq_ref import q_round_div
    with open(os.path.join(sim_dir, "inv_lut_lo.mem"), "w") as fh:
        for s in range(4096):
            fh.write(f"{q_round_div(1 << 24, max(s, 1)) & 0x1FFFFFF:07x}\n")
    with open(os.path.join(sim_dir, "inv_lut_hi.mem"), "w") as fh:
        for s in range(4096, 16512):
            fh.write(f"{q_round_div(1 << 24, s) & 0x1FFF:04x}\n")
    with open(os.path.join(sim_dir, "prompt.mem"), "w") as fh:
        for t in prompt_ids:
            fh.write(f"{int(t):03x}\n")
    with open(os.path.join(sim_dir, "wrom.mem")) as fh:
        wrom_n = sum(1 for ln in fh if ln.strip())

    vvp = os.path.join(sim_dir, "sim.vvp")
    defs = [f"-DPVAL={P}", f"-DLVAL={lanes}", f"-DWROMN={wrom_n}",
            f"-DTMAXVAL={tmax}", f"-DPLEN={plen}", f"-DNGEN={ngen}"]
    if os.environ.get("KVDBG"):
        defs.append("-DKVDBG")
    if os.environ.get("KVSTOP"):
        defs.append(f"-DKVSTOP={os.environ['KVSTOP']}")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp] + defs +
                        [TB] + [os.path.join(RTL, f) for f in RTL_FILES],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    with open(os.path.join(sim_dir, "vvp_stdout.log"), "w") as fh:
        fh.write(rp.stdout)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-3000:]); print(rp.stderr[-1500:]); return False

    def _ints(path):
        out = []
        with open(path) as fh:
            for ln in fh:
                s = ln.strip()
                if s:
                    out.append(None if ("x" in s.lower()) else int(s))
        return out

    got = _ints(os.path.join(sim_dir, "stream.out"))
    cycs = []
    with open(os.path.join(sim_dir, "cycs.out")) as fh:
        cycs = [int(ln.strip()) for ln in fh if ln.strip()]

    ok = (got == gold)
    if meta and os.path.exists(meta) and None not in got:
        txt = (f"[txt ] prompt={_decode(meta, prompt_ids)!r} "
               f"hw={_decode(meta, got)!r} gold={_decode(meta, gold)!r}")
        print(txt.encode("ascii", "replace").decode())   # cp1252-console safe
    print(f"[hw  ] gen={got}")
    print(f"[gold] gen={gold}")
    if cycs:
        print(f"[cyc ] per-pass min..max = {min(cycs)}..{max(cycs)}, "
              f"avg = {sum(cycs)/len(cycs):.0f} (T grows 1..{plen+ngen-1})")
    print(f"VEC_KV_VERDICT match={ok} plen={plen} ngen={ngen} "
          f"avg_cyc={sum(cycs)/len(cycs):.0f}" if cycs else
          f"VEC_KV_VERDICT match={ok} plen={plen} ngen={ngen}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_vec_kv")
    ap.add_argument("--prompt", default=None, help="prompt text (uses tokenizer)")
    ap.add_argument("--plen", type=int, default=4, help="random-ish prompt length if no --prompt")
    ap.add_argument("--ngen", type=int, default=6)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--tmax", type=int, default=256)
    ap.add_argument("--npz", default="fabric/export/goformer.npz")
    ap.add_argument("--meta", default="fabric/export/goformer_meta.json")
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_vec_kv"))
    a = ap.parse_args(argv)
    if a.prompt:
        ids = _encode(a.meta, a.prompt)
    else:
        ids = [48, 10, 100, 77, 5, 60, 120, 180][:a.plen]
    ok = run(a.dir, ids, a.ngen, a.p, a.lanes, a.tmax, a.npz, a.meta)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
