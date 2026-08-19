"""Export a baked Mamba-2 checkpoint to the .mem images the fabric loads.

The mamba twin of export_fabric.py, built on the mamba2_fixed contracts:
weights INT4 per-out-channel (quant_w) with FP16 scales, conv/norm gammas
INT16 fixed-point, SiLU + per-head fused softplus*exp decay as 256-entry
LUTs, and the calibrated activation / pow2 scan scales from
FixedMamba2.calibrate() dumped to shifts.json.

Outputs (to --out, default data/fabric_mamba/):
  l<i>_in_proj_w4.mem / _scale.mem    INT4 packed 8/32-bit word + FP16 scales
  l<i>_out_proj_w4.mem / _scale.mem
  head_w4.mem / head_scale.mem        (vocab, d_model)
  tok_emb_w4.mem / tok_emb_scale.mem  (vocab, d_model)
  l<i>_conv_w.mem / l<i>_conv_b.mem   INT16 Q1.14  (w row-major (conv_dim, k))
  silu_lut.mem                        256 x INT16 Q3.12, grid (i-128)/32
  l<i>_alut.mem                       n_heads*256 x UINT16 Q0.16 decay
                                      exp(-softplus(grid+dt_bias_h)*|A_h|);
                                      index = RAW (pre-bias) dt_raw Q3.12
                                      top 8 bits, (v_q312+32768)>>8, grid
                                      [-8,8) step 1/16 — bias folded in
  l<i>_dtlut.mem                      n_heads*256 x UINT16 Q1.14 dt =
                                      softplus(grid+dt_bias_h), same raw
                                      indexing, clamped [1,32767] (never 0)
  l<i>_ln_g.mem / l<i>_norm_g.mem     INT16 Q1.14 gammas
  normf_g.mem                         INT16 Q2.13 (values ~2.3-3.0 clip Q1.14;
                                      DEVIATION from the Q1.14 spec, flagged
                                      in the manifest)
  shifts.json                         per layer bX/bB/bC, in/out scales,
                                      dt_bias, negA (+D, qh), head_scale
  manifest.json                       every file with its element count

Self-check: every .mem is reloaded and compared against the checkpoint
(INT4*FP16scale < 1e-3; fixed-point files within 0.5 LSB or the saturation
bound; alut spot-checked 100 random (head,v) points within 1 LSB). Last
line printed is EXPORT_MAMBA_VERDICT: PASS/FAIL.

    .venv/bin/python -m model.export_mamba data/ckpt.mamba2.chat6q.baked.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .mamba2_fixed import FixedMamba2, quant_w

Q_A = 16                    # decay LUT fraction bits (UINT16 Q0.16)
Q_SILU = 12                 # silu LUT output Q3.12
Q_CONV = 14                 # conv w/b Q1.14
Q_NORM = 14                 # per-layer gammas Q1.14
Q_NORMF = 13                # final gamma Q2.13 (values 2.3-3.0 clip Q1.14)


# ------------------------------------------------------------ hex helpers ----

def _hex16(vals: np.ndarray) -> str:
    """INT16 (or UINT16) values -> one 4-hex-digit word per line."""
    v = np.asarray(vals).astype(np.int64) & 0xFFFF
    return "\n".join(f"{int(x):04x}" for x in v) + "\n"


def _load_hex(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array([int(ln, 16) for ln in f.read().split()], dtype=np.int64)


def _u16_to_i16(u: np.ndarray) -> np.ndarray:
    return np.where(u >= 1 << 15, u - (1 << 16), u)


def quant_fixed(vals: np.ndarray, frac: int) -> np.ndarray:
    """Float -> INT16 fixed point with `frac` fraction bits, saturating."""
    return np.clip(np.round(np.asarray(vals, dtype=np.float64) * (1 << frac)),
                   -(1 << 15), (1 << 15) - 1).astype(np.int64)


def pack_w4(q: np.ndarray) -> np.ndarray:
    """INT4 rows -> 32-bit words, 8 weights/word, row-major, LSB-first,
    two's complement nibbles. Row length must be a multiple of 8."""
    M, K = q.shape
    assert K % 8 == 0, f"row length {K} not a multiple of 8"
    nib = (q.astype(np.int64) & 0xF).reshape(M, K // 8, 8)
    shifts = np.arange(8, dtype=np.int64) * 4
    return (nib << shifts).sum(axis=2).reshape(-1).astype(np.uint32)


def unpack_w4(words: np.ndarray, M: int, K: int) -> np.ndarray:
    w = np.asarray(words, dtype=np.int64).reshape(M, K // 8)
    shifts = np.arange(8, dtype=np.int64) * 4
    nib = (w[:, :, None] >> shifts) & 0xF
    return np.where(nib >= 8, nib - 16, nib).reshape(M, K)


def _hex32(words: np.ndarray) -> str:
    return "\n".join(f"{int(x):08x}" for x in words) + "\n"


def fp16_bits(scale: np.ndarray) -> np.ndarray:
    """Per-channel float scale -> FP16 bit pattern as uint16."""
    return np.asarray(scale, dtype=np.float32).reshape(-1).astype(np.float16).view(np.uint16)


# ------------------------------------------------------------------ LUTs -----

def silu_lut_table() -> np.ndarray:
    grid = (np.arange(256) - 128) / 32.0
    return np.round(grid / (1.0 + np.exp(-grid)) * (1 << Q_SILU)).astype(np.int64)


def _softplus(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.where(v > 20.0, v, np.log1p(np.exp(np.minimum(v, 20.0))))


ALUT_GRID = (np.arange(256) - 128) / 16.0   # RAW dt_raw domain [-8,8) step
                                            # 1/16: idx = (v_q312+32768)>>8


def alut_table(negA: np.ndarray, dt_bias: np.ndarray) -> np.ndarray:
    """Per-head fused softplus*exp decay LUT, heads concatenated.
    Indexed by the RAW (pre-bias) Q3.12 dt projection's top 8 bits; the
    per-head dt_bias is folded INTO the table:
      entry_h[i] = round(exp(-softplus(grid[i]+dt_bias[h])*|A_h|)*2^16),
    clamped to 65535."""
    out = []
    for a, b in zip(np.asarray(negA, dtype=np.float64),
                    np.asarray(dt_bias, dtype=np.float64)):
        dt = _softplus(ALUT_GRID + b)
        e = np.round(np.exp(-dt * a) * (1 << Q_A))
        out.append(np.minimum(e, (1 << Q_A) - 1).astype(np.int64))
    return np.concatenate(out)


def dtlut_table(dt_bias: np.ndarray) -> np.ndarray:
    """Per-head dt LUT (same indexing as the alut: RAW dt_raw Q3.12 top 8
    bits, bias folded in): entry_h[i] = clamp(round(softplus(grid[i] +
    dt_bias[h]) * 2^14), 1, 32767) — UINT16 Q1.14, min 1 never 0 (the RTL
    derives a shift from bit_length; a zero entry would break it)."""
    out = []
    for b in np.asarray(dt_bias, dtype=np.float64):
        e = np.round(_softplus(ALUT_GRID + b) * (1 << 14))
        out.append(np.clip(e, 1, (1 << 15) - 1).astype(np.int64))
    return np.concatenate(out)


# ----------------------------------------------------------------- export ----

def export(ckpt_path: str, out_dir: str, val_bin: str, calib_tokens: int) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    fx = FixedMamba2(ckpt_path)
    L, H = fx.L, fx.H

    toks = np.fromfile(val_bin, dtype=np.uint16)[:calib_tokens]
    print(f"calibrating on {len(toks)} tokens of {val_bin} ...")
    fx.calibrate(toks)          # sets fx.xs + per-layer bX/bB/bC (reused, not
                                # reimplemented — the pow2 scales fall out of
                                # the instrumented step())

    manifest: dict = {
        "source_ckpt": os.path.abspath(ckpt_path),
        "cfg": fx.cfg,
        "n_heads": H,
        "packing": {
            "w4": "INT4 two's complement, 8 per 32-bit word, LSB-first, "
                  "row-major (row = output channel)",
            "scale": "per-output-channel FP16 bit pattern (uint16 hex)",
            "conv": f"INT16 Q1.{Q_CONV}", "silu_lut": f"INT16 Q3.{Q_SILU}",
            "alut": "UINT16 Q0.16, index = RAW dt_raw (pre-bias) top-8b of "
                    "Q3.12 ((v_q312+32768)>>8), domain [-8,8) step 1/16, "
                    "dt_bias folded into the table, heads concatenated",
            "dtlut": "UINT16 Q1.14 softplus(dt_raw+dt_bias), same indexing "
                     "as alut (raw pre-bias index, bias folded in), "
                     "clamped [1, 32767] (never 0: RTL bit_length shift)",
            "ln_g/norm_g": f"INT16 Q1.{Q_NORM}",
            "normf_g": f"INT16 Q2.{Q_NORMF} (DEVIATION: spec said Q1.14 but "
                       "the baked gamma_f is 2.3-3.0 everywhere and would "
                       "saturate; Q2.13 fits exactly)",
        },
        "files": {},
    }

    checks: list[tuple[str, float, float, bool]] = []   # (what, err, bound, ok)

    def emit(fname: str, text: str, count: int, **meta):
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(text)
        manifest["files"][fname] = {"elements": count, **meta}

    def emit_gemv(name: str, ref_w: np.ndarray, qs: tuple[np.ndarray, np.ndarray]):
        q, s = qs
        M, K = q.shape
        words = pack_w4(q)
        sbits = fp16_bits(s)
        emit(f"{name}_w4.mem", _hex32(words), int(M * K),
             shape=[int(M), int(K)], words=int(words.size))
        emit(f"{name}_scale.mem", _hex16(sbits), int(M), shape=[int(M)])
        # reload + reconstruct + compare vs checkpoint tensor
        rq = unpack_w4(_load_hex(os.path.join(out_dir, f"{name}_w4.mem")), M, K)
        rs = _load_hex(os.path.join(out_dir, f"{name}_scale.mem")).astype(np.uint16)
        rs = rs.view(np.float16).astype(np.float64)
        assert np.array_equal(rq, q), f"{name}: w4 pack/unpack round-trip failed"
        err = float(np.abs(rq * rs[:, None] - ref_w).max())
        checks.append((f"{name} INT4*FP16scale", err, 1e-3, err < 1e-3))

    def emit_fixed(fname: str, vals: np.ndarray, frac: int, what: str,
                   count: int | None = None, **meta):
        q = quant_fixed(vals, frac)
        emit(fname, _hex16(q), count or int(np.size(vals)),
             qformat=f"INT16 frac={frac}", **meta)
        r = _u16_to_i16(_load_hex(os.path.join(out_dir, fname)))
        err = float(np.abs(r / (1 << frac) - np.asarray(vals).reshape(-1)).max())
        bound = 0.5 / (1 << frac) + 1e-12
        checks.append((what, err, bound, err <= bound))

    # 1) GEMV weights (in_q/out_q/head_q already quantized by FixedMamba2;
    #    tok_emb quantized here with the same quant_w)
    import torch
    sd = {k: v.numpy().astype(np.float64)
          for k, v in torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False)["model"].items()}
    for l in range(L):
        p = f"layers.{l}.mixer."
        emit_gemv(f"l{l}_in_proj", sd[p + "in_proj.weight"], fx.layers[l]["in_q"])
        emit_gemv(f"l{l}_out_proj", sd[p + "out_proj.weight"], fx.layers[l]["out_q"])
    emit_gemv("head", sd["head.weight"], fx.head_q)
    emit_gemv("tok_emb", sd["tok_emb.weight"], quant_w(fx.emb))

    # 2) conv (Q1.14), 5) norms
    for l in range(L):
        lay = fx.layers[l]
        emit_fixed(f"l{l}_conv_w.mem", lay["conv_w"].reshape(-1), Q_CONV,
                   f"l{l}_conv_w", shape=list(lay["conv_w"].shape))
        emit_fixed(f"l{l}_conv_b.mem", lay["conv_b"], Q_CONV, f"l{l}_conv_b")
        emit_fixed(f"l{l}_ln_g.mem", lay["ln_g"], Q_NORM, f"l{l}_ln_g")
        emit_fixed(f"l{l}_norm_g.mem", lay["norm_g"], Q_NORM, f"l{l}_norm_g")
    emit_fixed("normf_g.mem", fx.normf_g, Q_NORMF, "normf_g")

    # 3) silu LUT
    slut = silu_lut_table()
    emit("silu_lut.mem", _hex16(slut), 256, qformat=f"INT16 Q3.{Q_SILU}")
    r = _u16_to_i16(_load_hex(os.path.join(out_dir, "silu_lut.mem")))
    ok = bool(np.array_equal(r, slut))
    checks.append(("silu_lut round-trip", 0.0 if ok else 1.0, 0.5, ok))

    # 4) per-layer per-head decay + dt LUTs (both indexed by RAW pre-bias
    #    dt_raw Q3.12 top-8b; dt_bias folded into the tables) + spot checks
    rng = np.random.default_rng(0)
    for l in range(L):
        negA = fx.layers[l]["negA"]
        bias = fx.layers[l]["dt_bias"]
        emit(f"l{l}_alut.mem", _hex16(alut_table(negA, bias)), int(H * 256),
             qformat="UINT16 Q0.16", heads=int(H))
        emit(f"l{l}_dtlut.mem", _hex16(dtlut_table(bias)), int(H * 256),
             qformat="UINT16 Q1.14", heads=int(H))
        ra = _load_hex(os.path.join(out_dir, f"l{l}_alut.mem"))
        rd = _load_hex(os.path.join(out_dir, f"l{l}_dtlut.mem"))
        worst_a = worst_d = 0
        for _ in range(100):
            h = int(rng.integers(0, H)); i = int(rng.integers(0, 256))
            v = (i - 128) / 16.0 + float(bias[h])     # grid is RAW dt_raw
            dt = float(_softplus(v))
            exp_a = min(round(np.exp(-dt * float(negA[h])) * (1 << Q_A)),
                        (1 << Q_A) - 1)
            exp_d = int(np.clip(round(dt * (1 << 14)), 1, (1 << 15) - 1))
            worst_a = max(worst_a, abs(int(ra[h * 256 + i]) - exp_a))
            worst_d = max(worst_d, abs(int(rd[h * 256 + i]) - exp_d))
        checks.append((f"l{l}_alut 100-pt spot", float(worst_a), 1.0, worst_a <= 1))
        checks.append((f"l{l}_dtlut 100-pt spot", float(worst_d), 1.0, worst_d <= 1))
        nz = bool((rd > 0).all())
        checks.append((f"l{l}_dtlut nonzero", 0.0 if nz else 1.0, 0.5, nz))

    # 6) shifts.json — everything the sequencer needs beyond the .mem images
    shifts = {
        "head_scale": float(fx.xs["head"]),
        "qh": 13,                        # scan state Q3.13 (Q_STATE default)
        "layers": [
            {
                "bX": int(fx.layers[l]["bX"]),
                "bB": int(fx.layers[l]["bB"]),
                "bC": int(fx.layers[l]["bC"]),
                "in_scale": float(fx.xs[("in", l)]),
                "out_scale": float(fx.xs[("out", l)]),
                "head_scale": float(fx.xs["head"]),
                "dt_bias": [float(v) for v in fx.layers[l]["dt_bias"]],
                "negA": [float(v) for v in fx.layers[l]["negA"]],
                "D": [float(v) for v in fx.layers[l]["Dh"]],   # spec addition:
                # the D-skip coefficients are needed by the fabric and had no
                # home in the listed files
            }
            for l in range(L)
        ],
    }
    with open(os.path.join(out_dir, "shifts.json"), "w") as f:
        json.dump(shifts, f, indent=2)
    manifest["files"]["shifts.json"] = {"elements": L}

    # 7) manifest
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------- verdict ---
    all_ok = True
    for what, err, bound, ok in checks:
        all_ok &= ok
        print(f"  {'ok ' if ok else 'FAIL'} {what:28s} err {err:.3e} "
              f"(bound {bound:.1e})")
    n_files = len(manifest["files"]) + 1     # + manifest itself
    total = sum(os.path.getsize(os.path.join(out_dir, fn))
                for fn in os.listdir(out_dir))
    print(f"exported {n_files} files -> {out_dir}  ({total / 1024:.1f} KB on disk)")
    print(f"EXPORT_MAMBA_VERDICT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.export_mamba",
                                description=__doc__.splitlines()[0])
    p.add_argument("ckpt", nargs="?", default="data/ckpt.mamba2.chat6q.baked.pt")
    p.add_argument("--out", default="data/fabric_mamba/")
    p.add_argument("--val-bin", default="data/bpe1024_chat6/val.bin")
    p.add_argument("--calib-tokens", type=int, default=64)
    args = p.parse_args(argv)
    ok = export(args.ckpt, args.out, args.val_bin, args.calib_tokens)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
