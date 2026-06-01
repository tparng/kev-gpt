"""Integrated sequencer reference — KV-cached decode + fabric-precision non-linears.

This is the bit-honest CONTRACT the Tier-3 hardware token sequencer must meet: given
a prompt, the streaming fixed-point KV-cached decoder emits the SAME token stream as
the float reference. It composes the two verified prerequisites — incremental KV
decode (goformer_kv, bit-exact) and fabric-precision non-linears (goformer_fixed,
cosine 1.0) — into the exact per-token dataflow the FSM runs:

    embed -> 4x[ LN, qkv, attn(+KV cache), proj, +res, LN, mlp(GELU), +res ]
          -> LN_f -> head -> argmax -> append -> loop

The whole point of Tier 3 is to run this with ZERO CPU between tokens; this module is
the golden token stream the RTL sequencer is validated against.

    python -m model.goformer_seq
"""

from __future__ import annotations

import os

import numpy as np

from .goformer_fixed import FixedConfig, FixedGoformer, _profile_gelu_range
from .goformer_full import GoformerFull, load_params
from .goformer_kv import KVDecoder, build_random_params


def build_engines(npz="fabric/export/goformer.npz"):
    if os.path.exists(npz):
        p, src = load_params(npz), npz
    else:
        p, src = build_random_params(), "random-params"
    idx = np.random.default_rng(7).integers(0, p["tok_emb"].shape[0], size=64)
    dom = float(np.ceil(_profile_gelu_range(p, idx) + 1))
    cfg = FixedConfig(gelu_lo=-dom, gelu_hi=dom)
    return p, src, GoformerFull(p), FixedGoformer(p, cfg)


def _validate(npz="fabric/export/goformer.npz", n_gen=40, prompt_len=8):
    p, src, float_engine, fixed_engine = build_engines(npz)
    rng = np.random.default_rng(0)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=prompt_len))

    # ground truth: float, full-recompute, greedy
    def float_greedy(idx0, n):
        out = list(idx0)
        for _ in range(n):
            out.append(int(np.argmax(float_engine.forward(out[-256:])[-1])))
        return out
    ref_stream = float_greedy(prompt, n_gen)

    # the integrated fixed-point KV-cached streaming decoder (the sequencer dataflow)
    fixed_stream = KVDecoder(engine=fixed_engine).generate_greedy(prompt, n_gen, 256)

    same = ref_stream == fixed_stream
    n_match = sum(1 for a, b in zip(ref_stream, fixed_stream) if a == b)

    # worst per-step logit cosine of the streaming fixed decoder vs float full-recompute
    dec = KVDecoder(engine=fixed_engine)
    worst = 1.0
    for t, tok in enumerate(ref_stream):
        lg = dec.step(int(tok))
        fl = float_engine.forward(ref_stream[:t + 1][-256:])[-1]
        worst = min(worst, float(lg @ fl / (np.linalg.norm(lg) * np.linalg.norm(fl))))

    print(f"src={src}  prompt_len={prompt_len} gen={n_gen}")
    print(f"sequencer (fixed+KV) vs float full-recompute: stream_match={n_match}/{len(ref_stream)} "
          f"identical={same} worst_step_cosine={worst:.7f}")
    print("GOFORMER_SEQ_" + ("PASS" if same else "FAIL"))
    return same


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
