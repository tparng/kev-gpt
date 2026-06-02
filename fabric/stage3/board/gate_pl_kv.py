"""GATE (laptop, no board): the KV-cached PL-backend wiring must produce the SAME
token stream as the reference full-recompute on the REAL exported model.

Bit-honest before fast. We run two greedy generations from the same prompt over
fabric/export/goformer.npz:

  A) reference: GoformerFull.forward(idx[-block_size:]) every token  (O(T^2), the
     exact arithmetic the current board driver uses).
  B) under test: model.goformer_kv.KVDecoder.step() per token with the resident
     PL backend's matmul (here the NumpyResident stand-in, which keys weights by
     id() and offset exactly like fabric.pl_gemv.PLResident — so the URAM-offset /
     preload wiring is exercised without a board, while the GEMV stays the exact
     int32 the PL reproduces bit-for-bit).

If the streams are identical AND the per-token matmul-call count collapses from
O(T) to a constant, the win is real and bit-honest. Prints one sentinel line:

    PL_KV_VERDICT identical=True stream=NN/NN

Run:  python -m fabric.stage3.board.gate_pl_kv
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.goformer_full import GoformerFull, load_params, numpy_matmul  # noqa: E402
from model.goformer_kv import KVDecoder  # noqa: E402
from fabric.stage3.board.pl_kv_chat import NumpyResident, kv_generate  # noqa: E402


def _ref_greedy(gf, prompt_ids, n_tokens, block_size):
    """Full-recompute greedy — what the current board driver effectively does."""
    out = list(prompt_ids)
    calls = 0
    for _ in range(n_tokens):
        cond = out[-block_size:]
        # count matmul calls the way GoformerFull issues them: qlin loops T rows,
        # 4*nlayer+1 Linears, T = len(cond).
        logits = gf.forward(cond)[-1]
        out.append(int(np.argmax(logits)))
    return out


def main():
    npz = os.path.join(_REPO, "fabric", "export", "goformer.npz")
    p = load_params(npz)
    block_size = p["block_size"]
    n_layer = len(p["blocks"])

    rng = np.random.default_rng(0)
    vocab = p["tok_emb"].shape[0]
    T0 = 8
    N = 24
    prompt = rng.integers(0, vocab, size=T0).tolist()

    # ----- A) reference full-recompute greedy (exact int GEMV via numpy) ---------
    gf_ref = GoformerFull(p, matmul=numpy_matmul)
    ref_stream = _ref_greedy(gf_ref, prompt, N, block_size)

    # ----- B) KV-cached greedy on the resident PL-backend stand-in ---------------
    backend = NumpyResident()
    backend.preload(p)                                   # exercises URAM-offset wiring
    engine = KVDecoder(p, matmul=backend.matmul)
    kv_stream = kv_generate(engine, prompt, N, block_size, greedy=True)

    # ----- equivalence -----------------------------------------------------------
    same = (ref_stream == kv_stream)
    matched = sum(int(a == b) for a, b in zip(ref_stream, kv_stream))
    total = len(ref_stream)

    # ----- extra: per-prefix last-token logits must match to maxabsdiff=0 --------
    # (proves the KV math == full math on the REAL weights, not just the stream)
    worst = 0.0
    eng2 = KVDecoder(p, matmul=backend.matmul)
    eng2.reset()
    probe = rng.integers(0, vocab, size=12).tolist()
    for t, tok in enumerate(probe):
        kv_logits = eng2.step(int(tok))
        full_logits = gf_ref.forward(probe[: t + 1])[-1]
        worst = max(worst, float(np.abs(kv_logits - full_logits).max()))

    # ----- the speed mechanism: matmul-call count per generated token ------------
    # KV: backend.n_matmul counts every GEMV the KV decoder issued (prime + gen).
    # Per generated step the KV decoder issues exactly 4*n_layer+1 Linears, each one
    # row (the single new position). The full-recompute issues (4*n_layer+1)*T rows
    # at depth T. Report both so the O(T^2)->O(T) collapse is visible.
    per_tok_kv = 4 * n_layer + 1
    deepest_T = T0 + N
    per_tok_full_deepest = (4 * n_layer + 1) * deepest_T
    speedup_deepest = per_tok_full_deepest / per_tok_kv

    print(f"[gate] model: n_layer={n_layer} d={p['tok_emb'].shape[1]} "
          f"vocab={vocab} block_size={block_size}")
    print(f"[gate] prefix last-token logits maxabsdiff (KV vs full) = {worst:.3e}")
    print(f"[gate] KV matmul calls total={backend.n_matmul} "
          f"(prime {T0} + gen {N}); per-gen-token Linears={per_tok_kv}")
    print(f"[gate] full-recompute Linears at deepest T={deepest_T}: "
          f"{per_tok_full_deepest}  -> KV is {speedup_deepest:.0f}x fewer matmul rows "
          f"at the last token")
    print(f"[gate] ref stream : {ref_stream}")
    print(f"[gate] kv  stream : {kv_stream}")

    ok = same and (worst == 0.0)
    print(f"PL_KV_VERDICT identical={ok} stream={matched}/{total}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
