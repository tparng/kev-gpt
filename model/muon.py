"""Muon optimizer (Jordan et al., modded-nanogpt) — the 2026 speedrun default.

Momentum + Newton-Schulz orthogonalization of the update for dense 2D matrix
params; everything else (scalars, norms, embeddings, depthwise conv) stays on
AdamW. Ported here as the drift ablation: onset is locked to ~45-50M tokens
across lr/schedule/precision/batch/window/corpus, which points at Adam's
update geometry, and Muon is the strongest known replacement at this scale.

Reference implementation follows the public modded-nanogpt one; fp32 (our
runs are --no-amp; the NS iteration is done in fp32 here, not bf16).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T for G = USV^T via quintic Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale so update RMS matches Adam-ish magnitudes across shapes
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)


class MultiOpt:
    """Wraps several optimizers as one (step/zero_grad/param_groups).

    Each param group carries `base_lr`; the training loop rescales all groups
    by the schedule ratio so Muon and AdamW keep their relative lrs.
    """

    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)
