"""A minimal Mamba2 (SSD) char-LM, sized for the Kevin-on-Kria budget.

Why this exists (doc 8): the transformer's on-chip cost that *grows* is the KV
cache — 512 KB/stream at T=256 INT8 (doc 7 §3). A Mamba2 layer replaces it with
a fixed-size recurrent state (d_inner x d_state + a (d_conv-1)-deep conv tail)
that never grows with context. This file is the trainable FP reference — the
top of a new reference ladder, before any Q-format or RTL work.

Deliberately minimal, no CUDA kernels, nothing exotic:
- Training path is the *quadratic* SSD form (the 1-semiseparable matrix
  materialised, O(T^2) like attention — fine at T=256) so it trains fast and
  parallel on the 3050 Ti.
- Decode path is the *recurrent* step (constant state, O(1) per token) — the
  form the fabric would implement.
- `python -m model.mamba2` self-tests that the two paths agree (the ladder's
  gate habit: equivalence before anything else is trusted).

Matches model.gpt.GPT's interface (forward(idx, targets) -> (logits, loss),
num_params(), generate()) so model.train drives it via --arch mamba2.

No positional embedding — the recurrence carries order. That deletes the
block_size x d_model pos table from the on-chip budget too.

State math runs in fp32 even under AMP (the cumsum/exp decay path is
precision-sensitive; every Mamba implementation does this).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Mamba2Config:
    block_size: int = 256        # training context; decode is unbounded (that's the point)
    vocab_size: int = 256        # set from the tokenizer at build time
    n_layer: int = 7             # ~3.0M at d256/expand2 — iso-param vs the L4 transformer
    d_model: int = 256
    d_state: int = 64            # N: recurrent state per head-channel
    d_conv: int = 4              # short causal depthwise conv
    expand: int = 2              # d_inner = expand * d_model
    headdim: int = 64            # P: channels per head
    dropout: float = 0.0
    z_loss: float = 0.0          # PaLM-style logsumexp^2 coeff. The M1/M1b runs
                                 # drift after ~5k iters and temperature-scaling
                                 # shows ~2/3 of it is logit-scale overconfidence;
                                 # this pins the scale. 0 = off.

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        assert self.d_inner % self.headdim == 0
        return self.d_inner // self.headdim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate=None):
        if gate is not None:
            x = x * F.silu(gate)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class Mamba2Block(nn.Module):
    """One SSD mixer. in_proj -> [gate z | conv'd (x,B,C) | dt] -> SSM -> gated norm -> out_proj."""

    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        d_in, nh, ds = cfg.d_inner, cfg.n_heads, cfg.d_state
        conv_dim = d_in + 2 * ds  # x, B, C all pass through the short conv
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_in + 2 * ds + nh, bias=False)
        self.conv = nn.Conv1d(conv_dim, conv_dim, cfg.d_conv, groups=conv_dim,
                              padding=cfg.d_conv - 1)
        self.dt_bias = nn.Parameter(torch.empty(nh))
        self.A_log = nn.Parameter(torch.empty(nh))
        self.D = nn.Parameter(torch.ones(nh))
        self.norm = RMSNorm(d_in)
        self.out_proj = nn.Linear(d_in, cfg.d_model, bias=False)

        # standard Mamba init: A ~ U[1,16]; dt ~ logU[1e-3, 1e-1] via softplus-inverse bias
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.empty(nh).uniform_(1.0, 16.0)))
            dt = torch.exp(torch.empty(nh).uniform_(math.log(1e-3), math.log(1e-1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inv softplus

    def _split(self, zxbcdt):
        d_in, ds = self.cfg.d_inner, self.cfg.d_state
        return torch.split(zxbcdt, [d_in, d_in + 2 * ds, self.cfg.n_heads], dim=-1)

    def forward(self, u):
        """Quadratic SSD over a full sequence. u: (B, T, d_model)."""
        cfg = self.cfg
        Bb, T, _ = u.shape
        nh, P, N = cfg.n_heads, cfg.headdim, cfg.d_state

        z, xBC, dt = self._split(self.in_proj(u))
        xBC = F.silu(self.conv(xBC.transpose(1, 2))[..., :T].transpose(1, 2))
        x, Bmat, Cmat = torch.split(xBC, [cfg.d_inner, N, N], dim=-1)

        # ---- state math in fp32 ----
        x32 = x.float().view(Bb, T, nh, P)
        B32, C32 = Bmat.float(), Cmat.float()
        dt32 = F.softplus(dt.float() + self.dt_bias.float())          # (B,T,H)
        A = -torch.exp(self.A_log.float())                            # (H,) negative
        # decay: log a_t = dt_t * A_h  (<= 0);  L[t,s] = exp(sum_{s<k<=t} log a_k)
        La = torch.cumsum(dt32 * A, dim=1)                            # (B,T,H)
        diff = La.permute(0, 2, 1).unsqueeze(-1) - La.permute(0, 2, 1).unsqueeze(-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=u.device))
        L = torch.exp(diff.masked_fill(~mask, float("-inf")))         # (B,H,T,S)
        CB = torch.einsum("btn,bsn->bts", C32, B32)                   # head-shared (1 group)
        M = CB.unsqueeze(1) * L * dt32.permute(0, 2, 1).unsqueeze(-2) # (B,H,T,S)
        y = torch.einsum("bhts,bshp->bthp", M, x32)
        y = y + self.D.float().view(1, 1, nh, 1) * x32
        # ----------------------------

        y = y.reshape(Bb, T, cfg.d_inner).to(u.dtype)
        return self.out_proj(self.norm(y, gate=z))

    # --- recurrent decode (the fabric-shaped path) ---

    def alloc_state(self, batch: int, device, dtype=torch.float32):
        cfg = self.cfg
        return {
            "conv": torch.zeros(batch, cfg.d_inner + 2 * cfg.d_state,
                                cfg.d_conv - 1, device=device, dtype=dtype),
            "ssm": torch.zeros(batch, cfg.n_heads, cfg.headdim, cfg.d_state,
                               device=device, dtype=dtype),
        }

    def step(self, u, state):
        """One token. u: (B, d_model). Mutates state in place, returns (B, d_model)."""
        cfg = self.cfg
        nh, P, N = cfg.n_heads, cfg.headdim, cfg.d_state

        z, xBC, dt = self._split(self.in_proj(u))
        # causal conv as a dot with the (d_conv-1)-deep tail
        conv_in = torch.cat([state["conv"], xBC.float().unsqueeze(-1)], dim=-1)
        state["conv"].copy_(conv_in[..., 1:])
        w = self.conv.weight.squeeze(1).float()                       # (conv_dim, d_conv)
        xBC = F.silu((conv_in * w).sum(-1) + self.conv.bias.float())
        x, Bv, Cv = torch.split(xBC, [cfg.d_inner, N, N], dim=-1)

        x32 = x.view(-1, nh, P)
        dt32 = F.softplus(dt.float() + self.dt_bias.float())          # (B,H)
        a = torch.exp(dt32 * -torch.exp(self.A_log.float()))          # (B,H)
        # h = a*h + dt * (x outer B);  y = h . C + D*x
        state["ssm"].mul_(a.unsqueeze(-1).unsqueeze(-1))
        state["ssm"].add_((dt32.unsqueeze(-1) * x32).unsqueeze(-1)
                          * Bv.view(-1, 1, 1, N))
        y = torch.einsum("bhpn,bn->bhp", state["ssm"], Cv)
        y = y + self.D.float().view(1, nh, 1) * x32

        y = y.reshape(-1, cfg.d_inner).to(u.dtype)
        return self.out_proj(self.norm(y, gate=z))


class Residual(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = Mamba2Block(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))

    def step(self, x, state):
        return x + self.mixer.step(self.norm(x), state)


class Mamba2(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Residual(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied, as in gpt.py
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_bytes(self, dtype_bytes: int = 1) -> int:
        """Per-stream decode state footprint (the number the memo argues from)."""
        cfg = self.cfg
        per_layer = (cfg.d_inner * cfg.d_state
                     + (cfg.d_inner + 2 * cfg.d_state) * (cfg.d_conv - 1))
        return cfg.n_layer * per_layer * dtype_bytes

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block {self.cfg.block_size}"
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.cfg.z_loss > 0:
                z = torch.logsumexp(logits.float(), dim=-1)
                loss = loss + self.cfg.z_loss * (z * z).mean()
        return logits, loss

    def alloc_state(self, batch: int, device):
        return [l.mixer.alloc_state(batch, device) for l in self.layers]

    @torch.no_grad()
    def step(self, idx, states):
        """One-token recurrent decode. idx: (B,) last token ids."""
        x = self.tok_emb(idx)
        for layer, st in zip(self.layers, states):
            x = layer.step(x, st)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Recurrent generation: prefill the prompt through step(), then decode.

        Unlike the transformer there is no context window to crop — the state
        just keeps integrating.
        """
        states = self.alloc_state(idx.size(0), idx.device)
        logits = None
        for t in range(idx.size(1)):
            logits = self.step(idx[:, t], states)
        for _ in range(max_new_tokens):
            lg = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            logits = self.step(nxt[:, 0], states)
        return idx


def _selftest():
    """Gate: recurrent step == quadratic forward, fp32, random weights."""
    torch.manual_seed(0)
    cfg = Mamba2Config(vocab_size=193, n_layer=2, block_size=64)
    m = Mamba2(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 48))
    logits_full, _ = m(idx)
    states = m.alloc_state(2, idx.device)
    step_logits = [m.step(idx[:, t], states) for t in range(48)]
    logits_step = torch.stack(step_logits, dim=1)
    d = (logits_full - logits_step).abs().max().item()
    denom = logits_full.flatten().norm() * logits_step.flatten().norm()
    cos = (logits_full.flatten() @ logits_step.flatten() / denom).item()
    n_par = m.num_params()
    print(f"params(L{cfg.n_layer})={n_par:,}  "
          f"state/stream={m.state_bytes(1):,} B (int8) {m.state_bytes(2):,} B (fp16)")
    full = Mamba2(Mamba2Config(vocab_size=193)).num_params()
    print(f"params(default L7 d256)={full:,}")
    print(f"recurrent-vs-quadratic: maxabsdiff={d:.3e}  cosine={cos:.9f}")
    ok = d < 1e-3 and cos > 0.999999
    print("MAMBA2_SELFTEST_VERDICT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
