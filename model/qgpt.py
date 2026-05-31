"""INT4-weight / INT8-activation QAT variant of the GPT in gpt.py.

The architecture mirrors gpt.GPT exactly so an FP best-val checkpoint loads as
a warm-start with strict=False (the .weight Parameters share keys; Brevitas's
extra quant scale params initialise fresh). Warm-starting from a converged FP
model is the standard QAT recipe and matters at this scale where a cold INT4
run wastes the early budget relearning what FP already knows.

Scope (deliberately narrow — this is the scaffold, not the final fabric path):
- Every nn.Linear in the FP model becomes a brevitas QuantLinear with INT4
  weights (per-channel, symmetric) and an INT8 input fake-quant. That captures
  ~all the weight-memory the fabric path cares about (qkv, attn proj, mlp.0,
  mlp.2, tied lm head).
- Embeddings, LayerNorm, and scaled_dot_product_attention stay FP. Embeddings
  are a lookup, not a matmul; LayerNorm and softmax become custom fabric ops in
  doc 2's plan and don't need fake-quant for the training story. Quantising the
  attention math is a future step (matters for the KV cache budget, but the
  weight story dominates first).

The validation that this is bit-honest against an FP reference (goformer,
cosine > 0.9999) is doc 3's next step and lives outside this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brevitas.nn import QuantLinear
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerChannelFloat,
)

from .gpt import GPT, GPTConfig


class Int4WeightPerChannel(Int8WeightPerChannelFloat):
    """Per-channel symmetric INT4 weight quant. Bit-width override on the
    Int8 base class is the documented brevitas idiom."""
    bit_width = 4


def _qlin(in_f: int, out_f: int, bias: bool = False) -> QuantLinear:
    return QuantLinear(
        in_f, out_f, bias=bias,
        weight_quant=Int4WeightPerChannel,
        input_quant=Int8ActPerTensorFloat,
        return_quant_tensor=False,
    )


class QBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)

        self.qkv = _qlin(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = _qlin(cfg.n_embd, cfg.n_embd)
        self.mlp_fc = _qlin(cfg.n_embd, 4 * cfg.n_embd)
        self.mlp_proj = _qlin(4 * cfg.n_embd, cfg.n_embd)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.mlp_drop = nn.Dropout(cfg.dropout)

    def _attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def _mlp(self, x):
        return self.mlp_drop(self.mlp_proj(F.gelu(self.mlp_fc(x))))

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self._mlp(self.ln2(x))
        return x


class QGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([QBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = _qlin(cfg.n_embd, cfg.vocab_size)
        # tied head: share the raw weight with the embedding. The QuantLinear's
        # forward fake-quantises the shared tensor; the embedding lookup uses it
        # in FP. The Parameter object is shared, so AdamW updates one tensor.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def load_fp_into_qat(qmodel: QGPT, fp_state_dict: dict) -> tuple[list, list]:
    """Warm-start a QGPT from an FP GPT checkpoint.

    QuantLinear.weight has the same key as nn.Linear.weight, so the FP weights
    drop straight in. Brevitas's input_quant scale parameters and the qkv/proj
    rename (mlp.0 -> mlp_fc, mlp.2 -> mlp_proj) won't match — both expected and
    handled here so the caller can see what didn't transfer.
    """
    remapped = {}
    for k, v in fp_state_dict.items():
        # FP block.mlp is an nn.Sequential: .0 = first Linear, .2 = second.
        nk = k.replace(".mlp.0.", ".mlp_fc.").replace(".mlp.2.", ".mlp_proj.")
        # FP attention sub-module is .attn.qkv / .attn.proj; QAT inlines them
        # onto the block as .qkv / .proj.
        nk = nk.replace(".attn.qkv.", ".qkv.").replace(".attn.proj.", ".proj.")
        remapped[nk] = v
    missing, unexpected = qmodel.load_state_dict(remapped, strict=False)
    return list(missing), list(unexpected)
