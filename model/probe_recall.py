"""Associative-recall probe: name stated early, queried late (doc 8 §6/M1).

The "Kevin remembers your name" demo is an induction/copy task — the known
weakness of fixed-state models and the native strength of attention. This
probe measures it directly instead of trusting NLL.

Method: 5-way forced choice. A story states a character's name, a filler of G
chars follows, then a re-mention point ("everyone shout look it be ") where we
teacher-force each candidate name and sum its char logprobs. The stated name
should out-score the four distractors; accuracy over names x fillers, chance
= 20%. The transformer physically forgets once the statement falls out of its
256-char window (that is the crop, not a bug); the SSM's state decays instead
— this probe draws both curves.

    python -m model.probe_recall data/ckpt.fp.pt data/ckpt.mamba2.pt
    python -m model.probe_recall data/ckpt.mamba2.pt --gaps 0,200,800,3200
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

NAMES = ["sam", "lily", "tom", "ben", "mia"]

OPEN = "once upon time there be little bear name {name} {name} very happy bear "

# neutral kevin-speak filler, no names — repeated/sliced to G chars
FILLER = ("one day sun shine bird sing tree grow tall flower smell sweet "
          "them walk park see big pond water very blue duck swim fast "
          "wind blow soft cloud move slow day feel warm nice grass green ")

QUERY = "then door open everyone shout look it be "


def load_model(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg_d, arch = ck["cfg"], ck.get("arch")
    if arch is None:  # older ckpts: infer from config keys
        arch = "mamba2" if "d_state" in cfg_d else "gpt"
    if arch == "mamba2":
        from .mamba2 import Mamba2, Mamba2Config
        model = Mamba2(Mamba2Config(**cfg_d))
    else:
        from .gpt import GPT, GPTConfig
        model = GPT(GPTConfig(**cfg_d))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, arch, ck["meta"]


def enc(text: str, stoi: dict) -> list[int]:
    ids = [stoi.get(c) for c in text]
    assert None not in ids, f"char outside vocab in: {text!r}"
    return ids


@torch.no_grad()
def cand_logprob_gpt(model, prompt_ids, cand_ids, device):
    """Sum logprob of cand chars after prompt, cropped to the model's window."""
    seq = prompt_ids + cand_ids
    seq = seq[-(model.cfg.block_size):]
    n_cand = len(cand_ids)
    x = torch.tensor([seq], device=device)
    logits, _ = model(x)
    lp = F.log_softmax(logits[0].float(), dim=-1)
    # position i predicts seq[i+1]
    return sum(lp[len(seq) - n_cand - 1 + j, seq[len(seq) - n_cand + j]].item()
               for j in range(n_cand))


@torch.no_grad()
def cand_logprob_mamba(model, prompt_states, last_logits, cand_ids, device):
    """Sum logprob of cand chars continuing from a saved prompt state."""
    import copy
    states = [{k: v.clone() for k, v in st.items()} for st in prompt_states]
    total, logits = 0.0, last_logits
    for j, cid in enumerate(cand_ids):
        lp = F.log_softmax(logits[0].float(), dim=-1)
        total += lp[cid].item()
        if j < len(cand_ids) - 1:
            logits = model.step(torch.tensor([cid], device=device), states)
    return total


@torch.no_grad()
def run_probe(model, arch, meta, gaps, device):
    stoi = meta["stoi"]
    rows = []
    for gap in gaps:
        filler = (FILLER * (gap // len(FILLER) + 1))[:gap]
        hits = 0
        for name in NAMES:
            prompt = OPEN.format(name=name) + filler + QUERY
            pids = enc(prompt, stoi)
            if arch == "mamba2":
                states = model.alloc_state(1, device)
                logits = None
                for t in pids:
                    logits = model.step(torch.tensor([t], device=device), states)
                scores = {c: cand_logprob_mamba(model, states, logits,
                                                enc(c, stoi), device)
                          for c in NAMES}
            else:
                scores = {c: cand_logprob_gpt(model, pids, enc(c, stoi), device)
                          for c in NAMES}
            pick = max(scores, key=scores.get)
            hits += (pick == name)
        # is the statement still inside the transformer's window?
        seen = "yes" if arch == "mamba2" else (
            "yes" if len(enc(OPEN.format(name=NAMES[0]) + filler + QUERY, stoi))
            <= model.cfg.block_size else "NO (cropped)")
        rows.append((gap, hits, len(NAMES), seen))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.probe_recall")
    p.add_argument("ckpts", nargs="+", help="checkpoint(s) to probe")
    p.add_argument("--gaps", default="0,100,200,400,800,1600",
                   help="filler lengths (chars) between statement and query")
    p.add_argument("--device", default="cpu",
                   help="cpu by default so it can run beside a GPU training job")
    args = p.parse_args(argv)
    gaps = [int(g) for g in args.gaps.split(",")]

    for path in args.ckpts:
        model, arch, meta = load_model(path, args.device)
        print(f"\n{path}  arch={arch}  params={model.num_params()/1e6:.2f}M")
        print(f"  gap  acc      name-in-window")
        for gap, hits, n, seen in run_probe(model, arch, meta, gaps, args.device):
            print(f"  {gap:4d}  {hits}/{n}      {seen}")
    print("\nchance = 1/5 = 20%")


if __name__ == "__main__":
    main()
