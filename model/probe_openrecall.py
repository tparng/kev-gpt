"""Open-schema recall gate (chatgen v4): did the copy circuit form?

Probes "my favourite {x} is {y}. ... what is my favourite {x}?" through the
chat register, with x drawn from (a) attributes SEEN in training and (b) the
HELD-OUT list never used as an attribute. If held-out recall tracks seen
recall, the model learned the schema (general); if held-out collapses while
seen holds, it memorized templates. Values y are always random nouns, so
value recall is copying either way.

    python -m model.probe_openrecall data/ckpt.mamba2.l10chat4.pt
    python -m model.probe_openrecall data/ckpt.mamba2.l10chat3.pt   # v3 baseline
"""

from __future__ import annotations

import argparse
import random
import re

import torch

from .chat_demo import ChatSession, Codec, load
from .chatgen import SMALLTALK_RAW
from .train import pick_device


def probe(session: ChatSession, attr: str, value: str, n_filler: int,
          rng: random.Random) -> str:
    session.reset()
    torch.manual_seed(rng.randrange(1 << 30))
    turns = [f"my favourite {attr} is {value}."]
    turns += [q for q, _ in rng.sample(SMALLTALK_RAW, k=n_filler)]
    turns += [f"what is my favourite {attr}?"]
    reply = ""
    for msg in turns:
        reply = ""
        for ev in session.stream(msg, temperature=0.3):
            if ev.get("text"):
                reply += ev["text"]
            if "replace" in ev:
                reply = ev["replace"]
    return reply.strip()


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.probe_openrecall")
    p.add_argument("ckpt")
    p.add_argument("--nouns-file", default="data/ts_nouns.txt")
    p.add_argument("--heldout-file", default="data/heldout_attrs.txt")
    p.add_argument("--trials", type=int, default=50, help="per condition")
    p.add_argument("--fillers", type=int, default=3, help="gap turns")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, meta, it, val = load(args.ckpt, device)
    session = ChatSession(model, Codec(meta), device)
    nouns = [l.strip() for l in open(args.nouns_file) if l.strip()]
    held = [l.strip() for l in open(args.heldout_file) if l.strip()]
    seen = [n for n in nouns if n not in set(held)]
    rng = random.Random(args.seed)
    print(f"# {args.ckpt} (iter {it}, val {val:.4f})  {device}  "
          f"trials={args.trials}/cond  fillers={args.fillers}")

    results = {}
    for cond, attrs in (("seen", seen), ("heldout", held)):
        hit = 0
        for i in range(args.trials):
            attr = attrs[i % len(attrs)] if cond == "heldout" else rng.choice(attrs)
            value = rng.choice(nouns)
            reply = probe(session, attr, value, args.fillers, rng)
            words = re.findall(r"[a-z]+", reply.lower())
            ok = value in words
            hit += ok
            if args.verbose or not ok:
                print(f"  [{cond}] {attr}={value} -> {reply!r}  "
                      f"{'OK' if ok else 'MISS'}")
        results[cond] = hit
        print(f"{cond}: {hit}/{args.trials} exact value recall")

    s, h = results["seen"], results["heldout"]
    verdict = ("GENERAL (copy circuit)" if h >= 0.8 * max(s, 1) and h > 0
               else "TEMPLATE (memorized)" if s > 2 * max(h, 1)
               else "WEAK (neither strong)")
    print(f"OPENRECALL_VERDICT: seen {s}/{args.trials}  "
          f"heldout {h}/{args.trials}  -> {verdict}")


if __name__ == "__main__":
    main()
