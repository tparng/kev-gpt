"""Synthetic multi-turn Kevin-speak chat corpus (doc 8 §5 M2, the recall data).

The recall probe showed the architecture retains facts past the transformer's
window — but TinyStories contains nothing worth remembering across 800 chars,
so nothing that long is ever *learned*. This generator manufactures exactly
that dependency: dialogues where a fact stated early (name, colour, pet...)
is queried after a controllable filler gap. Mixed ~5-10% into the story corpus
it teaches retention without disturbing the story distribution.

Style matches the kevinised corpus: lowercase telegraphic, no punctuation, no
function words. Speaker convention "user say ... kevin say ..." — the same
framing the live chat can prompt with.

    python -m model.chatgen -n 100000 -o data/kevchat.txt --stats
    python -m model.chatgen -n 5 --stats          # eyeball a few on stdout
"""

from __future__ import annotations

import argparse
import random
import sys

NAMES = ["sam", "lily", "tom", "ben", "mia", "max", "sue", "anna", "tim", "amy"]
COLOURS = ["red", "blue", "green", "yellow", "pink", "purple"]
PETS = ["dog", "cat", "bird", "fish", "bunny", "frog"]
TOYS = ["ball", "kite", "doll", "car", "block", "drum"]
PLACES = ["park", "beach", "forest", "garden", "farm", "lake"]

# fact key -> (statement template, query template, answer template)
FACTS = {
    "name":   ("me name be {v}",        "what be me name",       "you name be {v}"),
    "colour": ("me favourite colour be {v}", "what colour me like", "you like {v}"),
    "pet":    ("me have pet {v}",       "what pet me have",      "you have {v}"),
    "toy":    ("me best toy be {v}",    "what be me best toy",   "you best toy be {v}"),
    "place":  ("me love go {v}",        "where me love go",      "you love go {v}"),
}
VALUES = {"name": NAMES, "colour": COLOURS, "pet": PETS, "toy": TOYS, "place": PLACES}

SMALLTALK = [
    ("how be you", "kevin feel good today"),
    ("tell me story", "once upon time there be little bear him find honey him very happy"),
    ("what you do today", "kevin sit chip think fast thought"),
    ("you like sun", "yes sun feel warm nice"),
    ("me feel happy", "that be good happy be best thing"),
    ("it rain today", "rain make flower grow it okay"),
    ("me go school today", "school be fun place learn thing"),
    ("me eat cake", "cake taste sweet lucky you"),
    ("you be fast", "kevin think quick few word do trick"),
    ("me tire now", "rest be good idea sleep well"),
]

ACKS = ["nice meet you {name}", "that be nice", "kevin remember that", "good know",
        "sound fun", "kevin like that too"]


def dialogue(rng: random.Random, min_turns=4, max_turns=10) -> tuple[str, int]:
    """One dialogue. Returns (text, gap_chars between fact statement and query)."""
    fact = rng.choice(list(FACTS))
    value = rng.choice(VALUES[fact])
    stmt, query, answer = FACTS[fact]
    name = value if fact == "name" else rng.choice(NAMES)

    turns = [("user say hello kevin " + stmt.format(v=value),
              "kevin say hello " + rng.choice(ACKS).format(name=name))]
    n_filler = rng.randint(min_turns, max_turns) - 2
    fillers = rng.sample(SMALLTALK, k=min(n_filler, len(SMALLTALK)))
    for q, a in fillers:
        turns.append(("user say " + q, "kevin say " + a))
    turns.append(("user say " + query,
                  "kevin say " + answer.format(v=value)))

    flat = " ".join(u + " " + k for u, k in turns)
    # gap = chars between end of the fact statement and start of the query
    stmt_end = flat.index(turns[0][1])           # kevin's first reply starts
    query_start = flat.rindex("user say " + query)
    return flat, query_start - stmt_end


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.chatgen")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("-o", default=None, help="output file (default stdout)")
    p.add_argument("--min-turns", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats", action="store_true")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    out = open(args.o, "w", encoding="utf-8") if args.o else sys.stdout
    lens, gaps = [], []
    for _ in range(args.n):
        text, gap = dialogue(rng, args.min_turns, args.max_turns)
        out.write(text + "\n<|endoftext|>\n")
        lens.append(len(text)); gaps.append(gap)
    if args.o:
        out.close()

    if args.stats:
        import numpy as np
        L, G = np.array(lens), np.array(gaps)
        print(f"dialogues: {len(L):,}  total chars: {L.sum():,}", file=sys.stderr)
        print(f"len:  mean {L.mean():.0f}  p50 {np.median(L):.0f}  "
              f"p95 {np.percentile(L,95):.0f}", file=sys.stderr)
        print(f"gap:  mean {G.mean():.0f}  p50 {np.median(G):.0f}  "
              f"p95 {np.percentile(G,95):.0f}  max {G.max()}", file=sys.stderr)
        print(f"gaps > 256 chars: {(G>256).mean()*100:.0f}%   "
              f"> 400: {(G>400).mean()*100:.0f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
