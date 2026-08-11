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
FACTS_RAW = {
    "name":   ("my name is {v}.",             "what is my name?",       "your name is {v}."),
    "colour": ("my favourite colour is {v}.", "what colour do i like?", "you like {v}."),
    "pet":    ("i have a pet {v}.",           "what pet do i have?",    "you have a {v}."),
    "toy":    ("my best toy is my {v}.",      "what is my best toy?",   "your best toy is your {v}."),
    "place":  ("i love going to the {v}.",    "where do i love going?", "you love going to the {v}."),
}
SMALLTALK_RAW = [
    ("how are you today?", "i feel good today, thank you for asking."),
    ("tell me a story.", "once upon a time there was a little bear who found some honey. he was very happy."),
    ("what did you do today?", "i sat on my chip and thought some very fast thoughts."),
    ("do you like the sun?", "yes, the sun feels warm and nice."),
    ("i feel happy today.", "that is good. being happy is the best thing."),
    ("it is raining today.", "rain makes the flowers grow, so that is okay."),
    ("i went to school today.", "school is a fun place to learn new things."),
    ("i ate some cake.", "cake tastes sweet. lucky you!"),
    ("you are very fast.", "i think quickly. few words do the trick."),
    ("i am tired now.", "rest is a good idea. sleep well."),
]
ACKS_RAW = ["it is nice to meet you, {name}.", "that is nice.", "i will remember that.",
            "that is good to know.", "that sounds fun.", "i like that too."]

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


def dialogue(rng: random.Random, min_turns=4, max_turns=10,
             style: str = "kevin") -> tuple[str, int]:
    """One dialogue. Returns (text, gap_chars between fact statement and query)."""
    raw = style == "raw"
    facts, smalltalk, acks = ((FACTS_RAW, SMALLTALK_RAW, ACKS_RAW) if raw
                             else (FACTS, SMALLTALK, ACKS))
    upre, kpre = ("user: ", "kevin: ") if raw else ("user say ", "kevin say ")
    fact = rng.choice(list(facts))
    value = rng.choice(VALUES[fact])
    stmt, query, answer = facts[fact]
    name = value if fact == "name" else rng.choice(NAMES)

    hello = "hello kevin! " if raw else "hello kevin "
    turns = [(upre + hello + stmt.format(v=value),
              kpre + ("hello! " if raw else "hello ")
              + rng.choice(acks).format(name=name))]
    n_filler = rng.randint(min_turns, max_turns) - 2
    fillers = rng.sample(smalltalk, k=min(n_filler, len(smalltalk)))
    for q, a in fillers:
        turns.append((upre + q, kpre + a))
    turns.append((upre + query, kpre + answer.format(v=value)))

    flat = " ".join(u + " " + k for u, k in turns)
    # gap = chars between end of the fact statement and start of the query
    stmt_end = flat.index(turns[0][1])           # kevin's first reply starts
    query_start = flat.rindex(upre + query)
    return flat, query_start - stmt_end


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.chatgen")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("-o", default=None, help="output file (default stdout)")
    p.add_argument("--min-turns", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--style", choices=["kevin", "raw"], default="kevin",
                   help="kevin = telegraphic (kevinised corpus); raw = "
                        "grammatical English (raw/legit corpus)")
    p.add_argument("--names-file", default=None,
                   help="one name per line (e.g. data/ts_names.txt mined from "
                        "the corpus). Overrides the 10 built-ins — a big "
                        "inventory kills the guess-the-frequent-name shortcut "
                        "that v1 collapsed to.")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    if args.names_file:
        loaded = [l.strip().lower() for l in open(args.names_file) if l.strip()]
        VALUES["name"] = loaded
        NAMES[:] = loaded
    out = open(args.o, "w", encoding="utf-8") if args.o else sys.stdout
    lens, gaps = [], []
    for _ in range(args.n):
        text, gap = dialogue(rng, args.min_turns, args.max_turns, args.style)
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
