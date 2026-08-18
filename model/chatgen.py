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

# --- v3 (raw style only): wider fact shapes, paraphrased queries, facts
# stated anywhere in the dialogue, 1-3 facts per dialogue. The live demo
# showed v2's holes exactly: "i like bikes" (shape not in inventory) and
# facts stated after the greeting turn (position-locked binding) both fail.

LIKES = ["bikes", "trains", "dogs", "cats", "ice cream", "football", "drawing",
         "swimming", "singing", "apples", "robots", "the rain", "books",
         "flowers", "trucks", "dancing"]
FOODS = ["cake", "pizza", "apples", "soup", "bread", "ice cream", "cookies",
         "pasta", "cheese", "bananas"]
AGES = ["three", "four", "five", "six", "seven", "eight", "nine", "ten"]

# key -> (statement variants, query variants, answer). First query is the
# canonical phrasing (probe_recall depends on it staying put).
FACTS_V3 = {
    "name":   (["my name is {v}.", "i am called {v}."],
               ["what is my name?", "what is my name again?",
                "do you remember my name?", "who am i?"],
               "your name is {v}."),
    "colour": (["my favourite colour is {v}."],
               ["what colour do i like?", "what is my favourite colour?"],
               "you like {v}."),
    "pet":    (["i have a pet {v}."],
               ["what pet do i have?", "do you remember what pet i have?"],
               "you have a {v}."),
    "toy":    (["my best toy is my {v}."],
               ["what is my best toy?", "what is my best toy again?"],
               "your best toy is your {v}."),
    "place":  (["i love going to the {v}."],
               ["where do i love going?", "where do i like to go?"],
               "you love going to the {v}."),
    "like":   (["i like {v}.", "i really like {v}."],
               ["what do i like?", "what do i like again?",
                "do you remember what i like?"],
               "you like {v}."),
    "food":   (["my favourite food is {v}."],
               ["what is my favourite food?", "what food do i like?"],
               "your favourite food is {v}."),
    "age":    (["i am {v} years old."],
               ["how old am i?", "do you remember how old i am?"],
               "you are {v} years old."),
    "friend": (["my friend is called {v}."],
               ["what is my friend called?", "who is my friend?"],
               "your friend is called {v}."),
    "live":   (["i live near the {v}."],
               ["where do i live?", "do you remember where i live?"],
               "you live near the {v}."),
}
VALUES_V3 = {"name": NAMES, "colour": COLOURS, "pet": PETS, "toy": TOYS,
             "place": PLACES, "like": LIKES, "food": FOODS, "age": AGES,
             "friend": NAMES, "live": PLACES}


def _v3_inst(rng: random.Random, key: str):
    stmts, queries, answer = FACTS_V3[key]
    if key == "pet" and rng.random() < 0.4:  # pet-name: two-slot fact
        p, n = rng.choice(PETS), rng.choice(NAMES)
        return (f"my {p} is called {n}.",
                rng.choice([f"what is my {p} called?",
                            f"what is the name of my {p}?"]),
                f"your {p} is called {n}.", None)
    v = rng.choice(VALUES_V3[key])
    return (rng.choice(stmts).format(v=v),
            rng.choice(queries).format(v=v),
            answer.format(v=v),
            v if key in ("name", "friend") else None)


def dialogue_v3(rng: random.Random, min_turns=5, max_turns=12) -> tuple[str, int]:
    """One raw-style dialogue with 1-3 facts at random positions.

    Returns (text, max fact gap in chars).
    """
    insts = [_v3_inst(rng, key)
             for key in rng.sample(list(FACTS_V3), k=rng.randint(1, 3))]
    return _assemble(rng, insts, min_turns, max_turns)


# v4: open-schema facts. The attribute AND value slots draw from a
# corpus-mined noun inventory (~2.6k), so the (x, y) pair space is far too
# big to memorize — the only compressive solution is the copy circuit, which
# is what generalizes to attributes never seen in training. Gate: a held-out
# attribute list excluded from generation entirely, probed after training.

def dialogue_v4(rng: random.Random, nouns: list, min_turns=5, max_turns=12,
                p_typed=0.3) -> tuple[str, int]:
    insts, used = [], set()
    for _ in range(rng.randint(1, 3)):
        if rng.random() < p_typed:
            # v3 "like" would collide with the open-schema like query
            key = rng.choice([k for k in FACTS_V3 if k != "like"])
            insts.append(_v3_inst(rng, key))
            continue
        fam = rng.choice(("fav", "fav", "called", "like", "have"))
        if fam in used:  # "what do i like/have?" can't bind two answers
            fam = "fav"
        used.add(fam) if fam in ("like", "have") else None
        x = rng.choice(nouns)
        while x in used:
            x = rng.choice(nouns)
        used.add(x)
        if fam == "fav":
            y = rng.choice(nouns)
            insts.append((f"my favourite {x} is {y}.",
                          rng.choice([f"what is my favourite {x}?",
                                      f"what is my favourite {x} again?",
                                      f"do you remember my favourite {x}?"]),
                          f"your favourite {x} is {y}.", None))
        elif fam == "called":
            n = rng.choice(NAMES)
            insts.append((f"my {x} is called {n}.",
                          rng.choice([f"what is my {x} called?",
                                      f"what is the name of my {x}?",
                                      f"what is my {x}s name?"]),
                          f"your {x} is called {n}.", None))
        elif fam == "like":
            insts.append((rng.choice([f"i like {x}.", f"i really like {x}."]),
                          rng.choice(["what do i like?",
                                      "do you remember what i like?"]),
                          f"you like {x}.", None))
        else:  # have
            q, a = rng.choice([("what do i have?", f"you have a {x}."),
                               (f"do i have a {x}?", f"yes, you have a {x}.")])
            insts.append((f"i have a {x}.", q, a, None))
    return _assemble(rng, insts, min_turns, max_turns)


# v5: interleaved fact-story dialogues (the state-retention corpus). The
# payoff model recalls facts 28/30 but a story flowing through the recurrent
# state flushes them (post-story "what is my name?" -> wrong). No training
# example ever required retention ACROSS a generated story; these do: facts
# stated, a real story told mid-dialogue, facts queried after it.

def dialogue_v5(rng: random.Random, nouns: list, pairs: list,
                min_turns=5, max_turns=10) -> tuple[str, int]:
    insts = [_v3_inst(rng, rng.choice([k for k in FACTS_V3 if k != "like"]))
             if rng.random() < 0.4 else _open_inst(rng, nouns)
             for _ in range(rng.randint(1, 2))]
    topic, story = rng.choice(pairs)
    ask = rng.choice(["tell me a story about a {x}.",
                      "can you tell me a story about a {x}?",
                      "tell me a story."]).format(x=topic)
    story_turn = ("user: " + ask + " kevin: " + story)

    n_filler = max(0, rng.randint(min_turns, max_turns) - 2 * len(insts) - 1)
    fillers = rng.sample(SMALLTALK_RAW, k=min(n_filler, len(SMALLTALK_RAW)))
    turns = []
    for stmt, _, _, name in insts:            # facts first (with greeting)
        ack = rng.choice(ACKS_RAW)
        if "{name}" in ack:
            ack = ack.format(name=name) if name else rng.choice(ACKS_RAW[1:])
        turns.append("user: " + stmt + " kevin: " + ("hello! " if not turns else "") + ack)
    for q, a in fillers[: n_filler // 2]:
        turns.append("user: " + q + " kevin: " + a)
    turns.append(story_turn)                  # the story flows through the state
    for q, a in fillers[n_filler // 2:]:
        turns.append("user: " + q + " kevin: " + a)
    for _, query, answer, _ in insts:         # queries AFTER the story
        turns.append("user: " + query + " kevin: " + answer)
    turns[0] = "user: hello kevin! " + turns[0][len("user: "):]

    flat = " ".join(turns)
    gap = flat.rindex("user: " + insts[0][1]) if insts else 0
    return flat, gap


def _open_inst(rng: random.Random, nouns: list):
    fam = rng.choice(("fav", "called", "like"))
    x = rng.choice(nouns)
    if fam == "fav":
        y = rng.choice(nouns)
        return (f"my favourite {x} is {y}.",
                rng.choice([f"what is my favourite {x}?",
                            f"do you remember my favourite {x}?"]),
                f"your favourite {x} is {y}.", None)
    if fam == "called":
        n = rng.choice(NAMES)
        return (f"my {x} is called {n}.",
                rng.choice([f"what is my {x} called?", f"what is my {x}s name?"]),
                f"your {x} is called {n}.", None)
    return (rng.choice([f"i like {x}.", f"i really like {x}."]),
            rng.choice(["what do i like?", "do you remember what i like?"]),
            f"you like {x}.", None)


def _assemble(rng: random.Random, insts: list, min_turns: int,
              max_turns: int) -> tuple[str, int]:
    n_facts = len(insts)
    n_filler = max(0, rng.randint(min_turns, max_turns) - 2 * n_facts)
    # statements land early-ish, queries late-ish (long gaps are the point —
    # they teach retention), but positions still vary so binding isn't
    # greeting-locked like v2
    order = []
    for i in range(n_facts):
        s = rng.uniform(0.0, 0.6)
        order.append((s, ("s", i)))
        order.append((rng.uniform(max(s + 0.15, 0.55), 1.0), ("q", i)))
    for j in range(n_filler):
        order.append((rng.uniform(0.0, 1.0), ("f", j)))
    events = [e for _, e in sorted(order)]

    fillers = rng.sample(SMALLTALK_RAW, k=min(n_filler, len(SMALLTALK_RAW)))
    while len(fillers) < n_filler:
        fillers.append(rng.choice(SMALLTALK_RAW))

    turns, spans = [], {}
    for kind, i in events:
        if kind == "s":
            stmt, _, _, name = insts[i]
            ack = rng.choice(ACKS_RAW)
            if "{name}" in ack:
                ack = (ack.format(name=name) if name
                       else rng.choice(ACKS_RAW[1:]))
            turns.append((stmt, "hello! " + ack if not turns else ack))
            spans[i] = [len(turns) - 1, None]
        elif kind == "q":
            _, query, answer, _ = insts[i]
            turns.append((query, answer))
            spans[i][1] = len(turns) - 1
        else:
            q, a = fillers[i]
            turns.append((q, a))
    if not events or events[0][0] != "s":
        u, k = turns[0]
        turns[0] = (u, "hello! " + k if not k.startswith("hello") else k)
    turns[0] = ("hello kevin! " + turns[0][0], turns[0][1])

    flat_turns = ["user: " + u + " kevin: " + k for u, k in turns]
    flat = " ".join(flat_turns)
    gap = 0
    for si, qi in spans.values():
        start = sum(len(t) + 1 for t in flat_turns[:si]) + len(flat_turns[si])
        qstart = sum(len(t) + 1 for t in flat_turns[:qi])
        gap = max(gap, qstart - start)
    return flat, gap


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
    p.add_argument("--v3", action="store_true",
                   help="raw style only: 1-3 facts per dialogue at random "
                        "positions, paraphrased queries, wider fact shapes "
                        "(like/food/age/friend/pet-name/live)")
    p.add_argument("--v4", action="store_true",
                   help="raw style only: open-schema facts — attribute and "
                        "value slots drawn from --nouns-file so the pair "
                        "space can't be memorized; 30% v3 typed facts mixed in")
    p.add_argument("--v5", action="store_true",
                   help="raw style only: facts stated, a REAL story told "
                        "mid-dialogue, facts queried after it (state "
                        "retention through generation)")
    p.add_argument("--stories-file", default="data/story_pairs.tsv",
                   help="topic<TAB>story pairs for --v5 story turns")
    p.add_argument("--nouns-file", default="data/ts_nouns.txt",
                   help="corpus-mined noun inventory for the v4 open slots")
    p.add_argument("--heldout-file", default=None,
                   help="nouns NEVER used as v4 attributes (the "
                        "generalization gate probes these after training)")
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
    if (args.v3 or args.v4 or args.v5) and args.style != "raw":
        p.error("--v3/--v4/--v5 are raw style only")
    nouns = pairs = None
    if args.v5:
        nouns = [l.strip() for l in open(args.nouns_file) if l.strip()]
        if args.heldout_file:
            held = {l.strip() for l in open(args.heldout_file) if l.strip()}
            nouns = [n for n in nouns if n not in held]
        pairs = [tuple(l.rstrip("\n").split("\t", 1))
                 for l in open(args.stories_file, encoding="utf-8")]
        print(f"v5: {len(nouns)} nouns, {len(pairs)} stories", file=sys.stderr)
    elif args.v4:
        nouns = [l.strip() for l in open(args.nouns_file) if l.strip()]
        if args.heldout_file:
            held = {l.strip() for l in open(args.heldout_file) if l.strip()}
            nouns = [n for n in nouns if n not in held]
        print(f"v4 noun inventory: {len(nouns)}", file=sys.stderr)
    for _ in range(args.n):
        if args.v5:
            text, gap = dialogue_v5(rng, nouns, pairs,
                                    max(args.min_turns, 5), args.max_turns)
        elif args.v4:
            text, gap = dialogue_v4(rng, nouns, max(args.min_turns, 5),
                                    args.max_turns)
        elif args.v3:
            text, gap = dialogue_v3(rng, max(args.min_turns, 5), args.max_turns)
        else:
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
