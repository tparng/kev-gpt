"""Render a training run's garbage -> text progression into readable markdown.

Reads either the structured states.jsonl that train.py writes, or an older
train.log, and emits a loss table plus the sample at each checkpoint so you can
watch the model climb from random characters to telegraphic Kevin.

    python -m model.evolution data/states.jsonl -o data/evolution.md
    python -m model.evolution data/train.log            # prints to stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# matches: "iter   500 | train 1.249 | val 1.285 | lr 9.8e-04 | 4.8 min"
LOG_RE = re.compile(
    r"iter\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)"
    r"\s*\|\s*lr\s+(\S+)\s*\|\s*([\d.]+)\s*min"
)


def parse_jsonl(path: str) -> list[dict]:
    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))
    return states


def parse_log(path: str) -> list[dict]:
    """Parse train.py's printed format: an 'iter ...' line followed by a
    '   sample: ...' line."""
    states = []
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = LOG_RE.search(line)
        if not m:
            continue
        sample = ""
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt.startswith("sample:"):
                sample = nxt[len("sample:"):].strip()
        states.append({
            "iter": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "val_loss": float(m.group(3)),
            "lr": float(m.group(4)),
            "minutes": float(m.group(5)),
            "sample": sample,
        })
    return states


def load_states(path: str) -> list[dict]:
    states = parse_jsonl(path) if path.endswith(".jsonl") else parse_log(path)
    return sorted(states, key=lambda s: s["iter"])


def render(states: list[dict]) -> str:
    if not states:
        return "_no states found_\n"
    out = ["# Kevin GPT — from garbage to text", ""]
    out.append("How a 3M-param char-level model trained on Kevinised TinyStories "
               "went from random characters to telegraphic Kevin-speak.")
    out.append("")
    out.append("| iter | train | val | minutes |")
    out.append("|---:|---:|---:|---:|")
    for s in states:
        out.append(f"| {s['iter']} | {s['train_loss']:.3f} | "
                   f"{s['val_loss']:.3f} | {s['minutes']:.1f} |")
    out.append("")
    out.append("## Samples at each checkpoint")
    out.append("")
    out.append("_Fixed-seed continuations from an empty prompt; `|` marks a story "
               "boundary._")
    out.append("")
    for s in states:
        out.append(f"### iter {s['iter']}  ·  val {s['val_loss']:.3f}  ·  "
                   f"{s['minutes']:.1f} min")
        out.append("")
        out.append("```")
        out.append(s["sample"] or "(none)")
        out.append("```")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.evolution",
                                description="Render training progression to markdown.")
    p.add_argument("states", nargs="?", default="data/states.jsonl",
                   help="states.jsonl or train.log")
    p.add_argument("-o", "--output", default=None, help="markdown file (default: stdout)")
    args = p.parse_args(argv)

    md = render(load_states(args.states))
    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        sys.stderr.write(f"wrote {args.output}\n")
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
