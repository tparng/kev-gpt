"""Char-level tokeniser and data prep for the Kevin corpus.

Char-level on purpose for this first model: zero dependencies, no OOV, no
tokeniser training, and it keeps the vocab (hence the embedding/head params)
tiny — which is the whole game at the on-chip budget. The real FPGA model may
move to a small BPE sized to the URAM budget, but char-level proves the pipeline
and still produces telegraphic Kevin text.

The "<|endoftext|>" story delimiter is collapsed to a single newline, so the
model learns story boundaries as one char rather than a 13-char literal.
"""

from __future__ import annotations

import json
import os

import numpy as np

MARKER = "<|endoftext|>"
SEP = "\n"  # single-char story boundary


def build_text(corpus_path: str) -> str:
    """Read a *.kevin.txt corpus into one string, stories separated by SEP."""
    stories, current = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == MARKER:
                if current:
                    stories.append(" ".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        stories.append(" ".join(current))
    return SEP.join(stories) + SEP


def prepare(corpus_path: str, out_dir: str, val_frac: float = 0.01) -> dict:
    """Encode a corpus to train.bin / val.bin + meta.json under out_dir.

    Returns the meta dict. Skips re-encoding if the bins already exist and match
    this corpus path.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    if all(os.path.exists(p) for p in (meta_path, train_path, val_path)):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("source") == os.path.abspath(corpus_path):
            return meta

    text = build_text(corpus_path)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    n_val = int(len(ids) * val_frac)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "source": os.path.abspath(corpus_path),
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": {str(i): c for i, c in itos.items()},
        "total_chars": len(ids),
        "train_chars": len(train_ids),
        "val_chars": len(val_ids),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_split(out_dir: str, split: str) -> np.ndarray:
    path = os.path.join(out_dir, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_meta(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "meta.json")) as f:
        return json.load(f)


def encode(text: str, stoi: dict) -> list:
    return [stoi[c] for c in text if c in stoi]


def decode(ids, itos: dict) -> str:
    # itos keys are strings when loaded from json
    return "".join(itos[str(int(i))] if isinstance(itos.get(str(int(i))), str)
                   else itos[int(i)] for i in ids)


if __name__ == "__main__":
    import sys

    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/TinyStories-valid.kevin.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/char"
    meta = prepare(corpus, out)
    print(f"vocab_size={meta['vocab_size']}  "
          f"train_chars={meta['train_chars']:,}  val_chars={meta['val_chars']:,}")
    print("chars:", "".join(sorted(meta["stoi"], key=lambda c: meta["stoi"][c])))
