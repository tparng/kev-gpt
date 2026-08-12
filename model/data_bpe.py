"""BPE variant of the data pipeline (doc 8 — the smartness-per-byte lever).

Same bins/meta contract as model.data but token-level: a trained `tokenizers`
BPE json (e.g. data/bpe_rawchat_1024.json) encodes the corpus to uint16 bins.
meta carries bpe=True + the tokenizer path so train.py's sampler can round-trip
prompts. Measured on the kevin corpus: vocab 1024 ~= 3.8 chars/tok (raw ~3.25)
-> ~3x fewer fabric passes per reply and ~3x more chars per context window,
for +128 KB of embedding at INT4.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .data import build_text


def prepare_bpe(corpus_path: str, out_dir: str, tokenizer_file: str,
                val_frac: float = 0.01) -> dict:
    from tokenizers import Tokenizer

    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if all(os.path.exists(os.path.join(out_dir, f)) for f in
           ("meta.json", "train.bin", "val.bin")):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("source") == os.path.abspath(corpus_path)
                and meta.get("tokenizer_file") == os.path.abspath(tokenizer_file)):
            return meta

    tok = Tokenizer.from_file(tokenizer_file)
    text = build_text(corpus_path)
    # chunked encode (a merge broken at each boundary is ~1 token per chunk — noise)
    CH = 16 * 1024 * 1024
    ids = []
    for i in range(0, len(text), CH):
        ids.extend(tok.encode(text[i:i + CH]).ids)
    vocab = tok.get_vocab_size()
    assert vocab <= 65535
    ids = np.array(ids, dtype=np.uint16)

    n_val = int(len(ids) * val_frac)
    ids[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
    ids[-n_val:].tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "source": os.path.abspath(corpus_path),
        "tokenizer_file": os.path.abspath(tokenizer_file),
        "bpe": True,
        "vocab_size": vocab,
        "total_chars": len(text),
        "total_tokens": len(ids),
        "chars_per_token": round(len(text) / len(ids), 3),
        "train_chars": int((len(ids) - n_val) * len(text) / len(ids)),
        "train_tokens": len(ids) - n_val,
        "val_tokens": n_val,
        # placeholders so char-path consumers fail loudly rather than weirdly
        "stoi": {}, "itos": {},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
