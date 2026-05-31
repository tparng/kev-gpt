"""Download the TinyStories corpus.

TinyStories (Eldan & Li, 2023) is the lead corpus for this project: it is the
public dataset where a 2-4M parameter model still stays coherent. The plain-text
release stores one story per block, blocks separated by a "<|endoftext|>" line,
which the harness handles with --marker "<|endoftext|>".

stdlib only (urllib), so it runs unchanged on the Windows/Vivado box. Downloads
to data/ by default.

    python -m keviniser.fetch_tinystories            # validation split (~20 MB)
    python -m keviniser.fetch_tinystories --split train   # full train (~2 GB)
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"


def _ssl_context() -> ssl.SSLContext | None:
    """Build an SSL context that works around macOS python.org missing root
    certs by using certifi's bundle when available. Returns None to let urllib
    use its default (fine on Windows and Linux)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None
FILES = {
    "valid": "TinyStories-valid.txt",
    "train": "TinyStories-train.txt",
}


def _progress(done: int, total: int) -> None:
    if total <= 0:
        sys.stderr.write(f"\r  {done / 1e6:,.1f} MB")
    else:
        pct = 100 * done / total
        sys.stderr.write(f"\r  {done / 1e6:,.1f} / {total / 1e6:,.1f} MB ({pct:.0f}%)")
    sys.stderr.flush()


def fetch(split: str = "valid", dest_dir: str = "data", force: bool = False) -> str:
    if split not in FILES:
        raise SystemExit(f"unknown split {split!r}, choose from {list(FILES)}")
    os.makedirs(dest_dir, exist_ok=True)
    name = FILES[split]
    dest = os.path.join(dest_dir, name)

    if os.path.exists(dest) and not force:
        sys.stderr.write(f"{dest} already exists ({os.path.getsize(dest) / 1e6:.1f} MB), "
                         f"skipping. Use --force to redownload.\n")
        return dest

    url = f"{BASE}/{name}"
    sys.stderr.write(f"downloading {url}\n  -> {dest}\n")
    ctx = _ssl_context()
    try:
        with urllib.request.urlopen(url, context=ctx) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            chunk = 1 << 20  # 1 MB
            with open(dest, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    _progress(done, total)
    except Exception:
        # don't leave a half-written file behind
        if os.path.exists(dest):
            os.remove(dest)
        raise
    sys.stderr.write(f"\ndone: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)\n")
    sys.stderr.write(
        "\nnext: kevinise it (note the document marker)\n"
        f"  python -m keviniser.harness {dest} "
        f'-o {dest.replace(".txt", ".kevin.txt")} '
        '--marker "<|endoftext|>" --stats data/tinystories.stats.json\n'
    )
    return dest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="keviniser.fetch_tinystories",
        description="Download the TinyStories corpus to data/.",
    )
    p.add_argument("--split", choices=list(FILES), default="valid",
                   help="which split to download (default: valid, ~20 MB).")
    p.add_argument("--dest", default="data", help="destination directory.")
    p.add_argument("--force", action="store_true", help="redownload if present.")
    args = p.parse_args(argv)
    fetch(args.split, args.dest, args.force)


if __name__ == "__main__":
    main()
