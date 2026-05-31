"""Concatenate the exported packed-INT4 weights into one binary blob for the
A53 baseline benchmark (and as the canonical on-chip image size).

The blob is the whole model's weight bytes back-to-back, in sorted layer order —
the same ~1.5 MB the fabric holds on-chip and the same bytes the A53 streams from
DDR, so the two paths are compared moving identical data.

    python -m fabric.stage0.make_blob                 # -> fabric/export/weights.bin
    python fabric/stage0/make_blob.py --out w.bin
"""

from __future__ import annotations

import argparse
import glob
import os


def make_blob(export_dir: str = "fabric/export", out: str | None = None) -> str:
    out = out or os.path.join(export_dir, "weights.bin")
    mems = sorted(glob.glob(os.path.join(export_dir, "*.w.mem")))
    if not mems:
        raise SystemExit(f"no *.w.mem in {export_dir} — run model.export_fabric first")
    total = 0
    with open(out, "wb") as bf:
        for m in mems:
            with open(m) as f:
                blob = bytes(int(ln, 16) for ln in f if ln.strip())
            bf.write(blob)
            total += len(blob)
    print(f"wrote {out}: {total} bytes ({total/1024:.1f} KB) from {len(mems)} layers")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--export", default="fabric/export")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    make_blob(args.export, args.out)


if __name__ == "__main__":
    main()
