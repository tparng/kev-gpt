"""On-board: preload the whole model into URAM once, validate the resident PL
forward bit-exact vs numpy, and time generation (should be ~5-10x faster than the
re-streaming backend).  cd ~/kevpl/plchat && sudo ~/kevweb/venv/bin/python pl_resident.py
"""
import time
import json
import numpy as np

from model.goformer_full import GoformerFull, load_params, numpy_matmul
from fabric.pl_gemv import PLResident

p = load_params("goformer.npz")
meta = json.load(open("goformer_meta.json"))
stoi, itos = meta["stoi"], meta["itos"]
dec = lambda ids: "".join(itos[str(int(i))] for i in ids)

pl = PLResident()
t0 = time.time()
nb = pl.preload(p)
print(f"preloaded {nb} bytes ({nb/1024:.0f} KB) into URAM in {time.time()-t0:.1f}s (one-time)")

ids = [stoi.get(c, 0) for c in "once upon"][:6]
lo_np = GoformerFull(p, matmul=numpy_matmul).forward(ids)
t0 = time.time()
lo_pl = GoformerFull(p, matmul=pl.matmul).forward(ids)
tpl = time.time() - t0
print(f"forward: identical={bool(np.array_equal(lo_np, lo_pl))} "
      f"maxdiff={float(np.abs(lo_np-lo_pl).max()):.2e}  pl_forward={tpl:.2f}s")

gf = GoformerFull(p, matmul=pl.matmul)
ids = [stoi.get(c, 0) for c in "once upon time"]
t0 = time.time()
out = gf.generate(ids, 20, p["block_size"], temperature=0.85, top_k=40,
                  rng=np.random.default_rng(0))
dt = time.time() - t0
print("KEVIN(PL-resident):", dec(out))
print(f"{dt:.1f}s / 20 tok  =  {dt/20:.2f}s/tok  (was ~15s/tok re-streaming)")
