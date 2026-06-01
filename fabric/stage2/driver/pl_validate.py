"""On-board: isolate PL GEMV correctness (direct / cached / reloaded) then run the
full forward vs numpy.  cd ~/kevpl/plchat && sudo ~/kevweb/venv/bin/python pl_validate.py
"""
import sys
import time
import json
import numpy as np

from model.goformer_full import GoformerFull, load_params, numpy_matmul
from fabric.pl_gemv import PLGemv

p = load_params("goformer.npz")
meta = json.load(open("goformer_meta.json"))
stoi = meta["stoi"]
pl = PLGemv()
rng = np.random.default_rng(1)

def chk(tag, iw, x):
    yn = numpy_matmul(iw, x)
    yp = pl.matmul(iw, x)
    print(f"{tag:22s} match={np.array_equal(yn, yp)} maxdiff={int(np.abs(yn-yp).max())} "
          f"cyc={pl.last_cycles}")

print("== direct GEMV isolation ==")
iw_qkv = p["blocks"][0]["qkv"][0]
iw_mlp = p["blocks"][0]["mlp_fc"][0]
x1 = rng.integers(-128, 128, size=iw_qkv.shape[1]).astype(np.int64)
x2 = rng.integers(-128, 128, size=iw_qkv.shape[1]).astype(np.int64)
chk("qkv fresh", iw_qkv, x1)
chk("qkv cached(new x)", iw_qkv, x2)            # same weights, new activation
chk("mlp_fc reload", iw_mlp, rng.integers(-128, 128, size=iw_mlp.shape[1]).astype(np.int64))
chk("qkv re-reload", iw_qkv, x1)                # back to qkv -> must re-stream weights

print("== full forward ==")
prompt = sys.argv[1] if len(sys.argv) > 1 else "once upon"
ids = [stoi.get(c, 0) for c in prompt][:6] or [0]
lo_np = GoformerFull(p, matmul=numpy_matmul).forward(ids)
lo_pl = GoformerFull(p, matmul=pl.matmul).forward(ids)
diff = float(np.abs(lo_np - lo_pl).max())
print(f"forward maxabsdiff={diff:.3e} identical={bool(np.array_equal(lo_np, lo_pl))} "
      f"argmax np={int(lo_np[-1].argmax())} pl={int(lo_pl[-1].argmax())}")
print("PL_FORWARD_" + ("MATCH" if diff < 1e-9 else "MISMATCH"))
