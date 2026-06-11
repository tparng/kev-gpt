"""pl_kv256 — chat device for the doc-7 FAITHFUL single-stream KV bitstream.

Drives the R1 "Kevin remembers" sequencer (gemv_axi_seq_vec front end, IDCODE
"SQRV") whose KV banks PERSIST between GO pulses: one forward per GO at an
explicit POS, K8/V8 cache on-chip (the pinned doc-7 §4 R0 contract), so a
multi-token decode is just a train of GO pulses — prompt positions 0..L-1 fill
the KV window, then greedy feedback (TOK_OUT -> next TOK_ID) generates the
reply. Each request restarts at pos 0 and simply OVERWRITES the previous
request's KV (no reset needed between requests); the conversation memory lives
in the prompt = transcript tail, per doc-7 §6.

THE CRITICAL PROPERTY vs the t1/SQSB engines: this is REAL model-faithful text
at fabric speed — the whole decode (matmuls, attention over the growing window,
non-linears, argmax) runs in fabric; the host only feeds token ids. It is ONE
stream, so the daemon serves requests serially (the Precision-side batch
assembler degenerates to a queue, doc-7 §6).

Register map (gemv_axi_seq_vec.v):
  0x00 CTRL(b0 go,b1 wl_rst,b2 soft_reset)  0x04 STATUS(b0 done)
  0x08 TOK_ID  0x0C POS  0x10 W_DATA  0x24 TOK_OUT  0x28 CYCLES
  0x2C IDCODE(0x53515256 "SQRV")

Used by the A53 daemon:
    sudo ~/kevweb/venv/bin/python -m webchat.demo.a53_daemon --engine kv256 \
        --lanes 128 --fclk 200e6 --gen-chars 48 --tmax 256
"""

from __future__ import annotations

import asyncio
import mmap
import os
import sys
import threading
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fabric.stage3 import pack_banked, seq_ref  # noqa: E402
from fabric.stage3.board.pl_kv_chat import load_meta  # noqa: E402
from fabric.stage3.board.pl_seq_chat import set_and_verify_fclk  # noqa: E402

NLAYER = 4
IDCODE_SQRV = 0x53515256
BASE = 0xA0000000
R_CTRL, R_STATUS = 0x00, 0x04
R_TOKID, R_POS, R_WDATA = 0x08, 0x0C, 0x10
R_RDSEL, R_RDADDR, R_RDLO, R_RDHI = 0x14, 0x18, 0x1C, 0x20
R_TOKOUT, R_CYCLES, R_IDCODE = 0x24, 0x28, 0x2C
RD_SEL_HEAD = 8          # readback mux: final head logits (Q6.25), VOCAB lanes
HEAD_FRAC = 25           # Q6.25 fixed point


class Dev:
    """/dev/mem register window (same pattern as pl_seq_sb / pl_seq_vec)."""

    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)

    def wr(self, off, val): self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)
    def rd(self, off): return int(self.reg[off >> 2])

    def close(self):
        self.reg = None; self.mm.close(); os.close(self.fd)


def build_weight_image(intseq, lanes):
    """Full transposed INT4 image at `lanes`: per block (qkv|proj|mlp_fc|mlp_proj)
    for NLAYER blocks, then the head — the exact order the sequencer's w_base
    offsets assume (same as pl_seq_vec)."""
    p = intseq.p
    words = []
    for bi in range(NLAYER):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            words += pack_banked.pack_transposed(np.asarray(p["blocks"][bi][nm][0], np.int8), lanes)
    words += pack_banked.pack_transposed(np.asarray(p["head"][0], np.int8), lanes)
    # log SS36 fit-plan 2: the embed tables ride the weight image's spare depth
    # (shared packer with the sim writer so the two layouts cannot drift).
    from fabric.stage3.run_sequencer import wrom_embed_words
    emb = wrom_embed_words(intseq, lanes, len(words))
    if emb is not None:
        words += emb
    return words


class PLKV256Device:
    """Single-stream multi-token chat over the fabric-resident KV sequencer.

    Mirrors KVChatDevice's daemon-facing surface (infer / launches / submit_raw
    / close) so a53_daemon treats the engines interchangeably."""

    _STOPS = (". ", "! ", "? ", "\n")

    def __init__(self, *, lanes: int = 128, fclk: float = 125e6,
                 gen_chars: int = 48, tmax: int = 256,
                 temp: float = 0.0, top_k: int = 0,
                 npz: str | None = None, meta: str | None = None,
                 base: int = BASE, poll_timeout: float = 30.0):
        npz = npz or os.path.join(_REPO, "fabric", "export", "goformer.npz")
        meta = meta or os.path.join(_REPO, "fabric", "export", "goformer_meta.json")
        self.gen_chars = int(gen_chars)
        self.tmax = int(tmax)
        # SAMPLING (chat variety): the fabric decodes GREEDILY on-chip (argmax ->
        # TOK_OUT), so a fixed prompt is deterministic. temp>0 switches to
        # host-side temperature/top-k sampling: after each forward we read the
        # VOCAB head logits back through the debug readback port and sample on
        # the A53. temp==0 keeps the fast bit-exact greedy path (TOK_OUT, no
        # readback). The on-fabric decode and its gate are UNCHANGED — sampling
        # is a presentation layer, not a correctness claim.
        self.temp = float(temp)
        self.top_k = int(top_k)
        self._rng = np.random.default_rng()
        self.poll_timeout = float(poll_timeout)
        self._launches = 0
        self._lock = threading.Lock()           # one stream -> strictly serial

        self._enc, self._dec, self.vocab = load_meta(meta)
        self.fclk = set_and_verify_fclk(float(fclk))

        p, cfg = seq_ref.build(npz)
        intseq = seq_ref.IntSequencer(p, cfg)
        words = build_weight_image(intseq, int(lanes))
        subw = (int(lanes) * 4) // 32

        self._dev = Dev(int(base))
        idc = self._dev.rd(R_IDCODE)
        if idc != IDCODE_SQRV:
            self._dev.close()
            raise RuntimeError(f"IDCODE 0x{idc:08X} != SQRV 0x{IDCODE_SQRV:08X} "
                               "(wrong/missing kv256 bitstream?)")
        self._dev.wr(R_CTRL, 0x4); self._dev.wr(R_CTRL, 0x0)   # soft_reset
        self._dev.wr(R_CTRL, 0x2)                              # wl_rst
        t0 = time.time()
        for w in words:                          # boot-stream the weights ONCE
            w = int(w)
            for s in range(subw):
                self._dev.wr(R_WDATA, (w >> (32 * s)) & 0xFFFFFFFF)
        mode = (f"sample temp={self.temp} top_k={self.top_k}"
                if self.temp > 0 else "greedy")
        print(f"[kv256] weights: {len(words)*subw} chunks in {time.time()-t0:.2f}s "
              f"(fclk={self.fclk/1e6:.1f}MHz tmax={self.tmax} gen={self.gen_chars} "
              f"{mode})")

    # --- the fabric protocol ------------------------------------------------ #
    def _go(self, tok: int, pos: int) -> int:
        d = self._dev
        d.wr(R_TOKID, int(tok) & 0x1FF)
        d.wr(R_POS, int(pos) & 0x1FF)
        d.wr(R_CTRL, 0x1)                                      # go
        t0 = time.time()
        while not (d.rd(R_STATUS) & 0x1):
            if time.time() - t0 > self.poll_timeout:
                raise RuntimeError(f"kv256 TIMEOUT at pos={pos} "
                                   f"STATUS=0x{d.rd(R_STATUS):08X}")
        self._launches += 1
        return d.rd(R_TOKOUT) & 0x1FF

    def _read_logits(self) -> np.ndarray:
        """Read the VOCAB head logits of the just-finished forward back through
        the debug readback port (rd_sel=HEAD), as float (Q6.25 -> real).

        Tight inlined loop over the mmap register view: one addr write + one data
        read per logit. The 2-cycle registered readback settles inside the
        AXI-Lite round-trip (tens of fabric cycles), so no explicit settle reads
        are needed — verified argmax-equals-TOK_OUT on board."""
        reg = self._dev.reg
        a, lo = R_RDADDR >> 2, R_RDLO >> 2
        reg[R_RDSEL >> 2] = RD_SEL_HEAD
        out = np.empty(self.vocab, dtype=np.int64)
        for k in range(self.vocab):
            reg[a] = k
            out[k] = reg[lo]
        out = out.astype(np.float64)
        out[out >= (1 << 31)] -= (1 << 32)        # sign-extend the 32-bit logits
        return out / (1 << HEAD_FRAC)

    def _sample(self, logits: np.ndarray) -> int:
        """Temperature + optional top-k sample over the head logits."""
        z = logits / max(self.temp, 1e-6)
        z -= z.max()                              # stable softmax
        p = np.exp(z); p /= p.sum()
        if self.top_k and self.top_k < p.size:    # keep only the top-k mass
            drop = np.argpartition(p, -self.top_k)[:-self.top_k]
            p[drop] = 0.0; p /= p.sum()
        return int(self._rng.choice(p.size, p=p))

    def _next(self, argmax_tok: int) -> int:
        """The fabric's on-chip argmax (fast/greedy) or a host-side sample."""
        return self._sample(self._read_logits()) if self.temp > 0 else argmax_tok

    def _stream_gen(self, prompt: str):
        """Generator: yield each decoded char AS the fabric produces it. Holds NO
        lock (the caller serializes). Shared by the batch path (_gen_one) and the
        streaming path (stream_into) so the two cannot drift."""
        ids = self._enc((prompt or "").strip().lower())
        if not ids:
            ids = self._enc("\n") or [0]
        maxp = self.tmax - self.gen_chars        # prompt tail + reply <= window
        if len(ids) > maxp:
            ids = ids[-maxp:]
        am = 0
        for pos, tok in enumerate(ids):          # prompt phase: KV fills 0..L-1
            am = self._go(tok, pos)              # argmax out at pos L-1
        nxt = self._next(am)                     # 1st gen token (greedy or sampled)
        L = len(ids)
        text = ""
        for g in range(self.gen_chars):          # feedback phase
            ch = self._dec([nxt])
            text += ch
            yield ch
            if any(stp in text for stp in self._STOPS):
                break                            # _tidy drops the rest anyway
            if g + 1 < self.gen_chars:
                nxt = self._next(self._go(nxt, L + g))

    def _gen_one(self, prompt: str) -> str:
        return self._tidy("".join(self._stream_gen(prompt)))

    def stream_into(self, prompt: str, on_char) -> str:
        """Run ONE streaming generation under the serial lock, calling
        on_char(ch) as each char lands; return the tidied completion. The daemon
        uses this for the Enter-miss path so time-to-first-char is one token, not
        the full reply."""
        with self._lock:
            text = ""
            for ch in self._stream_gen(prompt):
                text += ch
                on_char(ch)
            return self._tidy(text)

    @staticmethod
    def _tidy(text: str) -> str:
        """One presentable utterance from a raw char-budget continuation (kept
        local to KVChatDevice._tidy's behavior so engines present alike, without
        a fabric->webchat import)."""
        text = text.split("\n", 1)[0]
        for stop in (". ", "! ", "? "):
            i = text.find(stop)
            if i != -1:
                return text[: i + 1].rstrip()
        if text.rstrip().endswith((".", "!", "?")):
            return text.rstrip()
        cut = text.rstrip()
        i = cut.rfind(" ")
        return cut[:i].rstrip() if i > 0 else cut

    # --- daemon-facing surface (parity with KVChatDevice / PLDevice) -------- #
    async def infer(self, prompts):
        """prompts: list of str, or (client_id, prompt) pairs — one stream, so
        client identity buys nothing here; requests run strictly serially."""
        loop = asyncio.get_running_loop()
        items = list(prompts)

        def work():
            with self._lock:
                return [self._gen_one(it[1] if isinstance(it, tuple) else it)
                        for it in items]

        return await loop.run_in_executor(None, work)

    @property
    def launches(self) -> int:
        return self._launches

    def submit_raw(self, tok_ids):
        raise NotImplementedError("--bench measures the T=1 PL pass; use --engine t1")

    def close(self):
        self._dev.close()
