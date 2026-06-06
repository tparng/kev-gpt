"""backend_stub.py — two InferenceBackend implementations.

(a) StubBackend — deterministic fake completions with configurable latency. The
    completion is a pure function of the prompt (hash-seeded telegraphic
    gibberish) so a test can assert exact strings, and the modelled per-batch
    latency lets off-box load tests push the server's debounce/coalesce/batch
    machinery to its plumbing limit without any hardware. This is what loadgen
    rehearses the death experiment against.

(b) PLSingleTokenBackend — drives the REAL current PL (the N=16 single-token
    sequencer) over the pl_seq_pp16 register protocol, generating a completion
    by iterative single-token feedback: argmax token -> append -> re-submit.

    HONESTY (read this): tokens beyond the first are NOT model-faithful. The
    current fabric has no KV cache and attention runs at T=1, so feeding a token
    back as the next prompt does not reproduce what a real KV-decode would
    compute. This backend exists to (1) exercise the entire software plane
    against real AXI/GigE timing and (2) measure the A53 launch rate honestly —
    NOT to produce real text. The first token of each completion IS bit-faithful
    to seq_ref; everything after it is plumbing exercise only. The completion is
    tagged so the UI/telemetry never present it as real output.

The PL backend imports pl_seq_pp16 WITHOUT modifying it (the original is owned by
another agent / the board flow). On a dev box with no /dev/mem it raises at
construction, which is why the stub is the default everywhere off-box.
"""

from __future__ import annotations

import asyncio
import hashlib
import time

from .backend import (BatchOutcome, InferenceBackend, InferRequest,
                      InferResult)

# Telegraphic-ish vocabulary so the fake completions LOOK like Kevin-speak even
# though they are gibberish. Keeps the demo visually honest about the register.
_KEVIN_WORDS = [
    "few", "word", "do", "trick", "me", "go", "fast", "chip", "no", "ddr",
    "tiny", "model", "big", "claim", "fit", "chip", "run", "hot", "win",
    "type", "thing", "see", "fast", "now", "kevin", "speak", "short", "good",
]


def _fake_completion(prompt: str, n_words: int = 6) -> str:
    """Deterministic gibberish seeded by the prompt. Same prompt -> same words,
    so freshness/blit logic and tests are reproducible."""
    h = hashlib.sha256(prompt.encode("utf-8", "ignore")).digest()
    words = []
    for i in range(n_words):
        idx = h[i % len(h)] % len(_KEVIN_WORDS)
        words.append(_KEVIN_WORDS[idx])
        # cheap rolling mix so consecutive words differ
        h = hashlib.sha256(h + bytes([i])).digest()
    return " ".join(words)


class StubBackend(InferenceBackend):
    """Deterministic completions + modelled latency for off-box load tests.

    latency_ms is per-batch (not per-stream): the fabric does one pass for the
    whole batch, so batching is what amortises it. A small per-stream term can be
    added via `per_stream_ms` to model the fact that wider batches cost a touch
    more to drain. jitter_ms adds uniform-ish noise via the prompt hash so p95/p99
    are not degenerate.
    """

    def __init__(self, latency_ms: float = 8.0, per_stream_ms: float = 0.2,
                 jitter_ms: float = 3.0, n_words: int = 6, capacity: int = 16):
        self.latency_ms = latency_ms
        self.per_stream_ms = per_stream_ms
        self.jitter_ms = jitter_ms
        self.n_words = n_words
        self.capacity = capacity

    async def infer_batch(self, reqs: list[InferRequest]) -> BatchOutcome:
        filled = len(reqs)
        # deterministic jitter from the first prompt so the modelled latency is
        # reproducible per test but still spreads the percentile distribution.
        seed = reqs[0].prompt if reqs else ""
        jit = (int(hashlib.sha256(seed.encode("utf-8", "ignore")).hexdigest(), 16)
               % 1000) / 1000.0
        batch_ms = (self.latency_ms + self.per_stream_ms * filled
                    + self.jitter_ms * jit)
        # perf_counter, not monotonic: on Windows time.monotonic() has ~15 ms
        # granularity, so a single-digit-ms sleep reads back 0 and occupancy/
        # latency would be a flat 0. perf_counter is the high-res clock.
        t0 = time.perf_counter()
        await asyncio.sleep(batch_ms / 1000.0)
        busy_s = time.perf_counter() - t0
        results = [
            InferResult(client_id=r.client_id, prompt=r.prompt, seq=r.seq,
                        completion=_fake_completion(r.prompt, self.n_words))
            for r in reqs
        ]
        return BatchOutcome(results=results, busy_s=busy_s,
                            filled=filled, capacity=self.capacity)


class PLSingleTokenBackend(InferenceBackend):
    """Drives the real N=16 single-token PL via the pl_seq_pp16 register protocol.

    NOT model-faithful past the first token (no fabric KV; attention T=1). It
    submits up to 16 prompts as 16 token-streams in ONE PL pass, reads the 16
    argmax tokens, then iterates: append each stream's new token to its own
    prompt-tail and resubmit, `gen_chars` times. Each iteration is exactly one
    PL submission == one A53 launch, which is the number we actually want to
    measure.

    Construction does the one-time weight/embed streaming; per-batch cost is the
    ~20 register pokes + result reads the PRD quotes. Import of pl_seq_pp16 is
    lazy + guarded so importing this module off-box (no numpy/dev-mem) is fine.
    """

    PROMPT_TAIL = 8   # the bitstream's PROMPT_LEN — we feed the last 8 chars

    def __init__(self, *, lanes: int = 128, fclk: float = 125e6,
                 gen_chars: int = 6, npz: str | None = None,
                 meta: str | None = None):
        self.lanes = lanes
        self.fclk = fclk
        self.gen_chars = gen_chars
        self.capacity = 16
        self._init(npz, meta)

    def _init(self, npz, meta):
        # Lazy heavy imports; only available on the board image.
        import os
        import sys
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from fabric.stage3.board import pl_seq_pp16 as pp16   # noqa: E402
        self._pp16 = pp16
        npz = npz or os.path.join(repo, "fabric", "export", "goformer.npz")
        meta = meta or os.path.join(repo, "fabric", "export", "goformer_meta.json")

        # reference + codec (reuse the driver's helpers verbatim; no edits there)
        self._enc, self._dec, _ = self._load_meta(meta)
        p, cfg = pp16.seq_ref.build(npz)
        self._intseq = pp16.seq_ref.IntSequencer(p, cfg)

        # force PL clock, open device, stream weights + embed exactly once.
        self._fclk_hz = pp16.set_and_verify_fclk(self.fclk)
        words = pp16.build_weight_image(self._intseq, self.lanes)
        emb = pp16.build_embed_chunks(self._intseq)
        self._dev = pp16.Dev()
        idc = self._dev.rd(pp16.R_IDCODE)
        if idc != pp16.IDCODE_SQ16:
            self._dev.close()
            raise RuntimeError(f"PL IDCODE 0x{idc:08X} != SQ16; wrong bitstream")
        self._dev.wr(pp16.R_CTRL, 0x4); self._dev.wr(pp16.R_CTRL, 0x0)  # soft reset
        self._dev.wr(pp16.R_CTRL, 0x2)                                  # wl_rst
        subw = (self.lanes * 4) // 32
        for w in words:
            w = int(w)
            for s in range(subw):
                self._dev.wr(pp16.R_WDATA, (w >> (32 * s)) & 0xFFFFFFFF)
        for v in emb:
            self._dev.wr(pp16.R_EDATA, v)
        self.launches = 0

    @staticmethod
    def _load_meta(meta_path):
        import json
        m = json.load(open(meta_path))
        stoi = m["stoi"]
        itos = {int(k): v for k, v in m["itos"].items()}
        enc = lambda s: [stoi.get(c, 0) for c in s]
        dec = lambda ids: "".join(itos.get(int(i), "?") for i in ids)
        return enc, dec, m["vocab_size"]

    def _submit_once(self, tok_ids: list[int]) -> tuple[list[int], int]:
        """One PL pass: write 16 TOK_IDs, GO, poll DONE, read 16 TOK_OUTs +
        CYCLES. Returns (out_tokens, cycles). One A53 launch."""
        pp16 = self._pp16
        dev = self._dev
        for b in range(16):
            dev.wr(pp16.R_TOKID[b], tok_ids[b] & 0x1FF)
        dev.wr(pp16.R_POS, 0)
        dev.wr(pp16.R_CTRL, 0x1)   # go
        # busy-poll DONE (the run is microseconds; no sleep)
        for _ in range(1_000_000):
            if dev.rd(pp16.R_STATUS) & 0x1:
                break
        cyc = dev.rd(pp16.R_CYCLES)
        outs = [dev.rd(pp16.R_TOKOUT[b]) & 0x1FF for b in range(16)]
        self.launches += 1
        return outs, cyc

    async def infer_batch(self, reqs: list[InferRequest]) -> BatchOutcome:
        # pad to 16 streams; the unused slots just compute a throwaway token.
        loop = asyncio.get_running_loop()
        filled = len(reqs)
        tails = [r.prompt[-self.PROMPT_TAIL:] for r in reqs]
        # per-stream growing char list (only the LAST char is fed as the t=0
        # token each pass — this is the T=1 limitation, documented above).
        completions = ["" for _ in reqs]

        def _run():
            t0 = time.monotonic()
            for _ in range(self.gen_chars):
                tok_ids = []
                for s in range(16):
                    if s < filled:
                        seed = tails[s] + completions[s]
                        ids = self._enc(seed[-self.PROMPT_TAIL:]) or [0]
                        tok_ids.append(ids[-1])
                    else:
                        tok_ids.append(0)
                outs, _cyc = self._submit_once(tok_ids)
                for s in range(filled):
                    completions[s] += self._dec([outs[s]])
            return time.monotonic() - t0

        busy_s = await loop.run_in_executor(None, _run)
        results = [
            InferResult(client_id=r.client_id, prompt=r.prompt, seq=r.seq,
                        completion="[T=1 stub] " + completions[i])
            for i, r in enumerate(reqs)
        ]
        return BatchOutcome(results=results, busy_s=busy_s,
                            filled=filled, capacity=16)

    async def close(self) -> None:
        try:
            self._dev.close()
        except Exception:
            pass
