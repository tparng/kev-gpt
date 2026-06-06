"""webchat.demo — the live-chat demo inference + telemetry planes (PRD 5).

Pure-software plane (P0): i7 asyncio server (debounce/coalesce/16-slot batch),
A53 thin daemon, telemetry tick + Cloudflare Worker dashboard, synthetic-typist
loadgen. Backends are pluggable behind InferenceBackend: StubBackend for off-box
load tests, PLSingleTokenBackend to drive the real (single-token, no-KV) PL.
"""
