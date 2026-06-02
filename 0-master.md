# Kevin on Kria, master notes

The single entry point for the whole project. Read this and the four component docs make sense in order. Each component doc holds the detail, this one holds the through-line.

- `1-keviniser.md`, the data tool
- `2-llm-on-kria.md`, the platform
- `3-kevin-on-kria.md`, the fusion
- `4-live-chatbot.md`, the public stress test

## What this is

I am putting a tiny language model into the programmable logic of a Kria KV260, with the weights baked into on-chip memory so it never touches DDR, and racing it against the same model running on the board's Arm cores. To make the necessarily small model into a feature rather than an apology, I train it on telegraphic text so it talks like Kevin Malone from the Office. The output is dumb on purpose, and the thing that makes it dumb turns out to be the same thing that makes it fast. So the joke and the engineering are one and the same, and every claim in the joke is measurable. Once it works, I put it on the internet, link it on the Hacker News post, and let a crowd hammer it, to see how many people one little board can serve at once before something gives.

## The idea in one breath

Two facts stacked on top of each other.

First, the wall. Generating one token at a time is bound by memory bandwidth, not by compute. Each token drags the whole weight set through the multipliers once, so throughput is roughly the bandwidth divided by the model size in bytes. On the KV260 the Arm cores and the fabric share the same roughly 20 GB/s DDR controller, so for a model that lives in DDR the fabric has no higher ceiling than the cores. They sit behind the same wall. The fabric only wins big if it leaves DDR entirely and keeps the whole model in on-chip BRAM and URAM, where bandwidth is hundreds of GB/s to TB/s. That escape is the entire source of the speedup, and it only works for a model small enough to fit on-chip, a few million parameters at INT4, roughly 1 to 2 MB.

Second, the fusion. A model that small is going to be dumb, so I lean in. Kevin's doctrine, why waste time say lot word when few word do trick, is not a bolt-on gag. Fewer words means fewer tokens means less weight streamed and less KV cache to hold, which means more of the model fits on-chip, which is the only thing that makes the fabric version fast. The dumbness and the speed are the same property seen from two sides. That coincidence is the reason this project is worth doing rather than just another FPGA-beats-CPU post.

## The four pieces

**1. The Keviniser, the data tool.** A part-of-speech-based preprocessor that strips function words and flattens inflection to turn ordinary English into dense telegraphic Kevin-speak. I strip the training corpus, not the runtime input, so the model learns the compressed distribution and generates that way on its own. It prints a compression ratio, which is the first real number in the project. Runnable code today. Detail in `1-keviniser.md`.

**2. LLM on Kria, the platform.** The serious hardware build. A nanoGPT-scale model, INT4, validated against goformer to cosine > 0.9999 before any speed number is trusted, baked fully on-chip, matmuls in a systolic array and the non-linearities (softmax, RMSNorm, activation) hand-rolled into fabric beside them. Staged so every step ships something. It also writes out the trap I am avoiding, the Vitis DPU graph-surgery route, which streams weights from DDR and so walks straight back into the wall. Detail in `2-llm-on-kria.md`.

**3. Kevin on Kria, the fusion.** The two halves joined. Train the doc 2 model on a doc 1 corpus, and the small size becomes the joke and the lever at once. This doc carries the framing, the measurable double win, and the honest writeup rule. Detail in `3-kevin-on-kria.md`.

**4. Live chatbot on Kria, the public stress test.** A web front end on the board, linked live on the Hacker News post, so a crowd can hit one little FPGA at once. The interesting twist is that concurrency suits the fabric, batching many users turns the idle systolic array into a busy one, so more users is more efficient per user up to the on-chip KV ceiling. The honest twist is that the bottleneck under load is probably the A53 network stack rather than the silicon, and reporting both the inference ceiling and the real serving ceiling is the point. Detail in `4-live-chatbot.md`.

## How they fit together

No new hardware between the pieces, the Kevin layer is entirely upstream in the data, and the platform does not care what the model says.

```
simple corpus (TinyStories)
        │
        ▼
   1. KEVINISER          strip function words, flatten inflection
        │                log the compression ratio
        ▼
   telegraphic corpus
        │
        ▼
   train, INT4 QAT       nanoGPT scale, on the 3050 Ti
        │
        ▼
   validate vs goformer  cosine > 0.9999, bit-honest first
        │
        ▼
   2. BAKE INTO FABRIC   weights in URAM, matmuls + non-linearities on-chip
        │
        ▼
   3. GENERATE           dense Kevin-speak streaming out of pure logic,
        │                tokens/sec and joules/token on screen
        ▼
   4. SERVE              web front end on the A53s, Cloudflare Tunnel,
                         linked on HN, see how many users one board holds
```

## What is actually true

The honest core, so the project stands up.

- **The double win is measurable.** The Keviniser compresses the corpus, probably to somewhere around 55 to 70 percent of the original tokens. That compression buys on-chip headroom, more effective context and vocabulary for a fixed budget, which raises throughput. So "few word do trick" is a logged efficiency result, not just a quote.
- **The uplift is real but bounded.** Realistically tens of x on decode for a model sized to live fully on-chip, more if the Arm cores do less per token. Not 100x once orchestration is counted. I report the roofline crossover, the model size where the fabric stops winning because it too has to go to DDR. The crossover is as much the result as the peak.
- **Where it stands on silicon (measured, not modelled).** The whole INT4 transformer now runs in fabric with the CPU out of the loop — the stage-3 sequencer. On the KV260 it generates a **token-stream-bit-exact** Kevin at **44.3 tok/s @ 40 MHz**, ~4× the optimised A53 (~11) wall and ~200× the first on-fabric run (0.22). The CPU-in-the-loop rungs (Python → resident weights → KV cache → C driver: 0.07 → 10.35) asymptote *to* the A53 — the architectural leap is taking it out of the loop. Resident-read and PE=256 widening are RTL bit-exact in sim (76 / 231 tok/s) with the wide bitstream building; the road past that is pipelining the LayerNorm DSP cascade and the serial datapath. The full ladder, tiered measured / sim / projected, is `fabric/progress.png`.
- **Where it loses, said up front.** Any model big enough to be DDR resident, long context once the KV cache spills, and far higher development effort than just running on the cores. All three stated out loud.
- **The writeup rule.** Mischief in the title, dry in the body. The reason the RAG post landed was honesty, not swagger. At no point do I pretend the prose is good, the dumbness is deliberate, and the speed numbers carry the post on their own.

## Build order

Condensed from doc 2, every stage ships something demonstrable.

| stage | ships | bottleneck |
|---|---|---|
| 0 | A53 baseline, and the proof it is bandwidth bound | DDR |
| 1 | raw on-chip matmul throughput | pipeline |
| 2 | the measured heterogeneous ping-pong tax | A53 round trips through DDR |
| 3 | the full zero-DRAM uplift and the crossover plot | per-token pipeline latency |
| 4 | the live chatbot under a Hacker News crowd, two ceilings measured | the A53 network stack, not the fabric |

Running alongside, on the data side: run the Keviniser on TinyStories, log the compression numbers, train the model on the stripped corpus, validate against goformer, then wire it into stage 3.

## The one sentence to remember

A language model so starved of memory it rediscovered Kevin Malone's communication philosophy, running entirely in FPGA fabric, where being dumb and being fast are the same thing, and all of it is measurable. Possible, practical, and properly funny.
