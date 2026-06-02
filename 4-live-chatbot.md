# Part 4, the live chatbot and the public stress test

Put it on the internet, link it on the Hacker News post, and find out how many people can hammer one little board at once. The hug of death, except the server is a roughly £250 board on a shelf drawing a few watts.

This is the fourth piece. It sits on top of the platform in `2-llm-on-kria.md` and serves the Kevin-speak model from `3-kevin-on-kria.md`. It does not change the fabric engine much, it wraps it in a web front end and points a crowd at it.

## The question

How many concurrent users can a single resident-model FPGA actually serve. And the honest sub-question that makes it interesting, when it does fall over, what falls over first. Because I have a strong suspicion it will not be the silicon.

## Why a crowd suits the fabric, and it is counterintuitive

Single-stream decode, one user typing, is the fabric's weakest case. The activation is a vector, the matmul is a GEMV, and the big systolic array is mostly idle, a huge MAC grid being fed one skinny vector at a time. All the compute the array can do, the 2 to 4 TOPS, goes unused under one user because the work is memory shaped, not compute shaped.

Concurrency fixes exactly that. Batch B users together and each decode step becomes a GEMM, a B-row matrix through the array instead of one vector. The weights, already resident on-chip, are read once and reused across all B users. The array fills up. So the thing that was idle under one user becomes busy under many, and the fabric's real compute edge finally gets spent. More users is more efficient per user, up to a ceiling. The fabric likes a crowd.

This is the inverse of the usual GPU serving problem. A GPU has to assemble large batches to hide its DDR latency and stay fed. Here the weights never leave on-chip, so there is no DDR latency to hide, and batching is pure upside, it just fills the array. The weakness under one user and the strength under many are the same fact seen from two sides, which is becoming the recurring shape of this whole project.

## The ceiling, and it is on-chip

The limit on B is memory. Each concurrent stream needs its own KV cache, and those caches live on-chip alongside the model. The model takes 1 to 2 MB of the roughly 3 MB on-chip budget, leaving on the order of 1 MB to split across B users. Short context, which the design already mandates, keeps each KV cache small, so I can fit a fair few streams, but it is a hard wall. Two regimes fall out of it:

- within batch capacity, users are served simultaneously at full speed
- beyond it, requests queue, and per-user latency grows with queue depth

The live dashboard watching that queue form under load is the drama.

## The architecture

Same heterogeneous split as doc 2, web front end instead of a local prompt.

```
                 the internet (Hacker News)
                          │
                  Cloudflare Tunnel            no static IP needed,
                          │                    DDoS buffer, rate limiting
                          ▼
   ┌───────────────────────────────────────────────┐
   │  KRIA KV260                                    │
   │                                                │
   │  A53s, PetaLinux                               │
   │   async web server, WebSocket per session      │
   │   batch scheduler, tokenise, sample            │
   │            │ AXI DMA                            │
   │            ▼                                    │
   │  PL fabric                                     │
   │   batched decode engine, weights in URAM        │
   │   B per-session KV caches on-chip               │
   └───────────────────────────────────────────────┘
                          │ pushes metrics out
                          ▼
                Cloudflare-hosted dashboard         stays up even when
                (static page + Worker)              the board is saturated
```

Two deliberate choices in there:

- **Cloudflare Tunnel** to expose a board sitting on my own network with no public static IP and no open ports, and it hands me a DDoS buffer and per-IP rate limiting for free. I already run the Cloudflare stack, so this is familiar ground.
- **Serve the dashboard from Cloudflare, not from the board.** The board pushes metrics out, the public status page is hosted separately. So when the board is being hammered and queuing, the page showing that it is being hammered stays up. Decouple the observer from the thing observed.

## The honest bottleneck

Here is the bit I will not pretend away. A single KV260 has one gigabit Ethernet port and four A53 cores. Under a real Hacker News hug, the limit is almost certainly the A53 network stack holding thousands of concurrent WebSocket connections, not the fabric running out of inference. The fabric does single-digit-microsecond tokens from on-chip, it has enormous headroom. The Linux box in front of it does not.

So the experiment honestly produces two numbers, not one:

- the **inference ceiling**, the fabric under synthetic local load, which will be large
- the **real-world serving ceiling**, the whole board under actual traffic through its own network stack, which will be modest

And the likely punchline is a genuinely good result rather than a failure. The silicon could serve thousands, but the Linux box tapped out at N connections. I built something so fast that the slow part was handling the sockets. I will say clearly which number I am quoting at any point, and show both, because the gap between them is the story.

There is a fork on how public to make it:

- **Pure**, board exposed directly through the tunnel, measures the true end-to-end limit of one board including its network stack. Smaller number, but it is honestly what one board does.
- **Buffered**, a small cloud proxy holds the connection storm and feeds the board over a single persistent channel, so the fabric's inference capacity is what shows. Bigger number, but I have added a non-FPGA component and have to say so.

The honest move is to do both and contrast them.

## The metrics

The live dashboard, which is itself the best HN bait in the project:

- concurrent active sessions, and how many are queued behind them
- tokens per second, aggregate and per user
- batch occupancy, how full the array is, which rises as users arrive
- on-chip KV utilisation, how close I am to the concurrency wall
- latency, time to first token, inter-token latency, and queue wait, reported separately
- board-level watts and joules per token, off a real power meter, which is the FPGA's home turf from doc 1
- a live bottleneck indicator, network bound or compute bound right now
- running totals, tokens served and users since launch, the counter ticking up through the spike

The screenshot that travels: a Hacker News spike landing on a board on a shelf, the queue forming, and the wattage sitting flat near nothing the whole time. A crowd served on the power of a lightbulb.

## Safety, light touch

The model is, helpfully, too dumb to be dangerous. It is a TinyStories-grade telegraphic Kevin model, it cannot really produce harmful content because it can barely produce coherent content. Prompt injection mostly defuses itself, push it and it just replies in broken Kevin-speak. That said, public input is public input, so basic guards: per-IP rate limiting at the tunnel, a max tokens per request, and an input length cap. Light, and proportionate to a model that struggles to finish a sentence.

## How it slots in

No change to the fabric engine from doc 2 except the one real addition, batched decode with per-session KV caches held on-chip. The model being served is the Keviniser-trained one from doc 3, so the dumbness is on public display, which is exactly the point, the crowd is meant to see the telegraphic output and the speed at the same time. This is stage 4, and only after stages 0 to 3 work.

## Making the latency disappear, the single-refresh trick (a design note for later)

A future-feature idea, only viable once the single-stream engine is fast, but worth
writing down because it turns the speed from a number into a feeling. The goal: the
answer should appear the instant you press Enter, fully formed, in a single monitor
refresh. No spinner, no typewriter streaming, just *bang*, the reply is there.

The trick rests on one property of autoregressive decode: typing a prompt forward is
*appending* tokens, and a KV cache is append-only. So as the user types, letter by
letter, the front end speculatively runs the prompt-so-far through the model — but each
keystroke is only **one new token to append** to the cache, not a fresh re-read of the
whole prompt. The cache stays caught up to the cursor for the cost of a single forward
step per keystroke, which at the target speed is well under the gap between keystrokes.
By the time the user hits Enter, the expensive part — prefill, the thing that is
normally the time-to-first-token — is already done. TTFT was spent during typing, for
free, and the user never saw it.

Two flavours, depending on how aggressive I want to be:

- **Warm cache.** Only the prefill runs while typing; the reply is generated on Enter.
  Because the model is so fast and Kevin is so terse, a whole reply generates in a few
  milliseconds — under one 60 Hz frame — so it still lands in a single refresh. Simple
  and robust, and probably enough.
- **Speculative generate.** Also generate the reply for the current prompt-so-far
  continuously, so on Enter the answer is *already computed* and the keypress is pure
  rendering, zero compute. It costs more fabric time — you are answering prompts that
  may still change — and the last keystroke leaves the speculation one character stale,
  so you re-run the delta. At these speeds that is cheap enough to redo per keystroke.

The "single refresh" part is then just a rendering choice: buffer the whole reply and
write it in one frame, do not stream it. The effect is the visceral version of the
speed claim — most fast-LLM demos still stream token by token; this one would just have
the answer appear, whole, the moment you commit the prompt.

The honest caveats, because this only works in a narrow regime. It needs the fast path
— at the current single-digit tok/s a reply takes tens of seconds, so this is a stage-4
feature gated on the sequencer, not something the driver-bound prototype can do.
Backspace and mid-prompt edits break the append-only assumption: invalidate the cache
from the edit point and recompute the tail, cheap but not free, so keep a per-position
cache and only redo the suffix. And under a crowd it fights the batch scheduler —
speculating for every half-typed prompt multiplies the load, so cap speculation to the
focused session or let it use only spare batch slots. It is, once again, the same fact
from two sides: the thing is only fast enough to pre-answer your prompt *because* it is
small, on-chip, and dumb.

## What I would report

- the two ceilings, inference versus serving, with the bottleneck named
- watts under full load, the crowd-on-a-lightbulb number
- the concurrency curve, per-user latency against number of users, showing the array filling and then the queue forming at the KV wall
- the moment the Hacker News spike hits, on the live graph

## Framing for the post

This is the demo that makes the speed claim visceral. I do not tell them it is fast, I let them hit it and watch the number move. Same rule as everywhere else, mischief in the title, dry in the body, no pretending the prose is good. The dumbness is the alibi, the live load is the proof.

Candidate angle: I put a language model on a £250 FPGA, linked it on Hacker News, and watched to see what broke first.

## Next step and the risk

Build stage 4 only after the single-stream engine works. The batched decode and per-session KV is the genuinely new fabric work, and it is where the time goes. Decide pure versus buffered, or commit to both. And soak-test with synthetic load, locust or k6 or vegeta, well before going live, so the Hacker News moment is not the first time the thing has ever seen concurrency. The one outcome I want to avoid is it falling over for a boring reason on the day, a misconfigured connection limit rather than an honest capacity wall.
