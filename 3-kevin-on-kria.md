# Kevin on Kria

Where the data tool from doc 1 and the fabric LLM from doc 2 come together into one project that is genuinely fun, technically honest, and has a built-in alibi for being dumb.

The pitch in one line: a language model so starved of memory that it independently rediscovered Kevin Malone's communication philosophy, and runs entirely in FPGA fabric while doing it.

## The joke is the thesis

This is the part that makes it more than a gag. Kevin's whole doctrine, why waste time say lot word when few word do trick, is not something I bolt onto the project for laughs. It is the actual engineering argument.

Few words means fewer tokens. Fewer tokens means less weight streamed per unit of output and less KV cache to hold. Less to hold means more of the model and more context fit in on-chip BRAM and URAM, which is the one thing that makes the fabric version fast in the first place (doc 2). So the thing that makes the model dumb, its tiny size and telegraphic output, is the same thing that makes it fast. The comedy and the optimisation are the identical property viewed from two angles. That is a rare and pleasing coincidence and the whole project hangs off it.

## How the two halves fuse

No new hardware, the platform is exactly doc 2. The Kevin layer is entirely upstream, in the data:

1. Take a simple corpus where a tiny model can stay coherent. TinyStories is the lead.
2. Run it through the Keviniser from doc 1, function words stripped, inflection flattened, telegraphic. Log the compression ratio.
3. Train the nanoGPT-scale model from doc 2 on the stripped corpus with Brevitas INT4 QAT. It now speaks Kevin natively, because that is the only language it has ever seen.
4. Validate against goformer to cosine > 0.9999, bake it into fabric, generate.

Note that I strip the training corpus, not the inference input. The model learns the compressed distribution and produces telegraphic text on its own, no preprocessing at runtime.

## The double win, and it is measurable

This is where the bit earns its keep as real work rather than just a funny demo.

- The Keviniser compresses the corpus, say to somewhere around 55 to 70 percent of the original token count. I have that number from doc 1.
- That compression directly buys on-chip headroom. A telegraphic model expresses the same content in fewer tokens, so for a fixed on-chip budget I get more effective context and a larger effective vocabulary, and decode throughput rises.
- So "few word do trick" stops being a Kevin quote and becomes a logged efficiency result. I can show, with numbers, that the compressed model does more useful work per joule and per millisecond than the same architecture trained on full English.

That is the difference between a meme and a measurement. The meme gets attention, the measurement is what makes it stand up.

## The quietly clever bit

Kevin-speak is a compressed, human-legible interface language. That is a crude, accidental version of the compressed-interface idea from the broader design notes, the one about co-designing a dense language between a planner and a renderer rather than passing full text around. Pushed properly that direction leads to soft prompts and latent thought vectors. Kevin obviously did not get that far, he just dropped the articles, but the shape of the idea is the same. The dumb joke is gesturing at a real research direction, and I can say so in one honest sentence without overclaiming that a toddler-grade telegraphic model is doing latent reasoning. It is not. It is just compressing, and compression is the first rung of that ladder.

## The demo

Type a prompt, watch dense broken content-word text stream out of pure fabric at absurd tokens per second, with a live tokens-per-second and joules-per-token readout next to it. The punchline writes itself on screen.

Expected shape of the output, illustrative not real since nothing is trained yet, but this is the texture a TinyStories-trained Kevin model should produce:

```
prompt:  the little dog wanted to play in the park
output:  little dog want play park. dog run run. ball there.
         dog happy. boy come throw ball. dog catch ball good dog.
```

Coherent enough to be charming, telegraphic enough to be funny, and clearly the work of a model that cannot hold many words in mind at once. The short on-chip context is doing that to it, and that limitation is exactly what keeps it on-chip and fast. Everything points the same way.

## Framing for a writeup

If this becomes a post, the rule from the strategy work holds: keep the mischief in the title and keep the body dry. The reason the RAG post landed was honesty, not swagger. So the headline can be playful, the body is measurement, tradeoffs and the roofline crossover, and at no point do I pretend the prose is good. The dumbness is deliberate and stated, which means the speed numbers carry the post on their own and never have to lean on the quality of the output. The Kevin persona is the thing that converts an obvious limitation into the entire point.

Candidate titles, mischief up front, honest underneath:

- Few Word Do Trick: a language model that runs entirely in FPGA fabric
- Why waste DDR: I baked a tiny LLM into on-chip BRAM and it talks like Kevin
- Zero-DRAM LLM: a telegraphic transformer with the weights wired into silicon
- The dumbest fast language model: Kevin-speak at a hundred thousand tokens a second

## Metrics to report

Same weights on both platforms, decode and prefill separately, board-level power.

- tokens per second, fabric versus A53, same model, the headline
- tokens per joule, fabric versus A53, and ideally versus a Pi and a phone for context
- the Keviniser compression ratio, words and tokens, the "few word do trick" number
- the on-chip headroom the compression buys, extra context or vocabulary for the same budget
- the roofline crossover, the model size where the fabric stops winning

## Why this is the right project

It uses the PL creatively in the truest sense, the model is the circuit. The uplift is real and large because it escapes the bandwidth wall, not because the fabric has more multipliers. The dumbness of a tiny model, which would be an embarrassment in any serious LLM project, is here both the joke and the engineering lever. And every individual claim is measurable, so the fun version and the rigorous version are the same version. Possible, practical, and as it happens, properly funny.

## Next step

Build doc 2's stage 1 to get a real on-chip number, run doc 1's Keviniser on TinyStories to get the compression number, train the model on the stripped corpus, and wire the two together. The hardware does not care that the model talks like Kevin, it just runs the circuit. Kevin is the reason the circuit is small enough to run fast.
