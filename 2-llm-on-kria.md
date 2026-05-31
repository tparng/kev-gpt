# A language model that lives entirely in FPGA fabric

The build plan for running a tiny transformer on a Kria KV260 with the weights baked into on-chip memory, never touching DDR, and beating the A53 cores by a margin that is real rather than imaginary.

This is doc 2 of three. Doc 1 is the data tool. Doc 3 fuses this with the Kevin angle. This one is the platform, and it stands on its own.

## The one paragraph thesis

Single stream autoregressive decode is memory bandwidth bound, not compute bound. Each token streams essentially the whole weight set through the multipliers once, so to first order the throughput is `tokens/sec for a given model size, the bandwidth divided by the bytes`. On the KV260 the quad A53 cores and the programmable logic pull from the same roughly 20 GB/s DDR4 controller, so the fabric does not have a higher memory ceiling for a model that lives in DDR. They sit behind the same wall. The only way the fabric wins big is to leave DDR entirely, by keeping the whole model in BRAM and URAM, where on-chip bandwidth is hundreds of GB/s to TB/s aggregate. That is the project. The uplift comes from escaping the wall, not from the fabric having more multipliers, and saying that out loud is what keeps the whole thing honest.

## The hardware, ballpark

Treat these as approximate, they are close enough to reason with.

| | |
|---|---|
| PS | quad Arm Cortex-A53 up to ~1.5 GHz, dual Cortex-R5F, Mali-400 MP2 |
| PL | ~256K logic cells, 1,248 DSP48E2 slices |
| BRAM | ~5 Mb (144 x 36 Kb) |
| URAM | ~18 Mb (64 x 288 Kb) |
| on-chip total | ~23 Mb BRAM+URAM, so roughly 3 MB of usable weight storage |
| DDR | 4 GB DDR4, 64-bit, ~20 GB/s, shared by PS and PL |

Three facts that decide everything:

- The PS and PL share that one DDR controller. This is the wall.
- ~3 MB on-chip is roughly 3M params at INT8 or 6M at INT4. That is the entire size budget.
- The A53 is ARMv8.0-A, which predates the SDOT and UDOT dot-product instructions (those are v8.2 and later). So INT8 on the A53 goes through ordinary widening multiplies and tops out around 50 to 100 GOPS peak, much less sustained. The Mali-400 is a vision era GPU with no compute API, so it is useless here. The A53s are a genuinely weak baseline, which I have to be honest about, because beating a weak baseline is not in itself impressive.

On the fabric side, INT8 packing gives roughly two MACs per DSP, and DSP48E2 in these speed grades clocks comfortably past 300 MHz, 500 plus is realistic. So the matmul engine is in the low single digit TOPS range, call it 2 to 4 TOPS with effort. That compute edge is fully felt in prefill and batching, and largely hidden during memory bound decode, which is exactly why the on-chip residency story matters more than the raw TOPS story.

## The non-negotiable constraint

The model must fit entirely on-chip. A few million parameters at INT4, roughly 1 to 2 MB, KV cache included. The moment the weights or the KV cache spill to DDR, I am back behind the 20 GB/s wall and the uplift evaporates. So I size the model deliberately to sit near the top of what URAM holds, because that is precisely the regime where the A53 has fallen off the bandwidth cliff and is crawling while the fabric version is flying. That maximises the honest gap.

Realistic headline: tens of x on decode tokens per second, more if I minimise how much the A53 has to do per token. Not 100x once orchestration overhead is counted. And I report the roofline crossover, the model size at which the fabric stops winning because it too has to go to DDR. The crossover plot is as much the result as the peak number.

## The trap I am not falling into

The obvious shortcut is to take the Vitis AI DPU overlay, do graph surgery to reshape the linear layers as 1x1 convolutions so the CNN compiler will eat them, and call it done. The trick is real, a 1x1 convolution is mathematically a matmul, but it walks straight into the wall. The DPU streams its weights from DDR over AXI, it does not bake them on-chip, so a model big enough to need the DPU is DDR bound by definition. On top of that the DPU has no softmax, no layernorm and no RoPE, so every one of those bounces back to the A53 through that same shared DDR, a ping-pong tax on top of the bandwidth tax. Its own honest conclusion, that you should hand-roll the non-linearities into the fabric next to the matmul core, is just an argument for the approach below. So I skip it, except possibly as a deliberately measured "here is the slow way" data point in the writeup.

## Tooling

I am hand-rolling this in HLS and RTL rather than leaning on a framework, because the fully on-chip pipelined transformer is not what the easy paths are built for. FINN is excellent for binarised CNNs with weights baked into fabric, which makes it a great warm-up, but its transformer support has historically been thin, so the attention block in particular wants hand-written logic. The stereo pipeline I built on this same board (census transform, four-path SGM, WTA, median filter, AXI DMA, R5 monitor, 150 plus MHz) is honestly a gnarlier dataflow problem than a clean systolic GEMV with streaming non-linearities, so the matmul engine here is well within range. FINN stays in the toolbox for the warm-up, not the headline.

## The model

- nanoGPT scale, char or word level, 2 to 4 layers, d_model around 128 to 256
- short context so the KV cache stays on-chip, this is a hard constraint not a preference
- Brevitas INT4 QAT, trained on the RTX 3050 Ti in an afternoon
- the golden reference is goformer. I validate the fabric output against it to cosine > 0.9999 before I trust a single speed number, the same correctness discipline as last time. Bit-honest first, fast second.

## The pathway

Staged so every step ships something demonstrable and I am never months in with nothing to show.

**Stage 0, baseline and the wall.** Run the model on the A53s, goformer-on-ARM or a tuned llama.cpp build, measure tokens per second and joules, confirm with a profiler that it is bandwidth bound. Cheap, and it is the "before" number the whole thing is measured against.

**Stage 1, on-chip matmul engine.** Hand-roll the INT8/INT4 systolic GEMV in fabric with the weights resident in URAM, and microbenchmark it in isolation. The raw matmul throughput jump is demonstrable here, before any of the fiddly parts. This is the satisfying checkpoint.

**Stage 2, heterogeneous midpoint, on purpose.** Matmuls on the PL from on-chip weights, softmax and norm still on the A53. Measure it. This is the ping-pong version, and quantifying the tax precisely, in microseconds per round trip and DDR bytes moved, is a genuine result that also motivates stage 3. It is the honest "here is what the easy approach costs" number.

**Stage 3, fabric-native non-linearities.** Move softmax (LUT-based exp with a running max for stability), RMSNorm and the activation into fabric right next to the matmul core, kill the round trips, full pipeline, weights never leave on-chip. This is the headline zero-DRAM number and the roofline crossover plot.

| stage | ships | bottleneck |
|---|---|---|
| 0 | the baseline and the proof it is bandwidth bound | DDR |
| 1 | raw on-chip matmul throughput | pipeline |
| 2 | the measured heterogeneous tax | A53 round trips through DDR |
| 3 | the full zero-DRAM uplift and crossover | per-token pipeline latency |

## The fiddly bits, named honestly

- The matmuls map cleanly to a systolic array. The rest is the work. Softmax, layernorm or RMSNorm and the activation are a small fraction of the compute but they shape the datapath, and getting LUT-based exp accurate enough to keep the validation passing is where the time goes.
- The KV cache is fine on-chip for short context and grows with sequence length. At long context it wants to spill to DDR, which is exactly the thing that eats the advantage, so context length stays short and that is a design constraint, not a limitation I am apologising for.
- A53 orchestration, tokenisation, sampling, sequence management and control flow, has to stay light or it becomes the per-token bottleneck on the fabric side and quietly caps the speedup. The less the A53 touches per token, the higher the real number.

## Where it loses, stated up front

So nobody has to ask. The fabric loses on any model big enough to be DDR resident, where it is back behind the same wall as the A53s with extra engineering for nothing. It loses at long context once the KV cache spills. And the development effort is far higher than just running the model on the cores. The honest version of this project says all three out loud, shows the crossover, and lets the on-chip number stand on its own.

## Next step

Train the candidate model, get it validated against goformer, then build stage 1 and get the first real on-chip matmul number on the board. The Kevin layer in doc 3 sits on top of this without changing any of the hardware, it just decides what the model says and why the small size is a feature.
