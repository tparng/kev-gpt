// =============================================================================
// mac_bank_dp — the DOUBLE-PUMPED MAC bank (DOUBLE-PUMP-100K Stage 0, Phase A).
//
// Arithmetically IDENTICAL to `mac_bank` (gemm_banked_resident_vec.sv), but it
// consumes TWO K-steps per `clk` cycle by running the accumulator in a 2x,
// phase-aligned `clk2x` domain. This is the GO/NO-GO arithmetic proof for the
// double-pump campaign: prove that folding 2 products/clk is bit-exact, then
// (Phase B) prove the DSP cascade can carry clk2x at >=400 MHz.
//
// --------------------------------------------------------------------------
// The 2x dataflow (why it is bit-exact)
// --------------------------------------------------------------------------
// `mac_bank` does, per lane, once per `clk` when `en`:
//       a <= a + w*x                       (K cycles -> K products)
// The total accumulated over the K-loop is  sum_{k=0..K-1} w[k]*x[k].
//
// `mac_bank_dp` is fed TWO operand pairs per `clk`:  (w0,x0) and (w1,x1),
// i.e. K-steps 2j and 2j+1 of the same K-loop, presented stable for the whole
// clk period. `clk2x` is 2x `clk` and 0-degree phase-aligned (same MMCM), so
// exactly TWO rising clk2x edges fall inside each clk period. A 1-bit phase
// register (toggled every clk2x edge, re-anchored each clk) selects:
//       phase 0 edge:  a <= a + w0*x0       (K-step 2j)
//       phase 1 edge:  a <= a + w1*x1       (K-step 2j+1)
// So over K/2 clk cycles the accumulator sees the SAME K products in the SAME
// order as the single-clock bank over K cycles. Addition is associative and
// the products are identical 4x8 signed integer multiplies, so the final `acc`
// is bit-identical, every lane. No saturation, no rounding -> no ordering
// subtlety can bite; this is a pure timing transform.
//
// Phase anchoring (the one timing detail that matters):
//   The parent's `clk`-domain control (en / w0/x0 / w1/x1 / clr) is sampled
//   into the clk2x domain. With clk2x exactly 2x and 0-deg aligned, the clk
//   LEVEL itself names the phase at each clk2x rising edge:
//       clk period 10 ns: clk high 0..5, low 5..10; clk2x edges at 0,5,10.
//       t=0   clk2x rising, clk==1  -> the start-of-period edge   = PHASE 0
//       t=5   clk2x rising, clk==0  -> the mid-period edge        = PHASE 1
//   So `phase = ~clk` sampled AT the clk2x edge: clk high -> phase 0 (w0,x0),
//   clk low -> phase 1 (w1,x1). This is a pure level sample with NO clk/clk2x
//   coincident-edge race (no NBA-ordering dependence on which clock's process
//   runs first) and needs no reset state -> deterministic from t=0. On silicon
//   the MMCM emits clk2x at 0-deg, so `clk` is a hard timing-named phase bit;
//   the synthesised form latches `clk` through the DSP control just the same.
//
// --------------------------------------------------------------------------
// LUT vs DSP variant
// --------------------------------------------------------------------------
// This file implements the LUT-based variant (the `mac_bank` twin) — it proves
// the ARITHMETIC and the 2x dataflow. The DSP-cascade variant (the real Fmax
// story) follows the SAME structure: the `mac_bank_dsp` packed-pair multiply
// (`a <= a + $signed({1'b0,wpak})*xs`, with sum_act recovery at readback) would
// likewise live in the clk2x domain, accumulating the packed pair for w0pak*x0
// on phase 0 and w1pak*x1 on phase 1 into the same 48-bit DSP accumulator. The
// DSP48E2's P-register IS the clk2x accumulator; PCIN/PCOUT carry the cascade
// at clk2x. sum_act would advance by xs each phase too (2 activations/clk),
// staying bit-exact with the single-clock sum_act. Built here LUT-first so the
// arithmetic gate is unambiguous before the DSP-pack encoding is layered on.
//
// NOTE: identical (* keep_hierarchy, use_dsp="no" *) attributes to `mac_bank`
// so the OOC behaves like the real leaf (the bulk-mult URAM-killer isolation).
// =============================================================================

(* keep_hierarchy = "yes", use_dsp = "no" *)
module mac_bank_dp #(
    parameter integer LANES = 128,
    parameter integer ABITS = 24       // |acc| range-proven <= 2^20, same as mac_bank
) (
    input  wire                      clk,    // fabric clock (200 MHz domain)
    input  wire                      clk2x,  // 2x clk, 0-deg phase-aligned (same MMCM)
    input  wire                      clr,    // sync clear (clk domain levels, sampled in clk2x)
    input  wire                      en,     // issue this clk: consume (w0,x0) AND (w1,x1)
    input  wire [LANES*4-1:0]        w0,     // weight word, K-step 2j   (one INT4 per lane)
    input  wire [LANES*4-1:0]        w1,     // weight word, K-step 2j+1
    input  wire signed [7:0]         x0,     // activation byte, K-step 2j   (shared across lanes)
    input  wire signed [7:0]         x1,     // activation byte, K-step 2j+1
    output wire [LANES*ABITS-1:0]    acc     // per-lane accumulator (sampled at clk boundaries)
);
    // ---- phase from the clk LEVEL sampled at the clk2x edge ------------------
    // phase 0 (use w0,x0) when clk is HIGH at the clk2x edge (start-of-period),
    // phase 1 (use w1,x1) when clk is LOW (mid-period). No reset state needed,
    // no clk/clk2x coincident-edge race -> deterministic from t=0.
    wire phase = ~clk;
    wire [LANES*4-1:0] wsel = phase ? w1 : w0;
    wire signed [7:0]  xsel = phase ? x1 : x0;

    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : g_l
            reg signed [ABITS-1:0] a;
            always @(posedge clk2x) begin
                if (clr) a <= {ABITS{1'b0}};            // clr held >= 1 clk -> seen by clk2x
                else if (en) a <= a + $signed(wsel[gl*4 +: 4]) * xsel;
            end
            assign acc[gl*ABITS +: ABITS] = a;
        end
    endgenerate
endmodule
