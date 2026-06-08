// =============================================================================
// mac_bank_dsp_dp — the DOUBLE-PUMPED DSP-packed MAC bank (DOUBLE-PUMP-100K
// Stage 1, the DSP twin of mac_bank_dp).
//
// Arithmetically IDENTICAL to `mac_bank_dsp` (gemm_banked_resident_vec.sv) — the
// WP487 packed-pair leaf (two +8-biased INT4 weights at a 22-bit gap share one
// INT8 activation, raw 48-bit pair accumulators, sum_act exported for the
// parent's readback recovery) — but it consumes TWO K-steps per `clk` by running
// the DSP accumulator (the P-register) in the phase-aligned `clk2x` domain, EXACTLY
// like mac_bank_dp does for the LUT bank.
//
// --------------------------------------------------------------------------
// The 2x dataflow (why it is bit-exact)
// --------------------------------------------------------------------------
// `mac_bank_dsp` does, per pair, once per `clk` when `en`:
//       a48 <= a48 + $unsigned(wpak)*xs           (K cycles -> K packed products)
//       sum_act <= sum_act + xs                    (one activation/clk)
// `mac_bank_dsp_dp` is fed TWO operand pairs/clk: (w0,x0) and (w1,x1) = K-steps 2j
// and 2j+1, presented stable for the whole clk period. clk2x is 2x and 0-deg
// phase-aligned, so exactly TWO rising clk2x edges fall in each clk period; the
// clk LEVEL names the phase (phase 0 = clk high uses w0/x0, phase 1 = clk low uses
// w1/x1 — see mac_bank_dp for the no-race phase-anchoring argument). So over K/2 clk
// cycles BOTH the packed accumulator AND sum_act see the SAME K products / K
// activations in the SAME order as the single-clock bank over K cycles. The
// readback recovery (y0/y1 from the raw 48-bit pair using 8*sum_act) is UNCHANGED
// and lives in the parent — it is exact because both accumulated sums are identical.
//
// sum_act advances by xs on EACH phase (2 activations/clk), staying bit-exact with
// the single-clock sum_act; the odd-K tail masks phase 1's activation to 0 (the
// parent drives x1=0 there), so the phantom step adds 0 to both a48 and sum_act.
//
// keep_hierarchy + use_dsp="yes" on the accumulator: same as mac_bank_dsp (the DSP
// P-register IS the clk2x accumulator; PCIN/PCOUT carry the cascade at clk2x).
// =============================================================================
`timescale 1ns / 1ps

(* keep_hierarchy = "yes" *)
module mac_bank_dsp_dp #(
    parameter integer LANES = 128,
    parameter integer ABITS = 24
) (
    input  wire                      clk,    // fabric clock (200 MHz domain)
    input  wire                      clk2x,  // 2x clk, 0-deg phase-aligned (same MMCM)
    input  wire                      clr,    // sync clear (clk levels, sampled in clk2x)
    input  wire                      en,     // issue this clk: consume (w0,x0) AND (w1,x1)
    input  wire [LANES*4-1:0]        w0,     // weight word, K-step 2j   (one INT4 per lane)
    input  wire [LANES*4-1:0]        w1,     // weight word, K-step 2j+1
    input  wire signed [7:0]         x0,     // activation byte, K-step 2j   (shared across lanes)
    input  wire signed [7:0]         x1,     // activation byte, K-step 2j+1
    output wire [LANES*ABITS-1:0]    acc,    // RAW pairs: pair p at p*48 (== mac_bank_dsp encoding)
    output wire signed [22:0]        sum_act_o   // for parent readback recovery
);
    // phase from the clk LEVEL sampled at the clk2x edge (mac_bank_dp convention):
    // clk high -> phase 0 (w0,x0); clk low -> phase 1 (w1,x1). No reset, no race.
    wire phase = ~clk;
    wire [LANES*4-1:0] wsel = phase ? w1 : w0;
    wire signed [7:0]  xsel = phase ? x1 : x0;
    wire signed [8:0]  xs   = xsel;                 // explicit sign-extend

    // shared per-stream sum of activations (+8 bias correction), advanced 2/clk
    reg signed [22:0] sum_act;
    always @(posedge clk2x) begin
        if (clr)      sum_act <= 23'sd0;
        else if (en)  sum_act <= sum_act + xs;
    end
    assign sum_act_o = sum_act;

    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 2) begin : g_d
            wire [3:0]  w0n  = wsel[gl*4 +: 4]     ^ 4'h8;   // +8 bias -> unsigned
            wire [3:0]  w1n  = wsel[(gl+1)*4 +: 4] ^ 4'h8;
            wire [25:0] wpak = {w1n, 18'b0, w0n};            // < 2^26: one DSP
            (* use_dsp = "yes" *) reg signed [47:0] a;
            always @(posedge clk2x) begin
                if (clr)     a <= 48'sd0;
                else if (en) a <= a + $signed({1'b0, wpak}) * xs;
            end
            // raw 48-bit pair accumulator: pair (gl/2) in acc[(gl/2)*48 +: 48].
            // (LANES/2)*48 == LANES*ABITS, so acc bus width is unchanged.
            assign acc[(gl/2)*48 +: 48] = a;
        end
    endgenerate
endmodule
