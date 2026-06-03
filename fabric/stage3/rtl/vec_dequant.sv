// -----------------------------------------------------------------------------
// vec_dequant — P-lane parallel per-channel dequant, BIT-EXACT to the scalar
// dequant the sequencer does in S_QKV_DEQ / S_PROJ_DEQ / S_FC_DEQ / S_MP_DEQ /
// S_HEAD_DEQ (rtl/sequencer_fast.sv) and seq_ref.IntSequencer._dequant_to_q.
//
// The scalar contract per element (gemvy INT32, per-channel mant 24-bit signed,
// per-channel exp signed 8-bit, target fraction FRAC):
//   dq_prod = $signed(gemvy) * $signed(mant)        // signed, up to ~96 bits
//   dq_shv  = $signed(exp) + FRAC
//   dq_val  = (dq_shv >= 0) ? dq_prod <<< dq_shv      // left shift
//                           : rsh_round(dq_prod, -dq_shv)  // round-half-away-from-zero
//   out     = dq_val[31:0]
//
// This module applies that to P lanes per cycle. Inputs arrive as PACKED buses
// (one wide vector with lane k in bits [k*W +: W]); each lane copies its slice to
// a plain vector before any arithmetic (iverilog-2012-safe: no variable part-select
// on unpacked-array elements, no over-wide part-selects). The compute is purely
// combinational per lane; a 1-cycle register stage publishes dq_out + carries valid.
//
// rsh_round is copied VERBATIM from sequencer_fast.sv (round-half-away-from-zero).
//
// iverilog -g2012 safe: constant-index part-selects from packed buses (in genvar
// loop), copy-to-plain-vector before shift, single registered write of each output.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module vec_dequant #(
    parameter integer P    = 8     // lanes processed per cycle
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  in_valid,
    input  wire signed [6:0]     frac,    // target fraction, RUNTIME (16 qkv-kv / 25 resid / 12 gelu)
    // PACKED per-lane buses: lane k occupies the k-th fixed-width slot.
    input  wire [P*32-1:0]       gemvy,   // P x signed [31:0]
    input  wire [P*24-1:0]       mant,    // P x signed [23:0]
    input  wire [P*8-1:0]        exp,     // P x signed [ 7:0]
    output reg                   out_valid,
    output reg  [P*32-1:0]       dq_out   // P x signed [31:0]  (== dq_val[31:0] per lane)
);
    // arithmetic right shift with round-half-away-from-zero (VERBATIM from sequencer_fast.sv)
    function signed [95:0] rsh_round(input signed [95:0] v, input integer s);
        reg signed [95:0] half;
        begin
            if (s <= 0) rsh_round = v <<< (-s);
            else begin
                half = (96'sd1 <<< (s-1));
                if (v >= 0) rsh_round = (v + half) >>> s;
                else        rsh_round = -(((-v) + half) >>> s);
            end
        end
    endfunction

    genvar gi;
    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : LANE
            // copy this lane's packed slice into plain vector regs BEFORE any arithmetic
            // (constant-index part-select from a packed bus is iverilog-safe).
            reg signed [31:0] gemvy_v;
            reg        [23:0] mant_v;                          // mant is UNSIGNED 24-bit (a
            reg signed [7:0]  exp_v;                           // positive mantissa, up to 2^24)
            reg signed [95:0] dq_prod;
            reg signed [8:0]  dq_shv;
            reg signed [95:0] dq_val;

            always @* begin
                gemvy_v = gemvy[gi*32 +: 32];
                mant_v  = mant [gi*24 +: 24];
                exp_v   = exp  [gi*8  +: 8 ];
                // zero-extend mant to a positive 25-bit signed before the signed product
                // (values >= 2^23 would flip negative under a 24-bit signed read)
                dq_prod = $signed(gemvy_v) * $signed({1'b0, mant_v});
                dq_shv  = $signed(exp_v) + $signed(frac);       // signed sum (runtime frac)
                if (dq_shv >= 0) dq_val = dq_prod <<< dq_shv;
                else             dq_val = rsh_round(dq_prod, -dq_shv);
            end

            // registered publish of this lane's low 32 bits
            always @(posedge clk) begin
                if (in_valid) dq_out[gi*32 +: 32] <= dq_val[31:0];
            end
        end
    endgenerate

    // valid pipeline (1-cycle latency, matches the registered dq_out stage)
    always @(posedge clk) begin
        if (rst) out_valid <= 1'b0;
        else     out_valid <= in_valid;
    end
endmodule
