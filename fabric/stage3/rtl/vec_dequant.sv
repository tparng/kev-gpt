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
// PIPELINING (250 campaign): the old stage-2 cone (dq_shv_r -> dq_out) was the
// 4.5ns-critical path — the 96b round-bias ADD (16x CARRY8) FOLLOWED by the
// variable barrel SHIFT FOLLOWED by negate/truncate, a 26-level CARRY8 chain.
// It is now SPLIT into two registered substages, ARITHMETIC BIT-IDENTICAL to
// the rsh_round contract above:
//   stage 2 (NEW reg): resolve direction/sign and do the 96b round-bias add ->
//                      a shift OPERAND op_r (non-negative for the right-shift
//                      cases), the shift amount sh_r, direction dir_r (0=left,
//                      1=right) and post-negate flag neg_r.
//   stage 3 (publish): apply the variable shift (<<< for left, >>> for right),
//                      conditionally negate, truncate to [31:0] -> dq_out.
// Latency 2 -> 3 cycles, throughput UNCHANGED (1 P-wide row/cycle, accepts
// in_valid every cycle). The GE_RB readback in cohort_engine is data-following-
// valid (it collects on dq_vout and issues independently on rv2), so the +1
// cycle drains as +1 cyc per GEMM-call readback fill, NOT per row.
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
    // NOTE: the round-half-away-from-zero contract (formerly the rsh_round function,
    // VERBATIM from sequencer_fast.sv) is now inlined and pipeline-split across stages
    // 2 and 3 below — see the stage-2/stage-3 comments for the bit-identical mapping.

    reg v1, v2;                   // stage-1 / stage-2 valid
    genvar gi;
    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : LANE
            // copy this lane's packed slice into plain vector regs BEFORE any arithmetic
            // (constant-index part-select from a packed bus is iverilog-safe).
            reg signed [31:0] gemvy_v;
            reg        [23:0] mant_v;                          // mant is UNSIGNED 24-bit (a
            reg signed [7:0]  exp_v;                           // positive mantissa, up to 2^24)
            reg signed [95:0] dq_prod_r;                       // STAGE-1 product register
            reg signed [8:0]  dq_shv_r;                        //   (multiplier retimes to DSPs)

            // stage-2 outputs (NEW register splitting the cone)
            reg signed [95:0] op_r;       // operand fed into the variable barrel shift
            reg        [8:0]  sh_r;        // shift amount (magnitude; dir_r selects sense)
            reg               dir_r;       // 0 = left shift, 1 = right shift
            reg               neg_r;       // negate the shifted result (neg-v right-shift case)
            // stage-2 combinational temporaries
            reg signed [8:0]  rs;          // right-shift amount = -dq_shv_r
            reg signed [95:0] half;        // round bias = 1 <<< (rs-1)
            reg signed [95:0] op_c;
            reg signed [8:0]  sh_c;
            reg               dir_c, neg_c;
            // stage-3 combinational
            reg signed [95:0] shifted;
            reg signed [95:0] res;

            always @* begin
                gemvy_v = gemvy[gi*32 +: 32];
                mant_v  = mant [gi*24 +: 24];
                exp_v   = exp  [gi*8  +: 8 ];
            end

            // stage 1: product + shift amount (registered)
            always @(posedge clk) begin
                if (in_valid) begin
                    // zero-extend mant to a positive 25-bit signed before the product
                    dq_prod_r <= $signed(gemvy_v) * $signed({1'b0, mant_v});
                    dq_shv_r  <= $signed(exp_v) + $signed(frac);
                end
            end

            // ---- stage 2: resolve direction/sign + 96b round-bias add (registered) ----
            // BIT-IDENTICAL decomposition of the dq_shv >= 0 / rsh_round(-dq_shv) contract:
            //   dq_shv_r >= 0  -> left shift of dq_prod_r by dq_shv_r, no bias, no negate.
            //   dq_shv_r <  0  -> rs = -dq_shv_r (>0); half = 1<<<(rs-1);
            //        v>=0: (v + half) >>> rs        ; op = v + half, no negate
            //        v< 0: -(((-v) + half) >>> rs)  ; op = (-v) + half, negate after
            // The (v +/- half) 96b adds (the CARRY8s) land HERE; the variable shift goes
            // to stage 3. The half expression is written VERBATIM (96'sd1 <<< (rs-1)) so any
            // wrap for large rs is bit-identical to the original rsh_round.
            always @* begin
                if (dq_shv_r >= 0) begin
                    op_c  = dq_prod_r;
                    sh_c  = dq_shv_r;           // dq_shv_r >= 0, fits magnitude
                    dir_c = 1'b0;               // left
                    neg_c = 1'b0;
                end else begin
                    rs    = -dq_shv_r;          // > 0
                    half  = (96'sd1 <<< (rs-1));
                    sh_c  = rs;
                    dir_c = 1'b1;               // right
                    if (dq_prod_r >= 0) begin
                        op_c  = dq_prod_r + half;
                        neg_c = 1'b0;
                    end else begin
                        op_c  = (-dq_prod_r) + half;
                        neg_c = 1'b1;           // negate after >>> rs
                    end
                end
            end
            always @(posedge clk) begin
                if (v1) begin
                    op_r  <= op_c;
                    sh_r  <= sh_c;
                    dir_r <= dir_c;
                    neg_r <= neg_c;
                end
            end

            // ---- stage 3: variable barrel shift + negate + truncate (registered publish) ----
            // op_r for the right cases is >= 0 by construction, so >>> (arithmetic) == >>>
            // (logical) — bit-identical to the original (v+half)>>>rs / ((-v)+half)>>>rs.
            always @* begin
                if (dir_r) shifted = op_r >>> sh_r;   // right
                else       shifted = op_r <<< sh_r;   // left
                res = neg_r ? -shifted : shifted;
            end
            always @(posedge clk) begin
                if (v2) dq_out[gi*32 +: 32] <= res[31:0];
            end
        end
    endgenerate

    // valid pipeline (3-cycle latency: product stage, round-bias stage, shift+publish)
    always @(posedge clk) begin
        if (rst) begin v1 <= 1'b0; v2 <= 1'b0; out_valid <= 1'b0; end
        else     begin v1 <= in_valid; v2 <= v1; out_valid <= v2; end
    end
endmodule
