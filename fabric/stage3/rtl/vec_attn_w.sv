// -----------------------------------------------------------------------------
// vec_attn_w — FULL-HEAD-WIDTH attention (doc-7 R2): one POSITION per cycle.
//
// The vec_attn (P=8) slope was 528 cyc/position at the sequencer level; this
// engine consumes a whole position's head row (HEAD_DIM lanes) per beat:
//
//   q     : NGRP beats of P Q.16 lanes (the qkv_bank rows, unchanged shape)
//   K     : T beats of HEAD_DIM*32 (kv_bank wide dequant stream) -> 64-MAC dot
//           per beat -> score -> softmax PASS1 inline (in_ready is high through
//           PASS1 and both producers run 1/cycle, so no skid is needed)
//   probs : collected to probmem (FULL-width index — the old vec_attn probmem
//           [ji[4:0]] silently wrapped at T>32; this engine owns T up to TMAX)
//   V     : T beats -> 64 per-dim accumulators (prob*v), position order j=0..T-1
//           identical to the scalar reference (per-dim sequential adds)
//   ctx   : NGRP P-wide strobes (interface identical to vec_attn's S_ACL shape)
//
// Bit-exactness: the 64-product dot is summed as 8 partial sums of 8, then
// two registered 4-way sums, then a final 2-way sum — integer addition with
// no mid-sum saturation in a 64-bit accumulator (|q|,|k| < 2^21 -> 64
// products < 2^48, full sum < 2^54), so ANY reduction order is bit-identical
// to the scalar sequential sum. Score and ctx rounding/sat are verbatim
// vec_attn: rsh_round(acc, 27) -> sat16 -> softmax; rsh_round(acc, 11) -> ctx.
//
// iverilog-2012 safe: unpacked elements copied to plain regs before part-
// selects; no over-wide part-selects.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module vec_attn_w #(
    parameter integer P        = 8,
    parameter integer HEAD_DIM = 64,
    parameter integer TMAX     = 256
) (
    input  wire                     clk,
    input  wire                     rst,

    input  wire                     start,      // pulse with tcount valid
    input  wire [8:0]               tcount,     // positions T (1..TMAX)

    input  wire                     q_valid,    // NGRP beats of P Q.16 lanes
    input  wire [P*32-1:0]          q_data,

    input  wire                     kv_valid,   // K beats, then (after k_done) V beats
    input  wire [HEAD_DIM*32-1:0]   kv_data,    // one position's head row, dequantised

    output reg                      k_done,     // pulses: probs ready, host may stream V
    output reg                      ctx_valid,  // NGRP P-wide ctx strobes
    output reg  [6:0]               ctx_idx,
    output reg  [P*32-1:0]          ctx_data,
    output reg                      done
);
    localparam integer NGRP     = HEAD_DIM / P;
    localparam integer SCORE_SH = 27;            // 2*VFRAC + ISQRT - SCORE_FRAC
    localparam integer CTX_SH   = 11;            // PROB_FRAC + VFRAC - RESID_FRAC
    localparam integer TW       = $clog2(TMAX);

    // ---- softmax (existing brick, verbatim wiring) ----
    reg                sm_start;
    reg  [8:0]         sm_tcount;
    reg                sm_in_valid;
    reg  signed [15:0] sm_score;
    wire               sm_in_ready;
    wire               sm_out_valid;
    wire [20:0]        sm_prob;
    wire               sm_done;
    // softmax_f = softmax.sv with PASS2/3 pipelined to 1 element/cycle (R4c);
    // bit-identical probs, ~6 -> ~2 cycles/position.
    softmax_f #(.TMAX(TMAX)) u_sm (
        .clk(clk), .rst(rst), .start(sm_start), .t_count(sm_tcount),
        .in_valid(sm_in_valid), .score(sm_score), .in_ready(sm_in_ready),
        .out_valid(sm_out_valid), .prob(sm_prob), .done(sm_done)
    );

    // ---- state ----
    localparam [2:0] W_IDLE=3'd0, W_Q=3'd1, W_K=3'd2, W_SMC=3'd3,
                     W_V=3'd4, W_EMIT=3'd5;
    reg [2:0] st;
    reg [8:0] T;
    reg [$clog2(NGRP+1)-1:0] qcnt;

    reg [HEAD_DIM*32-1:0] qreg;                 // the exact (unquantised) q vector
    reg [20:0] probmem [0:TMAX-1];              // Q1.20 probs, FULL-width indexed
    reg [TW-1:0] jc;                            // prob collect counter

    // ---- score pipeline (5 stages, 1 position/cycle) ----
    // TIMING: the 16x16 LUT multiplies are REGISTERED (stage B) before the
    // 8-of-8 partial sums (stage C); the 8-way final reduction + round/sat
    // cone (OOC 5ns WNS -0.776, ps_reg -> sm_score_reg) is SPLIT: stage D1
    // registers two 4-way sums, stage D2 does the 2-way sum + rsh_round +
    // sat16 into sm_score. Pure integer adds at full 64-bit width with no
    // intermediate saturation, so the regrouping is bit-exact; the stream
    // stays 1 position/cycle, +1 more cycle of latency.
    reg [HEAD_DIM*32-1:0] kreg;                 // stage A: registered K row
    reg                   kv0;
    reg signed [63:0]     pr   [0:HEAD_DIM-1];  // stage B: 64 registered products
    reg                   kvp;
    reg signed [63:0]     ps   [0:NGRP-1];      // stage C: 8 partial sums of 8
    reg                   kv1;
    reg signed [63:0]     sum_lo, sum_hi;       // stage D1: two 4-way sums
    reg                   kv2;
    reg [8:0]             scnt;                 // scores fed to softmax (0..T)
    // stage D2 feeds softmax (sm_in_valid/sm_score)

    // ---- ctx pipeline (V phase) ----
    reg [HEAD_DIM*32-1:0] vreg;                 // stage A: registered V row
    reg [20:0]            preg;                 //          prob for this position
    reg                   vv0, v_first;
    reg signed [63:0]     ctx_acc [0:HEAD_DIM-1];
    reg [TW:0]            vj;                   // V position counter (stage A)
    reg [TW:0]            vjc;                  // accumulated-position counter (stage B)
    reg [$clog2(NGRP+1)-1:0] eg;                // emit group counter

    // ---- combinational helpers ----
    integer l, g;
    reg signed [31:0] q_l, k_l, v_l;
    reg signed [63:0] part, tot;
    reg signed [63:0] prod;
    reg [20:0] p_t;
    reg signed [63:0] ca;

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

    function signed [15:0] sat16(input signed [63:0] v);
        begin
            if (v >  64'sd32767)      sat16 = 16'sd32767;
            else if (v < -64'sd32768) sat16 = -16'sd32768;
            else sat16 = v[15:0];
        end
    endfunction

    always @(posedge clk) begin
        sm_start <= 1'b0; sm_in_valid <= 1'b0;
        ctx_valid <= 1'b0; k_done <= 1'b0; done <= 1'b0;
        if (rst) begin
            st <= W_IDLE; kv0 <= 1'b0; kvp <= 1'b0; kv1 <= 1'b0; kv2 <= 1'b0;
            vv0 <= 1'b0;
        end else begin
            case (st)
                W_IDLE: if (start) begin
                    T <= tcount; qcnt <= 0; st <= W_Q;
                end
                W_Q: if (q_valid) begin
                    qreg[qcnt*P*32 +: P*32] <= q_data;
                    if (qcnt == NGRP-1) begin
                        sm_tcount <= T; sm_start <= 1'b1;     // PASS1 armed for K
                        kv0 <= 1'b0; kvp <= 1'b0; kv1 <= 1'b0; kv2 <= 1'b0;
                        scnt <= 9'd0;
                        st <= W_K;
                    end else qcnt <= qcnt + 1'b1;
                end
                // ---- K: one position/beat -> dot -> score -> softmax inline ----
                W_K: begin
                    // stage A: register the row
                    if (kv_valid) kreg <= kv_data;
                    kv0 <= kv_valid;
                    // stage B: 64 REGISTERED products (multiply only — timing split)
                    if (kv0) begin
                        for (l = 0; l < HEAD_DIM; l = l + 1) begin
                            q_l = qreg[l*32 +: 32];
                            k_l = kreg[l*32 +: 32];
                            pr[l] <= $signed(q_l) * $signed(k_l);
                        end
                    end
                    kvp <= kv0;
                    // stage C: NGRP partial sums of P (SAME addition order as before)
                    if (kvp) begin
                        for (g = 0; g < NGRP; g = g + 1) begin
                            part = 64'sd0;
                            for (l = 0; l < P; l = l + 1)
                                part = part + pr[g*P+l];
                            ps[g] <= part;
                        end
                    end
                    kv1 <= kvp;
                    // stage D1: two REGISTERED 4-way sums (timing split of the
                    // old 8-way sum + round/sat cone; exact 64-bit adds, any
                    // grouping is bit-identical)
                    if (kv1) begin
                        part = 64'sd0;
                        for (g = 0; g < NGRP/2; g = g + 1) part = part + ps[g];
                        sum_lo <= part;
                        part = 64'sd0;
                        for (g = NGRP/2; g < NGRP; g = g + 1) part = part + ps[g];
                        sum_hi <= part;
                    end
                    kv2 <= kv1;
                    // stage D2: final sum -> round/sat -> softmax (PASS1 1/cycle)
                    if (kv2) begin
                        tot = sum_lo + sum_hi;
                        sm_score    <= sat16(rsh_round({{32{tot[63]}}, tot}, SCORE_SH));
                        sm_in_valid <= 1'b1;
                        if (scnt == T - 1) st <= W_SMC;       // all T scores fed
                        else scnt <= scnt + 9'd1;
                    end
                end
                // ---- V starts at the FIRST prob (R4c): probmem stays >= 3 cycles
                // ahead of the V beats (prob j lands at first+j; the V stream the
                // host starts on k_done delivers beat j at first+3+j at best).
                // Prob collection itself is the UNCONDITIONAL block below the case.
                W_SMC: begin
                    if (sm_out_valid) begin
                        k_done <= 1'b1;            // host: stream V now
                        vj <= 0; vjc <= 0; vv0 <= 1'b0;
                        st <= W_V;
                    end
                end
                // ---- V: one position/beat -> 64 prob*v accumulates ----
                W_V: begin
                    // stage A: register row + its prob
                    if (kv_valid) begin
                        vreg <= kv_data;
                        p_t  = probmem[vj[TW-1:0]];
                        preg <= p_t;
                        v_first <= (vj == 0);
                        vj <= vj + 1'b1;
                    end
                    vv0 <= kv_valid;
                    // stage B: accumulate (position order preserved -> bit-exact)
                    if (vv0) begin
                        for (l = 0; l < HEAD_DIM; l = l + 1) begin
                            v_l  = vreg[l*32 +: 32];
                            prod = $signed({43'd0, preg}) * $signed({{32{v_l[31]}}, v_l});
                            ca   = v_first ? prod : (ctx_acc[l] + prod);
                            ctx_acc[l] <= ca;
                        end
                        if (vjc == {1'b0, T} - 1) begin
                            eg <= 0;
                            st <= W_EMIT;
                        end else vjc <= vjc + 1'b1;
                    end
                end
                // ---- emit NGRP P-wide ctx strobes (vec_attn-compatible) ----
                W_EMIT: begin
                    for (l = 0; l < P; l = l + 1) begin
                        ca = ctx_acc[eg*P + l];
                        ctx_data[l*32 +: 32] <= rsh_round({{32{ca[63]}}, ca}, CTX_SH);
                    end
                    ctx_idx   <= {{(7-$clog2(NGRP+1)){1'b0}}, eg};
                    ctx_valid <= 1'b1;
                    if (eg == NGRP-1) begin
                        done <= 1'b1;
                        st <= W_IDLE;
                    end else eg <= eg + 1'b1;
                end
                default: st <= W_IDLE;
            endcase
            // prob collection runs across W_SMC AND W_V (PASS3 overlaps the V phase)
            if (sm_out_valid) begin
                probmem[jc] <= sm_prob;
                jc <= jc + 1'b1;
            end
            if (start) jc <= 0;                  // prob counter reset per call
        end
    end
endmodule
