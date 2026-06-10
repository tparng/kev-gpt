// -----------------------------------------------------------------------------
// vec_attn_w — FULL-HEAD-WIDTH attention (doc-7 R2): one POSITION per cycle.
//
// The vec_attn (P=8) slope was 528 cyc/position at the sequencer level; this
// engine consumes a whole position's head row (HEAD_DIM lanes) per beat:
//
//   q     : NGRP beats of P Q.16 lanes (the qkv_bank rows, unchanged shape),
//           then HADAMARD-ROTATED in place (goformer_kvq rotate=True: 6-stage
//           Sylvester butterfly, 2 stages/cycle, then >>3 round-half-away-0) so
//           q.k is computed in the same rotated basis kv_bank stores K in. The
//           4 rotate cycles fit EXACTLY in the host's q-done -> first-K-beat gap.
//   K     : T beats of HEAD_DIM*32 (kv_bank wide dequant stream) -> 64-MAC dot
//           per beat -> score -> softmax PASS1 inline (in_ready is high through
//           PASS1 and both producers run 1/cycle, so no skid is needed)
//   probs : collected to probmem (FULL-width index — the old vec_attn probmem
//           [ji[4:0]] silently wrapped at T>32; this engine owns T up to TMAX)
//   V     : T beats -> 64 per-dim accumulators (prob*v), position order j=0..T-1
//           identical to the scalar reference (per-dim sequential adds)
//   ctx   : all 64 accumulators rounded (rsh_round(acc,11)) into a buffer, then
//           UN-ROTATED with the same butterfly + >>3 (H is symmetric, so
//           unrotate == rotate — verbatim goformer_kvq._attn_step order), then
//           NGRP P-wide strobes (interface identical to vec_attn's S_ACL shape)
//
// Bit-exactness: the 64-product dot is summed as 8 partial sums of 8 then a
// final sum-of-8 — integer addition with no mid-sum saturation in a 64-bit
// accumulator (|q|,|k| < 2^21 -> 64 products < 2^48), so the tree order is
// EXACTLY the scalar sequential sum. Score and ctx rounding/sat are verbatim
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
    localparam [3:0] W_IDLE=4'd0, W_Q=4'd1, W_QH=4'd2, W_QR=4'd3,
                     W_K=4'd4, W_SMC=4'd5, W_V=4'd6,
                     W_CR1=4'd7, W_CHAD=4'd8, W_CR2=4'd9, W_EMIT=4'd10;
    reg [3:0] st;
    reg [8:0] T;
    reg [$clog2(NGRP+1)-1:0] qcnt;

    reg [HEAD_DIM*32-1:0] qreg;                 // the q vector (rotated in place)
    reg [20:0] probmem [0:TMAX-1];              // Q1.20 probs, FULL-width indexed
    reg [TW-1:0] jc;                            // prob collect counter

    // ---- integer Hadamard (goformer_kvq): 2 butterfly stages/cycle ----
    // Exact integer adds (no mid-stage rounding), so the 2-per-cycle grouping is
    // bit-identical to the reference's 6 sequential stages. |q| < 2^21-class on
    // this model -> raw < 2^27: int32 lanes hold the q butterfly. The ctx
    // butterfly runs on 40-bit lanes (|ctx_q25 rounded| < 2^30 -> raw < 2^36).
    reg [1:0] had_i;                            // stage-pair counter (0..2)
    reg [HEAD_DIM*32-1:0] qhad_t, qhad_n;       // q butterfly next-state
    reg [HEAD_DIM*40-1:0] chad;                 // ctx rotate buffer (40-bit lanes)
    reg [HEAD_DIM*40-1:0] chad_t, chad_n;
    integer hb, hh1, hh2, hpx;
    reg signed [31:0] hqx, hqy;
    reg signed [39:0] hcx, hcy;
    always @* begin
        hh1 = 1 << (had_i * 2);                 // h = 1, 4, 16
        hh2 = 2 << (had_i * 2);                 // h = 2, 8, 32
        for (hb = 0; hb < HEAD_DIM; hb = hb + 1) begin
            hpx = hb ^ hh1;
            hqx = $signed(qreg[hb*32 +: 32]);
            hqy = $signed(qreg[hpx*32 +: 32]);
            qhad_t[hb*32 +: 32] = ((hb & hh1) == 0) ? (hqx + hqy) : (hqy - hqx);
        end
        for (hb = 0; hb < HEAD_DIM; hb = hb + 1) begin
            hpx = hb ^ hh2;
            hqx = $signed(qhad_t[hb*32 +: 32]);
            hqy = $signed(qhad_t[hpx*32 +: 32]);
            qhad_n[hb*32 +: 32] = ((hb & hh2) == 0) ? (hqx + hqy) : (hqy - hqx);
        end
        for (hb = 0; hb < HEAD_DIM; hb = hb + 1) begin
            hpx = hb ^ hh1;
            hcx = $signed(chad[hb*40 +: 40]);
            hcy = $signed(chad[hpx*40 +: 40]);
            chad_t[hb*40 +: 40] = ((hb & hh1) == 0) ? (hcx + hcy) : (hcy - hcx);
        end
        for (hb = 0; hb < HEAD_DIM; hb = hb + 1) begin
            hpx = hb ^ hh2;
            hcx = $signed(chad_t[hb*40 +: 40]);
            hcy = $signed(chad_t[hpx*40 +: 40]);
            chad_n[hb*40 +: 40] = ((hb & hh2) == 0) ? (hcx + hcy) : (hcy - hcx);
        end
    end

    // >>3 round-half-away-from-zero of the full q / ctx vectors (combinational)
    reg [HEAD_DIM*32-1:0] qrnd;
    reg [HEAD_DIM*40-1:0] crnd;
    integer rb;
    reg signed [31:0] rqv;
    reg signed [39:0] rcv;
    always @* begin
        for (rb = 0; rb < HEAD_DIM; rb = rb + 1) begin
            rqv = $signed(qreg[rb*32 +: 32]);
            if (rqv >= 0) qrnd[rb*32 +: 32] = (rqv + 32'sd4) >>> 3;
            else          qrnd[rb*32 +: 32] = -((-rqv + 32'sd4) >>> 3);
            rcv = $signed(chad[rb*40 +: 40]);
            if (rcv >= 0) crnd[rb*40 +: 40] = (rcv + 40'sd4) >>> 3;
            else          crnd[rb*40 +: 40] = -((-rcv + 40'sd4) >>> 3);
        end
    end

    // ---- score pipeline (3 stages, 1 position/cycle) ----
    reg [HEAD_DIM*32-1:0] kreg;                 // stage A: registered K row
    reg                   kv0;
    reg signed [63:0]     ps   [0:NGRP-1];      // stage B: 8 partial sums of 8
    reg                   kv1;
    reg [8:0]             scnt;                 // scores fed to softmax (0..T)
    // stage C feeds softmax (sm_in_valid/sm_score)

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
            st <= W_IDLE; kv0 <= 1'b0; kv1 <= 1'b0; vv0 <= 1'b0;
        end else begin
            case (st)
                W_IDLE: if (start) begin
                    T <= tcount; qcnt <= 0; st <= W_Q;
                end
                W_Q: if (q_valid) begin
                    qreg[qcnt*P*32 +: P*32] <= q_data;
                    if (qcnt == NGRP-1) begin
                        sm_tcount <= T;
                        kv0 <= 1'b0; kv1 <= 1'b0; scnt <= 9'd0;
                        had_i <= 2'd0; st <= W_QH;
                    end else qcnt <= qcnt + 1'b1;
                end
                // ---- q Hadamard: 3 butterfly cycles + 1 round cycle. The host's
                // first K beat lands EXACTLY in W_QR (q done at t -> kb_rstart t+1
                // -> kv_valid t+4), so the K capture stage runs here too and qreg
                // is final one cycle before the first dot (stage B at t+5).
                W_QH: begin
                    qreg <= qhad_n;
                    if (kv_valid) kreg <= kv_data;            // defensive (can't fire)
                    kv0 <= kv_valid;
                    if (had_i == 2'd2) st <= W_QR;
                    else had_i <= had_i + 2'd1;
                end
                W_QR: begin
                    qreg <= qrnd;                             // >>3 half-away-from-0
                    sm_start <= 1'b1;                         // PASS1 armed for K
                    if (kv_valid) kreg <= kv_data;            // first K beat lands here
                    kv0 <= kv_valid;
                    st <= W_K;
                end
                // ---- K: one position/beat -> dot -> score -> softmax inline ----
                W_K: begin
                    // stage A: register the row
                    if (kv_valid) kreg <= kv_data;
                    kv0 <= kv_valid;
                    // stage B: 64 products -> NGRP partial sums of P
                    if (kv0) begin
                        for (g = 0; g < NGRP; g = g + 1) begin
                            part = 64'sd0;
                            for (l = 0; l < P; l = l + 1) begin
                                q_l = qreg[(g*P+l)*32 +: 32];
                                k_l = kreg[(g*P+l)*32 +: 32];
                                part = part + $signed(q_l) * $signed(k_l);
                            end
                            ps[g] <= part;
                        end
                    end
                    kv1 <= kv0;
                    // stage C: total -> round/sat -> softmax (PASS1 takes 1/cycle)
                    if (kv1) begin
                        tot = 64'sd0;
                        for (g = 0; g < NGRP; g = g + 1) tot = tot + ps[g];
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
                            st <= W_CR1;
                        end else vjc <= vjc + 1'b1;
                    end
                end
                // ---- ctx un-rotate (reference order: round ALL 64 accumulators
                // to ctx_q25 FIRST, then Hadamard-rotate the rounded values) ----
                W_CR1: begin
                    for (l = 0; l < HEAD_DIM; l = l + 1) begin
                        ca = ctx_acc[l];
                        chad[l*40 +: 40] <= rsh_round({{32{ca[63]}}, ca}, CTX_SH);
                    end
                    had_i <= 2'd0; st <= W_CHAD;
                end
                W_CHAD: begin
                    chad <= chad_n;
                    if (had_i == 2'd2) st <= W_CR2;
                    else had_i <= had_i + 2'd1;
                end
                W_CR2: begin
                    chad <= crnd;                             // >>3 half-away-from-0
                    eg <= 0; st <= W_EMIT;
                end
                // ---- emit NGRP P-wide ctx strobes (vec_attn-compatible) ----
                W_EMIT: begin
                    for (l = 0; l < P; l = l + 1)
                        ctx_data[l*32 +: 32] <= chad[(eg*P + l)*40 +: 32];
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
