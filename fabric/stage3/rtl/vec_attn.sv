// -----------------------------------------------------------------------------
// vec_attn — single-head, P-wide attention kernel for "Kevin on Kria".
//
// For ONE attention head (HEAD_DIM=64) compute the context vector
//   ctx[d] = sum_j prob[j] * v_j[d]   (d in 0..63)
// from q[0..63], the K cache (T x 64) and the V cache (T x 64), bit-identical to
// the integer reference seq_ref._attn_step / rtl/sequencer_fast.sv S_SCORE/S_CTX.
//
// Pinned integer datapath (constants from seq_ref.py):
//   q,k,v : signed Q.16 (VFRAC=16), 32-bit.
//   score[j] = sat16( rsh_round( sum_d q[d]*k_j[d], 2*VFRAC+ISQRT-SCORE_FRAC=27 ) )  Q8.8
//   probs    = softmax.sv( scores[0..T-1] )                                          Q1.20
//   ctx[d]   = rsh_round( sum_j prob[j]*v_j[d], PROB_FRAC+VFRAC-RESID_FRAC=11 )       Q6.25
//   prob[j] is unsigned 21-bit, zero-extended ({11'd0, prob}) before the multiply.
//
// P-wide reductions: the 64 q*k products of a score are summed P-per-cycle over an
// adder tree (64/P cycles per key); the ctx sums do P output dims in parallel, one
// position per cycle (T cycles per group of P dims). Integer add is associative so
// the lane partitioning does not change the result; the rsh_round/sat are applied
// at exactly the same points with the same shifts as the scalar reference.
//
// iverilog-2012 note: every indexed read of an UNPACKED-array element is first
// copied to a plain vector reg before any part-select/shift (the X-on-unpacked rule).
//
// Load protocol (P-WIDE streaming, host asserts ld_valid for each wide word):
//   1. pulse `start` with `tcount` = T (1..TMAX).
//   2. stream HEAD_DIM/P q words, then T*HEAD_DIM/P K words (position-major), then
//      T*HEAD_DIM/P V words. Each wide word = P consecutive Q.16 lanes
//      (lane l in bits [l*32 +: 32]) on `ld_data`, qualified by `ld_valid`.
//   3. compute runs; `ctx_valid` strobes per output dim with (ctx_idx, ctx_data),
//      then `done` pulses once.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module vec_attn #(
    parameter integer P        = 8,      // reduction lanes (HEAD_DIM % P == 0)
    parameter integer HEAD_DIM = 64,
    parameter integer TMAX     = 32
) (
    input  wire                clk,
    input  wire                rst,        // sync reset

    input  wire                start,      // pulse with tcount valid; begins load
    input  wire [8:0]          tcount,     // number of valid positions T (1..TMAX)

    input  wire                ld_valid,   // a load word is presented on ld_data
    input  wire [P*32-1:0]     ld_data,    // P Q.16 lanes (q, then K, then V streams)
    output reg                 ld_ready,   // high while the loader is consuming words

    output reg                 ctx_valid,  // a ctx group is presented this cycle
    output reg  [6:0]          ctx_idx,    // which dim group (0..HEAD_DIM/P-1)
    output reg  [P*32-1:0]     ctx_data,   // P ctx lanes Q6.25 (dim = ctx_idx*P + lane)
    output reg                 done        // pulses 1 cycle when the head is complete
);
    // ---- pinned constants (seq_ref.py) ----
    localparam integer VFRAC      = 16;
    localparam integer ISQRT      = 3;
    localparam integer SCORE_FRAC = 8;
    localparam integer PROB_FRAC  = 20;
    localparam integer RESID_FRAC = 25;
    localparam integer SCORE_SH   = (2*VFRAC) + ISQRT - SCORE_FRAC;   // 27
    localparam integer CTX_SH     = PROB_FRAC + VFRAC - RESID_FRAC;   // 11

    localparam integer NGRP = HEAD_DIM / P;   // adder-tree passes per score / dim groups

    // ---- on-chip storage (wide words: P lanes per row, lane l = [l*32 +: 32]) ----
    reg [P*32-1:0]    qmem [0:HEAD_DIM/P-1];        // q
    reg [P*32-1:0]    kmem [0:TMAX*HEAD_DIM/P-1];   // K cache, position-major
    reg [P*32-1:0]    vmem [0:TMAX*HEAD_DIM/P-1];   // V cache, position-major
    reg [20:0]        probmem [0:TMAX-1];           // softmax probs Q1.20 (unsigned)

    // ---- softmax instance ----
    reg                sm_start;
    reg  [8:0]         sm_tcount;
    reg                sm_in_valid;
    reg  signed [15:0] sm_score;
    wire               sm_in_ready;
    wire               sm_out_valid;
    wire [20:0]        sm_prob;
    wire               sm_done;
    softmax #(.TMAX(TMAX)) u_sm (
        .clk(clk), .rst(rst), .start(sm_start), .t_count(sm_tcount),
        .in_valid(sm_in_valid), .score(sm_score), .in_ready(sm_in_ready),
        .out_valid(sm_out_valid), .prob(sm_prob), .done(sm_done)
    );

    // ---- FSM ----
    localparam [3:0]
        S_IDLE      = 4'd0,
        S_LD_Q      = 4'd1,
        S_LD_K      = 4'd2,
        S_LD_V      = 4'd3,
        S_SCORE     = 4'd4,   // P-wide accumulate one key's q.k
        S_SCORE_EMIT= 4'd5,   // round/sat -> feed softmax
        S_SM_COLL   = 4'd6,   // collect probs
        S_CTX       = 4'd7,   // accumulate prob.v for a group of P dims
        S_CTX_EMIT  = 4'd8,   // round + emit P ctx words
        S_DONE      = 4'd9;
    reg [3:0] st;

    reg [8:0]  T;             // latched tcount
    reg [15:0] ld_cnt;        // generic load counter

    // score accumulation
    reg [8:0]  ji;            // current key index
    reg [8:0]  grp;           // adder-tree pass / dim-group index
    reg signed [63:0] score_acc;
    reg signed [63:0] score_q88;

    // ctx accumulation: P parallel lane accumulators
    reg signed [95:0] ctx_acc [0:P-1];
    reg [8:0]  pj;            // position index within ctx sum
    reg [6:0]  emit_idx;      // which lane is being emitted in S_CTX_EMIT

    // pipeline registers: products registered before accumulate (Fmax)
    reg signed [63:0] sc_prod  [0:P-1];   // q*k lane products
    reg               sc_v;
    reg signed [95:0] ctx_prod [0:P-1];   // prob*v lane products
    reg               cx_v;

    integer lane;

    // ---- combinational P-wide score partial sum -------------------------------
    // sum over the P products in pass `grp`:  sum_l q[grp*P+l] * k_ji[grp*P+l]
    // copy unpacked elements to plain vectors before use (iverilog X rule).
    reg [P*32-1:0]    qw, kw, vw;
    reg signed [31:0] q_l;
    reg signed [31:0] k_l;
    reg signed [63:0] sc_part;

    // plain-vector scratch for unpacked reads in S_CTX / S_CTX_EMIT
    reg [20:0]        p_v;
    reg signed [31:0] v_l;
    reg signed [95:0] ca_v;

    // ---- combinational round/sat helpers (functions below) ----
    // (rsh_round / sat16 copied verbatim from sequencer_fast.sv)

    always @(posedge clk) begin
        sm_start    <= 1'b0;
        sm_in_valid <= 1'b0;
        ctx_valid   <= 1'b0;
        done        <= 1'b0;

        if (rst) begin
            st       <= S_IDLE;
            ld_ready <= 1'b0;
        end else begin
            case (st)
                S_IDLE: begin
                    ld_ready <= 1'b0;
                    if (start) begin
                        T        <= tcount;
                        ld_cnt   <= 16'd0;
                        ld_ready <= 1'b1;
                        st       <= S_LD_Q;
                    end
                end

                // ---- load q: HEAD_DIM/P wide words ----
                S_LD_Q: begin
                    ld_ready <= 1'b1;
                    if (ld_valid) begin
                        qmem[ld_cnt[2:0]] <= ld_data;
                        if (ld_cnt == NGRP-1) begin
                            ld_cnt <= 16'd0;
                            st     <= S_LD_K;
                        end else ld_cnt <= ld_cnt + 16'd1;
                    end
                end

                // ---- load K cache: T*HEAD_DIM/P wide words ----
                S_LD_K: begin
                    ld_ready <= 1'b1;
                    if (ld_valid) begin
                        kmem[ld_cnt] <= ld_data;
                        if (ld_cnt == (T*NGRP) - 1) begin
                            ld_cnt <= 16'd0;
                            st     <= S_LD_V;
                        end else ld_cnt <= ld_cnt + 16'd1;
                    end
                end

                // ---- load V cache: T*HEAD_DIM/P wide words ----
                S_LD_V: begin
                    ld_ready <= 1'b1;
                    if (ld_valid) begin
                        vmem[ld_cnt] <= ld_data;
                        if (ld_cnt == (T*NGRP) - 1) begin
                            ld_ready  <= 1'b0;
                            // kick off softmax for T scores
                            sm_tcount <= T;
                            sm_start  <= 1'b1;
                            ji        <= 9'd0;
                            grp       <= 9'd0;
                            score_acc <= 64'sd0;
                            sc_v      <= 1'b0;
                            st        <= S_SCORE;
                        end else ld_cnt <= ld_cnt + 16'd1;
                    end
                end

                // ---- score: P-wide accumulate q.k for key ji ----
                // 2-stage: lane products registered, adder tree accumulates the
                // delayed products (one extra cycle per key, multipliers retime to DSPs).
                S_SCORE: begin
                    qw   = qmem[(grp < NGRP) ? grp : 9'd0];
                    kw   = kmem[ji*NGRP + ((grp < NGRP) ? grp : 9'd0)];
                    for (lane = 0; lane < P; lane = lane + 1) begin
                        q_l = qw[lane*32 +: 32];
                        k_l = kw[lane*32 +: 32];
                        sc_prod[lane] <= $signed(q_l) * $signed(k_l);
                    end
                    sc_v <= (grp != NGRP);
                    if (grp != NGRP) grp <= grp + 9'd1;
                    if (sc_v) begin
                        sc_part = 64'sd0;
                        for (lane = 0; lane < P; lane = lane + 1)
                            sc_part = sc_part + sc_prod[lane];
                        score_acc <= score_acc + sc_part;
                        if (grp == NGRP) begin
                            grp  <= 9'd0;
                            sc_v <= 1'b0;
                            st   <= S_SCORE_EMIT;
                        end
                    end
                end

                // ---- round/sat the score and feed it to softmax ----
                S_SCORE_EMIT: begin
                    // only advance when softmax is ready to consume (PASS1)
                    if (sm_in_ready) begin
                        score_q88   = rsh_round(score_acc, SCORE_SH);
                        sm_score    <= sat16(score_q88);
                        sm_in_valid <= 1'b1;
                        if (ji == T-1) begin
                            // all scores fed; collect probs
                            ji        <= 9'd0;
                            st        <= S_SM_COLL;
                        end else begin
                            ji        <= ji + 9'd1;
                            grp       <= 9'd0;
                            score_acc <= 64'sd0;
                            st        <= S_SCORE;
                        end
                    end
                end

                // ---- collect softmax probs into probmem ----
                S_SM_COLL: begin
                    if (sm_out_valid) begin
                        probmem[ji[4:0]] <= sm_prob;
                        ji <= ji + 9'd1;
                    end
                    if (sm_done) begin
                        // start ctx: group of P dims, base dim = 0
                        grp  <= 9'd0;
                        pj   <= 9'd0;
                        cx_v <= 1'b0;
                        for (lane = 0; lane < P; lane = lane + 1)
                            ctx_acc[lane] <= 96'sd0;
                        st  <= S_CTX;
                    end
                end

                // ---- ctx: accumulate prob[pj]*v_pj[d] for P dims (d = grp*P+lane) ----
                // 2-stage: prob*v lane products registered, accumulate one cycle behind.
                S_CTX: begin
                    // prob[pj] zero-extended (unsigned Q1.20)
                    p_v  = probmem[(pj < T) ? pj[4:0] : 5'd0];
                    vw   = vmem[((pj < T) ? pj : 9'd0)*NGRP + grp];
                    for (lane = 0; lane < P; lane = lane + 1) begin
                        v_l = vw[lane*32 +: 32];
                        ctx_prod[lane] <= $signed({75'd0, p_v}) * $signed({{64{v_l[31]}}, v_l});
                    end
                    cx_v <= (pj != T);
                    if (pj != T) pj <= pj + 9'd1;
                    if (cx_v) begin
                        for (lane = 0; lane < P; lane = lane + 1)
                            ctx_acc[lane] <= ctx_acc[lane] + ctx_prod[lane];
                        if (pj == T) begin
                            pj       <= 9'd0;
                            cx_v     <= 1'b0;
                            emit_idx <= 7'd0;
                            st       <= S_CTX_EMIT;
                        end
                    end
                end

                // ---- round + emit ALL P ctx lanes of this group in one cycle ----
                S_CTX_EMIT: begin
                    for (lane = 0; lane < P; lane = lane + 1) begin
                        ca_v = ctx_acc[lane];
                        ctx_data[lane*32 +: 32] <= rsh_round(ca_v, CTX_SH);
                    end
                    ctx_idx   <= grp[6:0];
                    ctx_valid <= 1'b1;
                    if (grp == NGRP-1) begin
                        st <= S_DONE;
                    end else begin
                        grp <= grp + 9'd1;
                        pj  <= 9'd0;
                        for (lane = 0; lane < P; lane = lane + 1)
                            ctx_acc[lane] <= 96'sd0;
                        st  <= S_CTX;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    st   <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end

    // ---- round/sat (verbatim from sequencer_fast.sv) ----
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
            if (v >  64'sd32767)  sat16 = 16'sd32767;
            else if (v < -64'sd32768) sat16 = -16'sd32768;
            else sat16 = v[15:0];
        end
    endfunction
endmodule
