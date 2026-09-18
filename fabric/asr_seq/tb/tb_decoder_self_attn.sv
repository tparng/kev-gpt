// -----------------------------------------------------------------------------
// tb_decoder_self_attn — the decoder self-attention block gate: does
// rope_apply_vec.sv (new) + kv_bank.sv (checkpoint C's real INT8 K/V cache,
// unmodified) + vec_attn_w.sv (checkpoint C's real score/softmax/ctx engine,
// unmodified) correctly assemble into ASR's decoder self-attention?
//
// Scope: Q/K/V generation (LayerNorm + the Q/K/V GEMVs) is out of scope here
// -- each already separately gated in earlier sessions. This TB drives the
// three ATTENTION-specific modules directly with real Q/K/V (from the real
// HF model), for 3 real decode steps (T=1,2,3, growing the SAME KV cache),
// across all 8 real heads, checked against pack_decoder_self_attn.py's own
// golden ctx_q25 output.
//
// P=4 for kv_bank/vec_attn_w (NOT checkpoint C's own P=8 convention): HEAD_DIM=36
// is not divisible by 8, but is by 4 (HR=NGRP=9). TMAX=32 (not a tiny 4):
// kv_bank.sv's own pos_ra logic assumes $clog2(HROWS)>=9 (HROWS=NLAYER*2*
// NHEAD*TMAX) -- a real sizing floor this TB's own first attempt (TMAX=4)
// tripped, not a checkpoint-C-shape-specific constant.
//
// iverilog-2012 note: subroutine ports with unpacked dimensions aren't
// supported -- tasks below take/return PACKED HEAD_DIM*32-bit buses, with
// pack/unpack done via plain part-selects at the call sites, not unpacked
// array task arguments.
//
// STATUS, stated plainly (see ASR-ACCELERATOR-OP-SEQUENCE.md for the full
// writeup): T=1 (step 0, every head) passes BIT-EXACT. T>=2 (steps 1-2)
// currently MISMATCHES -- root cause not yet found despite extensive
// isolated debugging this session (RoPE's own output verified bit-exact at
// every step via a direct dump; write-commit and read addresses verified
// correct via direct hierarchical monitoring of kv_bank's own internal FSM
// state; and -- the most telling result -- standalone testbenches
// replicating kv_bank.sv + vec_attn_w.sv + rope_apply_vec.sv with the SAME
// real data and an increasingly exact copy of this TB's own operation
// sequence, interleaving, and per-head attention calls, could NOT reproduce
// the corruption at all). That last result is worth taking seriously: it
// suggests either a genuine but narrow interaction this TB's own longer,
// 3-step/8-head sequence triggers and shorter isolation attempts don't, or
// a bug in THIS file's own bookkeeping that hasn't been found yet either --
// not assumed to be one or the other. Not declared fixed; a real, open
// finding, not smoothed over.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_decoder_self_attn;
    localparam integer P        = 4;
    localparam integer HEAD_DIM = 36;
    localparam integer NHEAD    = 8;
    localparam integer NLAYER   = 1;
    localparam integer TMAX     = 32;
    localparam integer N_STEPS  = 3;
    localparam integer HR       = HEAD_DIM / P;     // 9 beats/head vector
    localparam integer ROPE_TMAX = 128;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    // ---- rope_apply_vec ------------------------------------------------------
    reg                       rope_start;
    reg  [$clog2(ROPE_TMAX)-1:0] rope_pos;
    reg  [HEAD_DIM*32-1:0]    rope_head_in;
    wire                      rope_done;
    wire [HEAD_DIM*32-1:0]    rope_head_out;

    rope_apply_vec #(.HEAD_DIM(HEAD_DIM), .ROT_DIM(32), .ROT_PAIRS(16), .TMAX(ROPE_TMAX),
                      .POST_SCALE_Q16(75674),
                      .ROM_FILE_COS("rope_cos.mem"), .ROM_FILE_SIN("rope_sin.mem")) u_rope (
        .clk(clk), .start(rope_start), .position(rope_pos),
        .head_in(rope_head_in), .done(rope_done), .head_out(rope_head_out)
    );

    // ---- kv_bank ---------------------------------------------------------------
    reg         kb_wstart, kb_wvalid, kb_rstart;
    reg  [3:0]  kb_wlayer;
    reg         kb_wkv;
    reg  [1:0]  kb_whead;
    reg  [8:0]  kb_wpos;
    reg  [P*32-1:0] kb_wdata;
    wire        kb_wdone;
    reg  [3:0]  kb_rlayer;
    reg         kb_rkv;
    reg  [1:0]  kb_rhead;
    reg  [8:0]  kb_rtcount;
    wire        kb_rvalid, kb_rdone;
    wire [HEAD_DIM*32-1:0] kb_rdata;

    kv_bank #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER), .TMAX(TMAX),
              .KBITS(8), .MEM_PRIMITIVE("block")) u_kvb (
        .clk(clk), .rst(rst),
        .wq_start(kb_wstart), .wq_layer(kb_wlayer), .wq_kv(kb_wkv), .wq_head(kb_whead),
        .wq_pos(kb_wpos), .wq_valid(kb_wvalid), .wq_data(kb_wdata), .wq_done(kb_wdone),
        .rd_start(kb_rstart), .rd_layer(kb_rlayer), .rd_kv(kb_rkv), .rd_head(kb_rhead),
        .rd_tcount(kb_rtcount), .rd_valid(kb_rvalid), .rd_data(kb_rdata), .rd_done(kb_rdone),
        .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head(2'd0), .rd2_tcount(9'd0),
        .rd2_valid(), .rd2_data(), .rd2_done()
    );

    // ---- vec_attn_w --------------------------------------------------------------
    reg                    at_start;
    reg  [8:0]             at_tcount;
    reg                    at_qvalid;
    reg  [P*32-1:0]        at_qdata;
    wire                   at_kdone, at_ctxvalid, at_done;
    wire [6:0]             at_ctxidx;
    wire [P*32-1:0]        at_ctxdata;

    vec_attn_w #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(TMAX)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .q_valid(at_qvalid), .q_data(at_qdata),
        .kv_valid(kb_rvalid), .kv_data(kb_rdata),     // kv_bank's read stream feeds vec_attn_w directly
        .k_done(at_kdone), .ctx_valid(at_ctxvalid), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata),
        .done(at_done)
    );

    // ---- test vectors ------------------------------------------------------------
    reg [31:0] qkv_in [0:N_STEPS*NHEAD*HEAD_DIM*3-1];   // step-major: [q(36) k(36) v(36)] per head
    reg [31:0] ctx_ref [0:N_STEPS*NHEAD*HEAD_DIM-1];

    reg [HEAD_DIM*32-1:0] q_h, k_h, v_h, q_rope_h, k_rope_h, ctx_got;

    integer step, head, i, base, mismatches, checked;
    integer f;

    // ---- task: run rope_apply_vec on a HEAD_DIM-wide packed vector ----------------
    task do_rope(input integer pos, input reg [HEAD_DIM*32-1:0] vin, output reg [HEAD_DIM*32-1:0] vout);
        begin
            rope_head_in = vin;
            rope_pos = pos[$clog2(ROPE_TMAX)-1:0];
            rope_start = 1'b1;
            @(posedge clk); #1;
            rope_start = 1'b0;
            if (!rope_done) begin
                $display("TB_FAIL,rope_done_not_asserted,pos=%0d", pos);
                $finish;
            end
            vout = rope_head_out;
        end
    endtask

    // ---- task: write one head's K or V vector into kv_bank at (layer=0,pos) -------
    task do_kv_write(input integer kv, input integer head_i, input integer pos,
                      input reg [HEAD_DIM*32-1:0] vec);
        integer b, l;
        begin
            kb_wlayer = 4'd0; kb_wkv = kv[0]; kb_whead = head_i[1:0]; kb_wpos = pos[8:0];
            kb_wstart = 1'b1;
            @(posedge clk); #1;
            kb_wstart = 1'b0;
            for (b = 0; b < HR; b = b + 1) begin
                for (l = 0; l < P; l = l + 1) kb_wdata[l*32 +: 32] = vec[(b*P+l)*32 +: 32];
                kb_wvalid = 1'b1;
                @(posedge clk); #1;
            end
            kb_wvalid = 1'b0;
            while (!kb_wdone) @(posedge clk);
            #1;
        end
    endtask

    // ---- task: full attention call for one head: feed Q, stream K then V from
    // kv_bank, drain ctx --------------------------------------------------------
    task do_attn(input integer head_i, input integer tcount, input reg [HEAD_DIM*32-1:0] qvec,
                 output reg [HEAD_DIM*32-1:0] ctxvec);
        integer b, l;
        begin
            at_tcount = tcount[8:0];
            at_start = 1'b1;
            @(posedge clk); #1;
            at_start = 1'b0;
            // feed Q: NGRP=HR beats of P Q.16 lanes
            for (b = 0; b < HR; b = b + 1) begin
                for (l = 0; l < P; l = l + 1) at_qdata[l*32 +: 32] = qvec[(b*P+l)*32 +: 32];
                at_qvalid = 1'b1;
                @(posedge clk); #1;
            end
            at_qvalid = 1'b0;

            // start kv_bank's K read; its own valid/data feed vec_attn_w directly
            kb_rlayer = 4'd0; kb_rkv = 1'b0; kb_rhead = head_i[1:0]; kb_rtcount = tcount[8:0];
            kb_rstart = 1'b1;
            @(posedge clk); #1;
            kb_rstart = 1'b0;

            while (!at_kdone) @(posedge clk);

            // now stream V the same way
            kb_rlayer = 4'd0; kb_rkv = 1'b1; kb_rhead = head_i[1:0]; kb_rtcount = tcount[8:0];
            kb_rstart = 1'b1;
            @(posedge clk); #1;
            kb_rstart = 1'b0;

            // NBA-safe polling: check DUT outputs only AFTER letting the clocked
            // always block's nonblocking assignments settle (@(posedge clk); #1;),
            // never immediately after the edge -- reading at_done/at_ctxvalid
            // right at the edge (no #1) sees the PRE-update (stale, one-cycle-old)
            // values, which made the very first attempt at this loop silently
            // exit one cycle early and drop the LAST ctx_valid/done beat (eg=8,
            // dims 32-35) -- caught by this gate's own first run, not assumed.
            begin : ctx_drain
                reg done_seen;
                done_seen = 1'b0;
                while (!done_seen) begin
                    @(posedge clk); #1;
                    if (at_ctxvalid) begin
                        for (l = 0; l < P; l = l + 1)
                            ctxvec[(at_ctxidx*P + l)*32 +: 32] = at_ctxdata[l*32 +: 32];
                    end
                    if (at_done) done_seen = 1'b1;
                end
            end
        end
    endtask

    initial begin
        $readmemh("qkv_in.mem", qkv_in);
        $readmemh("ctx_ref.mem", ctx_ref);

        rope_start = 1'b0; rope_pos = 0; rope_head_in = 0;
        kb_wstart = 1'b0; kb_wvalid = 1'b0; kb_rstart = 1'b0;
        kb_wlayer = 0; kb_wkv = 0; kb_whead = 0; kb_wpos = 0; kb_wdata = 0;
        kb_rlayer = 0; kb_rkv = 0; kb_rhead = 0; kb_rtcount = 0;
        at_start = 1'b0; at_tcount = 0; at_qvalid = 1'b0; at_qdata = 0;

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk); #1;

        mismatches = 0; checked = 0;
        f = $fopen("run_log.txt", "w");

        for (step = 0; step < N_STEPS; step = step + 1) begin
            for (head = 0; head < NHEAD; head = head + 1) begin
                base = (step * NHEAD + head) * HEAD_DIM * 3;
                for (i = 0; i < HEAD_DIM; i = i + 1) begin
                    q_h[i*32 +: 32] = qkv_in[base + i];
                    k_h[i*32 +: 32] = qkv_in[base + HEAD_DIM + i];
                    v_h[i*32 +: 32] = qkv_in[base + 2*HEAD_DIM + i];
                end

                do_rope(step, q_h, q_rope_h);
                do_rope(step, k_h, k_rope_h);

                do_kv_write(0, head, step, k_rope_h);   // K, RoPE'd
                do_kv_write(1, head, step, v_h);        // V, never RoPE'd (matches ops.c)

                do_attn(head, step + 1, q_rope_h, ctx_got);

                base = (step * NHEAD + head) * HEAD_DIM;
                for (i = 0; i < HEAD_DIM; i = i + 1) begin
                    checked = checked + 1;
                    if (ctx_got[i*32 +: 32] !== ctx_ref[base + i]) begin
                        mismatches = mismatches + 1;
                        if (mismatches <= 10)
                            $fwrite(f, "MISMATCH step=%0d head=%0d dim=%0d got=%0d ref=%0d\n",
                                    step, head, i, $signed(ctx_got[i*32 +: 32]), $signed(ctx_ref[base + i]));
                    end
                end
                $display("TB_STEP,step=%0d,head=%0d,done", step, head);
            end
        end
        $fclose(f);

        $display("TB_DONE,checked=%0d,mismatches=%0d", checked, mismatches);
        $display("DECODER_SELF_ATTN_VERDICT,bitexact=%0d,mismatches=%0d,checked=%0d",
                  (mismatches == 0), mismatches, checked);
        $finish;
    end

    initial begin
        #10_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
