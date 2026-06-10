// -----------------------------------------------------------------------------
// kv_bank — the on-chip K/V cache that makes the single-stream sequencer decode
// FAITHFULLY (doc 7 R1). Quantise-at-write, dequantise-at-read-stream, K4/V4 +
// per-head Hadamard, per-(head, position) asymmetric — the PINNED goformer_kvq
// rotate=True divfree contract (log §36 fit-plan item 1: halves the code bank):
//
//   write (one head's K or V vector, HEAD_DIM Q.16 ints, streamed HR rows wide):
//     x     = rsh_round(butterfly(x_raw), 3)        6-stage Sylvester Hadamard,
//                                                   then >>3 round-half-away-0
//     lo    = min(x),  span = max(x) - lo           over the ROTATED values
//     scale = max( rdiv(span, QMAX), 1 )            rdiv = round-half-up (a+b/2)/b
//             = ((span+7)*0x88888889)>>35 at QMAX=15 (EXACT magic, proven
//               exhaustively over the full uint32 domain)
//     inv   = rdiv(1<<INV_SH, scale)                one serial divide per head*pos
//             (scale reaches rdiv(2^21,15) ~ 2^17-class — an inv ROM is
//              infeasible at 4 bits, so the serial restoring divider returns)
//     code  = clip( (u*inv + HALF) >> INV_SH, QMAX) u = x - lo  (all non-negative)
//   read  (stream tcount positions of one head, 1 position/cycle after prefetch):
//     x_hat = code * scale16 + lo                   exact integer, kv_dma verbatim
//             (hdr stores scale & 0xFFFF, the position_ddr_row sc16 field — real
//              spans keep scale < 65,536 on this model, max observed ~36k)
//
// q is rotated the SAME way inside vec_attn_w (so q.k is preserved) and the ctx
// output is un-rotated there after the V sum (H is symmetric: unrotate==rotate).
//
// Storage (KBITS=4, TMAX=256, 4 layers, 4 heads): codes 256 KB + hdr 48 KB.
// Banks are row-addressed wide words (CLAUDE.md banking rule): the position is a
// memory ADDRESS, never a per-lane mux. Banks are NOT cleared between tokens or
// conversations — a conversation restarts at pos 0 and overwrites; attention at
// pos p reads only rows 0..p, all freshly written (the same argument that lets
// the bench skip KV resets).
//
// iverilog-2012 safe: variable part-selects only on plain regs (vecbuf, code_r),
// unpacked arrays read by element into a reg first.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_bank #(
    parameter integer P        = 8,
    parameter integer HEAD_DIM = 64,
    parameter integer NHEAD    = 4,
    parameter integer NLAYER   = 4,
    parameter integer TMAX     = 256,
    parameter integer KBITS    = 4,
    parameter integer INV_SH   = 24
) (
    input  wire        clk,
    input  wire        rst,

    // ---- quantise-write port: one head's K or V vector per wq_start ----------
    input  wire        wq_start,            // pulse; selectors sampled here
    input  wire [3:0]  wq_layer,
    input  wire        wq_kv,               // 0 = K, 1 = V
    input  wire [1:0]  wq_head,
    input  wire [8:0]  wq_pos,
    input  wire        wq_valid,            // HR beats of P Q.16 lanes follow
    input  wire [P*32-1:0] wq_data,
    output reg         wq_done,             // pulses when codes + hdr are stored

    // ---- dequant read-stream port A: tcount positions of one head ------------
    input  wire        rd_start,            // pulse; selectors sampled here
    input  wire [3:0]  rd_layer,
    input  wire        rd_kv,
    input  wire [1:0]  rd_head,
    input  wire [8:0]  rd_tcount,           // positions to stream (1..TMAX)
    output reg         rd_valid,            // a dequantised wide row is on rd_data
    output reg  [HEAD_DIM*32-1:0] rd_data,  // ONE POSITION's head row, dequantised
    output reg         rd_done,             // pulses after the last position

    // ---- dequant read-stream port B (doc-7 R4e): the URAM/BRAM second port ----
    input  wire        rd2_start,
    input  wire [3:0]  rd2_layer,
    input  wire        rd2_kv,
    input  wire [1:0]  rd2_head,
    input  wire [8:0]  rd2_tcount,
    output reg         rd2_valid,
    output reg  [HEAD_DIM*32-1:0] rd2_data,
    output reg         rd2_done
);
    localparam integer HR    = HEAD_DIM / P;
    localparam integer QMAX  = (1 << KBITS) - 1;
    localparam integer NHSEL = NLAYER * 2 * NHEAD;          // (layer,kv,head) combos
    localparam integer HROWS = NHSEL * TMAX;                // one row per (sel, pos)
    localparam integer DIVW  = INV_SH + 2;                  // divider width (26):
                                                            // num = 2^24 + scale/2 < 2^25

    // the scale magic below is /15-specific: this bank is the K4/V4 contract.
    initial if (KBITS != 4) begin
        $display("kv_bank: KBITS must be 4 (the /15 scale magic is pinned)");
        $finish;
    end

    // ---- banks (doc-7 R2: one POSITION per code row — KBITS=4 makes a head's
    // position exactly HEAD_DIM*4 = 256 bits, read 1 position/cycle).
    // DUAL-DIALECT (the weight_bank_tdp pattern): HDL inference of TDP UltraRAM
    // is dead in 2025.2, so the SYNTHESIS branch is xpm_memory_tdpram and the
    // sim branch a behavioral 2-port array. Gates verify sim; the board run is
    // the final bit-exactness check on the XPM side.
    wire [HEAD_DIM*KBITS-1:0] code_r,  code_r2;
    wire [47:0]               hdr_rd,  hdr_rd2;

    // ---- write-side state ------------------------------------------------------
    reg [3:0]  w_layer; reg w_kv; reg [1:0] w_head; reg [8:0] w_pos;
    reg [HEAD_DIM*32-1:0] vecbuf;            // plain reg: the head vector, collected
    reg signed [31:0] minv, maxv;
    reg [$clog2(HR+1)-1:0] w_vi;             // collect beat counter
    reg [$clog2(HR+1)-1:0] w_qi;             // round / quantise row counter
    reg signed [31:0] w_lo;
    reg [15:0] w_scale;
    reg [INV_SH:0] w_inv;                    // inv <= 2^INV_SH

    // ---- integer Hadamard: TWO butterfly stages per cycle over vecbuf ----------
    // (Sylvester order h = 1,2 | 4,8 | 16,32 — exact integer adds, no rounding
    // mid-stage, so the 2-per-cycle grouping is bit-identical to the reference's
    // 6 sequential stages). |x_raw| < 2^21 on this model -> raw < 2^27: int32 ok.
    reg [1:0] had_i;                          // 0..2 (stage-pair counter)
    reg [HEAD_DIM*32-1:0] had_t, had_n;
    integer hb, hh1, hh2, hp;
    reg signed [31:0] hx, hy;
    always @* begin
        hh1 = 1 << (had_i * 2);
        hh2 = 2 << (had_i * 2);
        for (hb = 0; hb < HEAD_DIM; hb = hb + 1) begin
            hp = hb ^ hh1;
            hx = $signed(vecbuf[hb*32 +: 32]);
            hy = $signed(vecbuf[hp*32 +: 32]);
            had_t[hb*32 +: 32] = ((hb & hh1) == 0) ? (hx + hy) : (hy - hx);
        end
        for (hb = 0; hb < HEAD_DIM; hb = hb + 1) begin
            hp = hb ^ hh2;
            hx = $signed(had_t[hb*32 +: 32]);
            hy = $signed(had_t[hp*32 +: 32]);
            had_n[hb*32 +: 32] = ((hb & hh2) == 0) ? (hx + hy) : (hy - hx);
        end
    end

    // ---- per-row >>3 round (half-away-from-zero) + row min/max ------------------
    // One vecbuf row per W_RND cycle: round in place, fold min/max of the ROTATED
    // ROUNDED values (the quantiser input — reference order: rotate, round, minmax).
    reg [P*32-1:0] rnd_row;
    reg signed [31:0] rn_v, rn_r, row_min, row_max;
    integer rp;
    always @* begin
        row_min = 32'sh7FFFFFFF; row_max = 32'sh80000000;
        for (rp = 0; rp < P; rp = rp + 1) begin
            rn_v = $signed(vecbuf[(w_qi*P + rp)*32 +: 32]);
            if (rn_v >= 0) rn_r = (rn_v + 32'sd4) >>> 3;
            else           rn_r = -((-rn_v + 32'sd4) >>> 3);
            rnd_row[rp*32 +: 32] = rn_r;
            if (rn_r < row_min) row_min = rn_r;
            if (rn_r > row_max) row_max = rn_r;
        end
    end

    // ---- scale = rdiv(span, 15) by EXACT magic multiply --------------------------
    // ((span+7) * 0x88888889) >> 35 == floor((span+7)/15), proven EXHAUSTIVELY over
    // the full uint32 domain in python (run_vec_kv campaign). Then the python sc16
    // truncation: hdr scale = scale & 0xFFFF (real spans keep scale < 2^16).
    reg [25:0] w_span;                       // span+7, explicitly unsigned (< 2^26)
    reg [57:0] w_magic;                      // w_span * 0x88888889 (26b x 32b, unsigned)
    wire [22:0] sc_full = w_magic[57:35];
    wire [15:0] sc_q    = (sc_full == 23'd0) ? 16'd1 : sc_full[15:0];

    // ---- serial unsigned restoring divider: inv = rdiv(2^24, scale) --------------
    // (restored from the pre-R4a kv_bank: at 4 bits scale spans up to ~2^17-class
    // values so the split inv ROM no longer fits; ~DIVW cycles per head*pos)
    reg [DIVW-1:0] dv_num, dv_quo;
    reg [DIVW:0]   dv_rem;
    reg [DIVW-1:0] dv_den;
    reg [$clog2(DIVW+1)-1:0] dv_i;
    // one divider step per cycle: rem' = {rem, num[msb]}; sub if >= den
    wire [DIVW:0] dv_shift = {dv_rem[DIVW-1:0], dv_num[DIVW-1]};
    wire          dv_ge    = (dv_shift >= {1'b0, dv_den});

    // quantise P lanes of vecbuf row w_qi (combinational; all operands registered)
    reg [P*KBITS-1:0] q_codes;
    reg signed [32:0] q_diff;
    reg [31:0]  q_u;
    reg [31+INV_SH+1:0] q_prod;              // u(<=2^21-class) * inv(<=2^24) + rounding
    reg [31:0]  q_code;
    integer qp;
    always @* begin
        q_codes = {(P*KBITS){1'b0}};
        for (qp = 0; qp < P; qp = qp + 1) begin
            q_diff = $signed(vecbuf[(w_qi*P + qp)*32 +: 32]) - w_lo;   // >= 0 by construction
            q_u    = q_diff[31:0];
            q_prod = q_u * w_inv + (1 << (INV_SH-1));
            q_code = q_prod >> INV_SH;
            if (q_code > QMAX) q_code = QMAX;
            q_codes[qp*KBITS +: KBITS] = q_code[KBITS-1:0];
        end
    end

    // ---- read-side state -------------------------------------------------------
    // ZERO-BUBBLE stream, one POSITION per cycle: the hdr read is pipelined
    // ALONGSIDE the code-row read (separate memories, same 1-cycle latency).
    // rowi walks positions 0..T-1. Total = T + 2 cycles per stream.
    reg [8:0]  r_rowi,  r2_rowi;               // position counters (0..TMAX)
    reg [8:0]  r_nrows, r2_nrows;              // T
    reg [8:0]  r_ecnt,  r2_ecnt;               // emitted-position counters
    reg        r_v0,    r2_v0;                 // addr-stage valids
    reg        r_v1,    r2_v1;                 // mem-out-register stage valids

    // TIMING (take-5 worst path): the hdr/code BRAM-out registers fed the 64-lane
    // dequant COMBINATIONALLY into the engines' DSP inputs (7-high BRAM cascade +
    // mult = -2.4ns @5ns). One register stage between the memory outputs and the
    // dequant splits the path; the stream grows by one cycle.
    reg [HEAD_DIM*KBITS-1:0] code_q, code_q2;
    reg [47:0]               hdr_q,  hdr_q2;
    always @(posedge clk) begin
        code_q  <= code_r;   hdr_q  <= hdr_rd;
        code_q2 <= code_r2;  hdr_q2 <= hdr_rd2;
    end

    // base addresses (registered at start; code and hdr share the per-position base)
    reg [$clog2(HROWS)-1:0] w_pbase, r_pbase, r2_pbase;
    reg [HEAD_DIM*KBITS-1:0] wstage;           // staged code row (P codes per W_QNT cycle)

    // dequant HEAD_DIM lanes of code_r with the CO-READ header (aligned):
    //   x_hat = code * scale + lo   (exact integer, kv_dma verbatim)
    reg [HEAD_DIM*32-1:0] deq_word, deq_word2;
    reg [KBITS-1:0] d_code;
    reg [KBITS+16-1:0] d_prod;
    reg signed [33:0] d_full;
    integer dp;
    always @* begin
        deq_word = {(HEAD_DIM*32){1'b0}};
        for (dp = 0; dp < HEAD_DIM; dp = dp + 1) begin
            d_code = code_q[dp*KBITS +: KBITS];
            d_prod = d_code * hdr_q[47:32];                  // unsigned mul (kv_dma shape)
            d_full = $signed({1'b0, d_prod}) + $signed(hdr_q[31:0]); // + signed lo
            deq_word[dp*32 +: 32] = d_full[31:0];
        end
    end
    reg [KBITS-1:0] d2_code;
    reg [KBITS+16-1:0] d2_prod;
    reg signed [33:0] d2_full;
    integer dq2;
    always @* begin
        deq_word2 = {(HEAD_DIM*32){1'b0}};
        for (dq2 = 0; dq2 < HEAD_DIM; dq2 = dq2 + 1) begin
            d2_code = code_q2[dq2*KBITS +: KBITS];
            d2_prod = d2_code * hdr_q2[47:32];
            d2_full = $signed({1'b0, d2_prod}) + $signed(hdr_q2[31:0]);
            deq_word2[dq2*32 +: 32] = d2_full[31:0];
        end
    end

    // ---- write FSM ---------------------------------------------------------------
    localparam [3:0] W_IDLE=4'd0, W_COLL=4'd1, W_HAD=4'd2, W_RND=4'd3,
                     W_SCALE=4'd4, W_MAG=4'd5, W_DIV=4'd6, W_DIVF=4'd7,
                     W_QNT=4'd8, W_CWR=4'd9;
    reg [3:0] wst;

    // ---- read FSMs (A and B) -------------------------------------------------
    localparam [1:0] R_IDLE=2'd0, R_RUN=2'd2;
    reg [1:0] rst_st, rst2_st;

    // sync reads: code row + hdr row for position rowi — co-read every cycle.
    // Stream B reads through the memories' SECOND port (URAM/BRAM TDP).
    // PORT DISCIPLINE: the quant-write (W_CWR) is FOLDED INTO port A — it commits
    // only when stream A is idle (W_CWR stalls on rst_st, see the write FSM), so
    // each memory is exactly write-else-read on A plus read on B = two ports.
    wire [$clog2(HROWS)-1:0] pos_ra  = r_pbase  + {{($clog2(HROWS)-9){1'b0}}, r_rowi};
    wire [$clog2(HROWS)-1:0] pos_ra2 = r2_pbase + {{($clog2(HROWS)-9){1'b0}}, r2_rowi};
    wire cwr_fire = (wst == W_CWR) && (rst_st == R_IDLE);
    // ONE ADDRESS NET PER PORT (the URAM contract): write and read share port A's
    // muxed address; port B is the second read stream.
    wire [$clog2(HROWS)-1:0] kva_a = cwr_fire ? w_pbase : pos_ra;
`ifdef SYNTHESIS
    xpm_memory_tdpram #(
        .ADDR_WIDTH_A($clog2(HROWS)), .ADDR_WIDTH_B($clog2(HROWS)),
        .BYTE_WRITE_WIDTH_A(HEAD_DIM*KBITS), .BYTE_WRITE_WIDTH_B(HEAD_DIM*KBITS),
        .CLOCKING_MODE("common_clock"), .MEMORY_PRIMITIVE("ultra"),
        .MEMORY_SIZE(HEAD_DIM*KBITS*HROWS),
        .READ_DATA_WIDTH_A(HEAD_DIM*KBITS), .READ_DATA_WIDTH_B(HEAD_DIM*KBITS),
        .READ_LATENCY_A(1), .READ_LATENCY_B(1),
        .WRITE_DATA_WIDTH_A(HEAD_DIM*KBITS), .WRITE_DATA_WIDTH_B(HEAD_DIM*KBITS),
        .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
    ) u_code (
        .clka(clk), .clkb(clk), .rsta(1'b0), .rstb(1'b0), .ena(1'b1), .enb(1'b1),
        .regcea(1'b1), .regceb(1'b1), .wea(cwr_fire), .web(1'b0),
        .addra(kva_a), .addrb(pos_ra2),
        .dina(wstage), .dinb({(HEAD_DIM*KBITS){1'b0}}),
        .douta(code_r), .doutb(code_r2),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb(), .sleep(1'b0));
    xpm_memory_tdpram #(
        .ADDR_WIDTH_A($clog2(HROWS)), .ADDR_WIDTH_B($clog2(HROWS)),
        .BYTE_WRITE_WIDTH_A(48), .BYTE_WRITE_WIDTH_B(48),
        .CLOCKING_MODE("common_clock"), .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(48*HROWS),
        .READ_DATA_WIDTH_A(48), .READ_DATA_WIDTH_B(48),
        .READ_LATENCY_A(1), .READ_LATENCY_B(1),
        .WRITE_DATA_WIDTH_A(48), .WRITE_DATA_WIDTH_B(48),
        .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
    ) u_hdr (
        .clka(clk), .clkb(clk), .rsta(1'b0), .rstb(1'b0), .ena(1'b1), .enb(1'b1),
        .regcea(1'b1), .regceb(1'b1), .wea(cwr_fire), .web(1'b0),
        .addra(kva_a), .addrb(pos_ra2),
        .dina({w_scale, w_lo}), .dinb(48'b0),
        .douta(hdr_rd), .doutb(hdr_rd2),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb(), .sleep(1'b0));
`else
    reg [HEAD_DIM*KBITS-1:0] code_bank [0:HROWS-1];
    reg [47:0]               hdr_bank  [0:HROWS-1];      // {scale16, lo32}
    reg [HEAD_DIM*KBITS-1:0] code_r_b,  code_r2_b;
    reg [47:0]               hdr_rd_b,  hdr_rd2_b;
    always @(posedge clk) begin
        if (cwr_fire) code_bank[kva_a] <= wstage;
        code_r_b  <= code_bank[kva_a];
        code_r2_b <= code_bank[pos_ra2];                  // port B: read
    end
    always @(posedge clk) begin
        if (cwr_fire) hdr_bank[kva_a] <= {w_scale, w_lo};
        hdr_rd_b  <= hdr_bank[kva_a];
        hdr_rd2_b <= hdr_bank[pos_ra2];
    end
    assign code_r  = code_r_b;
    assign code_r2 = code_r2_b;
    assign hdr_rd  = hdr_rd_b;
    assign hdr_rd2 = hdr_rd2_b;
`endif

    always @(posedge clk) begin
        wq_done <= 1'b0; rd_done <= 1'b0; rd2_done <= 1'b0;
        if (rst) begin
            wst <= W_IDLE; rst_st <= R_IDLE; rst2_st <= R_IDLE;
            rd_valid <= 1'b0; r_v0 <= 1'b0; r_v1 <= 1'b0;
            rd2_valid <= 1'b0; r2_v0 <= 1'b0; r2_v1 <= 1'b0;
        end else begin
            // =================== write side ===================
            case (wst)
                W_IDLE: if (wq_start) begin
                    w_layer <= wq_layer; w_kv <= wq_kv; w_head <= wq_head; w_pos <= wq_pos;
                    w_pbase <= ((wq_layer*2 + {3'b0,wq_kv})*NHEAD + {2'b0,wq_head})*TMAX
                                + {3'b0,wq_pos};
                    minv <= 32'sh7FFFFFFF; maxv <= 32'sh80000000;
                    w_vi <= 0; wst <= W_COLL;
                end
                W_COLL: if (wq_valid) begin
                    vecbuf[w_vi*P*32 +: P*32] <= wq_data;
                    if (w_vi == HR-1) begin
                        had_i <= 2'd0; wst <= W_HAD;
                    end else w_vi <= w_vi + 1'b1;
                end
                // 6-stage Hadamard butterfly, 2 stages/cycle (exact integer adds)
                W_HAD: begin
                    vecbuf <= had_n;
                    if (had_i == 2'd2) begin
                        w_qi <= 0; wst <= W_RND;
                    end else had_i <= had_i + 2'd1;
                end
                // >>3 round (half-away-from-zero) per row + min/max of the result
                W_RND: begin
                    vecbuf[w_qi*P*32 +: P*32] <= rnd_row;
                    if (row_min < minv) minv <= row_min;
                    if (row_max > maxv) maxv <= row_max;
                    if (w_qi == HR-1) wst <= W_SCALE;
                    else w_qi <= w_qi + 1'b1;
                end
                W_SCALE: begin
                    w_lo   <= minv;
                    w_span <= (maxv - minv + (QMAX >> 1));   // span+7, non-negative
                    wst <= W_MAG;
                end
                W_MAG: begin
                    // scale = rdiv(span,15) = ((span+7)*0x88888889)>>35 (EXACT;
                    // registered step, all-unsigned, no mixed-sign multiply)
                    w_magic <= {32'd0, w_span} * 58'd2290649225;
                    wst <= W_DIV;
                end
                W_DIV: begin
                    // hdr scale16 (= python sc16 = scale & 0xFFFF) and the divider
                    // setup: inv = rdiv(1<<INV_SH, scale), num = 2^24 + scale>>1
                    w_scale <= sc_q;
                    dv_num  <= (1 << INV_SH) + ({10'b0, sc_q} >> 1);
                    dv_den  <= {{(DIVW-16){1'b0}}, sc_q};
                    dv_rem  <= 0; dv_quo <= 0; dv_i <= DIVW[$clog2(DIVW+1)-1:0];
                    wst <= W_DIVF;
                end
                W_DIVF: begin
                    if (dv_i != 0) begin
                        dv_rem <= dv_ge ? (dv_shift - {1'b0, dv_den}) : dv_shift;
                        dv_quo <= {dv_quo[DIVW-2:0], dv_ge};
                        dv_num <= {dv_num[DIVW-2:0], 1'b0};
                        dv_i   <= dv_i - 1'b1;
                    end else begin
                        w_inv <= dv_quo[INV_SH:0];
                        w_qi  <= 0;
                        wst   <= W_QNT;
                    end
                end
                W_QNT: begin
                    // stage P codes/cycle into the position row, commit once whole
                    wstage[w_qi*P*KBITS +: P*KBITS] <= q_codes;
                    if (w_qi == HR-1) wst <= W_CWR;
                    else w_qi <= w_qi + 1'b1;
                end
                W_CWR: if (cwr_fire) begin
                    // commit happens in the port-A always block this same cycle
                    // (cwr_fire); stalls while stream A is mid-read (the feeder's
                    // writes land in the softmax gaps between K and V streams).
                    wq_done <= 1'b1;
                    wst <= W_IDLE;
                end
                default: wst <= W_IDLE;
            endcase

            // =================== read side (stream A) ===================
            rd_valid <= 1'b0;
            case (rst_st)
                R_IDLE: if (rd_start) begin
                    r_nrows <= rd_tcount;
                    r_rowi  <= 0;
                    r_ecnt  <= 0;
                    r_v0    <= 1'b0;
                    r_v1    <= 1'b0;
                    r_pbase <= ((rd_layer*2 + {3'b0,rd_kv})*NHEAD
                                 + {2'b0,rd_head})*TMAX;
                    rst_st <= R_RUN;
                end
                R_RUN: begin
                    // addr stage: present row rowi (code + its position's hdr together)
                    r_v0 <= (r_rowi != r_nrows);
                    r_v1 <= r_v0;
                    if (r_rowi != r_nrows) r_rowi <= r_rowi + 1'b1;
                    // emit stage: code_q/hdr_q (mem-out + one reg) land now
                    if (r_v1) begin
                        rd_valid <= 1'b1;
                        rd_data  <= deq_word;
                        if (r_ecnt == r_nrows - 1) begin
                            rd_done <= 1'b1;
                            rst_st  <= R_IDLE;
                        end else r_ecnt <= r_ecnt + 1'b1;
                    end
                end
                default: rst_st <= R_IDLE;
            endcase

            // =================== read side (stream B, port B) ===================
            rd2_valid <= 1'b0;
            case (rst2_st)
                R_IDLE: if (rd2_start) begin
                    r2_nrows <= rd2_tcount;
                    r2_rowi  <= 0;
                    r2_ecnt  <= 0;
                    r2_v0    <= 1'b0;
                    r2_v1    <= 1'b0;
                    r2_pbase <= ((rd2_layer*2 + {3'b0,rd2_kv})*NHEAD
                                  + {2'b0,rd2_head})*TMAX;
                    rst2_st <= R_RUN;
                end
                R_RUN: begin
                    r2_v0 <= (r2_rowi != r2_nrows);
                    r2_v1 <= r2_v0;
                    if (r2_rowi != r2_nrows) r2_rowi <= r2_rowi + 1'b1;
                    if (r2_v1) begin
                        rd2_valid <= 1'b1;
                        rd2_data  <= deq_word2;
                        if (r2_ecnt == r2_nrows - 1) begin
                            rd2_done <= 1'b1;
                            rst2_st  <= R_IDLE;
                        end else r2_ecnt <= r2_ecnt + 1'b1;
                    end
                end
                default: rst2_st <= R_IDLE;
            endcase
        end
    end
endmodule
