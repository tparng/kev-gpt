// -----------------------------------------------------------------------------
// kv_bank — the on-chip K/V cache that makes the single-stream sequencer decode
// FAITHFULLY (doc 7 R1). Quantise-at-write, dequantise-at-read-stream, K8/V8,
// per-(head, position) asymmetric — the PINNED goformer_kvq divfree contract:
//
//   write (one head's K or V vector, HEAD_DIM Q.16 ints, streamed HR rows wide):
//     lo    = min(x),  span = max(x) - lo
//     scale = max( rdiv(span, QMAX), 1 )            rdiv = round-half-up (a+b/2)/b
//     inv   = rdiv(1<<INV_SH, scale)                one serial divide per head*pos
//     code  = clip( (u*inv + HALF) >> INV_SH, QMAX) u = x - lo  (all non-negative)
//   read  (stream tcount positions of one head, HR rows/cycle after prefetch):
//     x_hat = code * scale + lo                      exact integer, kv_dma verbatim
//
// Storage (KBITS=8, TMAX=256, 4 layers, 4 heads): codes 512 KB + hdr 48 KB.
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
    parameter integer KBITS    = 8,
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

    // ---- dequant read-stream port: tcount positions of one head --------------
    input  wire        rd_start,            // pulse; selectors sampled here
    input  wire [3:0]  rd_layer,
    input  wire        rd_kv,
    input  wire [1:0]  rd_head,
    input  wire [8:0]  rd_tcount,           // positions to stream (1..TMAX)
    output reg         rd_valid,            // a dequantised wide row is on rd_data
    output reg  [HEAD_DIM*32-1:0] rd_data,  // ONE POSITION's head row, dequantised
    output reg         rd_done              // pulses after the last position
);
    localparam integer HR    = HEAD_DIM / P;
    localparam integer QMAX  = (1 << KBITS) - 1;
    localparam integer NHSEL = NLAYER * 2 * NHEAD;          // (layer,kv,head) combos
    localparam integer HROWS = NHSEL * TMAX;                // one row per (sel, pos)
    localparam integer DIVW  = INV_SH + 2;                  // divider width (26)

    // ---- banks (doc-7 R2: one POSITION per code row — KBITS=8 makes a head's
    // position exactly HEAD_DIM*8 = 512 bits, read 1 position/cycle) -----------
    (* ram_style = "ultra" *) reg [HEAD_DIM*KBITS-1:0] code_bank [0:HROWS-1];
    (* ram_style = "block" *) reg [47:0]               hdr_bank  [0:HROWS-1]; // {scale16, lo32}

    // sync-read registers
    reg [HEAD_DIM*KBITS-1:0] code_r;
    reg [47:0]               hdr_rd;

    // ---- write-side state ------------------------------------------------------
    reg [3:0]  w_layer; reg w_kv; reg [1:0] w_head; reg [8:0] w_pos;
    reg [HEAD_DIM*32-1:0] vecbuf;            // plain reg: the head vector, collected
    reg signed [31:0] minv, maxv;
    reg [$clog2(HR+1)-1:0] w_vi;             // collect beat counter
    reg [$clog2(HR+1)-1:0] w_qi;             // quantise row counter
    reg signed [31:0] w_lo;
    reg [15:0] w_scale;
    reg [INV_SH:0] w_inv;                    // inv <= 2^INV_SH

    // doc-7 R4a: NO divider. scale = rdiv(span,255) by the EXACT magic multiply
    // ((span+127)*0x80808081)>>39, proven == floor((span+127)/255) over the full
    // span range [0, 2^22) by exhaustive python check; inv = rdiv(2^24, scale)
    // from a constant ROM (one q_round_div per entry, data-independent, written
    // by the harness as inv_lut.mem). Same numbers as the serial divides, ~52
    // cycles fewer per (head, pos).
    localparam integer INVD = 16512;         // scale <= rdiv(2^22, 255) = 16,449
    (* rom_style = "block" *) reg [INV_SH:0] inv_lut [0:INVD-1];
    initial $readmemh("inv_lut.mem", inv_lut);
    reg [22:0] w_span;                       // span+127, explicitly unsigned
    reg [53:0] w_magic;                      // w_span * 0x80808081 (23b x 32b, unsigned)
    reg [INV_SH:0] inv_rd;
    // sync ROM read: address is the combinational scale during W_INVL (so inv_rd
    // lands in W_INVR), then the registered w_scale.
    wire [14:0] sc_now = (w_magic[53:39] == 0) ? 15'd1 : w_magic[53:39];
    wire [$clog2(INVD)-1:0] lut_a = (wst == 3'd3) ? {{($clog2(INVD)-15){1'b0}}, sc_now}
                                                  : w_scale[$clog2(INVD)-1:0];
    always @(posedge clk) inv_rd <= inv_lut[lut_a];

    // P-lane signed min/max of the incoming beat (combinational tree, P=8)
    integer mp;
    reg signed [31:0] lane_v, beat_min, beat_max;
    always @* begin
        beat_min = 32'sh7FFFFFFF; beat_max = 32'sh80000000;
        for (mp = 0; mp < P; mp = mp + 1) begin
            lane_v = $signed(wq_data[mp*32 +: 32]);
            if (lane_v < beat_min) beat_min = lane_v;
            if (lane_v > beat_max) beat_max = lane_v;
        end
    end

    // quantise P lanes of vecbuf row w_qi (combinational; all operands registered)
    reg [P*KBITS-1:0] q_codes;
    reg signed [32:0] q_diff;
    reg [31:0]  q_u;
    reg [31+INV_SH+1:0] q_prod;              // u(<=2^21) * inv(<=2^24) + rounding
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
    reg [8:0]  r_rowi;                         // position counter (0..TMAX)
    reg [8:0]  r_nrows;                        // T
    reg [8:0]  r_ecnt;                         // emitted-position counter
    reg        r_v0;                           // addr-stage valid (data lands next cyc)

    // base addresses (registered at start; code and hdr share the per-position base)
    reg [$clog2(HROWS)-1:0] w_pbase, r_pbase;
    reg [HEAD_DIM*KBITS-1:0] wstage;           // staged code row (P codes per W_QNT cycle)

    // dequant HEAD_DIM lanes of code_r with the CO-READ header (aligned):
    //   x_hat = code * scale + lo   (exact integer, kv_dma verbatim)
    reg [HEAD_DIM*32-1:0] deq_word;
    reg [KBITS-1:0] d_code;
    reg [KBITS+16-1:0] d_prod;
    reg signed [33:0] d_full;
    integer dp;
    always @* begin
        deq_word = {(HEAD_DIM*32){1'b0}};
        for (dp = 0; dp < HEAD_DIM; dp = dp + 1) begin
            d_code = code_r[dp*KBITS +: KBITS];
            d_prod = d_code * hdr_rd[47:32];                  // unsigned mul (kv_dma shape)
            d_full = $signed({1'b0, d_prod}) + $signed(hdr_rd[31:0]); // + signed lo
            deq_word[dp*32 +: 32] = d_full[31:0];
        end
    end

    // ---- write FSM ---------------------------------------------------------------
    localparam [2:0] W_IDLE=3'd0, W_COLL=3'd1, W_SCALE=3'd2, W_INVL=3'd3,
                     W_INVR=3'd4, W_SCALE2=3'd5, W_QNT=3'd6, W_CWR=3'd7;
    reg [2:0] wst;

    // ---- read FSM ----------------------------------------------------------------
    localparam [1:0] R_IDLE=2'd0, R_RUN=2'd2;
    reg [1:0] rst_st;

    // sync reads: code row + hdr row for position rowi — co-read every cycle.
    wire [$clog2(HROWS)-1:0] pos_ra = r_pbase + {{($clog2(HROWS)-9){1'b0}}, r_rowi};
    always @(posedge clk) begin
        code_r <= code_bank[pos_ra];
        hdr_rd <= hdr_bank[pos_ra];
    end

    always @(posedge clk) begin
        wq_done <= 1'b0; rd_done <= 1'b0;
        if (rst) begin
            wst <= W_IDLE; rst_st <= R_IDLE;
            rd_valid <= 1'b0; r_v0 <= 1'b0;
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
                    if (beat_min < minv) minv <= beat_min;
                    if (beat_max > maxv) maxv <= beat_max;
                    if (w_vi == HR-1) begin
                        wst <= W_SCALE;
                    end else w_vi <= w_vi + 1'b1;
                end
                W_SCALE: begin
                    // scale = rdiv(span,255) = ((span+127)*0x80808081)>>39 (EXACT;
                    // two registered steps, all-unsigned, no mixed-sign multiply)
                    w_lo   <= minv;
                    w_span <= (maxv - minv + (QMAX >> 1));
                    wst <= W_SCALE2;
                end
                W_SCALE2: begin
                    w_magic <= {31'd0, w_span} * 54'd2155905153;
                    wst <= W_INVL;
                end
                W_INVL: begin
                    w_scale <= {1'b0, sc_now};
                    wst <= W_INVR;                     // ROM read for sc_now lands next cycle
                end
                W_INVR: begin
                    // inv_rd (sync ROM read at w_scale) lands NOW
                    w_inv <= inv_rd;
                    w_qi  <= 0;
                    wst   <= W_QNT;
                end
                W_QNT: begin
                    // stage P codes/cycle into the position row, commit once whole
                    wstage[w_qi*P*KBITS +: P*KBITS] <= q_codes;
                    if (w_qi == HR-1) wst <= W_CWR;
                    else w_qi <= w_qi + 1'b1;
                end
                W_CWR: begin
                    code_bank[w_pbase] <= wstage;
                    hdr_bank [w_pbase] <= {w_scale, w_lo};
                    wq_done <= 1'b1;
                    wst <= W_IDLE;
                end
                default: wst <= W_IDLE;
            endcase

            // =================== read side ===================
            rd_valid <= 1'b0;
            case (rst_st)
                R_IDLE: if (rd_start) begin
                    r_nrows <= rd_tcount;
                    r_rowi  <= 0;
                    r_ecnt  <= 0;
                    r_v0    <= 1'b0;
                    r_pbase <= ((rd_layer*2 + {3'b0,rd_kv})*NHEAD
                                 + {2'b0,rd_head})*TMAX;
                    rst_st <= R_RUN;
                end
                R_RUN: begin
                    // addr stage: present row rowi (code + its position's hdr together)
                    r_v0 <= (r_rowi != r_nrows);
                    if (r_rowi != r_nrows) r_rowi <= r_rowi + 1'b1;
                    // emit stage: code_r/hdr_rd for the previous address land now
                    if (r_v0) begin
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
        end
    end
endmodule
