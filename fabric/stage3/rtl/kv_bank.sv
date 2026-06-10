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
    output reg  [P*32-1:0] rd_data,         // P sign-extended Q.16 lanes
    output reg         rd_done              // pulses after the last row
);
    localparam integer HR    = HEAD_DIM / P;
    localparam integer QMAX  = (1 << KBITS) - 1;
    localparam integer NHSEL = NLAYER * 2 * NHEAD;          // (layer,kv,head) combos
    localparam integer CROWS = NHSEL * TMAX * HR;           // code rows
    localparam integer HROWS = NHSEL * TMAX;                // hdr rows
    localparam integer DIVW  = INV_SH + 2;                  // divider width (26)

    // ---- banks ---------------------------------------------------------------
    (* ram_style = "ultra" *) reg [P*KBITS-1:0] code_bank [0:CROWS-1];
    (* ram_style = "block" *) reg [47:0]        hdr_bank  [0:HROWS-1]; // {scale16, lo32}

    // sync-read registers
    reg [P*KBITS-1:0] code_r;
    reg [47:0]        hdr_rd;

    // ---- write-side state ------------------------------------------------------
    reg [3:0]  w_layer; reg w_kv; reg [1:0] w_head; reg [8:0] w_pos;
    reg [HEAD_DIM*32-1:0] vecbuf;            // plain reg: the head vector, collected
    reg signed [31:0] minv, maxv;
    reg [$clog2(HR+1)-1:0] w_vi;             // collect beat counter
    reg [$clog2(HR+1)-1:0] w_qi;             // quantise row counter
    reg signed [31:0] w_lo;
    reg [15:0] w_scale;
    reg [INV_SH:0] w_inv;                    // inv <= 2^INV_SH

    // serial unsigned restoring divider (shared by the two write-side divides)
    reg [DIVW-1:0] dv_num, dv_quo;
    reg [DIVW:0]   dv_rem;
    reg [DIVW-1:0] dv_den;
    reg [$clog2(DIVW+1)-1:0] dv_i;
    reg            dv_run;
    // one divider step per cycle: rem' = {rem, num[msb]}; sub if >= den
    wire [DIVW:0] dv_shift = {dv_rem[DIVW-1:0], dv_num[DIVW-1]};
    wire          dv_ge    = (dv_shift >= {1'b0, dv_den});

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
    // ZERO-BUBBLE stream: the hdr read is pipelined ALONGSIDE the code read every
    // cycle (separate memories, same 1-cycle latency), so the per-position header
    // needs no latching FSM and no prologue. rowi walks 0..T*HR-1; the position is
    // rowi >> log2(HR) for the hdr address. Total = T*HR + 2 cycles per stream.
    reg [13:0] r_rowi;                         // up to TMAX*HR = 2048
    reg [13:0] r_nrows;                        // T*HR
    reg [13:0] r_ecnt;                         // emitted-row counter
    reg        r_v0;                           // addr-stage valid (data lands next cyc)

    // base addresses (registered at start)
    reg [$clog2(CROWS)-1:0] w_cbase, r_cbase;
    reg [$clog2(HROWS)-1:0] w_hbase, r_hbase;

    // dequant P lanes of code_r with the CO-READ header hdr_rd (aligned with code_r):
    //   x_hat = code * scale + lo   (exact integer, kv_dma verbatim)
    reg [P*32-1:0] deq_word;
    reg [KBITS-1:0] d_code;
    reg [KBITS+16-1:0] d_prod;
    reg signed [33:0] d_full;
    integer dp;
    always @* begin
        deq_word = {(P*32){1'b0}};
        for (dp = 0; dp < P; dp = dp + 1) begin
            d_code = code_r[dp*KBITS +: KBITS];
            d_prod = d_code * hdr_rd[47:32];                  // unsigned mul (kv_dma shape)
            d_full = $signed({1'b0, d_prod}) + $signed(hdr_rd[31:0]); // + signed lo
            deq_word[dp*32 +: 32] = d_full[31:0];
        end
    end

    // ---- write FSM ---------------------------------------------------------------
    localparam [2:0] W_IDLE=3'd0, W_COLL=3'd1, W_DIV1=3'd2, W_DIV1F=3'd3,
                     W_DIV2=3'd4, W_DIV2F=3'd5, W_QNT=3'd6;
    reg [2:0] wst;

    // ---- read FSM ----------------------------------------------------------------
    localparam [1:0] R_IDLE=2'd0, R_RUN=2'd2;
    reg [1:0] rst_st;

    // sync reads: code row rowi, hdr row = rowi's position — co-read every cycle.
    // NB zero-EXTEND r_rowi (never part-select wider than the reg — the X trap).
    wire [$clog2(CROWS)-1:0] code_ra = r_cbase + {{($clog2(CROWS)-14){1'b0}}, r_rowi};
    wire [$clog2(HROWS)-1:0] hdr_ra  = r_hbase
                                       + {{($clog2(HROWS)-11){1'b0}}, r_rowi[13:$clog2(HR)]};
    always @(posedge clk) begin
        code_r <= code_bank[code_ra];
        hdr_rd <= hdr_bank[hdr_ra];
    end

    always @(posedge clk) begin
        wq_done <= 1'b0; rd_done <= 1'b0;
        if (rst) begin
            wst <= W_IDLE; rst_st <= R_IDLE; dv_run <= 1'b0;
            rd_valid <= 1'b0; r_v0 <= 1'b0;
        end else begin
            // =================== write side ===================
            case (wst)
                W_IDLE: if (wq_start) begin
                    w_layer <= wq_layer; w_kv <= wq_kv; w_head <= wq_head; w_pos <= wq_pos;
                    w_cbase <= (((wq_layer*2 + {3'b0,wq_kv})*NHEAD + {2'b0,wq_head})*TMAX
                                + {3'b0,wq_pos})*HR;
                    w_hbase <= ((wq_layer*2 + {3'b0,wq_kv})*NHEAD + {2'b0,wq_head})*TMAX
                                + {3'b0,wq_pos};
                    minv <= 32'sh7FFFFFFF; maxv <= 32'sh80000000;
                    w_vi <= 0; wst <= W_COLL;
                end
                W_COLL: if (wq_valid) begin
                    vecbuf[w_vi*P*32 +: P*32] <= wq_data;
                    if (beat_min < minv) minv <= beat_min;
                    if (beat_max > maxv) maxv <= beat_max;
                    if (w_vi == HR-1) begin
                        wst <= W_DIV1;
                    end else w_vi <= w_vi + 1'b1;
                end
                W_DIV1: begin
                    // scale = rdiv(span, QMAX): num = span + QMAX>>1, den = QMAX
                    w_lo   <= minv;
                    dv_num <= (maxv - minv + (QMAX >> 1));
                    dv_den <= QMAX[DIVW-1:0];
                    dv_rem <= 0; dv_quo <= 0; dv_i <= DIVW[$clog2(DIVW+1)-1:0];
                    wst <= W_DIV1F;
                end
                W_DIV1F: begin
                    if (dv_i != 0) begin
                        dv_rem <= dv_ge ? (dv_shift - {1'b0, dv_den}) : dv_shift;
                        dv_quo <= {dv_quo[DIVW-2:0], dv_ge};
                        dv_num <= {dv_num[DIVW-2:0], 1'b0};
                        dv_i   <= dv_i - 1'b1;
                    end else begin
                        w_scale <= (dv_quo == 0) ? 16'd1 : dv_quo[15:0];
                        wst <= W_DIV2;
                    end
                end
                W_DIV2: begin
                    // inv = rdiv(1<<INV_SH, scale): num = (1<<INV_SH) + scale>>1
                    dv_num <= (1 << INV_SH) + ({10'b0, w_scale} >> 1);
                    dv_den <= {{(DIVW-16){1'b0}}, w_scale};
                    dv_rem <= 0; dv_quo <= 0; dv_i <= DIVW[$clog2(DIVW+1)-1:0];
                    wst <= W_DIV2F;
                end
                W_DIV2F: begin
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
                    code_bank[w_cbase + {{($clog2(CROWS)-$clog2(HR+1)){1'b0}}, w_qi}] <= q_codes;
                    if (w_qi == HR-1) begin
                        hdr_bank[w_hbase] <= {w_scale, w_lo};
                        wq_done <= 1'b1;
                        wst <= W_IDLE;
                    end else w_qi <= w_qi + 1'b1;
                end
                default: wst <= W_IDLE;
            endcase

            // =================== read side ===================
            rd_valid <= 1'b0;
            case (rst_st)
                R_IDLE: if (rd_start) begin
                    r_nrows <= rd_tcount * HR;
                    r_rowi  <= 0;
                    r_ecnt  <= 0;
                    r_v0    <= 1'b0;
                    r_cbase <= (((rd_layer*2 + {3'b0,rd_kv})*NHEAD
                                 + {2'b0,rd_head})*TMAX)*HR;
                    r_hbase <= ((rd_layer*2 + {3'b0,rd_kv})*NHEAD
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
