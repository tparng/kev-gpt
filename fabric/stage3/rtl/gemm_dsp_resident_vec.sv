// -----------------------------------------------------------------------------
// gemm_dsp_resident_vec — gemm_banked_resident_vec with DSP-PACKED MAC banks.
//
// WP487 weight-packing, adapted to INT4 weights x INT8 activations: each DSP
// multiplies TWO weight nibbles (wpak in the wide port at a 24-bit gap, each
// biased +8 to make it non-negative) by ONE INT8 activation:
//
//     wpak  = (w0 + 8)  +  (w1 + 8) << 24            (both in 0..15)
//     product = wpak * act        (signed 18-bit port; product < 2^40)
//     acc(48b)+= product             - up to 4096 accumulations, no overlap
//
// After the run, lane sums are recovered EXACTLY (integers):
//     y0 = acc[23:0] - 8*sum_act     (mod 2^24; |y0| < 2^21 - unambiguous)
//     y1 = (acc >> 24) - carry - 8*sum_act
//          carry = (y0 + 8*sum_act) >> 24
//
// sum_act is per stream per group. 2 MAC/DSP/cycle; LANES nibbles cost
// LANES/2 DSPs per stream. The whole core is bit-exact vs the LUT MAC core
// (same adds in the same order; bias-correction is exact integer algebra).
//
// Boundary, weight loader and ymem layout identical to gemm_banked_resident_vec.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemm_dsp_resident_vec #(
    parameter integer LANES  = 128,       // PE lanes (pow2)
    parameter integer N      = 8,         // batch streams
    parameter integer P      = 8,
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WWORDS = 25600,
    parameter integer RLAT   = 2
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS)-1:0]     w_base,
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [31:0]                   w_data,
    input  wire                          x_rst,
    input  wire                          x_we,
    input  wire [$clog2(N)-1:0]          x_stream,
    input  wire [P*8-1:0]                x_data,
    input  wire                          start,
    output reg                           done,
    input  wire [$clog2(N)-1:0]          rd_stream,
    input  wire [$clog2(MMAX/P)-1:0]     rd_addr,
    output reg  [P*32-1:0]               y_out
);
    localparam integer WBITS  = LANES*4;
    localparam integer YBITS  = LANES*32;
    localparam integer LSH    = $clog2(LANES);
    localparam integer LSHP   = $clog2(P);
    localparam integer GROUPS = (MMAX + LANES - 1) / LANES;
    localparam integer WAW    = $clog2(WWORDS);
    localparam integer XROWS  = KMAX / P;
    localparam integer SUBW   = WBITS / 32;
    localparam integer SSW    = (SUBW > 1) ? $clog2(SUBW) : 1;
    localparam integer BANKW  = (WBITS > 512) ? 72 : WBITS;
    localparam integer NB     = (WBITS + BANKW - 1) / BANKW;
    localparam integer WPAD   = NB * BANKW;

    reg [P*8-1:0]    xmem [0:N*XROWS-1];
    reg [YBITS-1:0]  ymem [0:N*GROUPS-1];

    // ---- one-time load (identical to the LUT core) -----------------------------
    reg [WAW-1:0]    wword;
    reg [SSW-1:0]    wsub;
    reg [WBITS-1:0]  wbuf;
    reg [$clog2(XROWS)-1:0] xptr;
    wire [WBITS-1:0] wnext = wbuf | ({{(WBITS-32){1'b0}}, w_data} << (wsub*32));
    wire [WPAD-1:0]  wnext_pad = {{(WPAD-WBITS){1'b0}}, wnext};
    wire             wcommit = w_we && (wsub == SUBW-1);
    always @(posedge clk) begin
        if (ld_rst) begin wword <= 0; wsub <= 0; wbuf <= {WBITS{1'b0}}; xptr <= 0; end
        else begin
            if (w_we) begin
                if (wsub == SUBW-1) begin
                    wword <= wword + 1'b1; wsub <= 0; wbuf <= {WBITS{1'b0}};
                end else begin
                    wbuf <= wnext; wsub <= wsub + 1'b1;
                end
            end
            if (x_rst) xptr <= 0;
            else if (x_we) begin
                xmem[x_stream*XROWS + xptr] <= x_data;
                xptr <= xptr + 1'b1;
            end
        end
    end

    wire [$clog2(WWORDS)-1:0] waddr;
    wire [WPAD-1:0] wword_pad;
    wire [WBITS-1:0] wword_rd = wword_pad[WBITS-1:0];
    genvar gb;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_w
            (* ram_style = "ultra" *) reg [BANKW-1:0] mem [0:WWORDS-1];
            reg [BANKW-1:0] rd;
            always @(posedge clk) begin
                if (wcommit) mem[wword] <= wnext_pad[gb*BANKW +: BANKW];
                rd <= mem[waddr];
            end
            assign wword_pad[gb*BANKW +: BANKW] = rd;
        end
    endgenerate

    // ---- run FSM ----------------------------------------------------------------
    localparam [2:0] IDLE = 3'd0, RUN = 3'd1, DRAIN = 3'd3, FIN = 3'd2, SETTLE = 3'd4;
    reg [2:0]            state;
    reg [1:0]            settle;
    reg [$clog2(GROUPS):0] g;
    reg [$clog2(KMAX):0] kc, kmac;
    reg [WAW-1:0]        grp_base;
    reg [$clog2(N):0]    db;
    reg                  acc_clr;

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    wire                 issue   = (kc < k_count);
    assign               waddr   = grp_base + kc;

    reg [WBITS-1:0]      word_p [0:RLAT-2];
    reg [P*8-1:0]        xrow_p [0:RLAT-1][0:N-1];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    integer i, b;

    wire                 mac_v = v_p[RLAT-1];
    wire [WBITS-1:0]     wsel = word_p[RLAT-2];

    // ---- N DSP-PACKED MAC banks --------------------------------------------------
    // Each genvar lane pair (gl, gl+1) of stream gm is ONE DSP:
    //   wpak = (w[gl]+8) + (w[gl+1]+8) << 24, product = wpak * act (signed)
    //   acc accumulates the wpak sums; recovery happens at DRAIN with sum_act.
    // one wire vector per stream � a flat N*YBITS concat crosses iverilog's
    // 16-bit part-select index space at stream 4 (bit 16384), silently wrapping
`ifdef SYNTHESIS
    wire [N*YBITS-1:0] acc_cat;          // packed concat � Vivado fails URAM
                                         // inference with unpacked wire arrays
`else
    wire [YBITS-1:0] acc_str [0:N-1];    // iverilog wraps >16k part-selects
`endif
    reg  [YBITS-1:0] y_lat [0:N-1];     // recovered outputs latched at SETTLE end
    reg  [YBITS-1:0] acc_sel;
    genvar gm, gl;
    generate
        for (gm = 0; gm < N; gm = gm + 1) begin : g_mac
            wire [P*8-1:0]    xrow = xrow_p[RLAT-1][gm];
            wire signed [8:0] xsel = $signed(xrow[xl_p[RLAT-1]*8 +: 8]);
            // per-stream sum of activations (for the +8 bias correction)
            reg signed [22:0] sum_act;
            always @(posedge clk) begin
                if (rst || acc_clr) sum_act <= 23'sd0;
                else if (mac_v)     sum_act <= sum_act + xsel;
            end
            for (gl = 0; gl < LANES; gl = gl + 2) begin : g_dsp
                // wpak unsigned weights: (w0+8) lives in bits 3:0, (w1+8) at 27:24
                wire [3:0]  w0 = wsel[gl*4 +: 4]    ^ 4'h8;     // +8 bias
                wire [3:0]  w1 = wsel[(gl+1)*4 +: 4] ^ 4'h8;
                wire [27:0] wpak = {w1, 20'b0, w0};
                reg signed [47:0] acc;
                always @(posedge clk) begin
                    if (rst || acc_clr) acc <= 48'sd0;
                    else if (mac_v)
                        acc <= acc + $signed({1'b0, wpak}) * xsel;
                end
                // recovery: y0 = (acc_lo - 8*sum_act) mod 2^24 (|y0| < 2^21 so the
                // 24-bit window is unambiguous); carry = (y0 + 8*sum_act) >> 24;
                // y1 = acc_hi - 8*sum_act - carry. Exact integer algebra.
                wire signed [25:0] sa8   = {sum_act, 3'b000};               // 8*sum_act
                wire [23:0]        y0m   = acc[23:0] - sa8[23:0];           // mod 2^24
                wire signed [31:0] y0    = {{8{y0m[23]}}, y0m};
                wire signed [27:0] hsum  = y0 + sa8;
                wire signed [3:0]  carry = hsum >>> 24;
                wire signed [31:0] y1    = $signed(acc[47:24]) - sa8 - carry;
`ifdef SYNTHESIS
                assign acc_cat[gm*YBITS + gl*32 +: 32]     = y0;
                assign acc_cat[gm*YBITS + (gl+1)*32 +: 32] = y1;
`else
                assign acc_str[gm][gl*32 +: 32]     = y0;
                assign acc_str[gm][(gl+1)*32 +: 32] = y1;
`endif
            end
        end
    endgenerate

    always @(posedge clk) begin
        word_p[0] <= wword_rd;
        for (b = 0; b < N; b = b + 1)
            xrow_p[0][b] <= xmem[b*XROWS + (kc[$clog2(KMAX)-1:0] >> LSHP)];
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT-1; i = i + 1)
            word_p[i] <= word_p[i-1];
        for (i = 1; i < RLAT; i = i + 1) begin
            for (b = 0; b < N; b = b + 1)
                xrow_p[i][b] <= xrow_p[i-1][b];
            xl_p[i]   <= xl_p[i-1];
            v_p[i]    <= v_p[i-1];
        end

        acc_clr <= 1'b0;
        if (rst) begin
            state <= IDLE; done <= 1'b0;
            g <= 0; kc <= 0; kmac <= 0; grp_base <= 0; db <= 0;
            acc_clr <= 1'b1;
            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        g <= 0; kc <= 0; kmac <= 0;
                        acc_clr <= 1'b1;
                        grp_base <= w_base;
                        for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + 1'b1;
                    if (mac_v) kmac <= kmac + 1'b1;
                    // settle: sum_act/acc must be FULLY drained before recovery —
                    // streams drained late would otherwise see trailing MAC terms
                    if (kmac == k_count) begin db <= 0; settle <= 0; state <= SETTLE; end
                end
                SETTLE: begin
                    settle <= settle + 1'b1;
                    if (settle == 2'd3) begin
                        for (b = 0; b < N; b = b + 1)
`ifdef SYNTHESIS
                            y_lat[b] <= acc_cat[b*YBITS +: YBITS];
`else
                            y_lat[b] <= acc_str[b];
`endif
                        state <= DRAIN;
                    end
                end
                DRAIN: begin
                    // reg-array runtime index is safe; recovery was latched at SETTLE
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= y_lat[db];
                    if (db == N-1) begin
                        acc_clr <= 1'b1;
                        if (g == gcount - 1) state <= FIN;
                        else begin
                            g <= g + 1'b1; kc <= 0; kmac <= 0;
                            grp_base <= grp_base + k_count;
                            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                            state <= RUN;
                        end
                    end else db <= db + 1'b1;
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    // ---- readback ---------------------------------------------------------------
    localparam integer PPG = LANES / P;
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    always @(posedge clk) begin
        rd_word <= ymem[rd_stream*GROUPS + rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_out   <= rd_word[rd_off*(P*32) +: P*32];
    end
endmodule
