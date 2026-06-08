// -----------------------------------------------------------------------------
// gemm_cohort_vec — ONE split-brain cohort's GEMM datapath: the run FSM + N MAC
// banks + per-stream act memories + fused readback, EXACTLY as in
// gemm_banked_resident_vec, but with the resident URAM weight banks HOISTED OUT
// to a shared weight_bank_tdp. This module drives a weight read ADDRESS (waddr)
// and consumes the registered weight word (wword_rd, 1-cycle latency = the old
// per-bank read reg = pipeline stage 0). Two cohorts each instance this, reading
// the shared TDP banks through independent ports — the split-brain core.
//
// Bit-identical arithmetic to gemm_banked_resident_vec (the MAC/drain/readback
// code is copied verbatim); only the weight storage moved out. The MAC banks
// here are this cohort's slice (8 of the 16), so N is the per-cohort stream
// count (=8) and the mac_bank/mac_bank_dsp leaves are shared with the original
// core's definitions (this file does NOT redefine them).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemm_cohort_vec #(
    parameter integer LANES  = 128,
    parameter integer N      = 8,         // streams in THIS cohort
    parameter integer ND     = 0,         // of which DSP-packed (streams N-ND..N-1)
    parameter integer P      = 8,
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WWORDS = 25600,
    parameter integer RLAT   = 2,
    // DOUBLE-PUMP-100K Stage 1: DP=1 runs the RUN MAC at 2 K-steps/clk via
    // mac_bank_dp. Identical mechanism to gemm_banked_resident_vec.DP; the only
    // difference is the weights are external — the shared weight_bank_tdp serves
    // the phase-1 word (wword1_rd, addr waddr+1) alongside wword_rd. DP=1 is
    // ND=0-only until the DSP leaf is double-pumped (the --nd 6 step).
    parameter integer DP     = 0,
    parameter integer ABITS  = 24
) (
    input  wire                          clk,
    input  wire                          clk2x,   // 2x clk, 0-deg aligned (DP=1 only)
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS)-1:0]     w_base,
    // per-call activation
    input  wire                          x_rst,
    input  wire                          x_we,
    input  wire [$clog2(N)-1:0]          x_stream,
    input  wire [$clog2(KMAX/P)-1:0]     x_row,      // EXPLICIT write row (row-major AQ)
    input  wire [P*8-1:0]                x_data,
    // DOUBLE-PUMP-100K: 2nd activation write port so the AQ can write TWO streams
    // /clk (a pair). Each per-stream xm is a separate array, so two writes to two
    // DIFFERENT streams are independent ports (the AQ never writes the same stream
    // twice in a clk). Tie x_we2=0 when unused.
    input  wire                          x_we2,
    input  wire [$clog2(N)-1:0]          x_stream2,
    input  wire [$clog2(KMAX/P)-1:0]     x_row2,
    input  wire [P*8-1:0]                x_data2,
    // overlap (AQ hiding behind RUN): when ovl_en, the RUN's MAC `issue` stalls
    // until the x row it needs is committed. x_rowcommit pulses once per fully
    // written row (all streams) and bumps rows_committed; `issue` is gated on
    // xrd_row < rows_committed so an under-run inserts MAC bubbles — bit-identical
    // accumulation, only delayed. x_rst clears rows_committed; ovl_en=0 -> classic.
    input  wire                          ovl_en,
    input  wire                          x_rowcommit,
    // run
    input  wire                          start,
    output reg                           done,
    // readback
    input  wire [$clog2(N)-1:0]          rd_stream,
    input  wire [$clog2(MMAX/P)-1:0]     rd_addr,
    output wire [P*32-1:0]               y_out,
    // DOUBLE-PUMP-100K: 2nd readback row (rd_addr+1) so GE_RB can dequant 2 rows
    // /clk. rd_addr is the pair base (even); rd_addr+1 shares the same group word
    // (PPG even), so it's a free 2nd P-group select off the SAME ymem read.
    output wire [P*32-1:0]               y_out2,
    // ---- shared weight bank read interface (replaces the in-core URAM) ----
    output wire [$clog2(WWORDS)-1:0]     waddr,        // read addr to weight_bank_tdp
    input  wire [LANES*4-1:0]            wword_rd,     // registered word (1-cyc latency)
    input  wire [LANES*4-1:0]            wword1_rd     // DP: word at waddr+1 (phase 1)
);
    localparam integer WBITS  = LANES*4;
    localparam integer YBITS  = LANES*ABITS;
    localparam integer LSH    = $clog2(LANES);
    localparam integer LSHP   = $clog2(P);
    localparam integer GROUPS = (MMAX + LANES - 1) / LANES;
    localparam integer WAW    = $clog2(WWORDS);
    localparam integer XROWS  = KMAX / P;

    (* ram_style = "distributed" *)
    reg [YBITS-1:0]  ymem [0:N*GROUPS-1];
    (* ram_style = "distributed" *)
    reg signed [22:0] sumact_mem [0:N*GROUPS-1];

    // ---- committed-row counter for the AQ/RUN overlap stall guard ------------
    // rows_committed = number of x rows (all streams) fully written and readable.
    // Bumped one cycle AFTER x_rowcommit so the xm[] write that produced the row
    // is already latched (write@T -> readable@T+1; commit registered @T+1 -> the
    // guarded read at T+2 is always safe). x_rst clears it.
    reg [$clog2(XROWS+1)-1:0] rows_committed;
    always @(posedge clk) begin
        if (x_rst) rows_committed <= 0;
        else if (x_rowcommit) rows_committed <= rows_committed + 1'b1;
    end

    // ---- run FSM + RLAT-deep read/mac pipeline -------------------------------
    localparam [2:0] IDLE = 3'd0, RUN = 3'd1, DRAIN = 3'd3, FIN = 3'd2, SETTLE = 3'd4;
    reg [2:0]            state;
    reg [1:0]            settle;
    reg [$clog2(GROUPS):0] g;
    reg [$clog2(KMAX):0] kc, kmac;
    reg [WAW-1:0]        grp_base;
    reg [$clog2(N):0]    db;
    reg                  acc_clr;

    localparam integer   STEP    = (DP != 0) ? 2 : 1;   // K-steps/clk
    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    // stall the RUN's K-iteration when the x row it would read isn't committed yet
    // (only under overlap; otherwise all rows are present before `start`).
    wire                 row_ready = !ovl_en || (xrd_row < rows_committed);
    wire                 issue   = (kc < k_count) && row_ready;       // phase-0 valid
    wire                 issue1  = (DP != 0) && ((kc + 1) < k_count) && row_ready;
    assign               waddr   = grp_base + kc;
    wire [WBITS-1:0]     wword_word  = wword_rd;    // 1-cycle registered (stage 0)
    wire [WBITS-1:0]     wword1_word = wword1_rd;   // DP: phase-1 word (addr waddr+1)

    reg [WBITS-1:0]      word_p  [0:RLAT-2];
    reg [WBITS-1:0]      word1_p [0:RLAT-2];   // DP: second weight word (phase 1)
    reg [LSHP-1:0]       xl_p    [0:RLAT-1];
    reg                  v_p     [0:RLAT-1];
    reg                  v1_p    [0:RLAT-1];   // DP: phase-1 valid (odd-K tail mask)
    integer i, b;
    wire [$clog2(XROWS)-1:0] xrd_row = kc[$clog2(KMAX)-1:0] >> LSHP;

    wire                 mac_v  = v_p[RLAT-1];
    wire                 mac_v1 = v1_p[RLAT-1];

    wire [WBITS-1:0] wsel  = word_p[RLAT-2];
    wire [WBITS-1:0] wsel1 = word1_p[RLAT-2];
`ifdef SYNTHESIS
    wire [4*YBITS-1:0] acc_q0, acc_q1, acc_q2, acc_q3;
    reg  [YBITS-1:0] sel_q0, sel_q1, sel_q2, sel_q3;
    wire [4:0] dbw = db;
`else
    wire [YBITS-1:0] acc_str [0:N-1];
    reg  [YBITS-1:0] y_lat [0:N-1];
`endif
    reg  [YBITS-1:0] acc_sel;
    wire signed [22:0] sa_str [0:N-1];
    reg  signed [22:0] sa_sel;
    genvar gm, gl;
    generate
        for (gm = 0; gm < N; gm = gm + 1) begin : g_mac
            reg [P*8-1:0] xm [0:XROWS-1];
            reg [P*8-1:0] xr [0:RLAT-1];
            integer xi;
            always @(posedge clk) begin
                if (x_we  && x_stream  == gm[$clog2(N)-1:0]) xm[x_row]  <= x_data;
                if (x_we2 && x_stream2 == gm[$clog2(N)-1:0]) xm[x_row2] <= x_data2;
                xr[0] <= xm[xrd_row];
                for (xi = 1; xi < RLAT; xi = xi + 1) xr[xi] <= xr[xi-1];
            end
            wire [P*8-1:0]    xrow = xr[RLAT-1];
            wire [LSHP-1:0]   xl0  = xl_p[RLAT-1];               // base (even) lane
            wire signed [7:0] xsel = xrow[xl0*8 +: 8];          // phase-0 act (and DP=0/DSP)
            // DP phase-1 act = next lane (xl0+1 in-row since xl0 even <= P-2);
            // masked to 0 on the odd-K tail so the phase-1 accumulate no-ops.
            wire signed [7:0] x1raw = xrow[(xl0 + 1'b1)*8 +: 8];
            wire signed [7:0] xsel1 = mac_v1 ? x1raw : 8'sd0;
            wire [YBITS-1:0] acc_bank;
            if (gm < N - ND) begin : g_lut
                if (DP != 0) begin : g_dpump
                    mac_bank_dp #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                        .clk(clk), .clk2x(clk2x), .clr(rst || acc_clr), .en(mac_v),
                        .w0(wsel), .w1(wsel1), .x0(xsel), .x1(xsel1), .acc(acc_bank));
                end else begin : g_single
                    mac_bank #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                        .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                        .w(wsel), .x(xsel), .acc(acc_bank));
                end
                assign sa_str[gm] = 23'sd0;
            end else begin : g_dsp
                if (DP != 0) begin : g_dsp_dp
                    mac_bank_dsp_dp #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                        .clk(clk), .clk2x(clk2x), .clr(rst || acc_clr), .en(mac_v),
                        .w0(wsel), .w1(wsel1), .x0(xsel), .x1(xsel1),
                        .acc(acc_bank), .sum_act_o(sa_str[gm]));
                end else begin : g_dsp_sp
                    mac_bank_dsp #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                        .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                        .w(wsel), .x(xsel), .acc(acc_bank),
                        .sum_act_o(sa_str[gm]));
                end
            end
`ifdef SYNTHESIS
            if (gm/4 == 0) begin : g_q0
                assign acc_q0[(gm%4)*YBITS +: YBITS] = acc_bank;
            end else if (gm/4 == 1) begin : g_q1
                assign acc_q1[(gm%4)*YBITS +: YBITS] = acc_bank;
            end else if (gm/4 == 2) begin : g_q2
                assign acc_q2[(gm%4)*YBITS +: YBITS] = acc_bank;
            end else begin : g_q3
                assign acc_q3[(gm%4)*YBITS +: YBITS] = acc_bank;
            end
`else
            assign acc_str[gm] = acc_bank;
`endif
        end
    endgenerate

    always @(posedge clk) begin
        word_p[0]  <= wword_word;
        word1_p[0] <= wword1_word;                  // DP (dead/trimmed when DP=0)
        xl_p[0]    <= kc[LSHP-1:0];
        v_p[0]     <= (state == RUN) && issue;
        v1_p[0]    <= (state == RUN) && issue1;
        for (i = 1; i < RLAT-1; i = i + 1) begin
            word_p[i]  <= word_p[i-1];
            word1_p[i] <= word1_p[i-1];
        end
        for (i = 1; i < RLAT; i = i + 1) begin
            xl_p[i]   <= xl_p[i-1];
            v_p[i]    <= v_p[i-1];
            v1_p[i]   <= v1_p[i-1];
        end

        acc_clr <= 1'b0;
        if (rst) begin
            state <= IDLE; done <= 1'b0;
            g <= 0; kc <= 0; kmac <= 0; grp_base <= 0; db <= 0;
            acc_clr <= 1'b1;
            for (i = 0; i < RLAT; i = i + 1) begin v_p[i] <= 1'b0; v1_p[i] <= 1'b0; end
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        g <= 0; kc <= 0; kmac <= 0;
                        acc_clr <= 1'b1;
                        grp_base <= w_base;
                        for (i = 0; i < RLAT; i = i + 1) begin v_p[i] <= 1'b0; v1_p[i] <= 1'b0; end
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + STEP;
                    // phase 0 (+1) and, under DP, phase 1 (+1) unless masked on
                    // the odd-K tail -> +2 or +1.
                    if (mac_v) kmac <= kmac + (mac_v1 ? 2'd2 : 2'd1);
`ifdef SYNTHESIS
                    if (kmac == k_count) begin db <= 0; state <= DRAIN; end
`else
                    if (kmac == k_count) begin db <= 0; settle <= 0; state <= SETTLE; end
`endif
                end
`ifndef SYNTHESIS
                SETTLE: begin
                    settle <= settle + 1'b1;
                    if (settle == 2'd3) begin
                        for (b = 0; b < N; b = b + 1)
                            y_lat[b] <= acc_str[b];
                        state <= DRAIN;
                    end
                end
`endif
                DRAIN: begin
`ifdef SYNTHESIS
                    case (db[1:0])
                        2'd0: sel_q0 = acc_q0[0*YBITS +: YBITS];
                        2'd1: sel_q0 = acc_q0[(N > 1 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q0 = acc_q0[(N > 2 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q0 = acc_q0[(N > 3 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_q1 = acc_q1[0*YBITS +: YBITS];
                        2'd1: sel_q1 = acc_q1[(N > 5 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q1 = acc_q1[(N > 6 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q1 = acc_q1[(N > 7 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_q2 = acc_q2[0*YBITS +: YBITS];
                        2'd1: sel_q2 = acc_q2[(N > 9 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q2 = acc_q2[(N > 10 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q2 = acc_q2[(N > 11 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_q3 = acc_q3[0*YBITS +: YBITS];
                        2'd1: sel_q3 = acc_q3[(N > 13 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q3 = acc_q3[(N > 14 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q3 = acc_q3[(N > 15 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (dbw[3:2])
                        2'd0: acc_sel = sel_q0;
                        2'd1: acc_sel = (N > 4)  ? sel_q1 : sel_q0;
                        2'd2: acc_sel = (N > 8)  ? sel_q2 : sel_q0;
                        default: acc_sel = (N > 12) ? sel_q3 : sel_q0;
                    endcase
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= acc_sel;
                    sa_sel = sa_str[db];
                    sumact_mem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= sa_sel;
`else
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= y_lat[db];
                    sa_sel = sa_str[db];
                    sumact_mem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= sa_sel;
`endif
                    if (db == N-1) begin
                        acc_clr <= 1'b1;
                        if (g == gcount - 1) state <= FIN;
                        else begin
                            g <= g + 1'b1; kc <= 0; kmac <= 0;
                            grp_base <= grp_base + k_count;
                            for (i = 0; i < RLAT; i = i + 1) begin v_p[i] <= 1'b0; v1_p[i] <= 1'b0; end
                            state <= RUN;
                        end
                    end else db <= db + 1'b1;
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    // ---- readback (identical to gemm_banked_resident_vec) --------------------
    localparam integer PPG = LANES / P;
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    reg [P*ABITS-1:0]      y_raw;
    reg [P*ABITS-1:0]      y_raw2;       // DOUBLE-PUMP: 2nd P-group (rd_off+1), same word
    reg signed [22:0]      sa_rd, sa_pipe;
    reg                    dsp0, dsp1;
    always @(posedge clk) begin
        rd_word <= ymem[rd_stream*GROUPS + rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_raw   <= rd_word[rd_off*(P*ABITS) +: P*ABITS];
        // 2nd row of the pair: next P-group of the SAME group word (rd_off even,
        // PPG even -> rd_off+1 is in-word; same sum_act/group, same 2-cyc latency).
        y_raw2  <= rd_word[(rd_off + 1'b1)*(P*ABITS) +: P*ABITS];
        sa_rd   <= sumact_mem[rd_stream*GROUPS + rd_addr / PPG];
        sa_pipe <= sa_rd;
        dsp0    <= (rd_stream >= (N - ND));
        dsp1    <= dsp0;
    end
    wire signed [25:0] sa8 = {sa_pipe, 3'b000};
    genvar gy, gp;
    generate
        wire [P*32-1:0] y_lut, y_lut2;
        for (gy = 0; gy < P; gy = gy + 1) begin : g_yext
            assign y_lut[gy*32 +: 32] =
                {{(32-ABITS){y_raw[gy*ABITS + ABITS-1]}}, y_raw[gy*ABITS +: ABITS]};
            assign y_lut2[gy*32 +: 32] =
                {{(32-ABITS){y_raw2[gy*ABITS + ABITS-1]}}, y_raw2[gy*ABITS +: ABITS]};
        end
        wire [P*32-1:0] y_dsp, y_dsp2;
        for (gp = 0; gp < P/2; gp = gp + 1) begin : g_rec
            wire [47:0]        a     = y_raw[gp*48 +: 48];
            wire [21:0]        y0m   = a[21:0] - sa8[21:0];
            wire signed [31:0] y0    = {{10{y0m[21]}}, y0m};
            wire signed [27:0] hsum  = y0 + sa8;
            wire signed [5:0]  carry = hsum >>> 22;
            wire signed [31:0] y1    = $signed(a[47:22]) - sa8 - carry;
            assign y_dsp[(2*gp)  *32 +: 32] = y0;
            assign y_dsp[(2*gp+1)*32 +: 32] = y1;
            // 2nd P-group recovery (same shared sum_act)
            wire [47:0]        a_2    = y_raw2[gp*48 +: 48];
            wire [21:0]        y0m_2  = a_2[21:0] - sa8[21:0];
            wire signed [31:0] y0_2   = {{10{y0m_2[21]}}, y0m_2};
            wire signed [27:0] hsum_2 = y0_2 + sa8;
            wire signed [5:0]  carry_2= hsum_2 >>> 22;
            wire signed [31:0] y1_2   = $signed(a_2[47:22]) - sa8 - carry_2;
            assign y_dsp2[(2*gp)  *32 +: 32] = y0_2;
            assign y_dsp2[(2*gp+1)*32 +: 32] = y1_2;
        end
        assign y_out  = dsp1 ? y_dsp  : y_lut;
        assign y_out2 = dsp1 ? y_dsp2 : y_lut2;
    endgenerate
endmodule
