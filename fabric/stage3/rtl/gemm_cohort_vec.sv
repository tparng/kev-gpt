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
    parameter integer ABITS  = 24
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS)-1:0]     w_base,
    // per-call activation
    input  wire                          x_rst,
    input  wire                          x_we,
    input  wire [$clog2(N)-1:0]          x_stream,
    input  wire [P*8-1:0]                x_data,
    // run
    input  wire                          start,
    output reg                           done,
    // readback
    input  wire [$clog2(N)-1:0]          rd_stream,
    input  wire [$clog2(MMAX/P)-1:0]     rd_addr,
    output wire [P*32-1:0]               y_out,
    // ---- shared weight bank read interface (replaces the in-core URAM) ----
    output wire [$clog2(WWORDS)-1:0]     waddr,        // read addr to weight_bank_tdp
    input  wire [LANES*4-1:0]            wword_rd      // registered word (1-cyc latency)
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

    // ---- act-write row pointer (the load assembler moved to weight_bank_tdp) ----
    reg [$clog2(XROWS)-1:0] xptr;
    always @(posedge clk) begin
        if (x_rst) xptr <= 0;
        else if (x_we) xptr <= xptr + 1'b1;
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

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    wire                 issue   = (kc < k_count);
    assign               waddr   = grp_base + kc;
    wire [WBITS-1:0]     wword_word = wword_rd;   // 1-cycle registered (stage 0)

    reg [WBITS-1:0]      word_p [0:RLAT-2];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    integer i, b;
    wire [$clog2(XROWS)-1:0] xrd_row = kc[$clog2(KMAX)-1:0] >> LSHP;

    wire                 mac_v = v_p[RLAT-1];

    wire [WBITS-1:0] wsel = word_p[RLAT-2];
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
                if (x_we && x_stream == gm[$clog2(N)-1:0]) xm[xptr] <= x_data;
                xr[0] <= xm[xrd_row];
                for (xi = 1; xi < RLAT; xi = xi + 1) xr[xi] <= xr[xi-1];
            end
            wire [P*8-1:0]    xrow = xr[RLAT-1];
            wire signed [7:0] xsel = xrow[xl_p[RLAT-1]*8 +: 8];
            wire [YBITS-1:0] acc_bank;
            if (gm < N - ND) begin : g_lut
                mac_bank #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                    .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                    .w(wsel), .x(xsel), .acc(acc_bank));
                assign sa_str[gm] = 23'sd0;
            end else begin : g_dsp
                mac_bank_dsp #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                    .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                    .w(wsel), .x(xsel), .acc(acc_bank),
                    .sum_act_o(sa_str[gm]));
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
        word_p[0] <= wword_word;
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT-1; i = i + 1)
            word_p[i] <= word_p[i-1];
        for (i = 1; i < RLAT; i = i + 1) begin
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

    // ---- readback (identical to gemm_banked_resident_vec) --------------------
    localparam integer PPG = LANES / P;
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    reg [P*ABITS-1:0]      y_raw;
    reg signed [22:0]      sa_rd, sa_pipe;
    reg                    dsp0, dsp1;
    always @(posedge clk) begin
        rd_word <= ymem[rd_stream*GROUPS + rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_raw   <= rd_word[rd_off*(P*ABITS) +: P*ABITS];
        sa_rd   <= sumact_mem[rd_stream*GROUPS + rd_addr / PPG];
        sa_pipe <= sa_rd;
        dsp0    <= (rd_stream >= (N - ND));
        dsp1    <= dsp0;
    end
    wire signed [25:0] sa8 = {sa_pipe, 3'b000};
    genvar gy, gp;
    generate
        wire [P*32-1:0] y_lut;
        for (gy = 0; gy < P; gy = gy + 1) begin : g_yext
            assign y_lut[gy*32 +: 32] =
                {{(32-ABITS){y_raw[gy*ABITS + ABITS-1]}}, y_raw[gy*ABITS +: ABITS]};
        end
        wire [P*32-1:0] y_dsp;
        for (gp = 0; gp < P/2; gp = gp + 1) begin : g_rec
            wire [47:0]        a     = y_raw[gp*48 +: 48];
            wire [21:0]        y0m   = a[21:0] - sa8[21:0];
            wire signed [31:0] y0    = {{10{y0m[21]}}, y0m};
            wire signed [27:0] hsum  = y0 + sa8;
            wire signed [5:0]  carry = hsum >>> 22;
            wire signed [31:0] y1    = $signed(a[47:22]) - sa8 - carry;
            assign y_dsp[(2*gp)  *32 +: 32] = y0;
            assign y_dsp[(2*gp+1)*32 +: 32] = y1;
        end
        assign y_out = dsp1 ? y_dsp : y_lut;
    endgenerate
endmodule
