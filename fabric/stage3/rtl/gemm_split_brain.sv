// -----------------------------------------------------------------------------
// gemm_split_brain — two independent gemm_cohort_vec datapaths sharing ONE
// dual-ported (TDP) resident weight image (weight_bank_tdp). Cohort 0 reads the
// weights on port B, cohort 1 on port A; the loader writes port A at boot. The
// two cohorts have fully independent run/act/readback interfaces — they may be
// on different calls/layers at any moment (that is the whole point: no shared
// bandwidth, no merge). MAC banks are physically partitioned 8+8 (N per cohort).
//
// This is the GEMM-level split-brain; the sequencer wraps it with one nl_engine
// + GE FSM per cohort. Bit-identical per-cohort to the single-engine core.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemm_split_brain #(
    parameter integer LANES  = 128,
    parameter integer N      = 8,         // streams PER COHORT
    parameter integer ND     = 0,         // DSP-packed streams per cohort
    parameter integer P      = 8,
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WWORDS = 25600,
    parameter integer RLAT   = 2,
    parameter integer ABITS  = 24
) (
    input  wire                          clk,
    input  wire                          rst,
    // ---- one-time shared weight load ----
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [31:0]                   w_data,
    // ---- cohort 0 ----
    input  wire [$clog2(MMAX+1)-1:0]     c0_m_count,
    input  wire [$clog2(KMAX+1)-1:0]     c0_k_count,
    input  wire [$clog2(WWORDS)-1:0]     c0_w_base,
    input  wire                          c0_x_rst,
    input  wire                          c0_x_we,
    input  wire [$clog2(N)-1:0]          c0_x_stream,
    input  wire [$clog2(KMAX/P)-1:0]     c0_x_row,
    input  wire [P*8-1:0]                c0_x_data,
    input  wire                          c0_start,
    output wire                          c0_done,
    input  wire [$clog2(N)-1:0]          c0_rd_stream,
    input  wire [$clog2(MMAX/P)-1:0]     c0_rd_addr,
    output wire [P*32-1:0]               c0_y_out,
    // ---- cohort 1 ----
    input  wire [$clog2(MMAX+1)-1:0]     c1_m_count,
    input  wire [$clog2(KMAX+1)-1:0]     c1_k_count,
    input  wire [$clog2(WWORDS)-1:0]     c1_w_base,
    input  wire                          c1_x_rst,
    input  wire                          c1_x_we,
    input  wire [$clog2(N)-1:0]          c1_x_stream,
    input  wire [$clog2(KMAX/P)-1:0]     c1_x_row,
    input  wire [P*8-1:0]                c1_x_data,
    input  wire                          c1_start,
    output wire                          c1_done,
    input  wire [$clog2(N)-1:0]          c1_rd_stream,
    input  wire [$clog2(MMAX/P)-1:0]     c1_rd_addr,
    output wire [P*32-1:0]               c1_y_out
);
    localparam integer WBITS = LANES*4;

    wire [$clog2(WWORDS)-1:0] waddr_b, waddr_a;
    wire [WBITS-1:0]          wword_b, wword_a;

    weight_bank_tdp #(.LANES(LANES), .WWORDS(WWORDS)) u_wbank (
        .clk(clk),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data),
        .raddr_b(waddr_b), .rword_b(wword_b),
        .raddr_a(waddr_a), .rword_a(wword_a));

    gemm_cohort_vec #(.LANES(LANES), .N(N), .ND(ND), .P(P), .MMAX(MMAX),
                      .KMAX(KMAX), .WWORDS(WWORDS), .RLAT(RLAT), .ABITS(ABITS)) u_c0 (
        .clk(clk), .rst(rst),
        .m_count(c0_m_count), .k_count(c0_k_count), .w_base(c0_w_base),
        .x_rst(c0_x_rst), .x_we(c0_x_we), .x_stream(c0_x_stream),
        .x_row(c0_x_row), .x_data(c0_x_data),
        .ovl_en(1'b0), .x_rowcommit(1'b0),
        .start(c0_start), .done(c0_done),
        .rd_stream(c0_rd_stream), .rd_addr(c0_rd_addr), .y_out(c0_y_out),
        .waddr(waddr_b), .wword_rd(wword_b));

    gemm_cohort_vec #(.LANES(LANES), .N(N), .ND(ND), .P(P), .MMAX(MMAX),
                      .KMAX(KMAX), .WWORDS(WWORDS), .RLAT(RLAT), .ABITS(ABITS)) u_c1 (
        .clk(clk), .rst(rst),
        .m_count(c1_m_count), .k_count(c1_k_count), .w_base(c1_w_base),
        .x_rst(c1_x_rst), .x_we(c1_x_we), .x_stream(c1_x_stream),
        .x_row(c1_x_row), .x_data(c1_x_data),
        .ovl_en(1'b0), .x_rowcommit(1'b0),
        .start(c1_start), .done(c1_done),
        .rd_stream(c1_rd_stream), .rd_addr(c1_rd_addr), .y_out(c1_y_out),
        .waddr(waddr_a), .wword_rd(wword_a));
endmodule
