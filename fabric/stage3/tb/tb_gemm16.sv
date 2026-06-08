// tb_gemm16 — standalone gate for the N-stream batch GEMM core (LUT + DSP
// leaves). Loads w.mem wide words via the 32-bit chunk loader, per-stream
// activation rows from xrows_s<i>.mem, runs one (M,K) call, reads every
// stream's outputs back through the P-wide port into y_s<i>.out.
`timescale 1ns / 1ps

`ifndef LANES
`define LANES 128
`endif
`ifndef NSTR
`define NSTR 16
`endif
`ifndef NDSP
`define NDSP 8
`endif
`ifndef MVAL
`define MVAL 256
`endif
`ifndef KVAL
`define KVAL 256
`endif

module tb_gemm16;
    localparam integer LANES = `LANES;
    localparam integer N     = `NSTR;
    localparam integer ND    = `NDSP;
    localparam integer P     = 8;
    localparam integer M     = `MVAL;
    localparam integer K     = `KVAL;
    localparam integer WBITS = LANES*4;
    localparam integer SUBW  = WBITS/32;
    localparam integer GROUPS= (M + LANES - 1)/LANES;
    localparam integer NW    = GROUPS*K;
    localparam integer XR    = K/P;

    // DOUBLE-PUMP-100K Stage 1: -DDPUMP exercises the double-pumped LUT path
    // (DP=1, mac_bank_dp at 2 K-steps/clk). clk2x is 2x clk, quarter-shifted so
    // its two rising edges per clk period sample distinct clk levels with NO
    // coincident-edge race — the exact convention proven by tb_macdp.
`ifdef DPUMP
    localparam integer DPV = 1;
`else
    localparam integer DPV = 0;
`endif
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;                 // 10 ns period, posedges at 5,15,...
    reg clk2x = 0;
    initial begin #2.5 forever #2.5 clk2x = ~clk2x; end   // rising at 2.5,7.5,...

    reg                     ld_rst = 0, w_we = 0, x_rst = 0, x_we = 0, start = 0;
    reg  [31:0]             w_data;
    reg  [$clog2(N)-1:0]    x_stream, rd_stream;
    reg  [P*8-1:0]          x_data;
    reg  [10:0]             m_count, k_count;
    reg  [$clog2(25600)-1:0] w_base = 0;
    reg  [$clog2(1024/P)-1:0] rd_addr;
    wire                    done;
    wire [P*32-1:0]         y_out;

    gemm_banked_resident_vec #(.LANES(LANES), .N(N), .ND(ND), .P(P),
                               .MMAX(1024), .KMAX(1024), .RLAT(2),
                               .WWORDS(25600), .DP(DPV)) dut (
        .clk(clk), .clk2x(clk2x), .rst(rst), .m_count(m_count), .k_count(k_count),
        .w_base(w_base), .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data),
        .x_rst(x_rst), .x_we(x_we), .x_stream(x_stream), .x_data(x_data),
        .start(start), .done(done),
        .rd_stream(rd_stream), .rd_addr(rd_addr), .y_out(y_out));

    reg [WBITS-1:0] wrom  [0:NW-1];
    reg [P*8-1:0]   xrows [0:N*XR-1];          // stream s rows at s*XR
    integer i, c, s, a, fd;
    reg [P*32-1:0] yw;
    reg [8*24-1:0] fname;

    initial begin
        $readmemh("w.mem", wrom);
        $readmemh("xrows.mem", xrows);
        repeat (4) @(posedge clk);
        rst <= 0; repeat (2) @(posedge clk);

        // weights: SUBW 32-bit chunks per wide word
        ld_rst <= 1; @(posedge clk); ld_rst <= 0; @(posedge clk);
        for (i = 0; i < NW; i = i + 1)
            for (c = 0; c < SUBW; c = c + 1) begin
                w_we <= 1; w_data <= wrom[i][c*32 +: 32]; @(posedge clk);
            end
        w_we <= 0; @(posedge clk);

        // activations, per stream
        for (s = 0; s < N; s = s + 1) begin
            x_rst <= 1; @(posedge clk); x_rst <= 0; @(posedge clk);
            for (i = 0; i < XR; i = i + 1) begin
                x_we <= 1; x_stream <= s[$clog2(N)-1:0];
                x_data <= xrows[s*XR + i]; @(posedge clk);
            end
            x_we <= 0; @(posedge clk);
        end

        // run
        m_count <= M[10:0]; k_count <= K[10:0]; w_base <= 0;
        start <= 1; @(posedge clk); start <= 0;
        wait (done); @(posedge clk);

        // readback: conservative 4-cycle settle per address
        for (s = 0; s < N; s = s + 1) begin
            $sformat(fname, "y_s%0d.out", s);
            fd = $fopen(fname, "w");
            for (a = 0; a < M/P; a = a + 1) begin
                rd_stream <= s[$clog2(N)-1:0]; rd_addr <= a[$clog2(1024/P)-1:0];
                repeat (4) @(posedge clk);
                yw = y_out;
                for (c = 0; c < P; c = c + 1)
                    $fdisplay(fd, "%08x", yw[c*32 +: 32]);
            end
            $fclose(fd);
        end
        $display("TB_DONE N=%0d ND=%0d M=%0d K=%0d", N, ND, M, K);
        $finish;
    end
endmodule
