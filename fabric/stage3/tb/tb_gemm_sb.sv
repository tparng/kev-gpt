// tb_gemm_sb — split-brain GEMM gate: two independent cohorts sharing one TDP
// weight image. Loads a shared w.mem (covering both cohorts' weight regions),
// per-cohort per-stream activations, then runs the two cohorts on DIFFERENT
// shapes / weight bases, STARTED A FEW CYCLES APART (deliberate desync — proves
// the two read ports are independent). Reads every stream of both cohorts back.
`timescale 1ns / 1ps

`ifndef LANES
`define LANES 128
`endif
`ifndef NSTR
`define NSTR 8
`endif
`ifndef NDSP
`define NDSP 0
`endif
`ifndef M0
`define M0 256
`endif
`ifndef K0
`define K0 256
`endif
`ifndef WB0
`define WB0 0
`endif
`ifndef M1
`define M1 256
`endif
`ifndef K1
`define K1 256
`endif
`ifndef WB1
`define WB1 512
`endif
`ifndef SKEW
`define SKEW 7
`endif

module tb_gemm_sb;
    localparam integer LANES = `LANES;
    localparam integer N     = `NSTR;
    localparam integer ND    = `NDSP;
    localparam integer P     = 8;
    localparam integer M0=`M0, K0=`K0, WB0=`WB0;
    localparam integer M1=`M1, K1=`K1, WB1=`WB1;
    localparam integer WBITS = LANES*4;
    localparam integer SUBW  = WBITS/32;
    // shared weight image: enough wide words to cover both regions
    localparam integer G0 = (M0 + LANES - 1)/LANES;
    localparam integer G1 = (M1 + LANES - 1)/LANES;
    localparam integer NW = (WB1 + G1*K1 > WB0 + G0*K0) ? (WB1 + G1*K1) : (WB0 + G0*K0);
    localparam integer XR0 = K0/P, XR1 = K1/P;

    // DOUBLE-PUMP-100K Stage 1: -DDPUMP exercises the double-pumped cohort path
    // (DP=1, 2 K-steps/clk). clk2x quarter-shifted per the tb_macdp convention.
`ifdef DPUMP
    localparam integer DPV = 1;
`else
    localparam integer DPV = 0;
`endif
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;                 // 10 ns period, posedges at 5,15,...
    reg clk2x = 0;
    initial begin #2.5 forever #2.5 clk2x = ~clk2x; end   // rising at 2.5,7.5,...

    reg ld_rst=0, w_we=0;
    reg [31:0] w_data;
    reg c0_x_rst=0,c0_x_we=0,c0_start=0; reg [$clog2(N)-1:0] c0_x_stream,c0_rd_stream;
    reg [P*8-1:0] c0_x_data; reg [10:0] c0_m,c0_k; reg [$clog2(25600)-1:0] c0_wb;
    reg [$clog2(1024/P)-1:0] c0_rd_addr; reg [$clog2(1024/P)-1:0] c0_x_row;
    reg c1_x_rst=0,c1_x_we=0,c1_start=0; reg [$clog2(N)-1:0] c1_x_stream,c1_rd_stream;
    reg [P*8-1:0] c1_x_data; reg [10:0] c1_m,c1_k; reg [$clog2(25600)-1:0] c1_wb;
    reg [$clog2(1024/P)-1:0] c1_rd_addr; reg [$clog2(1024/P)-1:0] c1_x_row;
    wire c0_done,c1_done; wire [P*32-1:0] c0_y,c1_y;
    // done is a ONE-CYCLE pulse from each cohort FSM (FIN->IDLE); with the
    // deliberate SKEW the two pulses never overlap, so latch them sticky.
    reg c0_fin=0, c1_fin=0;
    always @(posedge clk) begin
        if (c0_done) c0_fin <= 1'b1;
        if (c1_done) c1_fin <= 1'b1;
    end

    gemm_split_brain #(.LANES(LANES), .N(N), .ND(ND), .P(P), .MMAX(1024),
                       .KMAX(1024), .RLAT(2), .WWORDS(25600), .DP(DPV)) dut (
        .clk(clk), .clk2x(clk2x), .rst(rst), .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data),
        .c0_m_count(c0_m), .c0_k_count(c0_k), .c0_w_base(c0_wb),
        .c0_x_rst(c0_x_rst), .c0_x_we(c0_x_we), .c0_x_stream(c0_x_stream),
        .c0_x_row(c0_x_row), .c0_x_data(c0_x_data),
        .c0_start(c0_start), .c0_done(c0_done),
        .c0_rd_stream(c0_rd_stream), .c0_rd_addr(c0_rd_addr), .c0_y_out(c0_y),
        .c1_m_count(c1_m), .c1_k_count(c1_k), .c1_w_base(c1_wb),
        .c1_x_rst(c1_x_rst), .c1_x_we(c1_x_we), .c1_x_stream(c1_x_stream),
        .c1_x_row(c1_x_row), .c1_x_data(c1_x_data),
        .c1_start(c1_start), .c1_done(c1_done),
        .c1_rd_stream(c1_rd_stream), .c1_rd_addr(c1_rd_addr), .c1_y_out(c1_y));

    reg [WBITS-1:0] wrom  [0:NW-1];
    reg [P*8-1:0]   x0 [0:N*XR0-1];
    reg [P*8-1:0]   x1 [0:N*XR1-1];
    integer i, c, s, a, fd;
    reg [P*32-1:0] yw;
    reg [8*28-1:0] fname;

    initial begin
        $readmemh("w.mem",  wrom);
        $readmemh("x0.mem", x0);
        $readmemh("x1.mem", x1);
        repeat (4) @(posedge clk);
        rst <= 0; repeat (2) @(posedge clk);

        // shared weight image (port A loader)
        ld_rst <= 1; @(posedge clk); ld_rst <= 0; @(posedge clk);
        for (i = 0; i < NW; i = i + 1)
            for (c = 0; c < SUBW; c = c + 1) begin
                w_we <= 1; w_data <= wrom[i][c*32 +: 32]; @(posedge clk);
            end
        w_we <= 0; @(posedge clk);

        // cohort 0 activations
        for (s = 0; s < N; s = s + 1) begin
            c0_x_rst <= 1; @(posedge clk); c0_x_rst <= 0; @(posedge clk);
            for (i = 0; i < XR0; i = i + 1) begin
                c0_x_we <= 1; c0_x_stream <= s[$clog2(N)-1:0];
                c0_x_row <= i[$clog2(1024/P)-1:0];
                c0_x_data <= x0[s*XR0 + i]; @(posedge clk);
            end
            c0_x_we <= 0; @(posedge clk);
        end
        // cohort 1 activations
        for (s = 0; s < N; s = s + 1) begin
            c1_x_rst <= 1; @(posedge clk); c1_x_rst <= 0; @(posedge clk);
            for (i = 0; i < XR1; i = i + 1) begin
                c1_x_we <= 1; c1_x_stream <= s[$clog2(N)-1:0];
                c1_x_row <= i[$clog2(1024/P)-1:0];
                c1_x_data <= x1[s*XR1 + i]; @(posedge clk);
            end
            c1_x_we <= 0; @(posedge clk);
        end

        // run BOTH — start cohort 1 SKEW cycles after cohort 0 (desync proof)
        c0_m <= M0[10:0]; c0_k <= K0[10:0]; c0_wb <= WB0[$clog2(25600)-1:0];
        c1_m <= M1[10:0]; c1_k <= K1[10:0]; c1_wb <= WB1[$clog2(25600)-1:0];
        c0_start <= 1; @(posedge clk); c0_start <= 0;
        repeat (`SKEW) @(posedge clk);
        c1_start <= 1; @(posedge clk); c1_start <= 0;

        wait (c0_fin && c1_fin); @(posedge clk);

        // readback cohort 0
        for (s = 0; s < N; s = s + 1) begin
            $sformat(fname, "y0_s%0d.out", s);
            fd = $fopen(fname, "w");
            for (a = 0; a < M0/P; a = a + 1) begin
                c0_rd_stream <= s[$clog2(N)-1:0]; c0_rd_addr <= a[$clog2(1024/P)-1:0];
                repeat (4) @(posedge clk);
                yw = c0_y;
                for (c = 0; c < P; c = c + 1) $fdisplay(fd, "%08x", yw[c*32 +: 32]);
            end
            $fclose(fd);
        end
        // readback cohort 1
        for (s = 0; s < N; s = s + 1) begin
            $sformat(fname, "y1_s%0d.out", s);
            fd = $fopen(fname, "w");
            for (a = 0; a < M1/P; a = a + 1) begin
                c1_rd_stream <= s[$clog2(N)-1:0]; c1_rd_addr <= a[$clog2(1024/P)-1:0];
                repeat (4) @(posedge clk);
                yw = c1_y;
                for (c = 0; c < P; c = c + 1) $fdisplay(fd, "%08x", yw[c*32 +: 32]);
            end
            $fclose(fd);
        end
        $display("TB_DONE N=%0d ND=%0d M0=%0d K0=%0d M1=%0d K1=%0d SKEW=%0d",
                 N, ND, M0, K0, M1, K1, `SKEW);
        $finish;
    end
    initial begin #50000000; $display("TB_TIMEOUT"); $finish; end
endmodule
