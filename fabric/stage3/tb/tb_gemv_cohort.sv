// tb_gemv_cohort — one cohort GEMV call over N streams sharing the weight image.
// Files: gc_cfg.mem (word0=N), gc_w.mem (ROWS*WPR x 32b shared weights),
// gc_x.mem (N*D_IN x 8b, stream-major).  Out: gc_acc.out (N*ROWS INT32,
// stream-major).  Also prints GC_CYC = cycles from start->done (independent of
// N: the whole cohort costs one weight pass).
`timescale 1ns/1ps

module tb_gemv_cohort;
    localparam int NSTREAM = 8, ROWS = 1160, D_IN = 256, WPR = D_IN/8;
    localparam int WMEM = 262144;

    reg clk = 0, rst = 1, start = 0;
    wire done;
    reg wr_w = 0, wr_x = 0;
    reg [$clog2(WMEM)-1:0] wr_w_addr;
    reg [$clog2(NSTREAM)-1:0] wr_x_stream, rd_stream;
    reg [$clog2(D_IN)-1:0] wr_x_addr;
    reg [31:0] wr_w_data;
    reg signed [7:0] wr_x_data;
    reg [$clog2(ROWS)-1:0] rd_acc_addr;
    wire signed [31:0] rd_acc_data;
    reg [$clog2(WMEM)-1:0] base = 0;
    reg [$clog2(ROWS+1)-1:0] rows = ROWS;
    reg [$clog2(WPR+1)-1:0] wpr = WPR;

    gemv_i4i8_cohort #(.N(NSTREAM), .ROWS(ROWS), .D_IN(D_IN), .WMEM(WMEM)) dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .base(base), .rows(rows), .wpr(wpr),
        .wr_w(wr_w), .wr_w_addr(wr_w_addr), .wr_w_data(wr_w_data),
        .wr_x(wr_x), .wr_x_stream(wr_x_stream), .wr_x_addr(wr_x_addr),
        .wr_x_data(wr_x_data),
        .rd_stream(rd_stream), .rd_acc_addr(rd_acc_addr),
        .rd_acc_data(rd_acc_data));
    always #2 clk = ~clk;

    reg [31:0] wmem [0:ROWS*WPR-1];
    reg [7:0]  xmem [0:NSTREAM*D_IN-1];
    reg [31:0] cfg [0:0];
    integer N, s, i, fd, cyc = 0, c0 = 0;
    always @(posedge clk) cyc <= cyc + 1;

    initial begin
        $readmemh("gc_cfg.mem", cfg); N = cfg[0];
        $readmemh("gc_w.mem", wmem);
        $readmemh("gc_x.mem", xmem, 0, N*D_IN-1);
        fd = $fopen("gc_acc.out", "w");
        repeat (4) @(posedge clk);
        rst <= 0; @(posedge clk);
        // shared weights, written once
        for (i = 0; i < ROWS*WPR; i = i + 1) begin
            @(posedge clk); wr_w <= 1; wr_w_addr <= i; wr_w_data <= wmem[i];
        end
        @(posedge clk); wr_w <= 0;
        // per-stream activations
        for (s = 0; s < N; s = s + 1)
            for (i = 0; i < D_IN; i = i + 1) begin
                @(posedge clk); wr_x <= 1; wr_x_stream <= s[$clog2(NSTREAM)-1:0];
                wr_x_addr <= i; wr_x_data <= xmem[s*D_IN+i];
            end
        @(posedge clk); wr_x <= 0;
        // ONE cohort pass for all N streams
        @(posedge clk); start <= 1; c0 = cyc;
        @(posedge clk); start <= 0;
        wait (done);
        @(posedge clk);
        $display("GC_CYC N=%0d cyc=%0d", N, cyc - c0);
        for (s = 0; s < N; s = s + 1) begin
            rd_stream <= s[$clog2(NSTREAM)-1:0];
            for (i = 0; i < ROWS; i = i + 1) begin
                rd_acc_addr <= i;
                @(posedge clk); @(posedge clk);
                $fdisplay(fd, "%08x", rd_acc_data);
            end
        end
        $fclose(fd);
        $display("TB_GC_DONE N=%0d", N);
        $finish;
    end
endmodule
