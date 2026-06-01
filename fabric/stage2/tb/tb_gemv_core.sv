// Self-checking TB for gemv_core (Stage 2 keystone). Drives one GEMV using the
// exporter's WFILE/XFILE init (runtime load ports tied off), waits for done,
// dumps y to a file; the run script does the bit-exact Python compare vs golden.
`timescale 1ns / 1ps
`include "config.svh"

module tb_gemv_core;
    localparam integer M = `CFG_M;
    localparam integer K = `CFG_K;

    localparam integer WAW = $clog2(M*K/2);
    localparam integer XAW = $clog2(K);
    localparam integer MAW = $clog2(M);

    reg clk = 0, rst = 1, start = 0;
    wire done;
    always #5 clk = ~clk;   // 100 MHz

    gemv_core #(.M(M), .K(K), .WFILE(`CFG_WFILE), .XFILE(`CFG_XFILE)) dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .w_we(1'b0), .w_addr({WAW{1'b0}}), .w_data(8'd0),
        .x_we(1'b0), .x_addr({XAW{1'b0}}), .x_data(8'd0),
        .rd_addr({MAW{1'b0}}), .y_out()
    );

    reg [31:0] ycopy [0:M-1];
    integer m, cyc;

    initial begin
        repeat (4) @(posedge clk);
        rst <= 0;
        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;

        cyc = 0;
        while (!done) begin @(posedge clk); cyc = cyc + 1; end

        for (m = 0; m < M; m = m + 1) ycopy[m] = dut.y[m];
        $writememh(`CFG_YDUT, ycopy);
        $display("RESULT cycles=%0d M=%0d K=%0d", cyc, M, K);
        $finish;
    end

    initial begin
        #20000000;
        $display("RESULT TIMEOUT");
        $finish;
    end
endmodule
