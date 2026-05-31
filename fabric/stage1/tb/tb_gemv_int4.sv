// -----------------------------------------------------------------------------
// tb_gemv_int4 — self-checking testbench for the Stage 1 GEMV core.
//
// Drives one GEMV with weights/activations from the exporter's .mem images and
// dumps the result to a file. The authoritative bit-exact compare against the
// numpy golden is done in Python by the run script (so the verdict never depends
// on reading numbers off the console). This is the "bit-honest before fast" gate
// in hardware form.
//
// Dims and file paths come from config.svh, generated per-run by
// fabric/stage1/sim/run_iverilog.sh, so the same TB checks any exported layer.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "config.svh"

module tb_gemv_int4;
    localparam integer M  = `CFG_M;
    localparam integer K  = `CFG_K;
    localparam integer PE = `CFG_PE;

    reg clk = 0, rst = 1, start = 0;
    wire done;

    always #5 clk = ~clk;   // 100 MHz sim clock

    gemv_int4 #(
        .M(M), .K(K), .PE_LANES(PE),
        .WFILE(`CFG_WFILE), .XFILE(`CFG_XFILE)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .done(done)
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
        $display("RESULT cycles=%0d M=%0d K=%0d PE=%0d", cyc, M, K, PE);
        $finish;
    end

    initial begin
        #5000000;
        $display("RESULT TIMEOUT");
        $finish;
    end
endmodule
