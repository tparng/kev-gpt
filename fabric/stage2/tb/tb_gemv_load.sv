// Self-checking TB for gemv_load: streams a layer's weights+activation through
// the LOAD ports (not $readmemh init), sets runtime M/K, runs, dumps y. Proves
// the runtime-loadable, runtime-sized path bit-exact across layer shapes.
`timescale 1ns / 1ps
`include "config.svh"

module tb_gemv_load;
    localparam integer M = `CFG_M;
    localparam integer K = `CFG_K;
    localparam integer MMAX = 1024, KMAX = 1024, WBYTES = 131072;

    reg clk = 0, rst = 1, start = 0, ld_rst = 0, w_we = 0, x_we = 0;
    reg [7:0] w_data = 0, x_data = 0;
    reg [$clog2(MMAX+1)-1:0] m_count = 0;
    reg [$clog2(KMAX+1)-1:0] k_count = 0;
    wire done;
    always #5 clk = ~clk;

    gemv_load #(.MMAX(MMAX), .KMAX(KMAX), .WBYTES(WBYTES)) dut (
        .clk(clk), .rst(rst), .m_count(m_count), .k_count(k_count),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data), .x_we(x_we), .x_data(x_data),
        .start(start), .done(done), .rd_addr({$clog2(MMAX){1'b0}}), .y_out()
    );

    // source images loaded from the exporter's .mem files
    reg [7:0]        wsrc [0:WBYTES-1];
    reg signed [7:0] xsrc [0:KMAX-1];
    reg [31:0]       ycopy [0:M-1];
    integer i, m, cyc;

    initial begin
        $readmemh(`CFG_WFILE, wsrc);
        $readmemh(`CFG_XFILE, xsrc);

        repeat (4) @(posedge clk); rst <= 0; @(posedge clk);

        // stream weights then activation through the load ports
        ld_rst <= 1; @(posedge clk); ld_rst <= 0;
        for (i = 0; i < M*K/2; i = i + 1) begin
            w_we <= 1; w_data <= wsrc[i]; @(posedge clk);
        end
        w_we <= 0;
        for (i = 0; i < K; i = i + 1) begin
            x_we <= 1; x_data <= xsrc[i]; @(posedge clk);
        end
        x_we <= 0;

        // configure runtime dims and run
        m_count <= M[$clog2(MMAX+1)-1:0];
        k_count <= K[$clog2(KMAX+1)-1:0];
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;

        cyc = 0;
        while (!done) begin @(posedge clk); cyc = cyc + 1; end

        for (m = 0; m < M; m = m + 1) ycopy[m] = dut.y[m];
        $writememh(`CFG_YDUT, ycopy);
        $display("RESULT cycles=%0d M=%0d K=%0d", cyc, M, K);
        $finish;
    end

    initial begin #50000000; $display("RESULT TIMEOUT"); $finish; end
endmodule
