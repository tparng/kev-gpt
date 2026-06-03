// Testbench for layernorm_vec (P-wide-I/O LayerNorm). For each of NCASE vectors it pulses
// start, streams ROWS=256/P packed rows of (x,gamma) with valid_in, then captures the P-wide
// y_out for ROWS cycles (gated on y_valid) and dumps one 64-bit hex per element (element
// order = row-major lane-major = r*P+k), NCASE*256 lines, for the Python compare vs ln_int.
`timescale 1ns / 1ps
`ifndef NCASE
 `define NCASE 64
`endif
`ifndef PVAL
 `define PVAL 8
`endif
module tb;
    localparam integer P    = `PVAL;
    localparam integer D    = 256;
    localparam integer ROWS = D / P;
    localparam integer NC   = `NCASE;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, start, vin;
    reg [P*32-1:0] xin, gin;
    wire yv, done;
    wire [P*64-1:0] yout;

    layernorm_vec #(.P(P)) dut (
        .clk(clk), .rst(rst), .start(start), .valid_in(vin),
        .x_in(xin), .gamma_in(gin), .y_valid(yv), .y_out(yout), .done(done));

    // packed rows from the gate: row index = case*ROWS + r ; lane k = [32k +: 32]
    reg [P*32-1:0] xrows [0:ROWS*NC-1];
    reg [P*32-1:0] grows [0:ROWS*NC-1];
    integer c, r, f, k;
    reg [63:0] ylane;

    initial begin
        rst = 1'b1; start = 1'b0; vin = 1'b0; xin = 0; gin = 0;
        $readmemh("x.mem", xrows);
        $readmemh("g.mem", grows);
        @(posedge clk); #1; @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        f = $fopen("y.out", "w");
        for (c = 0; c < NC; c = c + 1) begin
            start = 1'b1; @(posedge clk); #1; start = 1'b0;
            // feed ROWS packed rows
            for (r = 0; r < ROWS; r = r + 1) begin
                vin = 1'b1; xin = xrows[c*ROWS + r]; gin = grows[c*ROWS + r];
                @(posedge clk); #1;
            end
            vin = 1'b0;
            // capture ROWS rows of P outputs, gated on y_valid
            r = 0;
            while (r < ROWS) begin
                if (yv) begin
                    for (k = 0; k < P; k = k + 1) begin
                        ylane = yout[k*64 +: 64];
                        $fwrite(f, "%016x\n", ylane);
                    end
                    r = r + 1;
                end
                @(posedge clk); #1;
            end
            repeat (3) begin @(posedge clk); #1; end   // let the FSM return to IDLE
        end
        $fclose(f);
        $display("TB_DONE");
        $finish;
    end

    initial begin #200000000; $display("TB_TIMEOUT"); $finish; end
endmodule
