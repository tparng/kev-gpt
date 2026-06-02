// Testbench for layernorm_par: for each of NCASE vectors, pulse start, stream 256
// (x,gamma) pairs, capture the 256 y_out values (on y_valid) to y.out, and record
// the start->done latency (lat_cycles) of each case to lat.out. Mirrors the serial
// tb so the SAME Python gate (re-pointed at this module) checks bit-trueness.
`timescale 1ns / 1ps
`ifndef NCASE
 `define NCASE 64
`endif

module tb;
    localparam integer NCASE = `NCASE;
    localparam integer D = 256;
    localparam integer NTOT = NCASE * D;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                rst = 1'b1;
    reg                start = 1'b0;
    reg                valid_in = 1'b0;
    reg signed [31:0]  x_in = 0;
    reg signed [31:0]  gamma_in = 0;
    wire               y_valid;
    wire signed [63:0] y_out;
    wire               done;
    wire [15:0]        lat_cycles;

    layernorm_par dut (
        .clk(clk), .rst(rst), .start(start), .valid_in(valid_in),
        .x_in(x_in), .gamma_in(gamma_in),
        .y_valid(y_valid), .y_out(y_out), .done(done),
        .lat_cycles(lat_cycles)
    );

    reg signed [31:0] xin  [0:NTOT-1];
    reg signed [31:0] gin  [0:NTOT-1];
    integer c, i, f, fl;

    initial begin
        $readmemh("x.mem", xin);
        $readmemh("gamma.mem", gin);
        f  = $fopen("y.out", "w");
        fl = $fopen("lat.out", "w");

        rst = 1'b1;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        for (c = 0; c < NCASE; c = c + 1) begin
            start = 1'b1;
            @(posedge clk); #1;
            start = 1'b0;
            for (i = 0; i < D; i = i + 1) begin
                valid_in = 1'b1;
                x_in     = xin[c*D + i];
                gamma_in = gin[c*D + i];
                @(posedge clk); #1;
            end
            valid_in = 1'b0;
            x_in = 0; gamma_in = 0;
            i = 0;
            while (i < D) begin
                if (y_valid) begin
                    $fwrite(f, "%016x\n", y_out);
                    i = i + 1;
                end
                @(posedge clk); #1;
            end
            while (!done) begin
                @(posedge clk); #1;
            end
            // done is high this cycle; lat_cycles is the start->done count
            $fwrite(fl, "%0d\n", lat_cycles);
            @(posedge clk); #1;        // let done deassert before next case
        end

        $fclose(f);
        $fclose(fl);
        $display("TB_DONE NCASE=%0d", NCASE);
        $finish;
    end
endmodule
