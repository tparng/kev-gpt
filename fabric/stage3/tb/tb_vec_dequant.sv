// Testbench for vec_dequant: stream N elements P-wide through the parallel dequant
// and dump each lane's dq_out to y.out for the Python bit-true compare
// (run_vec_dequant.py). The Python side groups N into ceil(N/P) cycles, pads the
// last partial cycle, and aligns the 1-cycle output latency.
//
// Stimulus comes as three flat per-element .mem files (one hex value per line):
//   gemvy.mem  : 32-bit two's-complement INT32  (8 hex nibbles)
//   mant.mem   : 24-bit two's-complement signed  (6 hex nibbles)
//   exp.mem    : 8-bit  two's-complement signed  (2 hex nibbles)
// The TB assembles P consecutive elements into the packed lane buses each cycle.
`timescale 1ns / 1ps
`ifndef NELEM
 `define NELEM 1024
`endif
`ifndef P
 `define P 8
`endif
`ifndef FRAC
 `define FRAC 16
`endif

module tb;
    localparam integer NELEM = `NELEM;
    localparam integer P     = `P;
    localparam integer FRAC  = `FRAC;
    localparam integer NCYC  = (NELEM + P - 1) / P;   // padded cycles
    localparam integer NPAD  = NCYC * P;              // padded element count

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                rst = 1'b1;
    reg                in_valid = 1'b0;
    reg [P*32-1:0]     gemvy = 0;
    reg [P*24-1:0]     mant  = 0;
    reg [P*8-1:0]      exp   = 0;
    wire               out_valid;
    wire [P*32-1:0]    dq_out;

    vec_dequant #(.P(P), .FRAC(FRAC)) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .gemvy(gemvy), .mant(mant), .exp(exp),
        .out_valid(out_valid), .dq_out(dq_out)
    );

    // per-element stimulus (flat, padded to NPAD with zeros)
    reg signed [31:0] gv_mem  [0:NPAD-1];
    reg signed [23:0] mt_mem  [0:NPAD-1];
    reg signed [7:0]  ex_mem  [0:NPAD-1];
    integer c, k, f, e;

    initial begin
        for (e = 0; e < NPAD; e = e + 1) begin
            gv_mem[e] = 0; mt_mem[e] = 0; ex_mem[e] = 0;
        end
        $readmemh("gemvy.mem", gv_mem);
        $readmemh("mant.mem",  mt_mem);
        $readmemh("exp.mem",   ex_mem);

        f = $fopen("y.out", "w");

        rst = 1'b1;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        // Drive NCYC valid cycles, then a few trailing cycles to flush the 1-cyc latency.
        for (c = 0; c < NCYC + 2; c = c + 1) begin
            if (c < NCYC) begin
                in_valid = 1'b1;
                for (k = 0; k < P; k = k + 1) begin
                    gemvy[k*32 +: 32] = gv_mem[c*P + k];
                    mant [k*24 +: 24] = mt_mem[c*P + k];
                    exp  [k*8  +: 8 ] = ex_mem[c*P + k];
                end
            end else begin
                in_valid = 1'b0;
                gemvy = 0; mant = 0; exp = 0;
            end
            @(posedge clk); #1;
            // capture: dq_out is valid this cycle when out_valid (1-cyc after in_valid)
            if (out_valid) begin
                for (k = 0; k < P; k = k + 1)
                    $fwrite(f, "%08x\n", dq_out[k*32 +: 32]);
            end
        end

        $fclose(f);
        $display("TB_DONE NELEM=%0d P=%0d FRAC=%0d NCYC=%0d", NELEM, P, FRAC, NCYC);
        $finish;
    end
endmodule
