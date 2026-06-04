// -----------------------------------------------------------------------------
// vec_gelu — P-lane parallel GELU. Applies GELU to P elements per cycle by
// instantiating P copies of the single-lane rtl/gelu_lut core (one per lane).
// It does NOT reimplement the math: each lane is bit-identical to gelu_lut, so
// the whole vector is bit-true to fabric/stage3/run_gelu.gelu_q.
//
// I/O = signed Q4.12 (16-bit, range +-8), P-wide. Streaming: accept a P-wide
// input every cycle (in_valid), emit the corresponding P-wide output 3 cycles
// later (out_valid). The valid is carried through a 3-deep shift register to
// match the gelu_lut pipeline latency exactly.
//
// The shared LUT (gelu_lut.mem, $readmemh) is loaded independently by each
// gelu_lut instance — fine in simulation; in hardware each lane infers its own
// BRAM ROM copy (or the synthesizer may share read ports).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module vec_gelu #(
    parameter integer P = 8
) (
    input  wire                       clk,
    input  wire                       in_valid,
    input  wire signed [16*P-1:0]     x,         // P x Q4.12, lane k = x[16*k +: 16]
    output wire                       out_valid,
    output wire signed [16*P-1:0]     y          // P x Q4.12, lane k = y[16*k +: 16]
);
    // P/2 paired cores: two lanes share one even/odd-banked LUT (same math,
    // half the BRAM). Bit-identical to P independent gelu_lut lanes.
    genvar k;
    generate
        for (k = 0; k < P; k = k + 2) begin : lane
            wire signed [15:0] xk0 = x[16*k +: 16];
            wire signed [15:0] xk1 = x[16*(k+1) +: 16];
            wire signed [15:0] yk0, yk1;
            gelu_lut2 u_gelu (.clk(clk), .x0(xk0), .y0(yk0), .x1(xk1), .y1(yk1));
            assign y[16*k +: 16]     = yk0;
            assign y[16*(k+1) +: 16] = yk1;
        end
    endgenerate

    // 3-deep valid shift register: gelu_lut has 3-cycle latency (stage 0/1/2),
    // so a P-wide input on cycle t emits on cycle t+3.
    reg [2:0] vsr = 3'b000;
    always @(posedge clk)
        vsr <= {vsr[1:0], in_valid};
    assign out_valid = vsr[2];
endmodule
