// -----------------------------------------------------------------------------
// gelu_lut2 — TWO GELU lanes sharing one even/odd-banked LUT pair (BRAM-saving
// version of gelu_lut). Bit-true to gelu_lut: same 8192-entry table + 3-bit
// interpolation, only the storage layout changes:
//     lut_e[i] = lut[2i], lut_o[i] = lut[2i+1]
// Each lookup reads adjacent entries (i, i+1) = one even + one odd, so each
// 4096-deep bank serves both reads of one lane on one port — two lanes use the
// two ports of each bank: 2 lanes per BRAM pair instead of 1 lane per BRAM.
// Latency is 3 cycles, identical to gelu_lut.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gelu_lut2 (
    input  wire                clk,
    input  wire signed [15:0]  x0,
    output reg  signed [15:0]  y0,
    input  wire signed [15:0]  x1,
    output reg  signed [15:0]  y1
);
    (* rom_style = "block" *) reg signed [15:0] lut_e [0:4095];
    (* rom_style = "block" *) reg signed [15:0] lut_o [0:4095];
    initial begin
        $readmemh("gelu_lut_e.mem", lut_e);
        $readmemh("gelu_lut_o.mem", lut_o);
    end

    // ---- stage 0: offset to unsigned index space -------------------------------
    reg [15:0] u0, u1;
    always @(posedge clk) begin
        u0 <= x0 + 16'h8000;
        u1 <= x1 + 16'h8000;
    end

    // lane 0 addresses
    wire [12:0] idx0  = u0[15:3];
    wire [12:0] idx0n = (idx0 == 13'd8191) ? idx0 : idx0 + 1'b1;
    wire [11:0] e0a   = idx0[0] ? idx0n[12:1] : idx0[12:1];   // even-bank address
    wire [11:0] o0a   = idx0[12:1];                           // odd-bank address
    // lane 1 addresses
    wire [12:0] idx1  = u1[15:3];
    wire [12:0] idx1n = (idx1 == 13'd8191) ? idx1 : idx1 + 1'b1;
    wire [11:0] e1a   = idx1[0] ? idx1n[12:1] : idx1[12:1];
    wire [11:0] o1a   = idx1[12:1];

    // ---- stage 1: registered LUT reads (each bank: port A lane0, port B lane1) --
    reg signed [15:0] e0, o0, e1, o1;
    reg               s0, s1, c0, c1;
    reg [2:0]         f0, f1;
    always @(posedge clk) begin
        e0 <= lut_e[e0a];  o0 <= lut_o[o0a];
        e1 <= lut_e[e1a];  o1 <= lut_o[o1a];
        s0 <= idx0[0];  c0 <= (idx0 == 13'd8191);  f0 <= u0[2:0];
        s1 <= idx1[0];  c1 <= (idx1 == 13'd8191);  f1 <= u1[2:0];
    end

    // ---- stage 2: select + linear interpolation --------------------------------
    wire signed [15:0] l0_0 = s0 ? o0 : e0;
    wire signed [15:0] l1_0 = c0 ? l0_0 : (s0 ? e0 : o0);
    wire signed [16:0] d0   = l1_0 - l0_0;
    wire signed [19:0] st0  = d0 * $signed({1'b0, f0});
    wire signed [15:0] l0_1 = s1 ? o1 : e1;
    wire signed [15:0] l1_1 = c1 ? l0_1 : (s1 ? e1 : o1);
    wire signed [16:0] d1   = l1_1 - l0_1;
    wire signed [19:0] st1  = d1 * $signed({1'b0, f1});
    always @(posedge clk) begin
        y0 <= l0_0 + (st0 >>> 3);
        y1 <= l0_1 + (st1 >>> 3);
    end
endmodule
