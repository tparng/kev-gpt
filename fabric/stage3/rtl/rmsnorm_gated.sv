// rmsnorm_gated.sv — Mamba-2 (optionally gated) RMSNorm (doc 9; reference =
// model/mamba2_fixed.rmsnorm_fixed at fabric LUT/rsqrt resolution).
//
//   gated=1:  g[i] = sat16( rnd( y[i] * silu(z[i]) >>> 12 ) )     (Q3.12)
//   gated=0:  g[i] = y[i]                (silu operand = 1.0 Q3.12; exact)
//   out[i]   = sat16( rnd( g[i] * rsqrt(mean(g^2)+eps) * gamma[i] >>> 40 ) )
//
// Formats (contract):
//   y, z        INT16 Q3.12       gamma  INT16 Q1.14
//   g           INT16 Q3.12 (rounded, saturated gated product)
//   sum(g^2)    Q6.24 accumulate (48-bit)
//   A = mean+eps Q.26  =  (sum >>> (log2(D)-2)) + round(1e-5*2^26)
//   rsqrt       Q.26 — 64-entry seed ROM ("seed.mem") + 2 Newton steps,
//               the silicon-proven layernorm.sv / layernorm_vec.sv datapath
//               WITHOUT the mean subtraction (RMS, not variance)
//   out         INT16 Q3.12: (g*Yr*gamma + 2^39) >>> 40, saturated
//
// SiLU: 256-entry LUT over [-4,4) Q3.12 input — the EXACT conv_silu.sv
// indexing contract: idx = top 8 bits of (z+16384) clamped to [0,32767]
// (i.e. floor), tails y=x for z>=4, 0 for z<-4. LUT loaded via write port.
//
// Scalar sequential (one lane): ~D cycles gate pass + rsqrt + ~D cycles
// output pass. Simplicity over speed — norm is a tiny slice of the token.
`timescale 1ns / 1ps
`default_nettype none

module rmsnorm_gated #(
    parameter integer D  = 512,      // vector length (power of 2)
    parameter integer QF = 12        // Q3.12 activations
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,      // pulse; y/z/gamma/lut written
    input  wire                    gated,      // 1: g=y*silu(z)  0: g=y
    input  wire                    short_len,  // 1: operate on D/2 elements
                                               // (the engine's pre/final norms
                                               // are 256-wide; gated norm 512)
    output reg                     done,

    // per-token inputs, written before start
    input  wire                    wr_y,
    input  wire [$clog2(D)-1:0]    wr_y_addr,
    input  wire signed [15:0]      wr_y_data,
    input  wire                    wr_z,
    input  wire [$clog2(D)-1:0]    wr_z_addr,
    input  wire signed [15:0]      wr_z_data,

    // gamma (once), Q1.14
    input  wire                    wr_g,
    input  wire [$clog2(D)-1:0]    wr_g_addr,
    input  wire signed [15:0]      wr_g_data,

    // silu LUT init (once), Q3.12 values
    input  wire                    wr_lut,
    input  wire [7:0]              wr_lut_addr,
    input  wire signed [15:0]      wr_lut_data,

    input  wire [$clog2(D)-1:0]    rd_o_addr,
    output wire signed [15:0]      rd_o_data
);
    localparam integer LOG2D  = $clog2(D);
    localparam integer A_FRAC = 26;
    localparam integer Y_FRAC = 26;
    localparam integer G_FRAC = 14;                    // gamma Q1.14
    localparam integer A_SH   = LOG2D + 2*QF - A_FRAC; // sum Q.2QF -> mean Q.26
    localparam integer OUT_SH = Y_FRAC + G_FRAC;       // 40: Q.(QF+26+14)->Q.QF
    localparam integer SEED_IDX_BITS = 6;
    localparam integer SEED_OUT_FRAC = 16;
    localparam [63:0]  EPS_A    = 64'd671;             // round(1e-5 * 2^26)
    localparam [31:0]  SQRT2Q15 = 32'd46341;           // round(sqrt(2)*2^15)
    localparam [127:0] ONE_P5   = 128'd3 <<< (2*Y_FRAC - 1);

    // ---- storage -------------------------------------------------------
    reg signed [15:0] yin  [0:D-1];
    reg signed [15:0] zin  [0:D-1];
    reg signed [15:0] grom [0:D-1];
    reg signed [15:0] lut  [0:255];
    reg signed [15:0] gbuf [0:D-1];       // gated product g (Q3.12)
    reg signed [15:0] obuf [0:D-1];

    assign rd_o_data = obuf[rd_o_addr];

    always @(posedge clk) begin
        if (wr_y)   yin [wr_y_addr]   <= wr_y_data;
        if (wr_z)   zin [wr_z_addr]   <= wr_z_data;
        if (wr_g)   grom[wr_g_addr]   <= wr_g_data;
        if (wr_lut) lut [wr_lut_addr] <= wr_lut_data;
    end

    // ---- seed ROM (64 x Q1.16) — same file/format as layernorm.sv -------
    (* rom_style = "block" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);

    // ---- FSM -------------------------------------------------------------
    localparam [4:0]
        S_IDLE=0, S_GATE=1, S_GDRAIN=2, S_A=3, S_MSB=4, S_SEED=5, S_SEED2=6,
        S_NEWT=7, S_NEWTB=8, S_NEWTC=9, S_OUT=10, S_DONE=11;
    reg [4:0] state;
    reg       g_mode;                      // sampled `gated` for this pass

    // ---- gate pass: 2-stage pipeline, one element/cycle ------------------
    reg [LOG2D-1:0] ci;
    wire issue = (state == S_GATE);
    reg               v1, v2;
    reg [LOG2D-1:0]   c1, c2;
    reg signed [15:0] y1, z1;
    reg signed [31:0] prod2;               // y*silu Q6.24

    // silu comb off z1 (conv_silu.sv lines 93-99 contract, verbatim)
    wire signed [16:0] zb   = z1 + $signed(17'sd16384);
    wire        [7:0]  sidx = (zb < 0) ? 8'd0 :
                              (zb > $signed(17'sd32767)) ? 8'd255 : zb[14:7];
    wire signed [15:0] silw = (z1 >= $signed(16'sd16384)) ? z1 :
                              (z1 < -$signed(16'sd16384)) ? 16'sd0 : lut[sidx];
    // ungated bypass: operand 1.0 (Q3.12) -> (y<<QF + 2^(QF-1))>>>QF == y exact
    wire signed [15:0] silsel = g_mode ? silw : 16'sd4096;

    // stage-2 comb: round >>> QF, saturate, square
    wire signed [20:0] pre = (prod2 + $signed(32'sd1 <<< (QF-1))) >>> QF;
    wire signed [15:0] gw  = (pre > $signed(21'sd32767))  ? 16'sd32767 :
                             (pre < -$signed(21'sd32768)) ? -16'sd32768 :
                             pre[15:0];
    wire signed [31:0] gsq = gw * gw;                         // Q6.24

    reg signed [47:0] sumsq;                                  // sum g^2 Q6.24

    // ---- rsqrt registers (layernorm.sv datapath, minus the mean) ---------
    reg signed [63:0] A, Yr, Y0;
    reg [1:0]  newt;
    reg [7:0]  msb;
    reg [5:0]  seed_idx;
    reg signed [8:0]  half, qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;
    reg [19:0]        seed_val;
    reg signed [127:0] yy, ayy, term;
    reg signed [191:0] ynew;

    integer b;
    reg [7:0] msb_c;
    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    // ---- output pass: 3-stage pipeline, one element/cycle ----------------
    reg [LOG2D:0]     oi;
    reg               ov1, ov2, ov3;
    reg [LOG2D-1:0]   oc1, oc2, oc3;
    reg signed [15:0] go1, gm1, gm2;
    reg signed [95:0]  prod_gy;            // g*Yr        Q.(QF+26)
    reg signed [111:0] prod_gyg;           // g*Yr*gamma  Q.(QF+40)
    localparam signed [111:0] ORND = 112'sd1 <<< (OUT_SH - 1);
    wire signed [112:0] osh  = (prod_gyg + ORND) >>> OUT_SH;
    wire signed [15:0]  osat = (osh > $signed(113'sd32767))  ? 16'sd32767 :
                               (osh < -$signed(113'sd32768)) ? -16'sd32768 :
                               osh[15:0];

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            state <= S_IDLE; v1 <= 0; v2 <= 0;
            ov1 <= 0; ov2 <= 0; ov3 <= 0; newt <= 0;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    ci <= '0; sumsq <= 48'sd0; g_mode <= gated;
                    v1 <= 1'b0; v2 <= 1'b0;
                    state <= S_GATE;
                end
                S_GATE: begin
                    if (ci == (short_len ? D/2-1 : D-1)) state <= S_GDRAIN;
                    ci <= ci + 1'b1;
                end
                S_GDRAIN: if (!v1 && !v2) state <= S_A;
                // ---- A = mean(g^2) + eps  (Q.26): the LN core minus mean --
                S_A: begin
                    A <= ($signed(sumsq) >>> (short_len ? A_SH-1 : A_SH))
                         + $signed(EPS_A);
                    state <= S_MSB;
                end
                S_MSB: begin
                    msb  <= msb_c;
                    half <= A_FRAC - $signed({1'b0, msb_c});
                    if (msb_c >= SEED_IDX_BITS)
                        seed_idx <= (A >> (msb_c - SEED_IDX_BITS)) & 6'h3F;
                    else
                        seed_idx <= (A << (SEED_IDX_BITS - msb_c)) & 6'h3F;
                    state <= S_SEED;
                end
                S_SEED: begin
                    seed_val <= seed_rom[seed_idx];
                    if (half[8] == 1'b0) begin
                        qsh  <= half >>> 1;
                        rbit <= half[0];
                    end else begin
                        qsh  <= -(((-half) + 1) >>> 1);
                        rbit <= half[0];
                    end
                    state <= S_SEED2;
                end
                S_SEED2: begin
                    seed_shifted = $signed({44'd0, seed_val}) <<< (Y_FRAC - SEED_OUT_FRAC);
                    if (qsh[8] == 1'b0) Y0 = seed_shifted <<< qsh;
                    else                Y0 = seed_shifted >>> (-qsh);
                    if (rbit) Y0 = (Y0 * $signed({33'd0, SQRT2Q15})) >>> 15;
                    Yr   <= Y0;
                    newt <= 0;
                    state <= S_NEWT;
                end
                // Newton y <- y*(1.5 - 0.5*a*y*y), 3 stages (layernorm.sv verbatim)
                S_NEWT:  begin yy  <= Yr * Yr;              state <= S_NEWTB; end
                S_NEWTB: begin ayy <= (A * yy) >>> A_FRAC;  state <= S_NEWTC; end
                S_NEWTC: begin
                    term = ONE_P5 - (ayy >>> 1);
                    ynew = (Yr * term) >>> (2*Y_FRAC);
                    Yr   <= ynew[63:0];
                    newt <= newt + 1'b1;
                    if (newt == 2'd1) begin
                        oi <= '0; ov1 <= 0; ov2 <= 0; ov3 <= 0;
                        state <= S_OUT;
                    end else state <= S_NEWT;
                end
                S_OUT: begin
                    // o-stage 0: fetch g / gamma
                    if (oi < (short_len ? D/2 : D)) begin
                        go1 <= gbuf[oi[LOG2D-1:0]];
                        gm1 <= grom[oi[LOG2D-1:0]];
                        oc1 <= oi[LOG2D-1:0];
                        ov1 <= 1'b1;
                        oi  <= oi + 1'b1;
                    end else ov1 <= 1'b0;
                    // o-stage 1: g * Yr
                    ov2 <= ov1; oc2 <= oc1; gm2 <= gm1;
                    if (ov1) prod_gy <= go1 * Yr;
                    // o-stage 2: * gamma
                    ov3 <= ov2; oc3 <= oc2;
                    if (ov2) prod_gyg <= prod_gy * gm2;
                    // o-stage 3: round >>> 40, sat, write
                    if (ov3) begin
                        obuf[oc3] <= osat;
                        if (oc3 == D-1) state <= S_DONE;
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase

            // gate pipeline (runs during S_GATE/S_GDRAIN)
            v1 <= issue; c1 <= ci;
            y1 <= yin[ci]; z1 <= zin[ci];
            v2 <= v1; c2 <= c1;
            prod2 <= y1 * silsel;                              // Q6.24
            if (v2) begin
                gbuf[c2] <= gw;
                sumsq    <= sumsq + gsq;
            end
        end
    end

endmodule

`default_nettype wire
