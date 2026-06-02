// -----------------------------------------------------------------------------
// layernorm — gamma-only LayerNorm over a d=256 vector, fabric fixed-point.
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * gamma[i]
//   mean = (1/256) sum x ; var = (1/256) sum (x-mean)^2   (population/biased)
//
// Formats (pinned, bit-true to fabric/stage3/run_layernorm.ln_int):
//   x      signed Q6.25 (32-bit residual stream)   gamma  signed Q4.20
//   mean   = sum >>> 8           (Q6.25, sum is 40-bit)
//   var    = (sum d*d) >>> 8     (Q12.50, d*d is 64-bit)
//   A      = (var >>> 24) + EPS  (Q.26)             rsqrt -> Q.26
//   y      = ((d*rsqrt)*gamma) >>> 49               (Q.22 output)
//
// rsqrt: 64-entry seed LUT (1/sqrt(mant), Q1.16) keyed on the 6 mantissa bits below
// the MSB of A; exponent handled by shifts (+ a sqrt2 multiply when E is odd); then
// 2 Newton-Raphson steps  y <- y*(1.5 - 0.5*a*y*y), all integer.
//
// Streaming protocol: pulse `start` for one cycle, then drive x_in/gamma_in with
// valid_in for 256 consecutive cycles (i = 0..255). After an internal compute, the
// 256 results stream out on y_out with y_valid (i = 0..255). `done` pulses at the end.
// Single-write memories; single registered ROM read; plain procedural SV.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module layernorm (
    input  wire                clk,
    input  wire                rst,
    input  wire                start,        // pulse before streaming a new vector
    input  wire                valid_in,     // x_in/gamma_in valid (256 cycles)
    input  wire signed [31:0]  x_in,         // Q6.25
    input  wire signed [31:0]  gamma_in,     // Q4.20 (value fits; sign-extended)
    output reg                 y_valid,
    output reg  signed [63:0]  y_out,        // Q.22 (sign-extended into 64 bits)
    output reg                 done
);
    localparam integer D        = 256;
    localparam integer QX       = 25;
    localparam integer G_FRAC   = 20;
    localparam integer A_FRAC   = 26;
    localparam integer Y_FRAC   = 26;
    localparam integer OUT_FRAC = 22;
    localparam integer VAR_FRAC = 50;
    localparam integer SEED_IDX_BITS = 6;
    localparam integer SEED_OUT_FRAC = 16;
    localparam integer OUT_SH   = (QX + Y_FRAC + G_FRAC) - OUT_FRAC;   // 49
    // eps = round(1e-5 * 2^26) = 671
    localparam [63:0] EPS_A   = 64'd671;
    localparam [31:0] SQRT2Q15 = 32'd46341;       // round(sqrt(2)*2^15)
    // 1.5 in Q(2*Y_FRAC) = 3 * 2^(2*Y_FRAC-1)
    localparam [127:0] ONE_P5 = 128'd3 <<< (2*Y_FRAC - 1);

    // ---- input storage ------------------------------------------------------
    reg signed [31:0] xmem [0:D-1];
    reg signed [31:0] gmem [0:D-1];
    reg [8:0] wptr;                              // 0..256 write pointer

    // ---- seed ROM (64 x Q1.16) ---------------------------------------------
    (* rom_style = "block" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);

    // ---- accumulators -------------------------------------------------------
    reg signed [39:0] sum;                       // sum of X  (Q6.25)
    reg signed [71:0] ssq;                       // sum of d*d (Q12.50)
    reg signed [39:0] mean;                      // Q6.25
    reg signed [71:0] var_q;                     // Q12.50
    reg [8:0] idx;                               // pass element counter 0..256

    // ---- rsqrt registers ----------------------------------------------------
    reg signed [63:0] A;                         // Q.26 (positive)
    reg signed [63:0] Yr;                        // Q.26 rsqrt result
    reg [1:0] newt;                              // Newton iteration counter
    reg [7:0] msb;                               // MSB index of A
    reg [5:0] seed_idx;
    reg signed [63:0] Y0;

    // ---- output pass --------------------------------------------------------
    reg signed [31:0] xrow;                      // plain vector copy of xmem[idx]
    reg signed [31:0] grow;                      // plain vector copy of gmem[idx]

    // ---- FSM ----------------------------------------------------------------
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_LOAD   = 4'd1,   // accept 256 (x,gamma); accumulate sum
        S_MEAN   = 4'd2,   // mean = sum>>8
        S_SSQ    = 4'd3,   // pass: accumulate d*d
        S_VAR    = 4'd4,   // var = ssq>>8 ; A = (var>>24)+eps
        S_MSB    = 4'd5,   // priority-encode MSB(A)
        S_SEED   = 4'd6,   // seed LUT read + exponent shift -> Y0
        S_SEED2  = 4'd7,   // apply seed ROM value (registered read) -> Y0
        S_NEWT   = 4'd8,   // Newton stage A:  yy  = Yr*Yr
        S_OUT    = 4'd9,   // stream 256 outputs
        S_DONE   = 4'd10,
        S_NEWTB  = 4'd11,  // Newton stage B:  ayy = (A*yy)>>>A_FRAC
        S_NEWTC  = 4'd12;  // Newton stage C:  Yr <- (Yr*term)>>>(2*Y_FRAC)
    reg [3:0] state;

    // combinational MSB index of A (A is positive, <2^31 or so)
    integer b;
    reg [7:0] msb_c;
    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    // exponent math for the seed:  E = msb - A_FRAC ; half = -E ; Y0 = seed<<(Y_FRAC-16)
    // then * 2^(half/2): half = 2*q + r.  Done with plain shifts; r==1 -> *sqrt2.
    reg signed [8:0]  Eexp;
    reg signed [8:0]  half;
    reg signed [8:0]  qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;              // seed << (Y_FRAC-16)
    reg [19:0]        seed_val;                  // registered ROM read

    // Newton temporaries (wide)
    reg signed [127:0] yy;
    reg signed [127:0] ayy;
    reg signed [127:0] term;
    reg signed [191:0] ynew;

    // Output temporaries (wide)
    reg signed [95:0]  prod;
    reg signed [127:0] prod2;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; wptr <= 0; idx <= 0; y_valid <= 0; done <= 0;
            sum <= 0; ssq <= 0; newt <= 0;
        end else begin
            y_valid <= 1'b0;
            done    <= 1'b0;
            case (state)
                // -------------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        wptr <= 0; sum <= 0;
                        state <= S_LOAD;
                    end
                end
                // -------------------------------------------------------------
                S_LOAD: begin
                    if (valid_in) begin
                        xmem[wptr] <= x_in;
                        gmem[wptr] <= gamma_in;
                        sum  <= sum + {{8{x_in[31]}}, x_in};     // sign-extend to 40
                        wptr <= wptr + 1'b1;
                        if (wptr == D-1) state <= S_MEAN;
                    end
                end
                // -------------------------------------------------------------
                S_MEAN: begin
                    mean <= sum >>> 8;                            // /256, Q6.25
                    ssq  <= 0;
                    idx  <= 0;
                    state <= S_SSQ;
                end
                // -------------------------------------------------------------
                S_SSQ: begin
                    // copy the array element to a plain vector first (iverilog-safe),
                    // then d = X - mean (Q6.25), accumulate d*d (Q12.50).
                    xrow = xmem[idx];
                    ssq <= ssq + ($signed({{8{xrow[31]}}, xrow}) - mean)
                                 * ($signed({{8{xrow[31]}}, xrow}) - mean);
                    idx <= idx + 1'b1;
                    if (idx == D-1) state <= S_VAR;
                end
                // -------------------------------------------------------------
                S_VAR: begin
                    var_q <= ssq >>> 8;                           // /256, Q12.50
                    // A = (var >>> (VAR_FRAC-A_FRAC)) + eps  = (var>>>24)+eps  (Q.26)
                    A <= ($signed(ssq >>> 8) >>> (VAR_FRAC - A_FRAC)) + $signed(EPS_A);
                    state <= S_MSB;
                end
                // -------------------------------------------------------------
                S_MSB: begin
                    msb <= msb_c;
                    // E = msb - A_FRAC ; half = -E = A_FRAC - msb
                    Eexp <= $signed({1'b0, msb_c}) - A_FRAC;
                    half <= A_FRAC - $signed({1'b0, msb_c});
                    // seed index = 6 mantissa bits below msb
                    if (msb_c >= SEED_IDX_BITS)
                        seed_idx <= (A >> (msb_c - SEED_IDX_BITS)) & 6'h3F;
                    else
                        seed_idx <= (A << (SEED_IDX_BITS - msb_c)) & 6'h3F;
                    state <= S_SEED;
                end
                // -------------------------------------------------------------
                S_SEED: begin
                    seed_val <= seed_rom[seed_idx];              // registered ROM read
                    // half = 2*qsh + rbit ; floor division toward -inf
                    if (half[8] == 1'b0) begin                   // half >= 0
                        qsh  <= half >>> 1;
                        rbit <= half[0];
                    end else begin                               // half < 0
                        qsh  <= -(((-half) + 1) >>> 1);
                        rbit <= half[0];                         // (-half)&1 == half&1
                    end
                    state <= S_SEED2;
                end
                // -------------------------------------------------------------
                S_SEED2: begin
                    // seed_shifted = seed_val << (Y_FRAC-16) = seed_val << 10
                    seed_shifted = $signed({44'd0, seed_val}) <<< (Y_FRAC - SEED_OUT_FRAC);
                    // apply 2^(qsh)
                    if (qsh[8] == 1'b0) Y0 = seed_shifted <<< qsh;
                    else                Y0 = seed_shifted >>> (-qsh);
                    // r==1 -> * sqrt2 (Q15)
                    if (rbit) Y0 = (Y0 * $signed({33'd0, SQRT2Q15})) >>> 15;
                    Yr   <= Y0;
                    newt <= 0;
                    state <= S_NEWT;
                end
                // -------------------------------------------------------------
                // Newton iteration y <- y*(1.5 - 0.5*a*y*y), PIPELINED into 3 single-
                // multiply stages so the old triple-chained-multiply (the 54-level DSP
                // cascade that capped Fmax ~50 MHz) becomes one multiply per stage. Yr is
                // not updated until stage C, so yy/ayy use the same old Yr and A -> the
                // computed value is BIT-IDENTICAL to the 1-cycle version; only +2 cyc/iter.
                S_NEWT: begin                                   // stage A
                    yy <= Yr * Yr;                              // Q(2*Y_FRAC)
                    state <= S_NEWTB;
                end
                S_NEWTB: begin                                  // stage B
                    ayy <= (A * yy) >>> A_FRAC;                 // Q(2*Y_FRAC)
                    state <= S_NEWTC;
                end
                S_NEWTC: begin                                  // stage C
                    term = ONE_P5 - (ayy >>> 1);                // 1.5 - 0.5*a*y*y
                    ynew = (Yr * term) >>> (2*Y_FRAC);         // back to Q.Y_FRAC
                    Yr   <= ynew[63:0];
                    newt <= newt + 1'b1;
                    if (newt == 2'd1) begin idx <= 0; state <= S_OUT; end
                    else state <= S_NEWT;
                end
                // -------------------------------------------------------------
                S_OUT: begin
                    // copy array elements to plain vectors, then operate (iverilog-safe)
                    xrow = xmem[idx];
                    grow = gmem[idx];
                    // d = X - mean (Q6.25)
                    // prod  = d * Yr            -> Q(QX+Y_FRAC)
                    prod  = ($signed({{8{xrow[31]}}, xrow}) - mean) * Yr;
                    // prod2 = prod * gamma      -> Q(QX+Y_FRAC+G_FRAC)
                    prod2 = prod * $signed({{96{grow[31]}}, grow});
                    y_out   <= prod2 >>> OUT_SH;                 // Q.OUT_FRAC
                    y_valid <= 1'b1;
                    idx <= idx + 1'b1;
                    if (idx == D-1) state <= S_DONE;
                end
                // -------------------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
