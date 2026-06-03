// -----------------------------------------------------------------------------
// layernorm_vec — P-WIDE-I/O gamma-only LayerNorm over d=256 (the 10k-datapath block).
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * gamma[i]
//
// Bit-EXACT to the SAME integer reference as rtl/layernorm.sv and run_layernorm.ln_int.
// The difference from layernorm.sv / layernorm_par is the I/O width: P (x,gamma) accepted
// per cycle and P y emitted per cycle (D/P rows instead of D cycles each way). The variance
// is built ALGEBRAICALLY during the load (no second pass), the rsqrt Newton is pipelined
// (3 single-multiply stages), and the output is pipelined (2 stages) — so the deep
// combinational paths that capped Fmax in the scalar core are broken here too.
//
// Bit-exactness rests on three integer identities (all exact, no approximation):
//   * sum and sum(x*x) accumulated P/cycle == the serial 1/cycle sums (integer add assoc.)
//   * sum_i (x_i-mean)^2 == sum(x*x) - 2*mean*sum + D*mean^2  for mean = sum>>>8 (floored)
//   * the rsqrt seed+2-Newton and the (d*Yr*gamma)>>>49 output are copied verbatim.
//
// Formats (pinned, identical to layernorm.sv):
//   x  signed Q6.25 (32b)   gamma signed Q4.20   mean=sum>>>8 (Q6.25)
//   ssq = sum(xx) - 2*mean*sum + D*mean^2 (Q12.50)   var=ssq>>>8   A=(var>>>24)+eps (Q.26)
//   rsqrt -> Q.26     y = ((d*rsqrt)*gamma) >>> 49  (Q.22)
//
// Protocol: pulse `start`; then drive x_in/gamma_in (each P*32-bit packed, lane k = [32k+:32])
// with valid_in for D/P cycles. Results stream on y_out (P*64-bit packed) with y_valid for
// D/P cycles; `done` pulses at the end. iverilog-2012 safe (plain-vector copies before any
// indexed read of an unpacked element; packed-vector +: part-selects only).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module layernorm_vec #(
    parameter integer P = 8                 // vector lanes (D must be divisible by P)
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  start,         // pulse before streaming a new vector
    input  wire                  valid_in,      // x_in/gamma_in valid (D/P cycles)
    input  wire [P*32-1:0]       x_in,          // P x Q6.25  (lane k = [32k +: 32])
    input  wire [P*32-1:0]       gamma_in,      // P x Q4.20
    output reg                   y_valid,
    output reg  [P*64-1:0]       y_out,         // P x Q.22 (each sign-extended into 64b)
    output reg                   done
);
    localparam integer D        = 256;
    localparam integer ROWS     = D / P;        // rows of P
    localparam integer QX       = 25;
    localparam integer G_FRAC   = 20;
    localparam integer A_FRAC   = 26;
    localparam integer Y_FRAC   = 26;
    localparam integer OUT_FRAC = 22;
    localparam integer VAR_FRAC = 50;
    localparam integer SEED_IDX_BITS = 6;
    localparam integer SEED_OUT_FRAC = 16;
    localparam integer OUT_SH   = (QX + Y_FRAC + G_FRAC) - OUT_FRAC;   // 49
    localparam [63:0] EPS_A   = 64'd671;                  // round(1e-5 * 2^26)
    localparam [31:0] SQRT2Q15 = 32'd46341;               // round(sqrt(2)*2^15)
    localparam [127:0] ONE_P5 = 128'd3 <<< (2*Y_FRAC - 1);

    // ---- P-banked input storage: xbank[p][row], gbank[p][row] -----------------
    reg signed [31:0] xbank [0:P-1][0:ROWS-1];
    reg signed [31:0] gbank [0:P-1][0:ROWS-1];
    reg [$clog2(ROWS+1)-1:0] wptr;              // 0..ROWS write/row pointer

    // ---- seed ROM (64 x Q1.16) -----------------------------------------------
    (* rom_style = "block" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);

    // ---- accumulators (built DURING the P-wide load) -------------------------
    reg signed [39:0]  sum;                     // sum of X (Q6.25)
    reg signed [71:0]  sumxx;                   // sum of x*x (Q.50)
    reg signed [39:0]  mean;                    // Q6.25
    reg signed [71:0]  ssq;                     // centered sum (Q.50)
    reg signed [71:0]  var_q;                   // Q12.50
    reg [$clog2(ROWS+1)-1:0] ridx, oidx;        // output row counters

    // ---- rsqrt registers (verbatim from the scalar core) ---------------------
    reg signed [63:0] A, Yr, Y0;
    reg [1:0]  newt;
    reg [7:0]  msb;
    reg [5:0]  seed_idx;
    reg signed [8:0]  Eexp, half, qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;
    reg [19:0]        seed_val;
    // Newton pipeline regs (single multiply per stage)
    reg signed [127:0] yy, ayy, term;
    reg signed [191:0] ynew;
    // var algebra temporaries
    reg signed [127:0] cterm, msq;

    // ---- load reduction: per-cycle P-input partial sums (combinational) -------
    integer lp;
    reg signed [39:0]  psum;                    // sum of the P lanes this row
    reg signed [71:0]  psumxx;                  // sum of x*x of the P lanes this row
    reg signed [31:0]  xl;                       // plain-vector lane copy
    always @(*) begin
        psum   = 40'sd0;
        psumxx = 72'sd0;
        for (lp = 0; lp < P; lp = lp + 1) begin
            xl     = x_in[lp*32 +: 32];          // packed-vector part-select (safe)
            psum   = psum   + $signed({{8{xl[31]}}, xl});
            psumxx = psumxx + ($signed(xl) * $signed(xl));
        end
    end

    // ---- combinational MSB index of A ----------------------------------------
    integer b;
    reg [7:0] msb_c;
    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    // ---- output pipeline (2 stages): stage1 prod=(x-mean)*Yr ; stage2 *gamma --
    reg signed [95:0]  prod   [0:P-1];
    reg signed [31:0]  grow_r [0:P-1];
    reg                s1v;
    reg signed [31:0]  xo, go;                   // plain-vector copies
    reg signed [95:0]  pj;
    reg signed [127:0] prod2;

    // ---- FSM ------------------------------------------------------------------
    localparam [3:0]
        S_IDLE=0, S_LOAD=1, S_VAR=2, S_MSB=3, S_SEED=4, S_SEED2=5,
        S_NEWT0=6, S_NEWTA=7, S_NEWTB=8, S_NEWTC=9, S_OUT=10, S_DONE=11;
    reg [3:0] state;
    integer wp;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; wptr <= 0; ridx <= 0; oidx <= 0;
            y_valid <= 1'b0; done <= 1'b0; s1v <= 1'b0;
            sum <= 0; sumxx <= 0; newt <= 0;
        end else begin
            y_valid <= 1'b0;
            done    <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        wptr <= 0; sum <= 0; sumxx <= 0; newt <= 0;
                        state <= S_LOAD;
                    end
                end
                // ---- P-wide load: store P lanes, fold the P partial sums in -----
                S_LOAD: begin
                    if (valid_in) begin
                        for (wp = 0; wp < P; wp = wp + 1) begin
                            xbank[wp][wptr] <= x_in[wp*32 +: 32];
                            gbank[wp][wptr] <= gamma_in[wp*32 +: 32];
                        end
                        sum   <= sum   + psum;
                        sumxx <= sumxx + psumxx;
                        wptr  <= wptr + 1'b1;
                        if (wptr == ROWS-1) state <= S_VAR;
                    end
                end
                // ---- algebraic centered sum (exact identity) -------------------
                S_VAR: begin
                    mean  <= sum >>> 8;                                  // /256, Q6.25
                    cterm = ($signed(sum >>> 8) * $signed(sum)) <<< 1;   // 2*mean*sum
                    msq   = D * ($signed(sum >>> 8) * $signed(sum >>> 8)); // D*mean^2
                    ssq   <= $signed(sumxx) - cterm + msq;               // centered Q.50
                    state <= S_MSB;
                end
                S_MSB: begin
                    var_q <= ssq >>> 8;                                  // Q12.50
                    A <= ($signed(ssq >>> 8) >>> (VAR_FRAC - A_FRAC)) + $signed(EPS_A);
                    state <= S_SEED;       // msb_c combinational off A; sampled next state
                end
                S_SEED: begin
                    msb  <= msb_c;
                    Eexp <= $signed({1'b0, msb_c}) - A_FRAC;
                    half <= A_FRAC - $signed({1'b0, msb_c});
                    if (msb_c >= SEED_IDX_BITS)
                        seed_idx <= (A >> (msb_c - SEED_IDX_BITS)) & 6'h3F;
                    else
                        seed_idx <= (A << (SEED_IDX_BITS - msb_c)) & 6'h3F;
                    state <= S_SEED2;
                end
                S_SEED2: begin
                    seed_val <= seed_rom[seed_idx];
                    if (half[8] == 1'b0) begin qsh <= half >>> 1; rbit <= half[0]; end
                    else begin qsh <= -(((-half) + 1) >>> 1); rbit <= half[0]; end
                    newt  <= 0;
                    state <= S_NEWT0;
                end
                // ---- build Y0 from the registered seed, then 2 pipelined Newtons --
                S_NEWT0: begin
                    seed_shifted = $signed({44'd0, seed_val}) <<< (Y_FRAC - SEED_OUT_FRAC);
                    if (qsh[8] == 1'b0) Y0 = seed_shifted <<< qsh;
                    else                Y0 = seed_shifted >>> (-qsh);
                    if (rbit) Y0 = (Y0 * $signed({33'd0, SQRT2Q15})) >>> 15;
                    Yr    <= Y0;
                    state <= S_NEWTA;
                end
                S_NEWTA: begin yy  <= Yr * Yr;               state <= S_NEWTB; end  // stage A
                S_NEWTB: begin ayy <= (A * yy) >>> A_FRAC;   state <= S_NEWTC; end  // stage B
                S_NEWTC: begin                                                       // stage C
                    term = ONE_P5 - (ayy >>> 1);
                    ynew = (Yr * term) >>> (2*Y_FRAC);
                    Yr   <= ynew[63:0];
                    newt <= newt + 1'b1;
                    if (newt == 2'd1) begin ridx <= 0; oidx <= 0; s1v <= 1'b0; state <= S_OUT; end
                    else state <= S_NEWTA;
                end
                // ---- P-wide pipelined output stream ----------------------------
                S_OUT: begin
                    // stage 1: P prods (x-mean)*Yr for row ridx
                    if (ridx < ROWS) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            xo = xbank[lp][ridx];               // plain-vector copy (iverilog-safe)
                            go = gbank[lp][ridx];
                            prod[lp]   <= ($signed({{8{xo[31]}}, xo}) - mean) * Yr;
                            grow_r[lp] <= go;
                        end
                        ridx <= ridx + 1'b1;
                        s1v  <= 1'b1;
                    end else s1v <= 1'b0;
                    // stage 2: P y = (prod*gamma)>>>OUT_SH for the row latched last cycle
                    if (s1v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            pj    = prod[lp];                   // plain-vector copies
                            go    = grow_r[lp];
                            prod2 = pj * $signed({{96{go[31]}}, go});
                            y_out[lp*64 +: 64] <= prod2 >>> OUT_SH;
                        end
                        y_valid <= 1'b1;
                        if (oidx == ROWS-1) state <= S_DONE;
                        oidx <= oidx + 1'b1;
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
