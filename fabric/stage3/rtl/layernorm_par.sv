// -----------------------------------------------------------------------------
// layernorm_par — throughput-optimized gamma-only LayerNorm over d=256.
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * gamma[i]
//   mean = (1/256) sum x ; var = (1/256) sum (x-mean)^2   (population/biased)
//
// Bit-true to the SAME integer reference as the serial rtl/layernorm.sv
// (fabric/stage3/run_layernorm._ln_int_quantized). The win over the serial core
// is the elimination of its 256-cycle second (d*d) pass and 256-cycle mean pass:
//
//   serial:  load(256) -> S_MEAN(1) -> S_SSQ(256) -> S_VAR(1) -> rsqrt(~6) -> out(256)
//            => ~264 cycles of NON-streaming compute between the two 256-cyc streams.
//   par   :  during the 256-cyc load we ALSO accumulate sum(x) and sum(x*x) via a
//            pipelined reduction; afterwards the only non-streaming work is the var
//            algebra (a few cycles) + rsqrt(~6).  => tens of cycles, not ~264.
//
// The key that makes this bit-EXACT (not approximate) to the serial core:
//   sum_i (x_i - mean)^2  ==  sum(x*x) - 2*mean*sum(x) + D*mean^2
// is an exact integer identity for the SAME floored integer mean = sum(x)>>8.
// Integer addition is associative, so an adder tree over x and x*x yields the
// identical wide-integer sums the serial accumulator produced. Verified 0 mismatch.
//
// Formats (pinned, identical to layernorm.sv):
//   x      signed Q6.25 (32-bit residual stream)   gamma  signed Q4.20
//   mean   = sum >>> 8           (Q6.25, sum is 40-bit)
//   ssq    = sum(x*x) - 2*mean*sum + D*mean*mean   (Q12.50, exact)
//   var    = ssq >>> 8           (Q12.50)
//   A      = (var >>> 24) + EPS  (Q.26)             rsqrt -> Q.26
//   y      = ((d*rsqrt)*gamma) >>> 49               (Q.22 output)
//
// rsqrt: 64-entry seed LUT (1/sqrt(mant), Q1.16) + 2 Newton-Raphson steps — the
// EXACT same arithmetic as the serial core (copied verbatim).
//
// Streaming protocol (identical to layernorm.sv): pulse `start` one cycle, then
// drive x_in/gamma_in with valid_in for 256 cycles. Results stream on y_out with
// y_valid for 256 cycles; `done` pulses at the end.
//
// Reduction implementation: a 5-stage pipelined adder tree reduces 8 lane-partial
// sums per "row" of the input. We stream the 256 elements through 8 lanes (32 rows)
// so that per-lane running sums are formed during the 256-cyc load with no extra
// passes, then an 8->1 pipelined tree (3 add levels, registered = O(log d) latency)
// finalizes sum_x and sum_xx. This is the pipelined-adder-tree reduction the
// throughput lever calls for, and is order-independent => bit-exact.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module layernorm_par (
    input  wire                clk,
    input  wire                rst,
    input  wire                start,        // pulse before streaming a new vector
    input  wire                valid_in,     // x_in/gamma_in valid (256 cycles)
    input  wire signed [31:0]  x_in,         // Q6.25
    input  wire signed [31:0]  gamma_in,     // Q4.20
    output reg                 y_valid,
    output reg  signed [63:0]  y_out,        // Q.22 (sign-extended into 64 bits)
    output reg                 done,
    // observability: latency in cycles from `start` to `done` (DERIVED, for the gate)
    output reg  [15:0]         lat_cycles
);
    localparam integer D        = 256;
    localparam integer LANES    = 8;            // 8 lanes -> 32 rows of 8
    localparam integer ROWS     = D / LANES;    // 32
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

    // ---- input storage (single-write mems) ---------------------------------
    reg signed [31:0] xmem [0:D-1];
    reg signed [31:0] gmem [0:D-1];
    reg [8:0] wptr;                                       // 0..256 write pointer

    // ---- seed ROM (64 x Q1.16) ---------------------------------------------
    (* rom_style = "block" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);

    // ---- per-lane running accumulators (built DURING the load stream) -------
    // 8 lanes; each lane sees ROWS=32 elements over the load. Per-lane sums are
    // independent partial sums; their order-independent total == serial sum.
    reg signed [39:0] lane_sx  [0:LANES-1];              // sum of x per lane (Q6.25)
    reg signed [63:0] lane_sxx [0:LANES-1];              // sum of x*x per lane (Q.50)

    integer li;
    reg [2:0]  lane_ptr;                                 // 0..7 round-robin lane select

    // ---- pipelined 8->1 adder tree (registered, 3 levels = O(log LANES)) ----
    // Level 0: 8 leaves (lane sums)        Level 1: 4 partials
    // Level 2: 2 partials                  Level 3: 1 total
    reg signed [42:0] t1_sx  [0:3];                      // +2 guard bits up the tree
    reg signed [43:0] t2_sx  [0:1];
    reg signed [44:0] t3_sx;
    reg signed [66:0] t1_sxx [0:3];
    reg signed [67:0] t2_sxx [0:1];
    reg signed [68:0] t3_sxx;
    reg [1:0] tree_stage;                                // 0..3 pipeline fill

    // ---- accumulators / var datapath ---------------------------------------
    reg signed [39:0] sum;                               // sum of X (Q6.25)
    reg signed [71:0] ssq;                               // sum of d*d (Q12.50)
    reg signed [39:0] mean;                              // Q6.25
    reg signed [71:0] var_q;                             // Q12.50
    reg [8:0] idx;                                       // output element counter

    // ---- rsqrt registers (verbatim from serial core) -----------------------
    reg signed [63:0] A;
    reg signed [63:0] Yr;
    reg [1:0] newt;
    reg [7:0] msb;
    reg [5:0] seed_idx;
    reg signed [63:0] Y0;

    reg signed [31:0] xrow;
    reg signed [31:0] grow;

    // ---- latency counter ----------------------------------------------------
    reg [15:0] cyc_cnt;
    reg        counting;

    // ---- FSM ----------------------------------------------------------------
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_LOAD   = 4'd1,   // accept 256 (x,gamma); per-lane sx/sxx accumulate
        S_TREE   = 4'd2,   // 3-level pipelined reduction of 8 lane partials
        S_VAR    = 4'd3,   // mean, ssq (algebraic), var, A
        S_MSB    = 4'd4,
        S_SEED   = 4'd5,
        S_SEED2  = 4'd6,
        S_NEWT   = 4'd7,
        S_OUT    = 4'd8,   // stream 256 outputs
        S_DONE   = 4'd9;
    reg [3:0] state;

    // combinational MSB index of A
    integer b;
    reg [7:0] msb_c;

    // seed exponent math
    reg signed [8:0]  Eexp;
    reg signed [8:0]  half;
    reg signed [8:0]  qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;
    reg [19:0]        seed_val;

    // Newton temporaries
    reg signed [127:0] yy;
    reg signed [127:0] ayy;
    reg signed [127:0] term;
    reg signed [191:0] ynew;

    // ssq algebra temporaries (wide, exact)
    reg signed [127:0] cterm;     // 2*mean*sum
    reg signed [127:0] msq;       // D*mean*mean

    // Output temporaries
    reg signed [95:0]  prod;
    reg signed [127:0] prod2;

    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; wptr <= 0; idx <= 0; y_valid <= 0; done <= 0;
            sum <= 0; ssq <= 0; newt <= 0; lane_ptr <= 0; tree_stage <= 0;
            cyc_cnt <= 0; counting <= 0; lat_cycles <= 0;
            for (li = 0; li < LANES; li = li + 1) begin
                lane_sx[li]  <= 0;
                lane_sxx[li] <= 0;
            end
        end else begin
            y_valid <= 1'b0;
            done    <= 1'b0;
            if (counting) cyc_cnt <= cyc_cnt + 1'b1;

            case (state)
                // -------------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        wptr <= 0; lane_ptr <= 0; newt <= 0;
                        for (li = 0; li < LANES; li = li + 1) begin
                            lane_sx[li]  <= 0;
                            lane_sxx[li] <= 0;
                        end
                        cyc_cnt  <= 0;
                        counting <= 1'b1;
                        state    <= S_LOAD;
                    end
                end
                // -------------------------------------------------------------
                // Load: store (x,gamma); fold x into its lane's running sx/sxx.
                // x*x done once per cycle (one DSP). No second pass over data.
                S_LOAD: begin
                    if (valid_in) begin
                        xmem[wptr] <= x_in;
                        gmem[wptr] <= gamma_in;
                        // sign-extend x to 40 / sxx product is 64-bit exact
                        lane_sx[lane_ptr]  <= lane_sx[lane_ptr]
                                              + {{8{x_in[31]}}, x_in};
                        lane_sxx[lane_ptr] <= lane_sxx[lane_ptr]
                                              + ($signed(x_in) * $signed(x_in));
                        lane_ptr <= lane_ptr + 1'b1;       // wraps 7->0 (3-bit)
                        wptr     <= wptr + 1'b1;
                        if (wptr == D-1) begin
                            tree_stage <= 0;
                            state <= S_TREE;
                        end
                    end
                end
                // -------------------------------------------------------------
                // Pipelined 8->1 reduction, 3 registered add-levels (O(log LANES)).
                S_TREE: begin
                    case (tree_stage)
                        2'd0: begin
                            // level1: 8 -> 4
                            t1_sx[0]  <= lane_sx[0]  + lane_sx[1];
                            t1_sx[1]  <= lane_sx[2]  + lane_sx[3];
                            t1_sx[2]  <= lane_sx[4]  + lane_sx[5];
                            t1_sx[3]  <= lane_sx[6]  + lane_sx[7];
                            t1_sxx[0] <= lane_sxx[0] + lane_sxx[1];
                            t1_sxx[1] <= lane_sxx[2] + lane_sxx[3];
                            t1_sxx[2] <= lane_sxx[4] + lane_sxx[5];
                            t1_sxx[3] <= lane_sxx[6] + lane_sxx[7];
                            tree_stage <= 2'd1;
                        end
                        2'd1: begin
                            // level2: 4 -> 2
                            t2_sx[0]  <= t1_sx[0]  + t1_sx[1];
                            t2_sx[1]  <= t1_sx[2]  + t1_sx[3];
                            t2_sxx[0] <= t1_sxx[0] + t1_sxx[1];
                            t2_sxx[1] <= t1_sxx[2] + t1_sxx[3];
                            tree_stage <= 2'd2;
                        end
                        default: begin
                            // level3: 2 -> 1 ; commit to sum/ssq accumulators
                            t3_sx  <= t2_sx[0]  + t2_sx[1];
                            t3_sxx <= t2_sxx[0] + t2_sxx[1];
                            sum    <= t2_sx[0]  + t2_sx[1];
                            // park the full sum(x*x) in ssq (Q.50) for the algebra
                            ssq    <= t2_sxx[0] + t2_sxx[1];
                            state  <= S_VAR;
                        end
                    endcase
                end
                // -------------------------------------------------------------
                // Algebraic var: ssq_centered = sum(x*x) - 2*mean*sum + D*mean^2,
                // with the SAME floored integer mean = sum>>>8 (exact identity).
                // `ssq` currently holds sum(x*x) (Q.50) from the tree.
                S_VAR: begin
                    mean <= sum >>> 8;                            // /256, Q6.25
                    cterm = ($signed(sum >>> 8) * $signed(sum)) <<< 1;     // 2*mean*sum
                    msq   = D * ($signed(sum >>> 8) * $signed(sum >>> 8)); // D*mean^2
                    ssq  <= $signed(ssq) - cterm + msq;                    // centered Q.50
                    state <= S_MSB;
                end
                // -------------------------------------------------------------
                // var = ssq>>8 ; A = (var>>24)+eps ; then MSB encode (one cycle late
                // is fine — A is registered before S_MSB consumes msb_c next state).
                S_MSB: begin
                    var_q <= ssq >>> 8;
                    A <= ($signed(ssq >>> 8) >>> (VAR_FRAC - A_FRAC)) + $signed(EPS_A);
                    state <= S_SEED;       // msb_c is combinational off A; sample in S_SEED
                end
                // -------------------------------------------------------------
                S_SEED: begin
                    msb <= msb_c;
                    Eexp <= $signed({1'b0, msb_c}) - A_FRAC;
                    half <= A_FRAC - $signed({1'b0, msb_c});
                    if (msb_c >= SEED_IDX_BITS)
                        seed_idx <= (A >> (msb_c - SEED_IDX_BITS)) & 6'h3F;
                    else
                        seed_idx <= (A << (SEED_IDX_BITS - msb_c)) & 6'h3F;
                    state <= S_SEED2;
                end
                // -------------------------------------------------------------
                S_SEED2: begin
                    seed_val <= seed_rom[seed_idx];              // registered ROM read
                    if (half[8] == 1'b0) begin
                        qsh  <= half >>> 1;
                        rbit <= half[0];
                    end else begin
                        qsh  <= -(((-half) + 1) >>> 1);
                        rbit <= half[0];
                    end
                    state <= S_NEWT;
                end
                // -------------------------------------------------------------
                // First S_NEWT cycle: build Y0 from the (now registered) seed_val,
                // then perform the two Newton steps on subsequent cycles.
                S_NEWT: begin
                    if (newt == 2'd0) begin
                        seed_shifted = $signed({44'd0, seed_val})
                                       <<< (Y_FRAC - SEED_OUT_FRAC);
                        if (qsh[8] == 1'b0) Y0 = seed_shifted <<< qsh;
                        else                Y0 = seed_shifted >>> (-qsh);
                        if (rbit) Y0 = (Y0 * $signed({33'd0, SQRT2Q15})) >>> 15;
                        Yr   <= Y0;
                        newt <= 2'd1;
                    end else begin
                        yy   = Yr * Yr;
                        ayy  = (A * yy) >>> A_FRAC;
                        term = ONE_P5 - (ayy >>> 1);
                        ynew = (Yr * term) >>> (2*Y_FRAC);
                        Yr   <= ynew[63:0];
                        newt <= newt + 1'b1;
                        if (newt == 2'd2) begin              // after 2 Newton updates
                            idx <= 0;
                            state <= S_OUT;
                        end
                    end
                end
                // -------------------------------------------------------------
                S_OUT: begin
                    xrow = xmem[idx];
                    grow = gmem[idx];
                    prod  = ($signed({{8{xrow[31]}}, xrow}) - mean) * Yr;
                    prod2 = prod * $signed({{96{grow[31]}}, grow});
                    y_out   <= prod2 >>> OUT_SH;
                    y_valid <= 1'b1;
                    idx <= idx + 1'b1;
                    if (idx == D-1) state <= S_DONE;
                end
                // -------------------------------------------------------------
                S_DONE: begin
                    done       <= 1'b1;
                    lat_cycles <= cyc_cnt;
                    counting   <= 1'b0;
                    state      <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
