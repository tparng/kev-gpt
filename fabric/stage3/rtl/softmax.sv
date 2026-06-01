// -----------------------------------------------------------------------------
// softmax — fabric softmax over a causal row of T scores (T up to 256).
//   probs[j] = exp(s[j]-max) / sum_k exp(s[k]-max)
//
// Datapath (pinned, bit-true to run_softmax.int_softmax):
//   score in   : signed Q8.8 (16-bit), real model range ~+-14  -> headroom +-128
//   z          : clip(s-max, -16, 0) in Q8.8 -> z-index = (-zq) in 0..4096
//   exp LUT    : 4097 entries, exp(-i/256) as Q1.20 (unsigned, LUT[0]=2^20)
//                init from exp_lut.mem ($readmemh, single registered read -> BRAM ROM)
//   sum        : 29-bit accumulator of the T exp values (Q1.20)
//   reciprocal : r = floor(2^40 / sum), 21-bit, by restoring long division (41 cyc)
//   probs out  : (e[j]*r) >> 20  -> Q1.20 (21-bit)
//
// Three passes over the on-chip exp buffer (expmem, 256 x 21-bit, single-write):
//   PASS1  load: stream T scores in; track running max; store each score in scoremem.
//   PASS2  exp : for j in 0..T-1: zq=clip(score[j]-max,-4096,0); e=LUT[-zq]; store in
//                expmem; sum += e.
//   RECIP      : r = floor(2^40 / sum) via 41-cycle restoring division.
//   PASS3 norm : for j in 0..T-1: prob[j] = (expmem[j]*r) >> 20; stream out.
//
// Plain procedural SV: one always @(posedge clk) FSM, single-write memories, a copy-
// to-plain-vector before any indexed read (iverilog X-on-unpacked-element rule).
// Latency is data dependent (~3T + 41 + a few); the host drives `start`/`in_valid`
// and reads `out_valid`/`prob`. Bit-true gate in run_softmax.py.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module softmax #(
    parameter integer TMAX = 256
) (
    input  wire                clk,
    input  wire                rst,        // sync reset
    input  wire                start,      // pulse with t_count valid; begins PASS1
    input  wire [8:0]          t_count,    // number of valid scores this row (1..256)
    input  wire                in_valid,   // a score word is presented on `score`
    input  wire signed [15:0]  score,      // Q8.8 score (host streams T of them)
    output reg                 in_ready,   // high while PASS1 is consuming scores
    output reg                 out_valid,  // a prob word is presented on `prob`
    output reg  [20:0]         prob,       // Q1.20 probability (unsigned)
    output reg                 done        // pulses 1 cycle when the row is complete
);
    // exp LUT: 4097 entries Q1.20 (21-bit). idx 0..4096 = exp(-idx/256).
    (* rom_style = "block" *) reg [20:0] explut [0:4096];
    initial $readmemh("exp_lut.mem", explut);

    // on-chip buffers (single-write each)
    reg signed [15:0] scoremem [0:TMAX-1];   // raw scores (Q8.8)
    reg [20:0]        expmem   [0:TMAX-1];    // exp values (Q1.20)

    // FSM states
    localparam [2:0] S_IDLE  = 3'd0,
                     S_LOAD  = 3'd1,   // PASS1: consume scores, track max
                     S_EXP   = 3'd2,   // PASS2: exp + accumulate sum
                     S_RECIP = 3'd3,   // long division r = 2^40 / sum
                     S_NORM  = 3'd4,   // PASS3: prob = (e*r)>>20
                     S_DONE  = 3'd5;
    reg [2:0] state;

    reg [8:0]  n;            // total valid scores (latched from t_count)
    reg [8:0]  i;            // load index / general counter
    reg signed [15:0] runmax;

    // PASS2 pipeline regs
    reg [8:0]  j;            // exp/norm element index
    reg [1:0]  exp_ph;       // sub-phase within EXP for an element (read->compute->store)
    reg signed [15:0] sc_v;  // plain-vector copy of scoremem[j]
    reg [28:0] sum;          // Q1.20 sum accumulator (<= 256*2^20, 29 bits)

    // reciprocal (restoring division): quo = floor(2^40 / sum)
    reg [40:0] rem;          // running remainder (41-bit)
    reg [20:0] quo;          // quotient r (21-bit: r in [2^12, 2^20])
    reg [5:0]  div_bit;      // counts 40..0
    reg [20:0] recip;        // latched r for PASS3

    // PASS3 pipeline
    reg [1:0]  norm_ph;
    reg [20:0] e_v;          // plain-vector copy of expmem[j]
    reg [41:0] er;           // e*r (<= 2^40)

    // helpers computed combinationally inside the FSM (kept as wires for clarity)
    wire signed [16:0] zq_full = sc_v - runmax;            // <= 0 for valid (max>=sc)
    // clip to [-4096, 0]; index = -zq (0..4096)
    wire [12:0] zidx = (zq_full <= -17'sd4096) ? 13'd4096
                     : (zq_full >= 17'sd0)     ? 13'd0
                     : (13'd0 - zq_full[12:0]);            // -zq, fits 0..4096

    always @(posedge clk) begin
        // defaults (pulsed signals)
        out_valid <= 1'b0;
        done      <= 1'b0;

        if (rst) begin
            state    <= S_IDLE;
            in_ready <= 1'b0;
            sum      <= 29'd0;
            i        <= 9'd0;
            j        <= 9'd0;
        end else begin
            case (state)
                // ----------------------------------------------------------------
                S_IDLE: begin
                    in_ready <= 1'b0;
                    if (start) begin
                        n        <= t_count;
                        i        <= 9'd0;
                        runmax   <= 16'sh8000;   // -32768, lowest Q8.8
                        sum      <= 29'd0;
                        in_ready <= 1'b1;
                        state    <= S_LOAD;
                    end
                end
                // ---------------- PASS1: load scores, running max ----------------
                S_LOAD: begin
                    in_ready <= 1'b1;
                    if (in_valid) begin
                        scoremem[i] <= score;
                        if (score > runmax) runmax <= score;
                        if (i == n - 1) begin
                            in_ready <= 1'b0;
                            i        <= 9'd0;
                            j        <= 9'd0;
                            exp_ph   <= 2'd0;
                            state    <= S_EXP;
                        end else begin
                            i <= i + 9'd1;
                        end
                    end
                end
                // ---------------- PASS2: exp(score-max), accumulate --------------
                S_EXP: begin
                    case (exp_ph)
                        2'd0: begin                  // read scoremem[j] to plain vector
                            sc_v   <= scoremem[j];
                            exp_ph <= 2'd1;
                        end
                        2'd1: begin                  // LUT read (zidx now valid from sc_v)
                            e_v    <= explut[zidx];
                            exp_ph <= 2'd2;
                        end
                        2'd2: begin                  // store exp, accumulate sum
                            expmem[j] <= e_v;
                            sum       <= sum + {8'd0, e_v};
                            if (j == n - 1) begin
                                // begin reciprocal: rem=0, quo=0, dividend = 2^40
                                rem     <= 41'd0;
                                quo     <= 21'd0;
                                div_bit <= 6'd40;
                                state   <= S_RECIP;
                            end else begin
                                j      <= j + 9'd1;
                                exp_ph <= 2'd0;
                            end
                        end
                        default: exp_ph <= 2'd0;
                    endcase
                end
                // ---------------- reciprocal: r = floor(2^40 / sum) --------------
                // Restoring long division of dividend 2^40 (bit 40 set) by `sum`.
                // 41 iterations (div_bit = 40..0). Each: shift remainder left, OR in
                // dividend bit, compare/subtract divisor, shift a quotient bit into quo.
                // Only the low 21 quotient bits matter (r <= 2^20).
                S_RECIP: begin : div_step
                    reg [41:0] shifted;
                    reg        dbit;
                    reg [40:0] rem_n;
                    reg        qbit;
                    dbit    = (div_bit == 6'd40);
                    shifted = {rem, dbit};                // 42-bit: (rem<<1)|dbit
                    if (shifted >= {13'd0, sum}) begin
                        rem_n = shifted[40:0] - {12'd0, sum};
                        qbit  = 1'b1;
                    end else begin
                        rem_n = shifted[40:0];
                        qbit  = 1'b0;
                    end
                    rem <= rem_n;
                    quo <= {quo[19:0], qbit};
                    if (div_bit == 6'd0) begin
                        recip   <= {quo[19:0], qbit};     // final 21-bit quotient
                        i       <= 9'd0;
                        j       <= 9'd0;
                        norm_ph <= 2'd0;
                        state   <= S_NORM;
                    end else begin
                        div_bit <= div_bit - 6'd1;
                    end
                end
                // ---------------- PASS3: prob = (e*r) >> 20 ----------------------
                S_NORM: begin
                    case (norm_ph)
                        2'd0: begin                  // read expmem[j] to plain vector
                            e_v     <= expmem[j];
                            norm_ph <= 2'd1;
                        end
                        2'd1: begin                  // multiply
                            er      <= e_v * recip;  // 21*21 -> 42-bit
                            norm_ph <= 2'd2;
                        end
                        2'd2: begin                  // shift + emit
                            prob      <= er[40:20];  // >>20, keep 21 bits
                            out_valid <= 1'b1;
                            if (j == n - 1) begin
                                state <= S_DONE;
                            end else begin
                                j       <= j + 9'd1;
                                norm_ph <= 2'd0;
                            end
                        end
                        default: norm_ph <= 2'd0;
                    endcase
                end
                // ----------------------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
