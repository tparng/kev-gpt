// -----------------------------------------------------------------------------
// softmax_f — softmax.sv with PASS2/PASS3 PIPELINED to 1 element/cycle (doc-7
// R4c). The shared softmax.sv runs each pass at 3 cycles/element (read ->
// compute -> store sub-phases); on the faithful single-stream sequencer that
// was 6 of the 8 slope cycles per position per head-call. This fork streams
// both passes through 3-STAGE PIPELINES at 1 element/cycle — the per-element
// arithmetic, LUT, accumulation ORDER (j ascending) and the divider are
// VERBATIM, so the emitted probs are bit-identical to softmax.sv; only the
// cycle count changes. The shared brick is untouched (the N=16 record designs'
// MEASURED cycle counts stay theirs).
//
// Interface and Q-formats identical to softmax.sv (drop-in).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module softmax_f #(
    parameter integer TMAX = 256,
    parameter integer DIV_STEPS = 2
) (
    input  wire                clk,
    input  wire                rst,
    input  wire                start,
    input  wire [8:0]          t_count,
    input  wire                in_valid,
    input  wire signed [15:0]  score,
    output reg                 in_ready,
    output reg                 out_valid,
    output reg  [20:0]         prob,
    output reg                 done
);
    (* rom_style = "block" *) reg [20:0] explut [0:4096];
    initial $readmemh("exp_lut.mem", explut);

    reg signed [15:0] scoremem [0:TMAX-1];
    reg [20:0]        expmem   [0:TMAX-1];

    localparam [2:0] S_IDLE  = 3'd0,
                     S_LOAD  = 3'd1,
                     S_EXP   = 3'd2,
                     S_RECIP = 3'd3,
                     S_NORM  = 3'd4,
                     S_DONE  = 3'd5;
    reg [2:0] state;

    reg [8:0]  n;
    reg [8:0]  i;
    reg signed [15:0] runmax;

    // ---- PASS2 pipeline (1 element/cycle): A read scoremem, B LUT, C store+sum
    reg [8:0]  j;                            // addr-stage index
    reg signed [15:0] sc_v;                  // stage-A out (scoremem[j])
    reg [8:0]  ja;  reg av;                  // stage-A tag/valid
    reg [20:0] e_v;                          // stage-B out (explut[zidx])
    reg [8:0]  jb;  reg bv;                  // stage-B tag/valid
    reg [28:0] sum;

    // reciprocal — verbatim from softmax.sv
    reg [40:0] rem;
    reg [20:0] quo;
    reg [5:0]  div_bit;
    reg [20:0] recip;
    (* dont_touch = "true" *) reg [20:0] recip_q;

    // ---- PASS3 pipeline (1 element/cycle): A read expmem, B multiply, C emit
    reg [20:0] e_v3;
    reg        av3;
    reg [41:0] er;
    reg        bv3;
    reg [8:0]  jc3;                          // emitted-element counter

    wire signed [16:0] zq_full = sc_v - runmax;
    wire [12:0] zidx = (zq_full <= -17'sd4096) ? 13'd4096
                     : (zq_full >= 17'sd0)     ? 13'd0
                     : (13'd0 - zq_full[12:0]);

    always @(posedge clk) begin
        out_valid <= 1'b0;
        done      <= 1'b0;

        if (rst) begin
            state    <= S_IDLE;
            in_ready <= 1'b0;
            sum      <= 29'd0;
            i        <= 9'd0;
            j        <= 9'd0;
            av <= 1'b0; bv <= 1'b0; av3 <= 1'b0; bv3 <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    in_ready <= 1'b0;
                    if (start) begin
                        n        <= t_count;
                        i        <= 9'd0;
                        runmax   <= 16'sh8000;
                        sum      <= 29'd0;
                        in_ready <= 1'b1;
                        state    <= S_LOAD;
                    end
                end
                // ---------------- PASS1 (verbatim) -------------------------------
                S_LOAD: begin
                    in_ready <= 1'b1;
                    if (in_valid) begin
                        scoremem[i] <= score;
                        if (score > runmax) runmax <= score;
                        if (i == n - 1) begin
                            in_ready <= 1'b0;
                            j <= 9'd0; av <= 1'b0; bv <= 1'b0;
                            state <= S_EXP;
                        end else begin
                            i <= i + 9'd1;
                        end
                    end
                end
                // ---------------- PASS2: pipelined 1 element/cycle ---------------
                S_EXP: begin
                    // stage A: read scoremem[j]
                    sc_v <= scoremem[(j < n) ? j : 9'd0];
                    ja   <= j;
                    av   <= (j < n);
                    if (j < n) j <= j + 9'd1;
                    // stage B: LUT read (zidx combinational from sc_v)
                    e_v  <= explut[zidx];
                    jb   <= ja;
                    bv   <= av;
                    // stage C: store + accumulate (j ascending, same order)
                    if (bv) begin
                        expmem[jb] <= e_v;
                        sum        <= sum + {8'd0, e_v};
                        if (jb == n - 1) begin
                            rem     <= 41'd0;
                            quo     <= 21'd0;
                            div_bit <= 6'd40;
                            av <= 1'b0; bv <= 1'b0;
                            state   <= S_RECIP;
                        end
                    end
                end
                // ---------------- reciprocal (verbatim from softmax.sv) ----------
                S_RECIP: begin : div_step
                    localparam integer PAD = (DIV_STEPS - (41 % DIV_STEPS)) % DIV_STEPS;
                    reg [41:0] sh;
                    reg [40:0] r_w;
                    reg [DIV_STEPS-1:0] qb;
                    reg [5:0]  pos;
                    reg        fin;
                    reg [2:0]  nstep;
                    integer    s;
                    r_w   = rem;
                    qb    = {DIV_STEPS{1'b0}};
                    pos   = div_bit;
                    fin   = 1'b0;
                    nstep = (div_bit == 6'd40) ? (DIV_STEPS - PAD) : DIV_STEPS;
                    for (s = 0; s < DIV_STEPS; s = s + 1) begin
                        if (s < nstep && !fin) begin
                            sh = {r_w, (pos == 6'd40)};
                            if (sh >= {13'd0, sum}) begin
                                r_w = sh[40:0] - {12'd0, sum};
                                qb  = {qb[DIV_STEPS-2:0], 1'b1};
                            end else begin
                                r_w = sh[40:0];
                                qb  = {qb[DIV_STEPS-2:0], 1'b0};
                            end
                            if (pos == 6'd0) fin = 1'b1;
                            else             pos = pos - 6'd1;
                        end else begin
                            qb = {qb[DIV_STEPS-2:0], 1'b0};
                        end
                    end
                    rem <= r_w;
                    quo <= {quo[20-DIV_STEPS:0], qb};
                    if (fin) begin
                        recip   <= {quo[20-DIV_STEPS:0], qb};
                        recip_q <= {quo[20-DIV_STEPS:0], qb};   // settled >=2 cyc before use
                        j       <= 9'd0;
                        jc3     <= 9'd0;
                        av3 <= 1'b0; bv3 <= 1'b0;
                        state   <= S_NORM;
                    end else begin
                        div_bit <= pos;
                    end
                end
                // ---------------- PASS3: pipelined 1 element/cycle ---------------
                S_NORM: begin
                    // stage A: read expmem[j]
                    e_v3 <= expmem[(j < n) ? j : 9'd0];
                    av3  <= (j < n);
                    if (j < n) j <= j + 9'd1;
                    // stage B: multiply (recip_q constant through the pass)
                    er  <= e_v3 * recip_q;
                    bv3 <= av3;
                    // stage C: emit
                    if (bv3) begin
                        prob      <= er[40:20];
                        out_valid <= 1'b1;
                        if (jc3 == n - 1) state <= S_DONE;
                        else jc3 <= jc3 + 9'd1;
                    end
                end
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
