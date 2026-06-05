// -----------------------------------------------------------------------------
// gemm_banked_resident_vec — gemv_banked_resident_vec with N BATCH STREAMS.
//
// One resident weight word read per cycle feeds N MAC banks: each stream MACs
// the same LANES nibbles against ITS OWN activation lane. Weight bandwidth is
// shared — N tokens per pass — the lever from 11k tok/s single-stream to 33k+.
//
// Boundary (per stream): act feed P INT8/cycle (x_we; x_stream selects, x_rst
// rewinds the row pointer); readback P INT32/cycle (rd_stream + rd_addr; same
// 2-cycle latency). Per-stream activations / outputs are SEGMENTED:
//     xmem row  = stream*XROWS  + r        (P INT8 lanes per row)
//     ymem word = stream*GROUPS + g        (LANES INT32 per group word)
//
// MAC: per cycle, N x LANES INT4xINT8 products into N accumulator banks.
// Bit-exact per stream vs the proven single-stream MAC: same adds in the same
// order within each stream, no cross-stream arithmetic.
//
// iverilog: variable +: part-selects only on plain-reg copies.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemm_banked_resident_vec #(
    parameter integer LANES  = 128,       // PE lanes = nibbles per wide word (pow2)
    parameter integer N      = 4,         // batch streams
    parameter integer ND     = 0,         // of which DSP-packed (streams N-ND..N-1)
    parameter integer P      = 8,         // boundary width (P divides LANES, KMAX, MMAX)
    parameter integer MMAX   = 1024,      // max output rows of any single layer
    parameter integer KMAX   = 1024,      // max reduction length of any single layer
    parameter integer WWORDS = 25600,     // resident capacity in wide words
    parameter integer RLAT   = 2,         // read->mac pipeline depth (cycles)
    // accumulator width: |acc| <= 8*128*1024 = 2^20 exactly (w=-8, x=-128,
    // k=1024 corner), so 24 bits holds every partial sum with 3 bits margin.
    // y_out stays P*32 (sign-extended) so the sequencer interface is unchanged.
    parameter integer ABITS  = 24
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS)-1:0]     w_base,
    // one-time load: 32-bit chunks assembled into LANES*4-bit wide words
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [31:0]                   w_data,
    // per-call activation: P INT8 lanes per write into stream x_stream
    input  wire                          x_rst,        // rewind act row pointer
    input  wire                          x_we,
    input  wire [$clog2(N)-1:0]          x_stream,
    input  wire [P*8-1:0]                x_data,
    // run (all N streams together)
    input  wire                          start,
    output reg                           done,
    // readback: P INT32 of stream rd_stream per address (2-cycle latency)
    input  wire [$clog2(N)-1:0]          rd_stream,
    input  wire [$clog2(MMAX/P)-1:0]     rd_addr,
    output wire [P*32-1:0]               y_out
);
    localparam integer WBITS  = LANES*4;
    localparam integer YBITS  = LANES*ABITS;   // internal acc/ymem lane width
    localparam integer LSH    = $clog2(LANES);
    localparam integer LSHP   = $clog2(P);
    localparam integer GROUPS = (MMAX + LANES - 1) / LANES;
    localparam integer WAW    = $clog2(WWORDS);
    localparam integer XROWS  = KMAX / P;
    localparam integer SUBW   = WBITS / 32;
    localparam integer SSW    = (SUBW > 1) ? $clog2(SUBW) : 1;

    // URAM-native banking (72b) only needed above 512 bits; L=128 keeps 512b word.
    localparam integer BANKW = (WBITS > 512) ? 72 : WBITS;
    localparam integer NB    = (WBITS + BANKW - 1) / BANKW;
    localparam integer WPAD  = NB * BANKW;

    // per-stream act memories: ONE xmem with N read ports made Vivado replicate
    // it N times at N=8 and refuse outright at N=16 ("Failed to dissolve the
    // memory... 131072 bits too large"). Each stream only ever reads its own
    // rows, so N small 1W/1R SDPs are the natural shape. Declared inside the
    // g_mac generate below; the write decodes on x_stream there.
    // distributed: shallow x very-wide in BRAM wastes a full RAMB36 per 72 bits
    // of width (57 tiles at N=8/32b); as LUTRAM it is a few k LUTs, tiles freed.
    (* ram_style = "distributed" *)
    reg [YBITS-1:0]  ymem [0:N*GROUPS-1];    // stream b group g at b*GROUPS+g
    // side memory: per (stream,group) sum_act for DSP-stream readback recovery.
    // Same N*GROUPS geometry/addressing as ymem; only DSP-stream rows are read.
    (* ram_style = "distributed" *)
    reg signed [22:0] sumact_mem [0:N*GROUPS-1];

    // ---- one-time load: assemble SUBW chunks -> commit all banks in one shot ----
    reg [WAW-1:0]    wword;
    reg [SSW-1:0]    wsub;
    reg [WBITS-1:0]  wbuf;
    reg [$clog2(XROWS)-1:0] xptr;
    wire [WBITS-1:0] wnext = wbuf | ({{(WBITS-32){1'b0}}, w_data} << (wsub*32));
    wire [WPAD-1:0]  wnext_pad = {{(WPAD-WBITS){1'b0}}, wnext};
    wire             wcommit = w_we && (wsub == SUBW-1);
    always @(posedge clk) begin
        if (ld_rst) begin wword <= 0; wsub <= 0; wbuf <= {WBITS{1'b0}}; xptr <= 0; end
        else begin
            if (w_we) begin
                if (wsub == SUBW-1) begin
                    wword <= wword + 1'b1; wsub <= 0; wbuf <= {WBITS{1'b0}};
                end else begin
                    wbuf <= wnext; wsub <= wsub + 1'b1;
                end
            end
            if (x_rst) xptr <= 0;
            else if (x_we) xptr <= xptr + 1'b1;   // per-stream mems write in g_mac
        end
    end

    // ---- per-bank URAM arrays + registered read (= pipeline stage 0) ------------
    wire [$clog2(WWORDS)-1:0] waddr;
    wire [WPAD-1:0] wword_pad;
    wire [WBITS-1:0] wword_rd = wword_pad[WBITS-1:0];
    genvar gb;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_w
            (* ram_style = "ultra" *) reg [BANKW-1:0] mem [0:WWORDS-1];
            reg [BANKW-1:0] rd;
            always @(posedge clk) begin
                if (wcommit) mem[wword] <= wnext_pad[gb*BANKW +: BANKW];
                rd <= mem[waddr];
            end
            assign wword_pad[gb*BANKW +: BANKW] = rd;
        end
    endgenerate

    // ---- run FSM + RLAT-deep read/mac pipeline -------------------------------
    localparam [2:0] IDLE = 3'd0, RUN = 3'd1, DRAIN = 3'd3, FIN = 3'd2, SETTLE = 3'd4;
    reg [2:0]            state;
    reg [1:0]            settle;
    reg [$clog2(GROUPS):0] g;
    reg [$clog2(KMAX):0] kc, kmac;
    reg [WAW-1:0]        grp_base;
    reg [$clog2(N):0]    db;                 // ymem drain stream counter
    reg                  acc_clr;            // broadcast accumulator clear

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    wire                 issue   = (kc < k_count);
    assign               waddr   = grp_base + kc;

    // pipeline: weight word (stage 0 = the per-bank URAM read reg) + acts + valid.
    // Per-stream act pipelines live INSIDE g_mac (per-stream xm memories +
    // their own xr stages) — one xmem with N read ports made Vivado replicate
    // at N=8 and refuse at N=16; per-stream 1W/1R SDPs are the natural shape.
    reg [WBITS-1:0]      word_p [0:RLAT-2];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    integer i, b;
    wire [$clog2(XROWS)-1:0] xrd_row = kc[$clog2(KMAX)-1:0] >> LSHP;

    wire                 mac_v = v_p[RLAT-1];

    // ---- N MAC banks (generate: each accumulator is its own plain reg, so the
    // per-lane variable part-select hits a PLAIN reg, never an array element —
    // accb[b][L*32 +: 32] reads/writes X under iverilog). wsel must be a WIRE:
    // an always@* copy made Vivado trim the URAM read register to 4 bits and
    // fall back to ~400k LUTRAM.
    wire [WBITS-1:0] wsel = word_p[RLAT-2];
    // per-stream wires: one flat N*YBITS concat wraps iverilog's 16-bit index
    // space at stream 4 (bit 16384) � streams 4..7 would read stream 0..3.
`ifdef SYNTHESIS
    // FOUR <=4-stream packed chunks + a 4:1 final select. Named wires (an
    // unpacked wire array broke URAM inference in the saga); each chunk stays
    // well under 16,384 bits (4 x LANES*ABITS = 12,288 at 24b), the shape of
    // every URAM-clean build. Unused chunks at small N dangle and trim.
    wire [4*YBITS-1:0] acc_q0, acc_q1, acc_q2, acc_q3;
    reg  [YBITS-1:0] sel_q0, sel_q1, sel_q2, sel_q3;
    wire [4:0] dbw = db;                  // db zero-extended (db[3:2] at any N)
`else
    wire [YBITS-1:0] acc_str [0:N-1];    // iverilog wraps >16k part-selects
    reg  [YBITS-1:0] y_lat [0:N-1];      // outputs latched at SETTLE (sim only:
                                         // dead regs in the synth view trip it)
`endif
    reg  [YBITS-1:0] acc_sel;
    // per-stream sum_act (only DSP streams drive a real value; LUT streams 0).
    wire signed [22:0] sa_str [0:N-1];
    reg  signed [22:0] sa_sel;                // selected stream's sum_act in DRAIN
    genvar gm, gl;
    generate
        for (gm = 0; gm < N; gm = gm + 1) begin : g_mac
            // per-stream act memory (1W/1R SDP) + its own RLAT-deep row pipeline.
            // Same cycle timing as the old shared-xmem stage-0 read.
            reg [P*8-1:0] xm [0:XROWS-1];
            reg [P*8-1:0] xr [0:RLAT-1];
            integer xi;
            always @(posedge clk) begin
                if (x_we && x_stream == gm[$clog2(N)-1:0]) xm[xptr] <= x_data;
                xr[0] <= xm[xrd_row];
                for (xi = 1; xi < RLAT; xi = xi + 1) xr[xi] <= xr[xi-1];
            end
            // stream act lane (shared by this stream's LANES MACs): copy the
            // unpacked element to a plain wire BEFORE the variable part-select
            // (the iverilog X rule).
            wire [P*8-1:0]    xrow = xr[RLAT-1];
            wire signed [7:0] xsel = xrow[xl_p[RLAT-1]*8 +: 8];
            // one accumulator per lane: all weight indexing is genvar-CONSTANT —
            // a runtime-L loop made Vivado fail to unroll and trim wsel to 4 bits
            // (the URAM read died, weights fell back to ~400k LUTRAM).
            // the stream's whole MAC array is ONE keep_hierarchy leaf: at N=8
            // (1,024 mults) the parent's bulk multiplier optimization detaches
            // the wsel loads before RAM mapping and the URAM weight banks fall
            // to ~450k LUTRAM — N=4 (512 mults) never tripped it, so 128-mult
            // leaves are safe. Per-LANE leaves also fixed URAM but cost +36k
            // LUT (no cross-lane xsel sharing): 127.5k > 117.1k capacity.
            // Streams >= N-ND use the DSP-packed leaf (2 lanes/DSP, WP487):
            // same weight word, same interface, recovered y values out.
            wire [YBITS-1:0] acc_bank;
            if (gm < N - ND) begin : g_lut
                mac_bank #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                    .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                    .w(wsel), .x(xsel), .acc(acc_bank));
                assign sa_str[gm] = 23'sd0;        // LUT streams: no recovery
            end else begin : g_dsp
                mac_bank_dsp #(.LANES(LANES), .ABITS(ABITS)) u_mac (
                    .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                    .w(wsel), .x(xsel), .acc(acc_bank),
                    .sum_act_o(sa_str[gm]));        // RAW-pair encoding now
            end
`ifdef SYNTHESIS
            if (gm/4 == 0) begin : g_q0
                assign acc_q0[(gm%4)*YBITS +: YBITS] = acc_bank;
            end else if (gm/4 == 1) begin : g_q1
                assign acc_q1[(gm%4)*YBITS +: YBITS] = acc_bank;
            end else if (gm/4 == 2) begin : g_q2
                assign acc_q2[(gm%4)*YBITS +: YBITS] = acc_bank;
            end else begin : g_q3
                assign acc_q3[(gm%4)*YBITS +: YBITS] = acc_bank;
            end
`else
            assign acc_str[gm] = acc_bank;
`endif
        end
    endgenerate

    always @(posedge clk) begin
        word_p[0] <= wword_rd;
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT-1; i = i + 1)
            word_p[i] <= word_p[i-1];
        for (i = 1; i < RLAT; i = i + 1) begin
            xl_p[i]   <= xl_p[i-1];
            v_p[i]    <= v_p[i-1];
        end

        acc_clr <= 1'b0;
        if (rst) begin
            state <= IDLE; done <= 1'b0;
            g <= 0; kc <= 0; kmac <= 0; grp_base <= 0; db <= 0;
            acc_clr <= 1'b1;
            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        g <= 0; kc <= 0; kmac <= 0;
                        acc_clr <= 1'b1;
                        grp_base <= w_base;
                        for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + 1'b1;
                    if (mac_v) kmac <= kmac + 1'b1;
`ifdef SYNTHESIS
                    // straight to DRAIN — the exact FSM of the URAM-clean
                    // a783c4a build. SETTLE is a sim-only mechanism (y_lat
                    // latch), so silicon finishes each call 4 cyc earlier
                    // than sim: sim cyc/tok is an upper bound (~0.8%).
                    if (kmac == k_count) begin db <= 0; state <= DRAIN; end
`else
                    if (kmac == k_count) begin db <= 0; settle <= 0; state <= SETTLE; end
`endif
                end
`ifndef SYNTHESIS
                SETTLE: begin
                    // sim-only: wait out the MAC pipeline tail, then latch the
                    // per-stream accumulators into y_lat for constant unroll
                    settle <= settle + 1'b1;
                    if (settle == 2'd3) begin
                        for (b = 0; b < N; b = b + 1)
                            y_lat[b] <= acc_str[b];
                        state <= DRAIN;
                    end
                end
`endif
                DRAIN: begin                  // one stream's group word per cycle
`ifdef SYNTHESIS
                    // per-chunk constant-base case — the EXACT idiom of the
                    // URAM-clean a783c4a build, once per <=4-stream chunk,
                    // then one 4:1 select. See the acc_q* declaration note.
                    case (db[1:0])
                        2'd0: sel_q0 = acc_q0[0*YBITS +: YBITS];
                        2'd1: sel_q0 = acc_q0[(N > 1 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q0 = acc_q0[(N > 2 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q0 = acc_q0[(N > 3 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_q1 = acc_q1[0*YBITS +: YBITS];
                        2'd1: sel_q1 = acc_q1[(N > 5 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q1 = acc_q1[(N > 6 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q1 = acc_q1[(N > 7 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_q2 = acc_q2[0*YBITS +: YBITS];
                        2'd1: sel_q2 = acc_q2[(N > 9 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q2 = acc_q2[(N > 10 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q2 = acc_q2[(N > 11 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_q3 = acc_q3[0*YBITS +: YBITS];
                        2'd1: sel_q3 = acc_q3[(N > 13 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_q3 = acc_q3[(N > 14 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_q3 = acc_q3[(N > 15 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (dbw[3:2])
                        2'd0: acc_sel = sel_q0;
                        2'd1: acc_sel = (N > 4)  ? sel_q1 : sel_q0;
                        2'd2: acc_sel = (N > 8)  ? sel_q2 : sel_q0;
                        default: acc_sel = (N > 12) ? sel_q3 : sel_q0;
                    endcase
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= acc_sel;
                    // sum_act is stable from end-of-RUN until acc_clr (asserted
                    // only at db==N-1 below), so DRAIN samples it safely.
                    sa_sel = sa_str[db];
                    sumact_mem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= sa_sel;
`else
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= y_lat[db];
                    sa_sel = sa_str[db];
                    sumact_mem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= sa_sel;
`endif
                    if (db == N-1) begin
                        acc_clr <= 1'b1;
                        if (g == gcount - 1) state <= FIN;
                        else begin
                            g <= g + 1'b1; kc <= 0; kmac <= 0;
                            grp_base <= grp_base + k_count;
                            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                            state <= RUN;
                        end
                    end else db <= db + 1'b1;
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    // ---- readback: P consecutive outputs of one stream per address (2-cycle) --
    // LUT streams: y_raw holds P x ABITS recovered lanes -> sign-extend (as
    // before). DSP streams: y_raw now holds P/2 RAW 48-bit pair accumulators;
    // the per-pair recovery that used to ride the always-on MAC datapath is
    // done HERE (4 pairs/cycle) using the stream/group's stored sum_act read
    // alongside ymem on the same address pipeline. Recovery is combinational
    // after y_raw (a few narrow adders) — no extra pipeline stage, so the GE
    // engine's read timing is unchanged. y_raw NBAs at the same edge y_out
    // used to; y_out is a wire off it, identical downstream values/timing.
    localparam integer PPG = LANES / P;            // P-groups per ymem word
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    reg [P*ABITS-1:0]      y_raw;
    // sum_act for the addressed (stream,group), aligned to y_raw (2-cycle path);
    // DSP-stream flag of rd_stream, delayed the same 2 cycles.
    reg signed [22:0]      sa_rd, sa_pipe;
    reg                    dsp0, dsp1;
    always @(posedge clk) begin
        rd_word <= ymem[rd_stream*GROUPS + rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_raw   <= rd_word[rd_off*(P*ABITS) +: P*ABITS];
        sa_rd   <= sumact_mem[rd_stream*GROUPS + rd_addr / PPG];
        sa_pipe <= sa_rd;
        dsp0    <= (rd_stream >= (N - ND));
        dsp1    <= dsp0;
    end
    // 8*sum_act, sign-extended for the recovery algebra
    wire signed [25:0] sa8 = {sa_pipe, 3'b000};
    genvar gy, gp;
    generate
        // LUT form: sign-extend each ABITS lane to 32 (unchanged).
        wire [P*32-1:0] y_lut;
        for (gy = 0; gy < P; gy = gy + 1) begin : g_yext
            assign y_lut[gy*32 +: 32] =
                {{(32-ABITS){y_raw[gy*ABITS + ABITS-1]}}, y_raw[gy*ABITS +: ABITS]};
        end
        // DSP form: recover y0/y1 from each raw 48-bit pair (P/2 pairs).
        // Lane l of the P-wide word maps to pair l/2: pair gp -> lanes 2gp,2gp+1.
        wire [P*32-1:0] y_dsp;
        for (gp = 0; gp < P/2; gp = gp + 1) begin : g_rec
            wire [47:0]        a     = y_raw[gp*48 +: 48];      // raw pair acc
            wire [21:0]        y0m   = a[21:0] - sa8[21:0];     // mod 2^22
            wire signed [31:0] y0    = {{10{y0m[21]}}, y0m};
            wire signed [27:0] hsum  = y0 + sa8;
            wire signed [5:0]  carry = hsum >>> 22;
            wire signed [31:0] y1    = $signed(a[47:22]) - sa8 - carry;
            assign y_dsp[(2*gp)  *32 +: 32] = y0;
            assign y_dsp[(2*gp+1)*32 +: 32] = y1;
        end
        assign y_out = dsp1 ? y_dsp : y_lut;
    endgenerate
endmodule


// one stream's LANES INT4xINT8 MAC+accumulate lanes. keep_hierarchy: an opaque
// leaf the parent's bulk multiplier optimization cannot restructure across
// (the N=8-in-context URAM killer; per-lane/per-stream/DSP variants all map
// URAM 64 — and all land ~126k LUT, so the cost is the N=8 MAC array itself,
// not the isolation style). use_dsp=no: a 4x8 mult is ~15 fabric LUTs, but
// Vivado auto-absorbed ~350 of them into DSPs — at N=16 that overflow (1,304
// > 1,248 with the packed banks) EVICTED the 32x32-class LN/attention/dequant
// multipliers to fabric at ~1.1k LUT each (366k total). Cheap mults stay in
// fabric; DSPs go to the expensive ones.
(* keep_hierarchy = "yes", use_dsp = "no" *)
module mac_bank #(
    parameter integer LANES = 128,
    parameter integer ABITS = 24       // see core note: |acc| <= 2^20 range-proven
) (
    input  wire                      clk,
    input  wire                      clr,
    input  wire                      en,
    input  wire [LANES*4-1:0]        w,
    input  wire signed [7:0]         x,
    output wire [LANES*ABITS-1:0]    acc
);
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : g_l
            reg signed [ABITS-1:0] a;
            always @(posedge clk) begin
                if (clr) a <= {ABITS{1'b0}};
                else if (en) a <= a + $signed(w[gl*4 +: 4]) * x;
            end
            assign acc[gl*ABITS +: ABITS] = a;
        end
    endgenerate
endmodule


// one stream's LANES lanes as LANES/2 DSP-packed MACs (WP487 adapted): two
// +8-biased INT4 weights at a 22-BIT gap share one INT8 activation —
//   wpak = {w1^8, 18'b0, w0^8};  acc48 += $unsigned(wpak) * $signed(x)
// The gap is 22, not 24: wpak must stay under 2^26 so the signed-extended
// operand fits the DSP48E2's 27-bit port — at a 24-bit gap (28-bit wpak)
// Vivado cascaded TWO DSPs per multiply (1,024 for 8 banks). The 22-bit lower
// field still exactly holds K<=1024 of 12-bit products (12+10), and |y0| <
// 2^21 keeps the mod-2^22 window unambiguous.
//
// RAW-PAIR datapath: this leaf now outputs the UNRECOVERED 48-bit pair
// accumulators packed into the acc bus (pair p in acc[p*48 +: 48]). The
// per-pair recovery (the ~5.4k LUT/bank that used to ride the always-on
// datapath) is hoisted to the parent's P-wide readback path (4 pairs/cycle)
// using the sum_act this leaf exports. LANES*ABITS == (LANES/2)*48 == 3072 at
// LANES=128/ABITS=24, so the acc bus width is unchanged — only its ENCODING.
//   recovery (done at readback, exact integer algebra; gated incl. 2^20 corner):
//     y0 = (acc[21:0] - 8*sum_act) mod 2^22
//     carry = (y0 + 8*sum_act) >>> 22
//     y1 = acc[47:22] - 8*sum_act - carry
// Same keep_hierarchy rationale as mac_bank (the bulk-mult URAM killer).
(* keep_hierarchy = "yes" *)
module mac_bank_dsp #(
    parameter integer LANES = 128,
    parameter integer ABITS = 24
) (
    input  wire                      clk,
    input  wire                      clr,
    input  wire                      en,
    input  wire [LANES*4-1:0]        w,
    input  wire signed [7:0]         x,
    output wire [LANES*ABITS-1:0]    acc,        // RAW pairs: pair p at p*48
    output wire signed [22:0]        sum_act_o   // for parent readback recovery
);
    wire signed [8:0] xs = x;                       // explicit sign-extend
    // shared per-stream sum of activations (the +8 bias correction term)
    reg signed [22:0] sum_act;
    always @(posedge clk) begin
        if (clr) sum_act <= 23'sd0;
        else if (en) sum_act <= sum_act + xs;
    end
    assign sum_act_o = sum_act;
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 2) begin : g_d
            wire [3:0]  w0 = w[gl*4 +: 4]     ^ 4'h8;   // +8 bias -> unsigned
            wire [3:0]  w1 = w[(gl+1)*4 +: 4] ^ 4'h8;
            wire [25:0] wpak = {w1, 18'b0, w0};         // < 2^26: one DSP
            (* use_dsp = "yes" *) reg signed [47:0] a;
            always @(posedge clk) begin
                if (clr) a <= 48'sd0;
                else if (en) a <= a + $signed({1'b0, wpak}) * xs;
            end
            // raw 48-bit pair accumulator: pair (gl/2) in acc[(gl/2)*48 +: 48].
            // (LANES/2)*48 == LANES*ABITS, so acc bus width is unchanged.
            assign acc[(gl/2)*48 +: 48] = a;
        end
    endgenerate
endmodule
