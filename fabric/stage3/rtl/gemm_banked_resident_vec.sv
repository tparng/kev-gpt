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
    parameter integer P      = 8,         // boundary width (P divides LANES, KMAX, MMAX)
    parameter integer MMAX   = 1024,      // max output rows of any single layer
    parameter integer KMAX   = 1024,      // max reduction length of any single layer
    parameter integer WWORDS = 25600,     // resident capacity in wide words
    parameter integer RLAT   = 2          // read->mac pipeline depth (cycles)
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
    output reg  [P*32-1:0]               y_out
);
    localparam integer WBITS  = LANES*4;
    localparam integer YBITS  = LANES*32;
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

    reg [P*8-1:0]    xmem [0:N*XROWS-1];     // stream b rows at b*XROWS
    // distributed: 64-deep x 4096-wide in BRAM wastes a full RAMB36 per 72 bits
    // of width (57 tiles); as LUTRAM it is ~4-5k LUTs and frees the tiles.
    (* ram_style = "distributed" *)
    reg [YBITS-1:0]  ymem [0:N*GROUPS-1];    // stream b group g at b*GROUPS+g

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
            else if (x_we) begin
                xmem[x_stream*XROWS + xptr] <= x_data;
                xptr <= xptr + 1'b1;
            end
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
    // Per-stream act rows are SEPARATE 2D arrays — a variable-base part-select
    // NBA write (xrow_p[0][b*W +: W] <=) is an iverilog trap: the base is
    // evaluated at update time, all N writes land in the LAST stream's slot.
    reg [WBITS-1:0]      word_p [0:RLAT-2];
    reg [P*8-1:0]        xrow_p [0:RLAT-1][0:N-1];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    integer i, b;

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
    // TWO N<=4-shaped packed halves. One N*YBITS concat is URAM-fatal at N=8
    // in any parent context (A/B: N=4 in-context 64 URAM, N=8 in-context 8
    // URAM, N=8 standalone 56) — past 16,384 bits Vivado loses the wsel loads,
    // trims rd/word_p 512->4b, and LUTRAMs the weights (~450k LUT). Each half
    // stays at <=16,384b, the exact shape of the URAM-clean a783c4a build.
    localparam integer NLO = (N > 4) ? 4 : N;
    localparam integer NHI = (N > NLO) ? N - NLO : 1;   // >=1 keeps widths legal
    wire [NLO*YBITS-1:0] acc_lo;
    wire [NHI*YBITS-1:0] acc_hi;
    reg  [YBITS-1:0] sel_lo, sel_hi;
`else
    wire [YBITS-1:0] acc_str [0:N-1];    // iverilog wraps >16k part-selects
    reg  [YBITS-1:0] y_lat [0:N-1];      // outputs latched at SETTLE (sim only:
                                         // dead regs in the synth view trip it)
`endif
    reg  [YBITS-1:0] acc_sel;
    genvar gm, gl;
    generate
        for (gm = 0; gm < N; gm = gm + 1) begin : g_mac
            // stream act lane (shared by this stream's LANES MACs) — combinational
            wire [P*8-1:0]    xrow = xrow_p[RLAT-1][gm];
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
            wire [YBITS-1:0] acc_bank;
            mac_bank #(.LANES(LANES)) u_mac (
                .clk(clk), .clr(rst || acc_clr), .en(mac_v),
                .w(wsel), .x(xsel), .acc(acc_bank));
`ifdef SYNTHESIS
            if (gm < NLO) begin : g_lo
                assign acc_lo[gm*YBITS +: YBITS] = acc_bank;
            end else begin : g_hi
                assign acc_hi[(gm-NLO)*YBITS +: YBITS] = acc_bank;
            end
`else
            assign acc_str[gm] = acc_bank;
`endif
        end
    endgenerate

    always @(posedge clk) begin
        word_p[0] <= wword_rd;
        for (b = 0; b < N; b = b + 1)
            xrow_p[0][b] <= xmem[b*XROWS + (kc[$clog2(KMAX)-1:0] >> LSHP)];
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT-1; i = i + 1)
            word_p[i] <= word_p[i-1];
        for (i = 1; i < RLAT; i = i + 1) begin
            for (b = 0; b < N; b = b + 1)
                xrow_p[i][b] <= xrow_p[i-1][b];
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
                    // per-half constant-base case — the EXACT idiom of the
                    // URAM-clean a783c4a build, once per <=4-stream half, then
                    // one 2:1 select. See the acc_lo/acc_hi declaration note.
                    case (db[1:0])
                        2'd0: sel_lo = acc_lo[0*YBITS +: YBITS];
                        2'd1: sel_lo = acc_lo[(NLO > 1 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_lo = acc_lo[(NLO > 2 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_lo = acc_lo[(NLO > 3 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    case (db[1:0])
                        2'd0: sel_hi = acc_hi[0*YBITS +: YBITS];
                        2'd1: sel_hi = acc_hi[(NHI > 1 ? 1 : 0)*YBITS +: YBITS];
                        2'd2: sel_hi = acc_hi[(NHI > 2 ? 2 : 0)*YBITS +: YBITS];
                        default: sel_hi = acc_hi[(NHI > 3 ? 3 : 0)*YBITS +: YBITS];
                    endcase
                    acc_sel = ((N > 4) && db[2]) ? sel_hi : sel_lo;
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= acc_sel;
`else
                    ymem[db*GROUPS + g[$clog2(GROUPS)-1:0]] <= y_lat[db];
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
    localparam integer PPG = LANES / P;            // P-groups per ymem word
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    always @(posedge clk) begin
        rd_word <= ymem[rd_stream*GROUPS + rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_out   <= rd_word[rd_off*(P*32) +: P*32];
    end
endmodule


// one stream's LANES INT4xINT8 MAC+accumulate lanes. keep_hierarchy: an opaque
// leaf the parent's bulk multiplier optimization cannot restructure across
// (the N=8-in-context URAM killer; per-lane/per-stream/DSP variants all map
// URAM 64 — and all land ~126k LUT, so the cost is the N=8 MAC array itself,
// not the isolation style; use_dsp burned 1,239 DSPs for zero LUT and was
// reverted to keep them for the packed 2-streams-per-DSP core, roadmap #13).
(* keep_hierarchy = "yes" *)
module mac_bank #(
    parameter integer LANES = 128
) (
    input  wire                  clk,
    input  wire                  clr,
    input  wire                  en,
    input  wire [LANES*4-1:0]    w,
    input  wire signed [7:0]     x,
    output wire [LANES*32-1:0]   acc
);
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : g_l
            reg signed [31:0] a;
            always @(posedge clk) begin
                if (clr) a <= 32'sd0;
                else if (en) a <= a + $signed(w[gl*4 +: 4]) * x;
            end
            assign acc[gl*32 +: 32] = a;
        end
    endgenerate
endmodule
