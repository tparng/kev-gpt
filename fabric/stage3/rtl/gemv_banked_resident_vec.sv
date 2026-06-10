// -----------------------------------------------------------------------------
// gemv_banked_resident_vec — gemv_banked_resident with a P-WIDE boundary.
//
// The resident MAC core is untouched (one wide URAM word = LANES nibbles/cycle).
// What changes is the BOUNDARY: act feed accepts P INT8 lanes per write, and
// readback returns P INT32 outputs per address. With the boundary the sequencer's
// G_AQ (act-quant) feeds in ceil(K/P) cycles and G_RB drains in ceil(M/P) — the
// boundary phases were ~16k of the 50,324 cyc/token at P=8.
//
// Activations are banked wide-word, like every sequencer scratch:
//     xmem row r holds P consecutive INT8 acts (lane l = bits [l*8 +: 8])
//
// Readback: group word YBITS = LANES*32 holds y[g*LANES .. (g+1)*LANES-1];
// rd_addr is a P-group index — y_out = lanes [rd_addr*P .. +P-1]. P must divide
// LANES, P-blocks never straddle a group word.
//
// iverilog: variable +: part-selects only on plain-reg copies, same as base.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_banked_resident_vec #(
    parameter integer LANES  = 128,       // PE lanes = nibbles per wide word (pow2)
    parameter integer P      = 8,         // boundary width (P divides LANES, KMAX, MMAX)
    parameter integer MMAX   = 1024,      // max output rows of any single layer
    parameter integer KMAX   = 1024,      // max reduction length of any single layer
    parameter integer WWORDS = 25600,     // resident capacity in wide words
    parameter integer RLAT   = 2,         // read->mac pipeline depth (cycles)
    parameter integer K2     = 0          // doc-7 R3: 2 K-steps/cycle via the URAM's
                                          // SECOND read port (free at N=1; the TDP claim
                                          // is silicon-proven by split-brain). Integer
                                          // associativity keeps the accumulate BIT-EXACT
                                          // (lane sums peak ~2^20, no mid-sum saturation).
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
    // per-call activation: P INT8 lanes per write
    input  wire                          x_we,
    input  wire [P*8-1:0]                x_data,
    // run
    input  wire                          start,
    output reg                           done,
    // readback: P INT32 outputs per address (2-cycle latency)
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
    localparam integer XAW    = $clog2(XROWS);
    localparam integer SUBW   = WBITS / 32;
    localparam integer SSW    = (SUBW > 1) ? $clog2(SUBW) : 1;

    // Weight memory in URAM banks (generate: one real array per bank).
    // GEOMETRY IS THE CONSTRAINT: a URAM is 4096 x 72b, and Vivado pads each
    // memory up to a multiple of 72b x 4096. One 1024b x 12800 memory -> 16 URAM
    // wide x 4 cascade = 64 = the whole device -> "infeasible" -> ~400k LUTRAM.
    // Even two 512b banks pad to 2 x (8 wide x 4) = 64. The dense packing is
    // 72b-wide banks: 15 banks x 4 cascade = 60 URAM for LANES=256.
    // For LANES<=128 the proven single 512b x 25600 memory (56 URAM) is kept.
    localparam integer BANKW = (WBITS > 512) ? 72 : WBITS;        // URAM-native width
    localparam integer NB    = (WBITS + BANKW - 1) / BANKW;       // banks
    localparam integer WPAD  = NB * BANKW;                        // padded word width

    reg [P*8-1:0]    xmem [0:XROWS-1];   // P acts per row
    reg [YBITS-1:0]  ymem [0:GROUPS-1];

    // ---- per-call activation pointer (the weight assembler now lives in the bank)
    reg [XAW-1:0]    xptr;
    always @(posedge clk) begin
        if (ld_rst) xptr <= 0;
        else if (x_we) begin xmem[xptr] <= x_data; xptr <= xptr + 1'b1; end
    end

    // ---- resident weights: the silicon-proven TDP URAM bank (weight_bank_tdp).
    // HDL inference of TDP UltraRAM is DEAD in 2025.2 (three OOC takes confirmed
    // it again: two-address ports, then "invalid write mode" for every template
    // variant). The bank's SYNTHESIS branch is xpm_memory_tdpram("ultra"), the
    // sim branch behavioral — gates verify sim, the board verifies XPM (the
    // codebase's dual-dialect pattern). K2 rides the bank's DP=1 COLUMN-PARITY
    // mode: ONE port-B read at the even pair address returns BOTH words kc
    // (rword_b) and kc+1 (rword1_b); grp_base and k_count are always even here.
    wire [$clog2(WWORDS)-1:0] waddr;
    wire [WBITS-1:0] wword_rd, wword2_rd;
    weight_bank_tdp #(.LANES(LANES), .WWORDS(WWORDS), .DP((K2 != 0) ? 1 : 0)) u_wb (
        .clk(clk), .clk2x(clk),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data),
        .raddr_b(waddr), .rword_b(wword_rd), .rword1_b(wword2_rd),
        .raddr_a({$clog2(WWORDS){1'b0}}), .rword_a(), .rword1_a());

    // ---- run FSM + RLAT-deep read/mac pipeline -------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]            state;
    reg [$clog2(GROUPS):0] g;
    reg [$clog2(KMAX):0] kc, kmac;
    reg [WAW-1:0]        grp_base;
    reg [YBITS-1:0]      accb;

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    wire                 issue   = (kc < k_count);
    wire                 issue2  = (K2 != 0) && (kc + 1 < k_count);
    assign               waddr   = grp_base + kc;   // even pair base (kc steps by 2)
    wire [$clog2(KMAX):0] kc1    = kc + 1;

    // pipeline: weight word (stage 0 = the per-bank URAM read reg) + act + valid
    reg [WBITS-1:0]      word_p [0:RLAT-2];
    reg [P*8-1:0]        xrow_p [0:RLAT-1];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    // K2 second lane (kc+1's word/act/valid ride the same depths)
    reg [WBITS-1:0]      word2_p [0:RLAT-2];
    reg [P*8-1:0]        xrow2_p [0:RLAT-1];
    reg [LSHP-1:0]       xl2_p   [0:RLAT-1];
    reg                  v2_p    [0:RLAT-1];
    integer i, L;
    reg signed [31:0]    prodL, sumL;
    reg signed [31:0]    prod2L;
    reg [WBITS-1:0]      wsel, wsel2;
    reg [P*8-1:0]        xrow, xrow2;
    reg signed [7:0]     xsel, xsel2;

    wire                 mac_v  = v_p[RLAT-1];
    wire                 mac_v2 = v2_p[RLAT-1];

    always @(posedge clk) begin
        word_p[0] <= wword_rd;
        xrow_p[0] <= xmem[kc[$clog2(KMAX)-1:0] >> LSHP];
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        word2_p[0] <= wword2_rd;
        xrow2_p[0] <= xmem[kc1[$clog2(KMAX)-1:0] >> LSHP];
        xl2_p[0]   <= kc1[LSHP-1:0];
        v2_p[0]    <= (state == RUN) && issue2;
        for (i = 1; i < RLAT-1; i = i + 1) begin
            word_p[i]  <= word_p[i-1];
            word2_p[i] <= word2_p[i-1];
        end
        for (i = 1; i < RLAT; i = i + 1) begin
            xrow_p[i] <= xrow_p[i-1];
            xl_p[i]   <= xl_p[i-1];
            v_p[i]    <= v_p[i-1];
            xrow2_p[i] <= xrow2_p[i-1];
            xl2_p[i]   <= xl2_p[i-1];
            v2_p[i]    <= v2_p[i-1];
        end

        if (rst) begin
            state <= IDLE; done <= 1'b0;
            g <= 0; kc <= 0; kmac <= 0; accb <= {YBITS{1'b0}}; grp_base <= 0;
            for (i = 0; i < RLAT; i = i + 1) begin v_p[i] <= 1'b0; v2_p[i] <= 1'b0; end
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        g <= 0; kc <= 0; kmac <= 0; accb <= {YBITS{1'b0}};
                        grp_base <= w_base;
                        for (i = 0; i < RLAT; i = i + 1) begin
                            v_p[i] <= 1'b0; v2_p[i] <= 1'b0;
                        end
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + ((K2 != 0 && issue2) ? 2'd2 : 2'd1);
                    if (mac_v) begin
                        wsel  = word_p[RLAT-2];
                        xrow  = xrow_p[RLAT-1];
                        xsel  = xrow[xl_p[RLAT-1]*8 +: 8];
                        wsel2 = word2_p[RLAT-2];
                        xrow2 = xrow2_p[RLAT-1];
                        xsel2 = xrow2[xl2_p[RLAT-1]*8 +: 8];
                        for (L = 0; L < LANES; L = L + 1) begin
                            prodL  = $signed(wsel[L*4 +: 4]) * xsel;
                            prod2L = mac_v2 ? $signed(wsel2[L*4 +: 4]) * xsel2 : 32'sd0;
                            sumL   = $signed(accb[L*32 +: 32]) + prodL + prod2L;
                            accb[L*32 +: 32] <= sumL;
                        end
                        kmac <= kmac + (mac_v2 ? 2'd2 : 2'd1);
                    end
                    if (kmac == k_count) begin
                        ymem[g[$clog2(GROUPS)-1:0]] <= accb;
                        if (g == gcount - 1) state <= FIN;
                        else begin
                            g <= g + 1'b1; kc <= 0; kmac <= 0;
                            accb <= {YBITS{1'b0}};
                            grp_base <= grp_base + k_count;
                            for (i = 0; i < RLAT; i = i + 1) begin
                                v_p[i] <= 1'b0; v2_p[i] <= 1'b0;
                            end
                        end
                    end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    // ---- readback: P consecutive outputs per address (2-cycle latency) --------
    localparam integer PPG = LANES / P;            // P-groups per ymem word
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    always @(posedge clk) begin
        rd_word <= ymem[rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_out   <= rd_word[rd_off*(P*32) +: P*32];
    end
endmodule
