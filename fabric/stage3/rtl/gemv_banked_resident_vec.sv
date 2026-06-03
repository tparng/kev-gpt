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

    (* ram_style = "ultra" *) reg [WBITS-1:0] wmem [0:WWORDS-1];
    reg [P*8-1:0]                             xmem [0:XROWS-1];   // P acts per row
    reg [YBITS-1:0]                           ymem [0:GROUPS-1];

    // ---- one-time load: assemble SUBW 32-bit chunks into one wide word ----------
    reg [WAW-1:0]    wword;
    reg [SSW-1:0]    wsub;
    reg [WBITS-1:0]  wbuf;
    reg [XAW-1:0]    xptr;
    wire [WBITS-1:0] wnext = wbuf | ({{(WBITS-32){1'b0}}, w_data} << (wsub*32));
    always @(posedge clk) begin
        if (ld_rst) begin wword <= 0; wsub <= 0; wbuf <= {WBITS{1'b0}}; xptr <= 0; end
        else begin
            if (w_we) begin
                wmem[wword] <= wnext;
                if (wsub == SUBW-1) begin
                    wword <= wword + 1'b1; wsub <= 0; wbuf <= {WBITS{1'b0}};
                end else begin
                    wbuf <= wnext; wsub <= wsub + 1'b1;
                end
            end
            if (x_we) begin xmem[xptr] <= x_data; xptr <= xptr + 1'b1; end
        end
    end

    // ---- run FSM + RLAT-deep read/mac pipeline -------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]            state;
    reg [$clog2(GROUPS):0] g;
    reg [$clog2(KMAX):0] kc, kmac;
    reg [WAW-1:0]        grp_base;
    reg [YBITS-1:0]      accb;

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    wire                 issue   = (kc < k_count);
    wire [WAW-1:0]       waddr   = grp_base + kc;

    // pipeline: weight word + the P-wide act row + lane-in-row + valid
    reg [WBITS-1:0]      word_p [0:RLAT-1];
    reg [P*8-1:0]        xrow_p [0:RLAT-1];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    integer i, L;
    reg signed [31:0]    prodL, sumL;
    reg [WBITS-1:0]      wsel;
    reg [P*8-1:0]        xrow;
    reg signed [7:0]     xsel;

    wire                 mac_v = v_p[RLAT-1];

    always @(posedge clk) begin
        word_p[0] <= wmem[waddr];
        xrow_p[0] <= xmem[kc[$clog2(KMAX)-1:0] >> LSHP];
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT; i = i + 1) begin
            word_p[i] <= word_p[i-1];
            xrow_p[i] <= xrow_p[i-1];
            xl_p[i]   <= xl_p[i-1];
            v_p[i]    <= v_p[i-1];
        end

        if (rst) begin
            state <= IDLE; done <= 1'b0;
            g <= 0; kc <= 0; kmac <= 0; accb <= {YBITS{1'b0}}; grp_base <= 0;
            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        g <= 0; kc <= 0; kmac <= 0; accb <= {YBITS{1'b0}};
                        grp_base <= w_base;
                        for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + 1'b1;
                    if (mac_v) begin
                        wsel = word_p[RLAT-1];
                        xrow = xrow_p[RLAT-1];
                        xsel = xrow[xl_p[RLAT-1]*8 +: 8];
                        for (L = 0; L < LANES; L = L + 1) begin
                            prodL = $signed(wsel[L*4 +: 4]) * xsel;
                            sumL  = $signed(accb[L*32 +: 32]) + prodL;
                            accb[L*32 +: 32] <= sumL;
                        end
                        kmac <= kmac + 1'b1;
                    end
                    if (kmac == k_count) begin
                        ymem[g[$clog2(GROUPS)-1:0]] <= accb;
                        if (g == gcount - 1) state <= FIN;
                        else begin
                            g <= g + 1'b1; kc <= 0; kmac <= 0;
                            accb <= {YBITS{1'b0}};
                            grp_base <= grp_base + k_count;
                            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
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
