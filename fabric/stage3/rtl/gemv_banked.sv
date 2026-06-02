// -----------------------------------------------------------------------------
// gemv_banked — Stage 3 THROUGHPUT CORE: transposed wider-word banked GEMV.
//
// This is the piece that breaks the PE=64 ceiling on the road to ~100k tok/s.
// The resident core (PE=1) reads weights row-major and does ONE MAC/cycle. Here
// the weights are stored TRANSPOSED so that one wide memory word holds the SAME
// column k of LANES consecutive output rows:
//
//     word(g, k)[L*4 +: 4] = nibble( W[g*LANES + L, k] )    L = 0..LANES-1
//
// So a single read at column k feeds ALL LANES lanes with their own weight, and
// every lane multiplies by the SAME shared activation x[k]. LANES MACs/cycle.
// A 64-bit word holds 16 nibbles -> 1 URAM feeds 16 lanes; LANES=256 spans 16
// URAM in parallel and feeds 256 lanes. That is the wider-word banking the
// roadmap calls the single gating Vivado item.
//
//   y[m] = sum_k W[m,k]*x[k]   W signed INT4, x signed INT8, y signed INT32 exact.
//   group g = LANES output rows; groups = ceil(M/LANES); each group streams K cols.
//
// Bit-exact to fabric/stage3/pack_banked.gemv_int by construction (each lane just
// accumulates its own row). Conservative SV for iverilog + ram_style=ultra synth.
// Weights/acts load over write ports at runtime (NOT $readmemh-init) so synthesis
// maps a real RAM to URAM instead of pruning a constant ROM (the stage-2 lesson).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_banked #(
    parameter integer LANES = 16,        // PE lanes = nibbles per wide word (pow2)
    parameter integer MMAX  = 1024,      // max output rows
    parameter integer KMAX  = 1024,      // max reduction length
    parameter integer RLAT  = 2,         // read->mac pipeline depth (cycles)
    // wmem (working weight buffer) DEPTH in wide words. Decoupled from MMAX*KMAX so the
    // physical URAM can be sized to the LARGEST single matmul actually streamed in, not
    // the worst-case MMAX*KMAX product (which over-allocates URAM). Defaults to exactly
    // the old GROUPS*KMAX value so every existing instantiation stays byte-identical; the
    // GEMV math/addressing is unchanged — for any valid layer (m<=MMAX,k<=KMAX) the word
    // address grp_base+kc stays < WWORDS. Pass a smaller WWORDS (>= max ceil(M/LANES)*K
    // over the layers a caller streams) to shrink the buffer.
    parameter integer WWORDS = ((MMAX + LANES - 1) / LANES) * KMAX
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    // one-time load (auto-incrementing pointers; assert ld_rst first)
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [LANES*4-1:0]            w_data,
    input  wire                          x_we,
    input  wire signed [7:0]             x_data,
    // run
    input  wire                          start,
    output reg                           done,
    // readback: one output element per call (2-cycle latency)
    input  wire [$clog2(MMAX)-1:0]       rd_addr,
    output reg  signed [31:0]            y_out
);
    localparam integer WBITS  = LANES*4;
    localparam integer YBITS  = LANES*32;
    localparam integer LSH    = $clog2(LANES);                  // LANES is pow2
    localparam integer GROUPS = (MMAX + LANES - 1) / LANES;
    localparam integer GWORDS = GROUPS * KMAX;     // worst-case MMAX*KMAX product (logical)
    localparam integer WAW    = $clog2(WWORDS);    // word-address width sized to the PHYSICAL buffer
    localparam integer XAW    = $clog2(KMAX);

    (* ram_style = "ultra" *) reg [WBITS-1:0] wmem [0:WWORDS-1];
    reg signed [7:0]                          xmem [0:KMAX-1];
    reg [YBITS-1:0]                           ymem [0:GROUPS-1];

    // ---- one-time load -------------------------------------------------------
    reg [WAW-1:0] wptr;
    reg [XAW-1:0] xptr;
    always @(posedge clk) begin
        if (ld_rst) begin wptr <= 0; xptr <= 0; end
        else begin
            if (w_we) begin wmem[wptr] <= w_data; wptr <= wptr + 1'b1; end
            if (x_we) begin xmem[xptr] <= x_data; xptr <= xptr + 1'b1; end
        end
    end

    // ---- run FSM + RLAT-deep read/mac pipeline -------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]            state;
    reg [$clog2(GROUPS):0] g;
    reg [XAW:0]          kc, kmac;
    reg [WAW-1:0]        grp_base;          // g * k_count, in words
    reg [YBITS-1:0]      accb;              // LANES x 32-bit accumulators, packed

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;  // ceil(M/LANES)
    wire                 issue   = (kc < k_count);
    wire [WAW-1:0]       waddr   = grp_base + kc;     // kc (col) zero-extends to WAW

    // pipeline registers
    reg [WBITS-1:0]      word_p [0:RLAT-1];
    reg signed [7:0]     x_p    [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    integer i, L;
    reg signed [31:0]    prodL, sumL;
    reg [WBITS-1:0]      wsel;          // plain-vector copies (variable part-select
    reg signed [7:0]     xsel;          // on an array element reads X in iverilog)

    wire                 mac_v = v_p[RLAT-1];

    always @(posedge clk) begin
        // read + control pipeline (shift)
        word_p[0] <= wmem[waddr];
        x_p[0]    <= xmem[kc[XAW-1:0]];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT; i = i + 1) begin
            word_p[i] <= word_p[i-1];
            x_p[i]    <= x_p[i-1];
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
                        grp_base <= 0;
                        for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + 1'b1;
                    if (mac_v) begin
                        wsel = word_p[RLAT-1];
                        xsel = x_p[RLAT-1];
`ifdef DBG_MAC
                        if (kmac < 3) $display("MAC kmac=%0d wsel=%h xsel=%h v_p1=%b", kmac, wsel, xsel, v_p[RLAT-1]);
`endif
                        for (L = 0; L < LANES; L = L + 1) begin
                            prodL = $signed(wsel[L*4 +: 4]) * xsel;
                            sumL  = $signed(accb[L*32 +: 32]) + prodL;
                            accb[L*32 +: 32] <= sumL;
                        end
                        kmac <= kmac + 1'b1;
                    end
                    if (kmac == k_count) begin
                        ymem[g[$clog2(GROUPS)-1:0]] <= accb;     // commit whole group
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

    // ---- readback: select lane element from its group word (2-cycle latency) --
    reg [YBITS-1:0] rd_word;
    reg [LSH-1:0]   rd_lane;
    always @(posedge clk) begin
        rd_word <= ymem[rd_addr >> LSH];
        rd_lane <= rd_addr[LSH-1:0];
        y_out   <= $signed(rd_word[rd_lane*32 +: 32]);
    end
endmodule
