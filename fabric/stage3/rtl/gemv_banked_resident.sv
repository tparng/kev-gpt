// -----------------------------------------------------------------------------
// gemv_banked_resident — Stage 3 DEPLOYABLE wide-GEMV core: the THROUGHPUT banked
// engine (LANES MACs/cycle, transposed wide word) made RESIDENT. The WHOLE model's
// transposed weights preload into URAM ONCE; per layer the host sets W_BASE (the
// word offset of that layer's transposed block) + M/K, streams only the activation,
// and runs. No per-token weight reload — the win the roadmap names.
//
// Fusion of two proven cores:
//   * gemv_banked.sv  — transposed layout: word(g,k)[L*4 +: 4] = nibble(W[g*LANES+L, k]).
//                       One read at column k feeds ALL LANES lanes sharing x[k]. LANES MACs/cyc.
//   * gemv_resident.sv — w_base layer offset into a big resident URAM; load once, run many.
//
// Resident transposed layout in the big URAM (one wide word per transposed column):
//     word(layer, g, k) = w_base[layer] + g*K_layer + k          (word units)
// where g = 0..ceil(M_layer/LANES)-1, k = 0..K_layer-1. w_base[layer] is the
// cumulative word count of all earlier layers (host computes it; see pack_banked_resident).
//
//   y[m] = sum_k W[m,k]*x[k]   W signed INT4, x signed INT8, y signed INT32 exact.
//
// Bit-exact to fabric/stage3/pack_banked.gemv_int by construction (each lane just
// accumulates its own row), independent of w_base — the offset only relocates the
// same transposed block. Conservative procedural SV for iverilog + ram_style=ultra.
//
// AXI bus is 32-bit, the wide word is LANES*4 bits. The load FSM assembles
// SUBW = LANES*4/32 incoming 32-bit chunks into one wide URAM word (low chunk first),
// exactly mirroring the 8-byte assembly in gemv_resident but at 32-bit granularity.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_banked_resident #(
    parameter integer LANES  = 256,       // PE lanes = nibbles per wide word (pow2, mult of 8)
    parameter integer MMAX   = 1024,      // max output rows of any single layer
    parameter integer KMAX   = 1024,      // max reduction length of any single layer
    parameter integer WWORDS = 16384,     // resident capacity in wide words (>= sum over layers)
    parameter integer RLAT   = 2          // read->mac pipeline depth (cycles)
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS)-1:0]     w_base,    // word offset of this layer's transposed block
    // one-time load: 32-bit chunks assembled into LANES*4-bit wide words
    input  wire                          ld_rst,
    input  wire                          w_we,      // pulses once per 32-bit chunk
    input  wire [31:0]                   w_data,
    // per-call activation
    input  wire                          x_we,
    input  wire signed [7:0]             x_data,
    // run
    input  wire                          start,
    output reg                           done,
    // readback: one output element per call (2-cycle latency)
    input  wire [$clog2(MMAX)-1:0]       rd_addr,
    output reg  signed [31:0]            y_out
);
    localparam integer WBITS  = LANES*4;                        // wide word width
    localparam integer YBITS  = LANES*32;
    localparam integer LSH    = $clog2(LANES);                  // LANES is pow2
    localparam integer GROUPS = (MMAX + LANES - 1) / LANES;     // max groups in one layer
    localparam integer WAW    = $clog2(WWORDS);                 // resident word-address width
    localparam integer XAW    = $clog2(KMAX);
    localparam integer SUBW   = WBITS / 32;                     // 32-bit chunks per wide word
    localparam integer SSW    = (SUBW > 1) ? $clog2(SUBW) : 1;

    (* ram_style = "ultra" *) reg [WBITS-1:0] wmem [0:WWORDS-1];
    reg signed [7:0]                          xmem [0:KMAX-1];
    reg [YBITS-1:0]                           ymem [0:GROUPS-1];

    // ---- one-time load: assemble SUBW 32-bit chunks into one wide word ----------
    reg [WAW-1:0]    wword;        // current wide-word index in the resident URAM
    reg [SSW-1:0]    wsub;         // which 32-bit chunk within the wide word
    reg [WBITS-1:0]  wbuf;
    reg [XAW-1:0]    xptr;
    // insert the incoming 32-bit chunk at position wsub (low chunk first)
    wire [WBITS-1:0] wnext = wbuf | ({{(WBITS-32){1'b0}}, w_data} << (wsub*32));
    always @(posedge clk) begin
        if (ld_rst) begin wword <= 0; wsub <= 0; wbuf <= {WBITS{1'b0}}; xptr <= 0; end
        else begin
            if (w_we) begin
                wmem[wword] <= wnext;                  // commit partial word every chunk
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
    reg [XAW:0]          kc, kmac;
    reg [WAW-1:0]        grp_base;          // w_base + g*k_count, in words
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
                        grp_base <= w_base;                  // first group starts at the layer base
                        for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + 1'b1;
                    if (mac_v) begin
                        wsel = word_p[RLAT-1];
                        xsel = x_p[RLAT-1];
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
                            grp_base <= grp_base + k_count;      // next group: + K words
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
