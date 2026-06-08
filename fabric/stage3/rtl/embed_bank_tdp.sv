// -----------------------------------------------------------------------------
// embed_bank_tdp — the SHARED resident token-embedding URAM image, read through
// BOTH true-dual-port (TDP) URAM ports so the two GEMM cohorts each get a REAL
// embed read port. This recovers the coh1/coh0 const-prop asymmetry: with a
// fixed-priority single-port arbiter (`emb_gnt0 = emb_req0`), Vivado const-prop's
// coh0's embed-wait away (coh0's nl_engine ~5.6k LUT smaller) while coh1 pays full
// freight AND an arbitration wait. With both ports gnt==req, BOTH nl_engines
// simplify and the asymmetry (+ coh1's embed stalls) disappears.
//
//   port A : embed loader WRITES at boot (we_a) + cohort-1 READS at runtime (the
//            loader is idle once boot completes, so the address mux is clean:
//            during load addr_a = write word; at runtime addr_a = cohort-1 read).
//   port B : cohort-0 READS (today's single-read timing, unchanged).
//
// Both ports return a REGISTERED read (1-cycle latency) = the embed pipeline stage
// the nl_engine expects: emb_q <= emb_w[emb_addr], one cycle after the granted
// address (the old single-port contract, now per-port).
//
// DUAL-DIALECT (the codebase pattern, see weight_bank_tdp + log §16/§2f3ba17):
//   `ifdef SYNTHESIS : xpm_memory_tdpram, MEMORY_PRIMITIVE="ultra" — the PROVEN
//        dual-port URAM mapping. HDL inference of TDP UltraRAM is dead in 2025.2
//        (two-address ports fall to BRAM, no budget); do NOT try to infer it.
//        WRITE_MODE "no_change" BOTH ports (XPM_MEMORY 40-14 requires it for TDP
//        UltraRAM; read_first is a synth ERROR).
//   else (iverilog) : a behavioral 2-port array, $readmemh-initialised at boot —
//        bit-identical to the old inferred emb_w. Gates verify THIS side.
//
// The write data (el_next) and per-word commit/address are assembled in
// sequencer_sb (the el loader, which ALSO produces the pos_emb broadcast path);
// this module only owns the dual-port image and its two read ports.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module embed_bank_tdp #(
    parameter integer P     = 8,
    parameter integer VOCAB = 193,
    parameter integer EROWS = 32,
    // DOUBLE-PUMP-100K: DP=1 maps embed to BRAM (block) instead of URAM, freeing
    // the 8 URAM the 2-K-wide weight image needs to fit the 64-URAM budget.
    parameter integer DP    = 0,
    parameter integer EBITS = P*32
) (
    input  wire                          clk,
    // ---- boot write (port A), assembled by the el loader in sequencer_sb ----
    input  wire                          w_we,        // commit one wide word
    input  wire [$clog2(VOCAB*EROWS)-1:0] w_addr,
    input  wire [EBITS-1:0]              w_data,
    // ---- port B read (cohort 0) ----
    input  wire [$clog2(VOCAB*EROWS)-1:0] raddr_b,
    output wire [EBITS-1:0]              rword_b,
    // ---- port A read (cohort 1) — only sampled at runtime (loader idle) ----
    input  wire [$clog2(VOCAB*EROWS)-1:0] raddr_a,
    output wire [EBITS-1:0]              rword_a
);
    localparam integer WORDS = VOCAB*EROWS;
    localparam integer EAW   = $clog2(WORDS);
    // URAM for the single-pump record; BRAM for the double-pump (frees URAM).
    localparam EPRIM = (DP != 0) ? "block" : "ultra";

    // URAM-native banking (72b) above 512 bits; EBITS=256 (P=8) keeps one word.
    localparam integer BANKW = (EBITS > 512) ? 72 : EBITS;
    localparam integer NB    = (EBITS + BANKW - 1) / BANKW;
    localparam integer EPAD  = NB * BANKW;

    wire [EPAD-1:0] wpad = {{(EPAD-EBITS){1'b0}}, w_data};

    // port A address: write address at boot, cohort-1 read address at runtime.
    wire [EAW-1:0] aaddr = w_we ? w_addr : raddr_a;

    wire [EPAD-1:0] rpad_a, rpad_b;
    assign rword_a = rpad_a[EBITS-1:0];
    assign rword_b = rpad_b[EBITS-1:0];

`ifndef SYNTHESIS
    // behavioral side needs the flat image for $readmemh (the old emb_w layout):
    // pack the per-bank reads from one wide array so init matches tok_emb_w.mem.
    reg [EBITS-1:0] emb_w [0:WORDS-1];
    initial begin $readmemh("tok_emb_w.mem", emb_w); end
    reg [EBITS-1:0] rd_a, rd_b;
    always @(posedge clk) begin
        if (w_we) emb_w[w_addr] <= w_data;
        rd_a <= emb_w[aaddr];          // port A: write or cohort-1 read (loader idle)
        rd_b <= emb_w[raddr_b];        // port B: cohort-0 read
    end
    assign rpad_a = {{(EPAD-EBITS){1'b0}}, rd_a};
    assign rpad_b = {{(EPAD-EBITS){1'b0}}, rd_b};
`else
    genvar gb;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_e
            // PROVEN: xpm_memory_tdpram ultra, both ports independent.
            // Port A = write(boot)/read(cohort1); Port B = read(cohort0).
            wire [BANKW-1:0] douta, doutb;
            xpm_memory_tdpram #(
                .ADDR_WIDTH_A(EAW), .ADDR_WIDTH_B(EAW),
                .BYTE_WRITE_WIDTH_A(BANKW), .BYTE_WRITE_WIDTH_B(BANKW),
                .CLOCKING_MODE("common_clock"),
                .MEMORY_PRIMITIVE(EPRIM),
                .MEMORY_SIZE(BANKW*WORDS),
                .READ_DATA_WIDTH_A(BANKW), .READ_DATA_WIDTH_B(BANKW),
                .READ_LATENCY_A(1), .READ_LATENCY_B(1),
                .WRITE_DATA_WIDTH_A(BANKW), .WRITE_DATA_WIDTH_B(BANKW),
                // TDP UltraRAM REQUIRES no_change (XPM_MEMORY 40-14). The loader is
                // idle (wea=0) once boot ends, so runtime read == behavioral model.
                .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
            ) u_tdp (
                .clka(clk), .clkb(clk),
                .rsta(1'b0), .rstb(1'b0),
                .ena(1'b1), .enb(1'b1),
                .regcea(1'b1), .regceb(1'b1),
                .wea(w_we), .web(1'b0),
                .addra(aaddr), .addrb(raddr_b),
                .dina(wpad[gb*BANKW +: BANKW]), .dinb({BANKW{1'b0}}),
                .douta(douta), .doutb(doutb),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb(),
                .sleep(1'b0)
            );
            assign rpad_a[gb*BANKW +: BANKW] = douta;
            assign rpad_b[gb*BANKW +: BANKW] = doutb;
        end
    endgenerate
`endif
endmodule
