// -----------------------------------------------------------------------------
// weight_bank_tdp — the SHARED resident URAM weight image, read through BOTH
// true-dual-port (TDP) URAM ports so two independent GEMM cohorts stream it at
// once. This is the only resource split-brain shares.
//
//   port A : loader WRITES at boot (we_a) + cohort-1 READS at runtime (the
//            loader is idle once boot completes, so the address mux is clean:
//            during load addr_a = wword/write; at runtime addr_a = cohort-1 read).
//   port B : cohort-0 READS (today's single-read path, unchanged timing).
//
// Both ports return a REGISTERED read (1-cycle latency) = the pipeline stage 0
// the cohort datapath expects (the old per-bank `rd <= mem[waddr]`).
//
// DUAL-DIALECT (the codebase pattern, see gemm_banked_resident_vec + log §16):
//   `ifdef SYNTHESIS : xpm_memory_tdpram, MEMORY_PRIMITIVE="ultra" at bank
//        geometry — the PROVEN dual-port URAM mapping (C:/kevbuild/uram_dual:
//        URAM 56 / 0 LUT / 0 BRAM, both ports independent). HDL inference of
//        TDP UltraRAM is dead in 2025.2 (two-address ports fall to BRAM); do
//        NOT try to infer it.
//   else (iverilog) : a behavioral 2-port array. Gates verify THIS side; the
//        board run is the final bit-exactness check on the XPM side.
//
// The write path mirrors the old core: SUBW 32-bit chunks are assembled into a
// WBITS wide word and committed to all NB banks in one shot at `wword`.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module weight_bank_tdp #(
    parameter integer LANES  = 128,
    parameter integer WWORDS = 25600,
    // DOUBLE-PUMP-100K Stage 1: DP=1 each cohort consumes 2 weight words/clk, so
    // each read port serves raddr AND raddr+1. In sim (Stage 1a) this is modelled
    // as a second behavioral read of the same array (the plan-authorised "two
    // reads of the URAM" model). On silicon (Stage 1b) it is the SAME URAM port
    // clocked at clk2x — one word per clk2x edge — NOT a second physical port
    // (TDP UltraRAM has only two ports, both already used). The SYNTHESIS branch
    // here is unchanged (single read); the clk2x dual-beat read lands in Stage 1b.
    parameter integer DP     = 0,
    parameter integer WBITS  = LANES*4
) (
    input  wire                          clk,
    input  wire                          clk2x,    // 2x clk (DP=1 silicon; unused in sim model)
    // ---- boot load (drives port A writes) ----
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [31:0]                   w_data,
    // ---- port B read (cohort 0) ----
    input  wire [$clog2(WWORDS)-1:0]     raddr_b,
    output wire [WBITS-1:0]              rword_b,
    output wire [WBITS-1:0]              rword1_b,   // DP: word at raddr_b+1 (phase 1)
    // ---- port A read (cohort 1) — only sampled at runtime (loader idle) ----
    input  wire [$clog2(WWORDS)-1:0]     raddr_a,
    output wire [WBITS-1:0]              rword_a,
    output wire [WBITS-1:0]              rword1_a    // DP: word at raddr_a+1 (phase 1)
);
    localparam integer WAW  = $clog2(WWORDS);
    localparam integer SUBW = WBITS / 32;
    localparam integer SSW  = (SUBW > 1) ? $clog2(SUBW) : 1;

    // URAM-native banking (72b) above 512 bits; L=128 keeps one 512b word.
    localparam integer BANKW = (WBITS > 512) ? 72 : WBITS;
    localparam integer NB    = (WBITS + BANKW - 1) / BANKW;
    localparam integer WPAD  = NB * BANKW;

    // ---- write-word assembler (identical to the old core) ----
    reg [WAW-1:0]   wword;
    reg [SSW-1:0]   wsub;
    reg [WBITS-1:0] wbuf;
    wire [WBITS-1:0] wnext = wbuf | ({{(WBITS-32){1'b0}}, w_data} << (wsub*32));
    wire [WPAD-1:0]  wnext_pad = {{(WPAD-WBITS){1'b0}}, wnext};
    wire             wcommit = w_we && (wsub == SUBW-1);
    always @(posedge clk) begin
        if (ld_rst) begin wword <= 0; wsub <= 0; wbuf <= {WBITS{1'b0}}; end
        else if (w_we) begin
            if (wsub == SUBW-1) begin
                wword <= wword + 1'b1; wsub <= 0; wbuf <= {WBITS{1'b0}};
            end else begin
                wbuf <= wnext; wsub <= wsub + 1'b1;
            end
        end
    end

    // port A address: write address at boot, cohort-1 read address at runtime.
    wire [WAW-1:0] aaddr = wcommit ? wword : raddr_a;
    // DP phase-1 read addresses (raddr+1). Port A's +1 is don't-care during a
    // boot write (the loader never reads); at runtime aaddr==raddr_a.
    wire [WAW-1:0] aaddr1 = aaddr   + 1'b1;
    wire [WAW-1:0] baddr1 = raddr_b + 1'b1;

    wire [WPAD-1:0] rpad_a, rpad_b, rpad1_a, rpad1_b;
    assign rword_a  = rpad_a[WBITS-1:0];
    assign rword_b  = rpad_b[WBITS-1:0];
    assign rword1_a = rpad1_a[WBITS-1:0];
    assign rword1_b = rpad1_b[WBITS-1:0];

    genvar gb;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_w
`ifdef SYNTHESIS
            // PROVEN: xpm_memory_tdpram ultra, both ports independent.
            // Port A = write(boot)/read(cohort1); Port B = read(cohort0).
            wire [BANKW-1:0] douta, doutb;
            xpm_memory_tdpram #(
                .ADDR_WIDTH_A(WAW), .ADDR_WIDTH_B(WAW),
                .BYTE_WRITE_WIDTH_A(BANKW), .BYTE_WRITE_WIDTH_B(BANKW),
                .CLOCKING_MODE("common_clock"),
                .MEMORY_PRIMITIVE("ultra"),
                .MEMORY_SIZE(BANKW*WWORDS),
                .READ_DATA_WIDTH_A(BANKW), .READ_DATA_WIDTH_B(BANKW),
                .READ_LATENCY_A(1), .READ_LATENCY_B(1),
                .WRITE_DATA_WIDTH_A(BANKW), .WRITE_DATA_WIDTH_B(BANKW),
                // TDP UltraRAM REQUIRES no_change (XPM_MEMORY 40-14) — and it is
                // what the proven uram_dual experiment used. Runtime-identical to
                // the behavioral model: the loader is idle (wea=0) once boot ends.
                .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
            ) u_tdp (
                .clka(clk), .clkb(clk),
                .rsta(1'b0), .rstb(1'b0),
                .ena(1'b1), .enb(1'b1),
                .regcea(1'b1), .regceb(1'b1),
                .wea(wcommit), .web(1'b0),
                .addra(aaddr), .addrb(raddr_b),
                .dina(wnext_pad[gb*BANKW +: BANKW]), .dinb({BANKW{1'b0}}),
                .douta(douta), .doutb(doutb),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb(),
                .sleep(1'b0)
            );
            assign rpad_a[gb*BANKW +: BANKW] = douta;
            assign rpad_b[gb*BANKW +: BANKW] = doutb;
            // DP phase-1 read: Stage 1b implements this as the SAME port clocked
            // at clk2x (a second beat), NOT a third XPM port. Stubbed 0 here so
            // the synth view still elaborates; no synth build happens in Stage 1a.
            assign rpad1_a[gb*BANKW +: BANKW] = {BANKW{1'b0}};
            assign rpad1_b[gb*BANKW +: BANKW] = {BANKW{1'b0}};
`else
            // behavioral 2-port (sim): independent registered reads, one write
            // site on port A. Both reads are 1-cycle = the cohort pipeline stage 0.
            reg [BANKW-1:0] mem [0:WWORDS-1];
            reg [BANKW-1:0] rd_a, rd_b;
            always @(posedge clk) begin
                if (wcommit) mem[wword] <= wnext_pad[gb*BANKW +: BANKW];
                rd_a <= mem[raddr_a];
                rd_b <= mem[raddr_b];
            end
            assign rpad_a[gb*BANKW +: BANKW] = rd_a;
            assign rpad_b[gb*BANKW +: BANKW] = rd_b;
            // DP: second behavioral read at raddr+1, same 1-cycle latency (the
            // plan-authorised "two reads of the URAM" sim model). DP=0 -> 0.
            if (DP != 0) begin : g_rd2
                reg [BANKW-1:0] rd1_a, rd1_b;
                always @(posedge clk) begin
                    rd1_a <= mem[aaddr1];
                    rd1_b <= mem[baddr1];
                end
                assign rpad1_a[gb*BANKW +: BANKW] = rd1_a;
                assign rpad1_b[gb*BANKW +: BANKW] = rd1_b;
            end else begin : g_no_rd2
                assign rpad1_a[gb*BANKW +: BANKW] = {BANKW{1'b0}};
                assign rpad1_b[gb*BANKW +: BANKW] = {BANKW{1'b0}};
            end
`endif
        end
    endgenerate
endmodule
