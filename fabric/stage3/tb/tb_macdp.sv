// =============================================================================
// tb_macdp — bit-exact gate for mac_bank_dp (the double-pumped MAC bank).
//
// Drives BOTH:
//   * the REFERENCE single-clock `mac_bank` (the real leaf, compiled in from
//     gemm_banked_resident_vec.sv) over K cycles, and
//   * `mac_bank_dp` over K/2 clk cycles (2 K-steps/clk, accumulator in clk2x),
// from the SAME random (weight-word, activation-byte) K-step sequence, and
// asserts the final `acc` words are bit-identical for every lane.
//
// Clocks (MMCM 2x, 0-deg-aligned model):
//     always #5  clk   = ~clk;     // 100 MHz time-base -> 10 ns period
//     always #2.5 clk2x = ~clk2x;  // 2x, edges 0-deg aligned to clk
// The exact frequency is irrelevant for sim correctness; only the 2x ratio and
// the 0-deg alignment matter, which is exactly the silicon MMCM case.
//
// Data: w.mem holds K weight words (LANES*4 bits each, one hex token/line);
//       x.mem holds K activation bytes (8 bits each). pack_macdp writes both
//       and computes the golden acc; the TB dumps acc_ref.out / acc_dp.out and
//       Python does the bit-exact compare + prints MACDP_VERDICT.
// =============================================================================
`timescale 1ns / 1ps

`ifndef LANES
 `define LANES 128
`endif
`ifndef ABITS
 `define ABITS 24
`endif
`ifndef KVAL
 `define KVAL 1024
`endif

module tb;
    localparam integer LANES = `LANES;
    localparam integer ABITS = `ABITS;
    localparam integer KVAL  = `KVAL;   // must be even (2 K-steps/clk)

    // ---- 2x clocks (MMCM 0-deg, 2x ratio) ------------------------------------
    // clk edges at 0,5,10,...  clk2x edges at 2.5,7.5,12.5,...  -> clk2x is a
    // pure 2x of clk, sampling the clk LEVEL unambiguously (no coincident-edge
    // race): clk2x rising at 2.5 sees clk HIGH (phase 0), at 7.5 sees clk LOW
    // (phase 1). This models the MMCM 2x output; the half-period skew is benign
    // (any phase where each clk period contains exactly one high-clk and one
    // low-clk clk2x edge gives the same deterministic (w0,x0)->(w1,x1) order).
    reg clk   = 1'b0;
    reg clk2x = 1'b0;
    initial begin #2.5 forever #2.5 clk2x = ~clk2x; end
    always #5 clk = ~clk;         // 10 ns period; clk2x quarter-shifted

    // ---- DUT I/O -------------------------------------------------------------
    reg                 clr_ref, en_ref;
    reg [LANES*4-1:0]   w_ref;
    reg signed [7:0]    x_ref;
    wire [LANES*ABITS-1:0] acc_ref;

    reg                 clr_dp, en_dp;
    reg [LANES*4-1:0]   w0, w1;
    reg signed [7:0]    x0, x1;
    wire [LANES*ABITS-1:0] acc_dp;

    // reference single-clock bank (the REAL leaf)
    mac_bank #(.LANES(LANES), .ABITS(ABITS)) u_ref (
        .clk(clk), .clr(clr_ref), .en(en_ref), .w(w_ref), .x(x_ref), .acc(acc_ref));

    // double-pumped bank under test
    mac_bank_dp #(.LANES(LANES), .ABITS(ABITS)) u_dp (
        .clk(clk), .clk2x(clk2x), .clr(clr_dp), .en(en_dp),
        .w0(w0), .w1(w1), .x0(x0), .x1(x1), .acc(acc_dp));

    // ---- stimulus storage ----------------------------------------------------
    reg [LANES*4-1:0] wload [0:KVAL-1];
    reg [7:0]         xload [0:KVAL-1];
    integer i, f;

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("x.mem", xload);

        // -------- reset / clear both banks (clr held a full clk) --------------
        clr_ref = 1; en_ref = 0; w_ref = 0; x_ref = 0;
        clr_dp  = 1; en_dp  = 0; w0 = 0; w1 = 0; x0 = 0; x1 = 0;
        @(posedge clk); #1;
        @(posedge clk); #1;          // clr seen by both clk and clk2x domains
        clr_ref = 0; clr_dp = 0;

        // -------- drive the two banks in LOCKSTEP -----------------------------
        // The reference consumes 1 K-step/clk; the DP consumes 2 K-steps/clk.
        // We fork them so both run off the same `clk` over the same wall time,
        // each from the same K sequence.
        fork
            begin : ref_drive
                integer k;
                for (k = 0; k < KVAL; k = k + 1) begin
                    en_ref = 1; w_ref = wload[k]; x_ref = xload[k];
                    @(posedge clk); #1;
                end
                en_ref = 0;
            end
            begin : dp_drive
                integer j;
                for (j = 0; j < KVAL; j = j + 2) begin
                    en_dp = 1;
                    w0 = wload[j];   x0 = xload[j];
                    w1 = wload[j+1]; x1 = xload[j+1];
                    @(posedge clk); #1;
                end
                en_dp = 0;
            end
        join

        // -------- drain the clk2x pipeline tail, then sample ------------------
        @(posedge clk); @(posedge clk); #1;

        f = $fopen("acc_ref.out", "w");
        for (i = 0; i < LANES; i = i + 1)
            $fwrite(f, "%06x\n", acc_ref[i*ABITS +: ABITS]);
        $fclose(f);

        f = $fopen("acc_dp.out", "w");
        for (i = 0; i < LANES; i = i + 1)
            $fwrite(f, "%06x\n", acc_dp[i*ABITS +: ABITS]);
        $fclose(f);

        $display("TB_DONE lanes=%0d k=%0d", LANES, KVAL);
        $finish;
    end

    initial begin
        #5000000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
