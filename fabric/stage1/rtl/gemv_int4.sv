// -----------------------------------------------------------------------------
// gemv_int4 — Stage 1 on-chip INT4 x INT8 matrix-vector engine.
//
// Computes  y[m] = sum_k  W[m,k] * x[k]   for m in 0..M-1, k in 0..K-1
//   W : INT4 signed [-8,7], resident on-chip (the whole point — never DDR)
//   x : INT8 signed, the activation vector for one decode step
//   y : INT32 exact integer accumulation (dequant scales applied downstream)
//
// This is the bit-exact core the numpy golden (fabric/golden.py) checks against
// before any speed number is trusted. Correctness first; the throughput story is
// the PE_LANES parallelism plus closing timing at the fabric clock.
//
// Weight memory layout is the fabric/pack.py contract: packed bytes, row-major
// over (M, K), two nibbles per byte, low nibble = even k. The core reads the
// packed byte and selects the nibble in-datapath, so the URAM holds exactly the
// $readmemh image the exporter wrote.
//
// PE_LANES output rows accumulate in parallel; M is processed in
// ceil(M/PE_LANES) row-groups, one activation broadcast to all lanes per cycle,
// so a run is ~ ceil(M/PE_LANES) * K cycles. Written in a conservative
// procedural style so Icarus (sim) and Vivado (synth) both accept it.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_int4 #(
    parameter integer M        = 256,   // output features
    parameter integer K        = 256,   // input features (must be even)
    parameter integer PE_LANES = 16,    // output rows computed in parallel
    parameter         WFILE    = "",     // $readmemh image: packed INT4 weights
    parameter         XFILE    = ""      // $readmemh image: INT8 activations
) (
    input  wire        clk,
    input  wire        rst,     // synchronous, active high
    input  wire        start,   // pulse to begin a GEMV
    output reg         done,
    // Result read-out. Exposing y through a port keeps the whole accumulation
    // datapath + weight memory observable, so out-of-context synthesis reports
    // real utilisation/timing instead of pruning it all away. In a real system
    // this is the tap point for the AXI-stream reader / next stage.
    input  wire [$clog2(M)-1:0] rd_addr,
    output reg  signed [31:0]   y_out
);
    localparam integer KB     = K / 2;                       // packed bytes/row
    localparam integer GROUPS = (M + PE_LANES - 1) / PE_LANES;
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FLUSH = 2'd2, FIN = 2'd3;

    // Result vector — internal (unpacked array ports aren't portable across
    // simulators); the testbench reads it hierarchically as dut.y[m], and in a
    // real system a downstream reader/AXI adapter taps it the same way.
    reg signed [31:0] y [0:M-1];

    // On-chip weight image (packed bytes) and activation vector.
    (* rom_style = "ultra" *) reg [7:0] wmem [0:M*KB-1];
    reg signed [7:0]                     xmem [0:K-1];

    initial begin
        if (WFILE != "") $readmemh(WFILE, wmem);
        if (XFILE != "") $readmemh(XFILE, xmem);
    end

    reg [1:0]           state;
    integer             kcnt, gcnt, i;
    reg signed [31:0]   acc [0:PE_LANES-1];

    // procedural per-lane temporaries (blocking, used within one iteration)
    integer             row, waddr;
    reg [7:0]           wbyte;
    reg signed [3:0]    w4;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; done <= 1'b0; kcnt <= 0; gcnt <= 0;
            for (i = 0; i < PE_LANES; i = i + 1) acc[i] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        kcnt <= 0; gcnt <= 0;
                        for (i = 0; i < PE_LANES; i = i + 1) acc[i] <= 0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    for (i = 0; i < PE_LANES; i = i + 1) begin
                        row = gcnt * PE_LANES + i;
                        if (row < M) begin
                            waddr = row * KB + (kcnt >> 1);
                            wbyte = wmem[waddr];
                            w4    = kcnt[0] ? wbyte[7:4] : wbyte[3:0];  // odd:even
                            acc[i] <= acc[i] + w4 * xmem[kcnt];          // signed*signed
                        end
                    end
                    if (kcnt == K-1) state <= FLUSH;
                    else             kcnt <= kcnt + 1;
                end
                FLUSH: begin
                    for (i = 0; i < PE_LANES; i = i + 1)
                        if (gcnt * PE_LANES + i < M)
                            y[gcnt * PE_LANES + i] <= acc[i];
                    for (i = 0; i < PE_LANES; i = i + 1) acc[i] <= 0;
                    kcnt <= 0;
                    if (gcnt == GROUPS-1) state <= FIN;
                    else begin gcnt <= gcnt + 1; state <= RUN; end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    // Registered read of the result array — the observability tap (see ports).
    always @(posedge clk) y_out <= y[rd_addr];
endmodule
