// -----------------------------------------------------------------------------
// gemv_core — Stage 2 keystone: a synthesis-clean, runtime-LOADABLE, bit-exact
// INT4xINT8 GEMV. Replaces gemv_int4 for the on-fabric path.
//
// Why this exists: gemv_int4's weight ROM was a combinational `initial $readmemh`
// array read by all lanes at once. That is sim-correct but synthesis DROPS the
// init ("Net wmem does not have driver" -> pruned to zero) and the multi-port read
// won't map to BRAM/URAM. This core fixes both:
//   * REGISTERED read (1-cycle latency) -> infers a real BRAM/URAM.
//   * single read port (PE_LANES=1 for now) -> maps cleanly; per-lane banking is
//     the parallel follow-up.
//   * WRITE ports (w_we/x_we) so weights+activation load at runtime over AXI
//     (URAM cannot be bitstream-initialised, so the real model DMAs flash->URAM
//     once at boot). WFILE/XFILE `$readmemh` init is kept for sim + the BRAM demo.
//
//   y[m] = sum_k W[m,k]*x[k]   W INT4 signed, x INT8 signed, y INT32 exact.
// Packed-byte layout is the fabric/pack.py contract (low nibble = even k).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_core #(
    parameter integer M     = 256,
    parameter integer K     = 256,    // even
    parameter         WFILE = "",
    parameter         XFILE = ""
) (
    input  wire                     clk,
    input  wire                     rst,      // synchronous, active high
    input  wire                     start,
    output reg                      done,
    // runtime load (BRAM/URAM write); leave idle to use WFILE/XFILE init
    input  wire                     w_we,
    input  wire [WAW-1:0]           w_addr,
    input  wire [7:0]               w_data,
    input  wire                     x_we,
    input  wire [XAW-1:0]           x_addr,
    input  wire [7:0]               x_data,
    // result readback (registered)
    input  wire [MAW-1:0]           rd_addr,
    output reg  signed [31:0]       y_out
);
    localparam integer KB     = K / 2;          // packed bytes per row
    localparam integer WDEPTH = M * KB;
    localparam integer WAW    = $clog2(WDEPTH);
    localparam integer XAW    = $clog2(K);
    localparam integer MAW    = $clog2(M);

    (* ram_style = "block" *) reg [7:0] wmem [0:WDEPTH-1];
    reg signed [7:0]                    xmem [0:K-1];
    reg signed [31:0]                   y    [0:M-1];

    initial begin
        if (WFILE != "") $readmemh(WFILE, wmem);
        if (XFILE != "") $readmemh(XFILE, xmem);
    end

    // ---- write ports (runtime load), separate process -> simple dual-port -----
    always @(posedge clk) begin
        if (w_we) wmem[w_addr] <= w_data;
        if (x_we) xmem[x_addr] <= x_data;
    end

    // ---- FSM + 1-deep registered-read pipeline --------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]         state;
    reg [MAW:0]       mrow;        // 0..M
    reg [XAW:0]       kc;          // issue counter 0..K
    reg [XAW:0]       kmac;        // mac counter 0..K
    reg               rvalid;      // a read issued last cycle -> MAC it now
    reg               kpar_r;      // nibble select, aligned with the read
    reg        [7:0]  wbyte_r;
    reg signed [7:0]  x_r;
    reg signed [31:0] acc;

    wire [WAW-1:0]    raddr_w = mrow[MAW-1:0] * KB + kc[XAW:1];   // m*KB + (k>>1)
    wire [XAW-1:0]    raddr_x = kc[XAW-1:0];
    wire signed [3:0] w4   = kpar_r ? $signed(wbyte_r[7:4]) : $signed(wbyte_r[3:0]);
    wire signed [31:0] prod = w4 * x_r;

    // registered BRAM read (1-cycle latency); meaningful only when rvalid
    always @(posedge clk) begin
        wbyte_r <= wmem[raddr_w];
        x_r     <= xmem[raddr_x];
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; done <= 1'b0; rvalid <= 1'b0;
            mrow <= 0; kc <= 0; kmac <= 0; acc <= 0; kpar_r <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        mrow <= 0; kc <= 0; kmac <= 0; acc <= 0; rvalid <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    // issue stage: present a read for (mrow, kc)
                    if (kc < K) begin
                        rvalid <= 1'b1; kpar_r <= kc[0]; kc <= kc + 1'b1;
                    end else begin
                        rvalid <= 1'b0;
                    end
                    // mac stage: consume the read issued last cycle
                    if (rvalid) begin
                        acc  <= acc + prod;
                        kmac <= kmac + 1'b1;
                    end
                    // row complete: all K products accumulated
                    if (kmac == K) begin
                        y[mrow[MAW-1:0]] <= acc;
                        if (mrow == M-1) begin
                            state <= FIN;
                        end else begin
                            mrow <= mrow + 1'b1; kc <= 0; kmac <= 0;
                            acc  <= 0; rvalid <= 1'b0;
                        end
                    end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    always @(posedge clk) y_out <= y[rd_addr];   // registered readback
endmodule
