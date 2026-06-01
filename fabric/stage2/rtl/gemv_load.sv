// -----------------------------------------------------------------------------
// gemv_load — Stage 2 step 2: runtime-LOADABLE, runtime-SIZED INT4xINT8 GEMV.
//
// One engine for ALL 17 layers from a single bitstream. Weights + activation
// stream in over AXI (auto-incrementing load pointers, reset by ld_rst); M and K
// are runtime inputs (layers vary: 256x256, 768x256, 1024x256, 256x1024, 193x256).
// Memories are sized for the largest single layer (max M*K/2 = 131072 bytes), and
// load once into BRAM now / URAM at boot for the real model.
//
//   y[m] = sum_k W[m,k]*x[k]   W INT4 signed, x INT8 signed, y INT32 exact.
// Packed-byte layout = fabric/pack.py (low nibble = even k). Registered BRAM read
// (1-cycle latency) so synthesis bakes real block RAM, not a pruned ROM.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_load #(
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WBYTES = 131072   // max packed weight bytes over all layers
) (
    input  wire                        clk,
    input  wire                        rst,
    // per-layer config (set before start)
    input  wire [$clog2(MMAX+1)-1:0]   m_count,   // M (rows)
    input  wire [$clog2(KMAX+1)-1:0]   k_count,   // K (cols, even)
    // streaming load (auto-incrementing pointers; pulse ld_rst to rewind)
    input  wire                        ld_rst,
    input  wire                        w_we,
    input  wire [7:0]                  w_data,
    input  wire                        x_we,
    input  wire [7:0]                  x_data,
    // run
    input  wire                        start,
    output reg                         done,
    // result readback (registered)
    input  wire [$clog2(MMAX)-1:0]     rd_addr,
    output reg  signed [31:0]          y_out
);
    localparam integer WAW = $clog2(WBYTES);
    localparam integer XAW = $clog2(KMAX);
    localparam integer MAW = $clog2(MMAX);

    (* ram_style = "block" *) reg [7:0] wmem [0:WBYTES-1];
    reg signed [7:0]                    xmem [0:KMAX-1];
    reg signed [31:0]                   y    [0:MMAX-1];

    // ---- streaming load: auto-incrementing write pointers --------------------
    reg [WAW-1:0] wptr;
    reg [XAW-1:0] xptr;
    always @(posedge clk) begin
        if (ld_rst) begin
            wptr <= 0; xptr <= 0;
        end else begin
            if (w_we) begin wmem[wptr] <= w_data; wptr <= wptr + 1'b1; end
            if (x_we) begin xmem[xptr] <= x_data; xptr <= xptr + 1'b1; end
        end
    end

    // ---- FSM + 1-deep registered-read pipeline -------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]         state;
    reg [MAW:0]       mrow;
    reg [XAW:0]       kc;
    reg [XAW:0]       kmac;
    reg [WAW-1:0]     row_base;     // running m*KB (avoids a runtime multiply)
    reg               rvalid, kpar_r;
    reg        [7:0]  wbyte_r;
    reg signed [7:0]  x_r;
    reg signed [31:0] acc;

    wire [XAW-1:0]    kb = k_count[XAW:1];                 // K/2 bytes per row
    wire [WAW-1:0]    raddr_w = row_base + kc[XAW:1];
    wire [XAW-1:0]    raddr_x = kc[XAW-1:0];
    wire signed [3:0] w4 = kpar_r ? $signed(wbyte_r[7:4]) : $signed(wbyte_r[3:0]);
    wire signed [31:0] prod = w4 * x_r;

    always @(posedge clk) begin
        wbyte_r <= wmem[raddr_w];
        x_r     <= xmem[raddr_x];
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; done <= 1'b0; rvalid <= 1'b0;
            mrow <= 0; kc <= 0; kmac <= 0; acc <= 0; row_base <= 0; kpar_r <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        mrow <= 0; kc <= 0; kmac <= 0; acc <= 0;
                        row_base <= 0; rvalid <= 1'b0; state <= RUN;
                    end
                end
                RUN: begin
                    if (kc < k_count) begin
                        rvalid <= 1'b1; kpar_r <= kc[0]; kc <= kc + 1'b1;
                    end else begin
                        rvalid <= 1'b0;
                    end
                    if (rvalid) begin
                        acc  <= acc + prod;
                        kmac <= kmac + 1'b1;
                    end
                    if (kmac == k_count) begin
                        y[mrow[MAW-1:0]] <= acc;
                        if (mrow == m_count - 1) begin
                            state <= FIN;
                        end else begin
                            mrow <= mrow + 1'b1; kc <= 0; kmac <= 0;
                            acc  <= 0; rvalid <= 1'b0; row_base <= row_base + kb;
                        end
                    end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    always @(posedge clk) y_out <= y[rd_addr];
endmodule
