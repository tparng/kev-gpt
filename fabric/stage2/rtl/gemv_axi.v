// -----------------------------------------------------------------------------
// gemv_axi — AXI4-Lite shell around the Stage 1 gemv_int4 core so the A53 (PS)
// can drive it and read results back over the PS-PL AXI bridge. This is the
// Stage 2 "first PL silicon" wrapper: weights + one activation vector are baked
// into the core's ROMs (WFILE/XFILE), software pulses START, polls DONE, then
// reads y[m] one element at a time. A hardware cycle counter (START->DONE) gives
// the on-fabric latency directly.
//
// Register map (AXI-Lite, 32-bit, byte offsets):
//   0x00 CTRL    (W)  bit0 = start (write 1, self-clearing pulse)
//                     bit1 = soft reset (held while 1)
//   0x04 STATUS  (R)  bit0 = done   bit1 = busy
//   0x08 RD_ADDR (W)  index m of y to read back (0..M-1)
//   0x0C RD_DATA (R)  y[RD_ADDR], signed INT32 (write RD_ADDR first, then read)
//   0x10 CYCLES  (R)  START->DONE cycle count (latched at DONE)
//   0x14 IDCODE  (R)  0x4745_4D56 ("GEMV") — sanity check the bitstream is live
//
// Conservative style; Vivado synthesis target (not iverilog).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_axi #(
    parameter integer M        = 256,
    parameter integer K        = 256,
    parameter integer PE_LANES = 16,
    parameter         WFILE    = "",
    parameter         XFILE    = "",
    parameter integer C_S_AXI_ADDR_WIDTH = 6
) (
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output reg                               S_AXI_AWREADY,
    input  wire [31:0]                       S_AXI_WDATA,
    input  wire [3:0]                        S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output reg                               S_AXI_WREADY,
    output reg  [1:0]                        S_AXI_BRESP,
    output reg                               S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output reg                               S_AXI_ARREADY,
    output reg  [31:0]                       S_AXI_RDATA,
    output reg  [1:0]                        S_AXI_RRESP,
    output reg                               S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);
    wire clk = S_AXI_ACLK;
    wire aresetn = S_AXI_ARESETN;

    // ---- write channel -----------------------------------------------------
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr;
    wire wr_en = S_AXI_AWVALID & S_AXI_WVALID & ~S_AXI_BVALID;

    reg        start_pulse;
    reg        soft_reset;
    reg [$clog2(M)-1:0] rd_addr;

    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            S_AXI_BVALID  <= 1'b0; S_AXI_BRESP  <= 2'b00;
            awaddr <= 0; start_pulse <= 1'b0; soft_reset <= 1'b0; rd_addr <= 0;
        end else begin
            start_pulse <= 1'b0;                 // one-cycle pulse by default
            // address+data accept (fixed 1-cycle handshake)
            if (wr_en && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1; S_AXI_WREADY <= 1'b1; awaddr <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            end
            // commit the write one cycle after accept
            if (S_AXI_AWREADY) begin
                case (awaddr[5:2])
                    4'h0: begin start_pulse <= S_AXI_WDATA[0]; soft_reset <= S_AXI_WDATA[1]; end
                    4'h2: rd_addr <= S_AXI_WDATA[$clog2(M)-1:0];
                    default: ;
                endcase
                S_AXI_BVALID <= 1'b1; S_AXI_BRESP <= 2'b00;
            end else if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // ---- read channel ------------------------------------------------------
    reg [3:0] araddr_idx;
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_ARREADY <= 1'b0; S_AXI_RVALID <= 1'b0; S_AXI_RRESP <= 2'b00;
            S_AXI_RDATA <= 32'b0; araddr_idx <= 4'h0;
        end else begin
            if (S_AXI_ARVALID && !S_AXI_ARREADY && !S_AXI_RVALID) begin
                S_AXI_ARREADY <= 1'b1; araddr_idx <= S_AXI_ARADDR[5:2];
            end else begin
                S_AXI_ARREADY <= 1'b0;
            end
            if (S_AXI_ARREADY) begin
                S_AXI_RVALID <= 1'b1; S_AXI_RRESP <= 2'b00;
                case (araddr_idx)
                    4'h1: S_AXI_RDATA <= {30'b0, core_busy, core_done};
                    4'h3: S_AXI_RDATA <= core_y_out;
                    4'h4: S_AXI_RDATA <= cycles_latched;
                    4'h5: S_AXI_RDATA <= 32'h4745_4D56;       // "GEMV"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

    // ---- core + cycle counter ---------------------------------------------
    wire               core_done;
    wire signed [31:0] core_y_out;
    reg                core_busy;
    reg  [31:0]        cycles_run, cycles_latched;
    wire               core_rst = ~aresetn | soft_reset;

    always @(posedge clk) begin
        if (core_rst) begin
            core_busy <= 1'b0; cycles_run <= 32'b0; cycles_latched <= 32'b0;
        end else begin
            if (start_pulse) begin
                core_busy <= 1'b1; cycles_run <= 32'b0;
            end else if (core_busy) begin
                cycles_run <= cycles_run + 1'b1;
                if (core_done) begin
                    core_busy <= 1'b0; cycles_latched <= cycles_run;
                end
            end
        end
    end

    // Stage 2 keystone core (registered-read BRAM, runtime-loadable, bit-exact).
    // Load ports tied off for the BRAM-init demo; AXI-driven load is the next step.
    localparam integer CORE_WAW = $clog2(M*K/2);
    localparam integer CORE_XAW = $clog2(K);
    gemv_core #(.M(M), .K(K), .WFILE(WFILE), .XFILE(XFILE)) u_gemv (
        .clk    (clk),
        .rst    (core_rst),
        .start  (start_pulse),
        .done   (core_done),
        .w_we   (1'b0), .w_addr({CORE_WAW{1'b0}}), .w_data(8'b0),
        .x_we   (1'b0), .x_addr({CORE_XAW{1'b0}}), .x_data(8'b0),
        .rd_addr(rd_addr),
        .y_out  (core_y_out)
    );
endmodule
