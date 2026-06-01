// -----------------------------------------------------------------------------
// gemv_axi_load — AXI4-Lite shell around gemv_load so the A53 can stream a
// layer's weights+activation, set runtime M/K, run, and read results. One
// bitstream runs all 17 layers by re-loading weights per layer.
//
// Register map (AXI-Lite, 32-bit byte offsets):
//   0x00 CTRL    (W) b0=start  b1=ld_rst (rewind load ptrs)  b2=soft reset
//   0x04 STATUS  (R) b0=done(latched)  b1=busy
//   0x08 M_COUNT (W) rows M for the loaded layer
//   0x0C K_COUNT (W) cols K (even)
//   0x10 W_DATA  (W) low byte -> wmem[wptr++]  (stream weights, one byte/write)
//   0x14 X_DATA  (W) low byte -> xmem[xptr++]  (stream activation)
//   0x18 RD_ADDR (W) result index m
//   0x1C RD_DATA (R) y[RD_ADDR] (signed INT32)
//   0x20 CYCLES  (R) START->DONE cycle count (latched)
//   0x24 IDCODE  (R) 0x4745_4D4C ("GEML")
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_axi_load #(
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WBYTES = 131072,
    parameter integer C_S_AXI_ADDR_WIDTH = 6
) (
    input  wire                          S_AXI_ACLK,
    input  wire                          S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  wire [2:0]                    S_AXI_AWPROT,
    input  wire                          S_AXI_AWVALID,
    output reg                           S_AXI_AWREADY,
    input  wire [31:0]                   S_AXI_WDATA,
    input  wire [3:0]                    S_AXI_WSTRB,
    input  wire                          S_AXI_WVALID,
    output reg                           S_AXI_WREADY,
    output reg  [1:0]                    S_AXI_BRESP,
    output reg                           S_AXI_BVALID,
    input  wire                          S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  wire [2:0]                    S_AXI_ARPROT,
    input  wire                          S_AXI_ARVALID,
    output reg                           S_AXI_ARREADY,
    output reg  [31:0]                   S_AXI_RDATA,
    output reg  [1:0]                    S_AXI_RRESP,
    output reg                           S_AXI_RVALID,
    input  wire                          S_AXI_RREADY
);
    localparam integer MCW = $clog2(MMAX+1);
    localparam integer KCW = $clog2(KMAX+1);
    localparam integer MAW = $clog2(MMAX);

    wire clk = S_AXI_ACLK;
    wire aresetn = S_AXI_ARESETN;

    // ---- write channel -----------------------------------------------------
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr;
    wire wr_en = S_AXI_AWVALID & S_AXI_WVALID & ~S_AXI_BVALID;

    reg              start_pulse, ld_rst, soft_reset, w_we, x_we;
    reg [7:0]        w_data, x_data;
    reg [MCW-1:0]    m_count;
    reg [KCW-1:0]    k_count;
    reg [MAW-1:0]    rd_addr;

    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            S_AXI_BVALID  <= 1'b0; S_AXI_BRESP <= 2'b00; awaddr <= 0;
            start_pulse <= 1'b0; ld_rst <= 1'b0; soft_reset <= 1'b0;
            w_we <= 1'b0; x_we <= 1'b0; w_data <= 8'b0; x_data <= 8'b0;
            m_count <= 0; k_count <= 0; rd_addr <= 0;
        end else begin
            start_pulse <= 1'b0; ld_rst <= 1'b0; w_we <= 1'b0; x_we <= 1'b0;
            if (wr_en && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1; S_AXI_WREADY <= 1'b1; awaddr <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            end
            if (S_AXI_AWREADY) begin
                case (awaddr[5:2])
                    4'h0: begin start_pulse <= S_AXI_WDATA[0]; ld_rst <= S_AXI_WDATA[1];
                                soft_reset <= S_AXI_WDATA[2]; end
                    4'h2: m_count <= S_AXI_WDATA[MCW-1:0];
                    4'h3: k_count <= S_AXI_WDATA[KCW-1:0];
                    4'h4: begin w_we <= 1'b1; w_data <= S_AXI_WDATA[7:0]; end
                    4'h5: begin x_we <= 1'b1; x_data <= S_AXI_WDATA[7:0]; end
                    4'h6: rd_addr <= S_AXI_WDATA[MAW-1:0];
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
                    4'h1: S_AXI_RDATA <= {30'b0, core_busy, done_latched};
                    4'h7: S_AXI_RDATA <= core_y_out;
                    4'h8: S_AXI_RDATA <= cycles_latched;
                    4'h9: S_AXI_RDATA <= 32'h4745_4D4C;       // "GEML"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

    // ---- core + cycle/done latch ------------------------------------------
    wire               core_done;
    wire signed [31:0] core_y_out;
    reg                core_busy, done_latched;
    reg  [31:0]        cycles_run, cycles_latched;
    wire               core_rst = ~aresetn | soft_reset;

    always @(posedge clk) begin
        if (core_rst) begin
            core_busy <= 1'b0; cycles_run <= 0; cycles_latched <= 0; done_latched <= 1'b0;
        end else if (start_pulse) begin
            core_busy <= 1'b1; cycles_run <= 0; done_latched <= 1'b0;
        end else if (core_busy) begin
            cycles_run <= cycles_run + 1'b1;
            if (core_done) begin
                core_busy <= 1'b0; cycles_latched <= cycles_run; done_latched <= 1'b1;
            end
        end
    end

    gemv_load #(.MMAX(MMAX), .KMAX(KMAX), .WBYTES(WBYTES)) u_core (
        .clk(clk), .rst(core_rst),
        .m_count(m_count), .k_count(k_count),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data), .x_we(x_we), .x_data(x_data),
        .start(start_pulse), .done(core_done),
        .rd_addr(rd_addr), .y_out(core_y_out)
    );
endmodule
