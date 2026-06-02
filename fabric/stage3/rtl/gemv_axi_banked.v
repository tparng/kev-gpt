// -----------------------------------------------------------------------------
// gemv_axi_banked — AXI4-Lite shell around gemv_banked_resident. The WHOLE model's
// TRANSPOSED weights load ONCE into URAM; per layer the host sets W_BASE/M/K, streams
// the activation, and runs. This is the deployable WIDE-GEMV (LANES MACs/cycle) engine.
//
// Adapts gemv_axi_resident's register map. Two differences vs the PE=1 resident shell:
//   * W_DATA carries a FULL 32-bit chunk of the wide word (not one byte). The core
//     assembles SUBW = LANES*4/32 chunks into one wide URAM word, low chunk first.
//   * W_BASE is a WORD offset (wide-word index) of the layer's transposed block,
//     not a byte offset.
//
//   0x00 CTRL(b0 start,b1 ld_rst,b2 reset) 04 STATUS(b0 done,b1 busy) 08 M_COUNT 0C K_COUNT
//   10 W_DATA(32-bit load chunk) 14 X_DATA(int8) 18 RD_ADDR 1C RD_DATA 20 CYCLES
//   24 IDCODE(0x4745_4D42 "GEMB") 28 W_BASE(wide-word offset of layer)
//
// Module-ref top for Vivado BD MUST be .v (this file); the core is .sv.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_axi_banked #(
    parameter integer LANES  = 256,
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WWORDS = 16384,
    parameter integer RLAT   = 2,
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
    localparam integer WAW = $clog2(WWORDS);

    wire clk = S_AXI_ACLK;
    wire aresetn = S_AXI_ARESETN;

    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr;
    wire wr_en = S_AXI_AWVALID & S_AXI_WVALID & ~S_AXI_BVALID;

    reg              start_pulse, ld_rst, soft_reset, w_we, x_we;
    reg [31:0]       w_data;
    reg [7:0]        x_data;
    reg [MCW-1:0]    m_count;
    reg [KCW-1:0]    k_count;
    reg [MAW-1:0]    rd_addr;
    reg [WAW-1:0]    w_base;

    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            S_AXI_BVALID  <= 1'b0; S_AXI_BRESP <= 2'b00; awaddr <= 0;
            start_pulse <= 1'b0; ld_rst <= 1'b0; soft_reset <= 1'b0;
            w_we <= 1'b0; x_we <= 1'b0; w_data <= 32'b0; x_data <= 8'b0;
            m_count <= 0; k_count <= 0; rd_addr <= 0; w_base <= 0;
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
                    4'h4: begin w_we <= 1'b1; w_data <= S_AXI_WDATA; end          // full 32-bit chunk
                    4'h5: begin x_we <= 1'b1; x_data <= S_AXI_WDATA[7:0]; end
                    4'h6: rd_addr <= S_AXI_WDATA[MAW-1:0];
                    4'hA: w_base <= S_AXI_WDATA[WAW-1:0];                          // wide-word offset
                    default: ;
                endcase
                S_AXI_BVALID <= 1'b1; S_AXI_BRESP <= 2'b00;
            end else if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

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
                    4'h9: S_AXI_RDATA <= 32'h4745_4D42;       // "GEMB"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

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

    gemv_banked_resident #(.LANES(LANES), .MMAX(MMAX), .KMAX(KMAX),
                           .WWORDS(WWORDS), .RLAT(RLAT)) u_core (
        .clk(clk), .rst(core_rst),
        .m_count(m_count), .k_count(k_count), .w_base(w_base),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data), .x_we(x_we), .x_data($signed(x_data)),
        .start(start_pulse), .done(core_done),
        .rd_addr(rd_addr), .y_out(core_y_out)
    );
endmodule
