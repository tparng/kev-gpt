// -----------------------------------------------------------------------------
// mamba_seq_axi — AXI4-Lite shell around mamba_seq (the full Mamba-2 token
// engine). Skeleton per ssm_scan_axi.v / gemv_axi_seq.v.
//
// Flow:
//   1. Load tables: write TSEL, then stream (TADDR autoincrements on TDATA
//      writes — 550k words load in ~2 writes/word over AXI).
//   2. Poll STATUS.ready (scan state clear sweep).
//   3. Per token: write TOK (also pulses go), poll STATUS.done, read TOKOUT.
//      CYCLES latches the step's cycle count.
//   4. Debug: DBGSEL/DBGADDR then read DBGDATA (per-phase buffers).
//
// Register map (addr[7:2]):
//   0x00 CTRL  (b0 go, b1 soft_reset)      0x04 STATUS (b0 done, b1 busy, b2 ready)
//   0x08 TOK   (write: tok_in + go)        0x0C TOKOUT
//   0x10 TSEL  (table selector)            0x14 TADDR (auto-inc base)
//   0x18 TDATA (write pulses wr_en)        0x1C CYCLES
//   0x20 DBGSEL 0x24 DBGADDR 0x28 DBGDATA  0x2C IDCODE (0x4D414D42 "MAMB")
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module mamba_seq_axi #(
    parameter integer C_S_AXI_ADDR_WIDTH = 8
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
    wire clk = S_AXI_ACLK;
    wire aresetn = S_AXI_ARESETN;

    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr;
    wire wr_en_axi = S_AXI_AWVALID & S_AXI_WVALID & ~S_AXI_BVALID;

    reg        go_pulse, soft_reset;
    reg [9:0]  tok_in;
    reg [3:0]  tsel;
    reg [18:0] taddr;
    reg        t_we;
    reg [18:0] t_wa;      // address latched BEFORE the auto-increment
    reg [31:0] t_wd;
    reg [3:0]  dbgsel;
    reg [10:0] dbgaddr;

    wire        ready, done;
    wire [9:0]  tok_out;
    wire signed [31:0] dbg_data;

    // ---- write channel ------------------------------------------------------
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            S_AXI_BVALID  <= 1'b0; S_AXI_BRESP <= 2'b00; awaddr <= 0;
            go_pulse <= 1'b0; soft_reset <= 1'b0; t_we <= 1'b0;
            tok_in <= 0; tsel <= 0; taddr <= 0; t_wd <= 0;
            dbgsel <= 0; dbgaddr <= 0;
        end else begin
            go_pulse <= 1'b0; soft_reset <= 1'b0; t_we <= 1'b0;
            if (wr_en_axi && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1; S_AXI_WREADY <= 1'b1; awaddr <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            end
            if (S_AXI_AWREADY) begin
                case (awaddr[7:2])
                    6'h0: begin go_pulse <= S_AXI_WDATA[0];
                                soft_reset <= S_AXI_WDATA[1]; end
                    6'h2: begin tok_in <= S_AXI_WDATA[9:0]; go_pulse <= 1'b1; end
                    6'h4: tsel  <= S_AXI_WDATA[3:0];
                    6'h5: taddr <= S_AXI_WDATA[18:0];
                    6'h6: begin t_we <= 1'b1; t_wd <= S_AXI_WDATA;
                                t_wa <= taddr; taddr <= taddr + 1'b1; end
                    6'h8: dbgsel  <= S_AXI_WDATA[3:0];
                    6'h9: dbgaddr <= S_AXI_WDATA[10:0];
                    default: ;
                endcase
                S_AXI_BVALID <= 1'b1; S_AXI_BRESP <= 2'b00;
            end else if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // ---- run state ------------------------------------------------------------
    reg        busy, done_l;
    reg [31:0] cycles, cycles_l;
    always @(posedge clk) begin
        if (!aresetn || soft_reset) begin
            busy <= 1'b0; done_l <= 1'b0; cycles <= 0; cycles_l <= 0;
        end else if (go_pulse) begin
            busy <= 1'b1; done_l <= 1'b0; cycles <= 0;
        end else if (busy) begin
            cycles <= cycles + 1'b1;
            if (done) begin busy <= 1'b0; done_l <= 1'b1; cycles_l <= cycles; end
        end
    end

    // ---- read channel -----------------------------------------------------------
    reg [5:0] aridx;
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_ARREADY <= 1'b0; S_AXI_RVALID <= 1'b0; S_AXI_RRESP <= 2'b00;
            S_AXI_RDATA <= 32'b0; aridx <= 6'h0;
        end else begin
            if (S_AXI_ARVALID && !S_AXI_ARREADY && !S_AXI_RVALID) begin
                S_AXI_ARREADY <= 1'b1; aridx <= S_AXI_ARADDR[7:2];
            end else S_AXI_ARREADY <= 1'b0;
            if (S_AXI_ARREADY) begin
                S_AXI_RVALID <= 1'b1; S_AXI_RRESP <= 2'b00;
                case (aridx)
                    6'h1: S_AXI_RDATA <= {29'b0, ready, busy, done_l};
                    6'h3: S_AXI_RDATA <= {22'b0, tok_out};
                    6'h7: S_AXI_RDATA <= cycles_l;
                    6'hA: S_AXI_RDATA <= dbg_data;
                    6'hB: S_AXI_RDATA <= 32'h4D414D42;   // "MAMB"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

    mamba_seq u_seq (
        .clk(clk), .rst(~aresetn | soft_reset), .ready(ready),
        .start(go_pulse), .tok_in(tok_in), .done(done), .tok_out(tok_out),
        .wr_en(t_we), .wr_sel(tsel), .wr_addr(t_wa), .wr_data(t_wd),
        .dbg_sel(dbgsel), .dbg_addr(dbgaddr), .dbg_data(dbg_data));

endmodule
