// -----------------------------------------------------------------------------
// gemv_axi_seq_vec — AXI4-Lite shell around sequencer_vec (the P-WIDE datapath
// sequencer). Same skeleton as gemv_axi_seq_fast.v. IDCODE "SQRV" (0x53515256).
//
// Flow: CTRL.wl_rst, stream the transposed INT4 image to W_DATA; set TOK_ID/POS;
// CTRL.go; poll STATUS.done; read TOK_OUT (the argmax token) + CYCLES (go..done).
// RD_SEL/RD_ADDR + RD_DATA_LO/HI expose the phase banks for on-silicon debug.
//
// Register map (addr[7:2]):
//  0x00 CTRL (b0 go, b1 wl_rst, b2 soft_reset)   0x04 STATUS (b0 done, b1 busy)
//  0x08 TOK_ID   0x0C POS    0x10 W_DATA (wl_we) 0x14 RD_SEL  0x18 RD_ADDR
//  0x1C RD_DATA_LO   0x20 RD_DATA_HI   0x24 TOK_OUT   0x28 CYCLES   0x2C IDCODE
//  0x30 SEED (write-only): nonzero => load xorshift state + enable on-chip Gumbel-max
//       sampling (persists across GOs); 0/never-written => greedy argmax (bit-exact).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_axi_seq_vec #(
    parameter integer P      = 16,
    parameter integer LANES  = 128,
    parameter integer NLAYER = 4,
    parameter integer WWORDS = 262144,
    parameter integer TMAX   = 64,    // pos-table depth; 64 fits the embed ROMs in BRAM
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
    wire wr_en = S_AXI_AWVALID & S_AXI_WVALID & ~S_AXI_BVALID;

    reg          go_pulse, wl_rst, soft_reset, wl_we;
    reg [1:0]    dbg_stop;                           // CTRL b4:3: debug halt point (1/2/3)
    reg [31:0]   wl_data;
    reg [8:0]    tok_id, pos;
    reg [3:0]    rd_sel;
    reg [10:0]   rd_addr;
    reg [31:0]   seed;                               // on-chip sampling seed (0 => greedy)
    reg          seed_we;                            // 1-cyc pulse: load xorshift + enable sampling

    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY<=0; S_AXI_WREADY<=0; S_AXI_BVALID<=0; S_AXI_BRESP<=0; awaddr<=0;
            go_pulse<=0; wl_rst<=0; soft_reset<=0; wl_we<=0; wl_data<=0;
            tok_id<=0; pos<=0; rd_sel<=0; rd_addr<=0; dbg_stop<=0;
            seed<=0; seed_we<=0;
        end else begin
            go_pulse<=0; wl_rst<=0; wl_we<=0; seed_we<=0;
            if (wr_en && !S_AXI_AWREADY) begin
                S_AXI_AWREADY<=1; S_AXI_WREADY<=1; awaddr<=S_AXI_AWADDR;
            end else begin S_AXI_AWREADY<=0; S_AXI_WREADY<=0; end
            if (S_AXI_AWREADY) begin
                case (awaddr[7:2])
                    6'h0: begin go_pulse<=S_AXI_WDATA[0]; wl_rst<=S_AXI_WDATA[1]; soft_reset<=S_AXI_WDATA[2]; dbg_stop<=S_AXI_WDATA[4:3]; end
                    6'h2: tok_id  <= S_AXI_WDATA[8:0];
                    6'h3: pos     <= S_AXI_WDATA[8:0];
                    6'h4: begin wl_we<=1; wl_data<=S_AXI_WDATA; end
                    6'h5: rd_sel  <= S_AXI_WDATA[3:0];
                    6'h6: rd_addr <= S_AXI_WDATA[10:0];
                    6'hC: begin seed <= S_AXI_WDATA; seed_we <= 1'b1; end   // 0x30 SEED
                    default: ;
                endcase
                S_AXI_BVALID<=1; S_AXI_BRESP<=0;
            end else if (S_AXI_BVALID && S_AXI_BREADY) S_AXI_BVALID<=0;
        end
    end

    reg [5:0] araddr_idx;
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_ARREADY<=0; S_AXI_RVALID<=0; S_AXI_RRESP<=0; S_AXI_RDATA<=0; araddr_idx<=0;
        end else begin
            if (S_AXI_ARVALID && !S_AXI_ARREADY && !S_AXI_RVALID) begin
                S_AXI_ARREADY<=1; araddr_idx<=S_AXI_ARADDR[7:2];
            end else S_AXI_ARREADY<=0;
            if (S_AXI_ARREADY) begin
                S_AXI_RVALID<=1; S_AXI_RRESP<=0;
                case (araddr_idx)
                    6'h1: S_AXI_RDATA <= {30'b0, core_busy, done_latched};
                    6'h7: S_AXI_RDATA <= core_rd_data[31:0];
                    6'h8: S_AXI_RDATA <= core_rd_data[63:32];
                    6'h9: S_AXI_RDATA <= {23'b0, core_tok_out};
                    6'hA: S_AXI_RDATA <= cycles_latched;
                    6'hB: S_AXI_RDATA <= 32'h5351_5256;          // "SQRV"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) S_AXI_RVALID<=0;
        end
    end

    wire               core_done_w;
    wire [8:0]         core_tok_out;
    wire signed [63:0] core_rd_data;
    reg                core_busy, done_latched;
    reg  [31:0]        cycles_run, cycles_latched;
    wire               core_rst = ~aresetn | soft_reset;

    always @(posedge clk) begin
        if (core_rst) begin core_busy<=0; cycles_run<=0; cycles_latched<=0; done_latched<=0; end
        else if (go_pulse) begin core_busy<=1; cycles_run<=0; done_latched<=0; end
        else if (core_busy) begin
            cycles_run<=cycles_run+1'b1;
            if (core_done_w) begin core_busy<=0; cycles_latched<=cycles_run; done_latched<=1; end
        end
    end

    sequencer_vec #(.P(P), .LANES(LANES), .NLAYER(NLAYER), .WWORDS(WWORDS), .TMAX(TMAX)) u_seq (
        .clk(clk), .rst(core_rst), .go(go_pulse),
        .tok_id(tok_id), .pos(pos), .done(core_done_w), .tok_out(core_tok_out),
        .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(core_rd_data),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data), .dbg_stop(dbg_stop),
        .seed(seed), .seed_we(seed_we)
    );
endmodule
