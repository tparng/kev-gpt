// -----------------------------------------------------------------------------
// gemv_axi_seq_sb — AXI4-Lite shell around sequencer_sb (SPLIT-BRAIN: two
// independent N=8 cohorts on the TDP weight image). IDCODE "SQSB" (0x53515342).
// The SQ16 register map is kept VERBATIM (same board driver, new idcode);
// ND is DSP-packed streams PER COHORT (6 -> 12 of 16 total):
//  0x00 CTRL (b0 go, b1 wl_rst, b2 soft_reset, b4:3 dbg_stop)
//  0x04 STATUS (b0 done, b1 busy)
//  0x08 TOK_ID0  0x0C POS    0x10 W_DATA (wl_we)  0x14 RD_SEL  0x18 RD_ADDR
//  0x1C RD_DATA_LO  0x20 RD_DATA_HI  0x24 TOK_OUT0  0x28 CYCLES  0x2C IDCODE
//  0x30 TOK_ID1  0x34 TOK_ID2  0x38 TOK_ID3   0x3C/0x40/0x44 TOK_OUT1..3
//  0x48 RD_STREAM  0x4C E_DATA  0x50-0x5C TOK_ID4..7  0x60-0x6C TOK_OUT4..7
//  0x80-0x9C TOK_ID8..15 (writes)   0xA0-0xBC TOK_OUT8..15 (reads)
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_axi_seq_sb #(
    parameter integer P      = 8,
    parameter integer LANES  = 128,
    parameter integer N      = 16,        // total streams (two cohorts of NC)
    parameter integer NC     = 8,         // streams per cohort
    parameter integer ND     = 6,         // DSP-packed streams PER COHORT
    parameter integer NLAYER = 4,
    parameter integer WWORDS = 25600,
    parameter integer TMAX   = 32,
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

    reg          go_pulse, wl_rst, soft_reset, wl_we, el_we;
    reg [1:0]    dbg_stop;
    reg [31:0]   wl_data;
    reg [8:0]    tok_id [0:15];
    reg [8:0]    pos;
    reg [3:0]    rd_sel;
    reg [10:0]   rd_addr;
    reg [3:0]    rd_stream;
    integer ti;

    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY<=0; S_AXI_WREADY<=0; S_AXI_BVALID<=0; S_AXI_BRESP<=0; awaddr<=0;
            go_pulse<=0; wl_rst<=0; soft_reset<=0; wl_we<=0; wl_data<=0; pos<=0;
            for (ti = 0; ti < 16; ti = ti + 1) tok_id[ti] <= 9'd0;
            rd_sel<=0; rd_addr<=0; rd_stream<=0; dbg_stop<=0;
        end else begin
            go_pulse<=0; wl_rst<=0; wl_we<=0; el_we<=0;
            if (wr_en && !S_AXI_AWREADY) begin
                S_AXI_AWREADY<=1; S_AXI_WREADY<=1; awaddr<=S_AXI_AWADDR;
            end else begin S_AXI_AWREADY<=0; S_AXI_WREADY<=0; end
            if (S_AXI_AWREADY) begin
                case (awaddr[7:2])
                    6'h00: begin go_pulse<=S_AXI_WDATA[0]; wl_rst<=S_AXI_WDATA[1]; soft_reset<=S_AXI_WDATA[2]; dbg_stop<=S_AXI_WDATA[4:3]; end
                    6'h02: tok_id[0] <= S_AXI_WDATA[8:0];
                    6'h03: pos       <= S_AXI_WDATA[8:0];
                    6'h04: begin wl_we<=1; wl_data<=S_AXI_WDATA; end
                    6'h05: rd_sel  <= S_AXI_WDATA[3:0];
                    6'h06: rd_addr <= S_AXI_WDATA[10:0];
                    6'h0C: tok_id[1] <= S_AXI_WDATA[8:0];
                    6'h0D: tok_id[2] <= S_AXI_WDATA[8:0];
                    6'h0E: tok_id[3] <= S_AXI_WDATA[8:0];
                    6'h12: rd_stream <= S_AXI_WDATA[3:0];
                    6'h13: begin el_we<=1; wl_data<=S_AXI_WDATA; end
                    6'h14: tok_id[4] <= S_AXI_WDATA[8:0];     // 0x50
                    6'h15: tok_id[5] <= S_AXI_WDATA[8:0];
                    6'h16: tok_id[6] <= S_AXI_WDATA[8:0];
                    6'h17: tok_id[7] <= S_AXI_WDATA[8:0];     // 0x5C
                    6'h20, 6'h21, 6'h22, 6'h23,
                    6'h24, 6'h25, 6'h26, 6'h27:               // 0x80-0x9C
                        tok_id[8 + (awaddr[4:2] & 3'h7)] <= S_AXI_WDATA[8:0];
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
                    6'h01: S_AXI_RDATA <= {30'b0, core_busy, done_latched};
                    6'h07: S_AXI_RDATA <= core_rd_data[31:0];
                    6'h08: S_AXI_RDATA <= core_rd_data[63:32];
                    6'h09: S_AXI_RDATA <= {23'b0, tok_outs[8:0]};
                    6'h0A: S_AXI_RDATA <= cycles_latched;
                    6'h0B: S_AXI_RDATA <= 32'h5351_5342;          // "SQSB"
                    6'h0F: S_AXI_RDATA <= {23'b0, tok_outs[17:9]};
                    6'h10: S_AXI_RDATA <= {23'b0, tok_outs[26:18]};
                    6'h11: S_AXI_RDATA <= {23'b0, tok_outs[35:27]};
                    6'h18: S_AXI_RDATA <= {23'b0, tok_outs[44:36]};   // 0x60
                    6'h19: S_AXI_RDATA <= {23'b0, tok_outs[53:45]};
                    6'h1A: S_AXI_RDATA <= {23'b0, tok_outs[62:54]};
                    6'h1B: S_AXI_RDATA <= {23'b0, tok_outs[71:63]};
                    6'h28: S_AXI_RDATA <= {23'b0, tok_outs[80:72]};   // 0xA0
                    6'h29: S_AXI_RDATA <= {23'b0, tok_outs[89:81]};
                    6'h2A: S_AXI_RDATA <= {23'b0, tok_outs[98:90]};
                    6'h2B: S_AXI_RDATA <= {23'b0, tok_outs[107:99]};
                    6'h2C: S_AXI_RDATA <= {23'b0, tok_outs[116:108]};
                    6'h2D: S_AXI_RDATA <= {23'b0, tok_outs[125:117]};
                    6'h2E: S_AXI_RDATA <= {23'b0, tok_outs[134:126]};
                    6'h2F: S_AXI_RDATA <= {23'b0, tok_outs[143:135]};  // 0xBC
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) S_AXI_RVALID<=0;
        end
    end

    wire               core_done_w;
    wire [N*9-1:0]     tok_outs;
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

    sequencer_sb #(.P(P), .LANES(LANES), .N(N), .NC(NC), .ND(ND),
                   .NLAYER(NLAYER), .WWORDS(WWORDS), .TMAX(TMAX)) u_seq (
        .clk(clk), .rst(core_rst), .go(go_pulse),
        .tok_ids({tok_id[15], tok_id[14], tok_id[13], tok_id[12],
                  tok_id[11], tok_id[10], tok_id[9],  tok_id[8],
                  tok_id[7],  tok_id[6],  tok_id[5],  tok_id[4],
                  tok_id[3],  tok_id[2],  tok_id[1],  tok_id[0]}),
        .pos(pos),
        .done(core_done_w), .tok_outs(tok_outs),
        .rd_stream(rd_stream), .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(core_rd_data),
        .wl_rst(wl_rst), .wl_we(wl_we), .el_we(el_we), .wl_data(wl_data), .dbg_stop(dbg_stop)
    );
endmodule
