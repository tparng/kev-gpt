// -----------------------------------------------------------------------------
// gemv_axi_seq — AXI4-Lite shell around the Tier-3 SEQUENCER (sequencer.sv). This is
// the CAPSTONE: the whole single-stream autoregressive transformer (embed -> NLAYER
// blocks -> ln_f -> head -> argmax -> token stream) runs in fabric with the CPU OUT
// of the loop between `go` and the N-token result.
//
// Boot/run flow over this shell:
//   1. LOAD WEIGHTS (once): write CTRL.wl_rst=1, then stream the transposed INT4 image
//      as 32-bit chunks to W_DATA (low chunk of each LANES*4-bit wide word first). The
//      core assembles SUBW chunks/word into the runtime-loadable URAM — NOT $readmemh,
//      so synth maps a real URAM instead of pruning a constant ROM (the bitten gotcha).
//   2. WRITE PROMPT: for each prompt slot, write PL_ADDR then PL_DATA (PL_DATA write
//      pulses the prompt write-enable at PL_ADDR).
//   3. (optional) set TOK_ID/POS for the legacy single-token path.
//   4. GO: write CTRL.go=1. Poll STATUS.done. Read NGEN.
//   5. DRAIN STREAM: for i in 0..NGEN-1: write TS_ADDR=i, read TS_DATA (the i-th token).
//      Residual x can be inspected the same way via RD_ADDR/RD_DATA.
//
// Register map (addr[7:2] index; AXI4-Lite, 32-bit data):
//   0x00 CTRL   (b0 go, b1 wl_rst, b2 soft_reset)   0x04 STATUS (b0 done, b1 busy)
//   0x08 TOK_ID (legacy single-token id)            0x0C POS    (legacy single-token pos)
//   0x10 W_DATA (32-bit weight load chunk; write pulses wl_we)
//   0x14 PL_DATA(prompt token id; write pulses pl_we at PL_ADDR)
//   0x18 PL_ADDR(prompt write address)
//   0x1C RD_ADDR(residual x readback addr)          0x20 RD_DATA(residual x, Q6.25)
//   0x24 TS_ADDR(token-stream readback addr)        0x28 TS_DATA(emitted token id)
//   0x2C NGEN   (number of generated tokens)        0x30 CYCLES (run cycle count)
//   0x34 IDCODE (0x5345_5152 "SEQR")
//
// Module-ref top for the Vivado BD MUST be .v (this file); the core stays .sv.
// Adapted from gemv_axi_banked.v (same handshake/cycle-counter skeleton).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_axi_seq #(
    parameter integer D          = 256,
    parameter integer D3         = 768,
    parameter integer D_MLP      = 1024,
    parameter integer NHEAD      = 4,
    parameter integer HEAD_DIM   = 64,
    parameter integer NLAYER     = 4,
    parameter integer VOCAB      = 193,
    parameter integer LANES      = 16,
    parameter integer TMAX       = 256,
    parameter integer KVMAX      = 32,
    parameter integer PROMPT_LEN = 8,
    parameter integer NGEN       = 8,
    parameter integer WWORDS     = 262144,
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

    // control / load registers
    reg          go_pulse, wl_rst, soft_reset, wl_we, pl_we;
    reg [31:0]   wl_data;
    reg [8:0]    pl_addr, pl_data;
    reg [7:0]    tok_id;
    reg [8:0]    pos;
    reg [8:0]    rd_addr, ts_addr;

    // ---- write channel ------------------------------------------------------
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            S_AXI_BVALID  <= 1'b0; S_AXI_BRESP <= 2'b00; awaddr <= 0;
            go_pulse <= 1'b0; wl_rst <= 1'b0; soft_reset <= 1'b0;
            wl_we <= 1'b0; pl_we <= 1'b0; wl_data <= 32'b0;
            pl_addr <= 0; pl_data <= 0; tok_id <= 0; pos <= 0;
            rd_addr <= 0; ts_addr <= 0;
        end else begin
            // one-cycle pulses default low
            go_pulse <= 1'b0; wl_rst <= 1'b0; wl_we <= 1'b0; pl_we <= 1'b0;
            if (wr_en && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1; S_AXI_WREADY <= 1'b1; awaddr <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            end
            if (S_AXI_AWREADY) begin
                case (awaddr[7:2])
                    6'h0: begin go_pulse <= S_AXI_WDATA[0]; wl_rst <= S_AXI_WDATA[1];
                                soft_reset <= S_AXI_WDATA[2]; end          // CTRL
                    6'h2: tok_id  <= S_AXI_WDATA[7:0];                     // TOK_ID
                    6'h3: pos     <= S_AXI_WDATA[8:0];                     // POS
                    6'h4: begin wl_we <= 1'b1; wl_data <= S_AXI_WDATA; end // W_DATA (32-bit chunk)
                    6'h5: begin pl_we <= 1'b1; pl_data <= S_AXI_WDATA[8:0]; end // PL_DATA -> write@PL_ADDR
                    6'h6: pl_addr <= S_AXI_WDATA[8:0];                     // PL_ADDR
                    6'h7: rd_addr <= S_AXI_WDATA[8:0];                     // RD_ADDR
                    6'h9: ts_addr <= S_AXI_WDATA[8:0];                     // TS_ADDR
                    default: ;
                endcase
                S_AXI_BVALID <= 1'b1; S_AXI_BRESP <= 2'b00;
            end else if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // ---- read channel -------------------------------------------------------
    reg [5:0] araddr_idx;
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_ARREADY <= 1'b0; S_AXI_RVALID <= 1'b0; S_AXI_RRESP <= 2'b00;
            S_AXI_RDATA <= 32'b0; araddr_idx <= 6'h0;
        end else begin
            if (S_AXI_ARVALID && !S_AXI_ARREADY && !S_AXI_RVALID) begin
                S_AXI_ARREADY <= 1'b1; araddr_idx <= S_AXI_ARADDR[7:2];
            end else begin
                S_AXI_ARREADY <= 1'b0;
            end
            if (S_AXI_ARREADY) begin
                S_AXI_RVALID <= 1'b1; S_AXI_RRESP <= 2'b00;
                case (araddr_idx)
                    6'h1: S_AXI_RDATA <= {30'b0, core_busy, done_latched};   // STATUS
                    6'h8: S_AXI_RDATA <= core_x_out;                         // RD_DATA (Q6.25)
                    6'hA: S_AXI_RDATA <= {23'b0, core_ts_out};               // TS_DATA
                    6'hB: S_AXI_RDATA <= {23'b0, core_ngen_out};             // NGEN
                    6'hC: S_AXI_RDATA <= cycles_latched;                     // CYCLES
                    6'hD: S_AXI_RDATA <= 32'h5345_5152;                      // IDCODE "SEQR"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

    // ---- core run state (cycle counter + done latch) ------------------------
    wire               core_busy_w, core_done_w;
    wire signed [31:0] core_x_out;
    wire [8:0]         core_tok_out, core_ts_out, core_ngen_out;
    reg                core_busy, done_latched;
    reg  [31:0]        cycles_run, cycles_latched;
    wire               core_rst = ~aresetn | soft_reset;

    always @(posedge clk) begin
        if (core_rst) begin
            core_busy <= 1'b0; cycles_run <= 0; cycles_latched <= 0; done_latched <= 1'b0;
        end else if (go_pulse) begin
            core_busy <= 1'b1; cycles_run <= 0; done_latched <= 1'b0;
        end else if (core_busy) begin
            cycles_run <= cycles_run + 1'b1;
            if (core_done_w) begin
                core_busy <= 1'b0; cycles_latched <= cycles_run; done_latched <= 1'b1;
            end
        end
    end

    sequencer #(.D(D), .D3(D3), .D_MLP(D_MLP), .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM),
                .NLAYER(NLAYER), .VOCAB(VOCAB), .LANES(LANES), .TMAX(TMAX),
                .KVMAX(KVMAX), .PROMPT_LEN(PROMPT_LEN), .NGEN(NGEN),
                .WWORDS(WWORDS)) u_seq (
        .clk(clk), .rst(core_rst), .go(go_pulse),
        .tok_id(tok_id), .pos(pos),
        .busy(core_busy_w), .done(core_done_w),
        .rd_addr(rd_addr), .x_out(core_x_out), .tok_out(core_tok_out),
        .ts_addr(ts_addr), .ts_out(core_ts_out), .ngen_out(core_ngen_out),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data),
        .pl_we(pl_we), .pl_addr(pl_addr), .pl_data(pl_data)
    );
endmodule
