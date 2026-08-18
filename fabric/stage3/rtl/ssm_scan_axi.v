// -----------------------------------------------------------------------------
// ssm_scan_axi — AXI4-Lite shell around ssm_scan (the Mamba-2 scan core), the
// first mamba silicon bring-up (doc 9 §7 rung 4). Skeleton adapted from
// gemv_axi_seq.v (same handshake + cycle-counter pattern).
//
// Flow:
//   1. soft_reset (CTRL b1) -> core clear-FSM sweeps state; poll STATUS.ready.
//   2. Load vectors: *_ADDR then *_DATA (data write pulses the we) for DTX
//      (16b), B (8b), C (8b); write A_Q.
//   3. One step: CTRL.go, poll done. Bench: write STEPS=N, CTRL.go_loop —
//      runs N back-to-back steps on the loaded vectors (state evolves each
//      step, deterministically comparable to the Python reference), CYCLES
//      latches the total -> silicon cycles/step + tok-rate in one register.
//   4. Read y: Y_ADDR then Y_DATA.
//
// Register map (addr[7:2]; AXI4-Lite, 32-bit):
//   0x00 CTRL  (b0 go, b1 soft_reset, b2 go_loop)
//   0x04 STATUS(b0 done, b1 busy, b2 ready)
//   0x08 A_Q    0x0C DTX_ADDR  0x10 DTX_DATA  0x14 B_ADDR  0x18 B_DATA
//   0x1C C_ADDR 0x20 C_DATA    0x24 STEPS     0x28 Y_ADDR  0x2C Y_DATA
//   0x30 CYCLES 0x34 IDCODE (0x53534D53 "SSMS")
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module ssm_scan_axi #(
    parameter integer P = 64,
    parameter integer N = 64,
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

    reg        go_pulse, soft_reset, loop_go;
    reg        dtx_we, b_we, c_we;
    reg [15:0] a_q;
    reg [5:0]  dtx_addr, b_addr, c_addr, y_addr;
    reg [15:0] dtx_data;
    reg [7:0]  b_data, c_data;
    reg [31:0] steps;

    // ---- write channel ------------------------------------------------------
    always @(posedge clk) begin
        if (!aresetn) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            S_AXI_BVALID  <= 1'b0; S_AXI_BRESP <= 2'b00; awaddr <= 0;
            go_pulse <= 1'b0; soft_reset <= 1'b0; loop_go <= 1'b0;
            dtx_we <= 1'b0; b_we <= 1'b0; c_we <= 1'b0;
            a_q <= 0; dtx_addr <= 0; b_addr <= 0; c_addr <= 0; y_addr <= 0;
            dtx_data <= 0; b_data <= 0; c_data <= 0; steps <= 32'd1;
        end else begin
            go_pulse <= 1'b0; soft_reset <= 1'b0; loop_go <= 1'b0;
            dtx_we <= 1'b0; b_we <= 1'b0; c_we <= 1'b0;
            if (wr_en && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1; S_AXI_WREADY <= 1'b1; awaddr <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0;
            end
            if (S_AXI_AWREADY) begin
                case (awaddr[7:2])
                    6'h0: begin go_pulse <= S_AXI_WDATA[0];
                                soft_reset <= S_AXI_WDATA[1];
                                loop_go <= S_AXI_WDATA[2]; end
                    6'h2: a_q      <= S_AXI_WDATA[15:0];
                    6'h3: dtx_addr <= S_AXI_WDATA[5:0];
                    6'h4: begin dtx_we <= 1'b1; dtx_data <= S_AXI_WDATA[15:0]; end
                    6'h5: b_addr   <= S_AXI_WDATA[5:0];
                    6'h6: begin b_we <= 1'b1; b_data <= S_AXI_WDATA[7:0]; end
                    6'h7: c_addr   <= S_AXI_WDATA[5:0];
                    6'h8: begin c_we <= 1'b1; c_data <= S_AXI_WDATA[7:0]; end
                    6'h9: steps    <= S_AXI_WDATA;
                    6'hA: y_addr   <= S_AXI_WDATA[5:0];
                    default: ;
                endcase
                S_AXI_BVALID <= 1'b1; S_AXI_BRESP <= 2'b00;
            end else if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // ---- read channel -------------------------------------------------------
    wire        core_ready, core_done;
    wire signed [15:0] y_data;
    reg         busy, done_latched;
    reg  [31:0] cycles_run, cycles_latched;

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
                    6'h1: S_AXI_RDATA <= {29'b0, core_ready, busy, done_latched};
                    6'hB: S_AXI_RDATA <= {{16{y_data[15]}}, y_data};
                    6'hC: S_AXI_RDATA <= cycles_latched;
                    6'hD: S_AXI_RDATA <= 32'h53534D53;   // "SSMS"
                    default: S_AXI_RDATA <= 32'b0;
                endcase
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

    // ---- run/loop controller ------------------------------------------------
    reg        start_core;
    reg [31:0] steps_left;
    reg        looping;

    always @(posedge clk) begin
        if (!aresetn || soft_reset) begin
            busy <= 1'b0; looping <= 1'b0; start_core <= 1'b0;
            cycles_run <= 0; cycles_latched <= 0; done_latched <= 1'b0;
            steps_left <= 0;
        end else begin
            start_core <= 1'b0;
            if (go_pulse && !busy) begin
                busy <= 1'b1; looping <= 1'b0; steps_left <= 32'd1;
                cycles_run <= 0; done_latched <= 1'b0; start_core <= 1'b1;
            end else if (loop_go && !busy) begin
                busy <= 1'b1; looping <= 1'b1; steps_left <= steps;
                cycles_run <= 0; done_latched <= 1'b0; start_core <= 1'b1;
            end else if (busy) begin
                cycles_run <= cycles_run + 1'b1;
                if (core_done) begin
                    if (steps_left <= 32'd1) begin
                        busy <= 1'b0; done_latched <= 1'b1;
                        cycles_latched <= cycles_run;
                    end else begin
                        steps_left <= steps_left - 1'b1;
                        start_core <= 1'b1;
                    end
                end
            end
        end
    end

    ssm_scan #(.P(P), .N(N)) u_core (
        .clk(clk), .rst(~aresetn | soft_reset), .ready(core_ready),
        .start(start_core), .done(core_done),
        .a_q(a_q),
        .wr_dtx(dtx_we), .wr_dtx_addr(dtx_addr), .wr_dtx_data(dtx_data),
        .wr_b(b_we), .wr_b_addr(b_addr), .wr_b_data(b_data),
        .wr_c(c_we), .wr_c_addr(c_addr), .wr_c_data(c_data),
        .rd_y_addr(y_addr), .rd_y_data(y_data)
    );
endmodule
