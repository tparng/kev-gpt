// AXI-LEVEL testbench for the Tier-3 sequencer wrapped in gemv_axi_seq. Drives the
// ENTIRE flow over the AXI4-Lite bus exactly as the host will on the board:
//   1. CTRL.wl_rst, then stream wrom.mem to W_DATA as 32-bit chunks (the LOAD PORT,
//      NOT $readmemh) -> the runtime URAM weight image.
//   2. write the prompt over PL_ADDR/PL_DATA.
//   3. CTRL.go, poll STATUS.done.
//   4. read NGEN, then drain the token stream via TS_ADDR/TS_DATA.
// Dumps tokstream.out for the Python compare vs seq_ref.IntSequencer.generate_greedy.
// This is the binding SEQ_AXI gate: weights through the port + prompt over the bus +
// autoregressive emit == reference token stream.
`timescale 1ns / 1ps
`ifndef LANES
 `define LANES 16
`endif
`ifndef PNLAYER
 `define PNLAYER 4
`endif
`ifndef PPROMPT
 `define PPROMPT 4
`endif
`ifndef PNGEN
 `define PNGEN 6
`endif
`ifndef PKVMAX
 `define PKVMAX 16
`endif

module tb;
    localparam integer D     = 256;
    localparam integer D3    = 768;
    localparam integer D_MLP = 1024;
    localparam integer VOCAB = 193;
    localparam integer LANES = `LANES;
    localparam integer NLAYER= `PNLAYER;
    localparam integer WBITS = LANES*4;
    localparam integer SUBW  = (WBITS > 32) ? (WBITS/32) : 1;
    localparam integer AW    = 8;             // S_AXI address width

    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;
    localparam integer GW_MP   = ((D    + LANES-1)/LANES) * D_MLP;
    localparam integer GW_BLK  = GW_QKV + GW_PROJ + GW_FC + GW_MP;
    localparam integer GW_HEAD = ((VOCAB + LANES-1)/LANES) * D;
    localparam integer WROM_N  = GW_BLK*NLAYER + GW_HEAD;
    localparam integer WWORDS  = 1 << ($clog2(WROM_N));

    // register byte offsets (addr[7:2] index)
    localparam [AW-1:0] R_CTRL=8'h00, R_STATUS=8'h04, R_TOKID=8'h08, R_POS=8'h0C,
                        R_WDATA=8'h10, R_PLDATA=8'h14, R_PLADDR=8'h18, R_RDADDR=8'h1C,
                        R_RDDATA=8'h20, R_TSADDR=8'h24, R_TSDATA=8'h28, R_NGEN=8'h2C,
                        R_CYCLES=8'h30, R_IDCODE=8'h34;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg aresetn;

    // AXI4-Lite master signals
    reg  [AW-1:0] awaddr;  reg awvalid;  wire awready;
    reg  [31:0]   wdata;   reg wvalid;   wire wready;
    wire [1:0]    bresp;   wire bvalid;  reg bready;
    reg  [AW-1:0] araddr;  reg arvalid;  wire arready;
    wire [31:0]   rdata;   wire [1:0] rresp; wire rvalid; reg rready;

    gemv_axi_seq #(.NLAYER(NLAYER), .LANES(LANES), .KVMAX(`PKVMAX),
                   .PROMPT_LEN(`PPROMPT), .NGEN(`PNGEN), .WWORDS(WWORDS),
                   .C_S_AXI_ADDR_WIDTH(AW)) dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(aresetn),
        .S_AXI_AWADDR(awaddr), .S_AXI_AWPROT(3'b0), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(4'hF), .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr), .S_AXI_ARPROT(3'b0), .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp), .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready)
    );

    // ---- AXI4-Lite master tasks ----
    task axi_write(input [AW-1:0] addr, input [31:0] data);
        begin
            @(posedge clk); #1;
            awaddr = addr; awvalid = 1'b1; wdata = data; wvalid = 1'b1; bready = 1'b1;
            // hold until both address+data accepted
            wait (awready && wready);
            @(posedge clk); #1;
            awvalid = 1'b0; wvalid = 1'b0;
            wait (bvalid);
            @(posedge clk); #1;
            bready = 1'b0;
        end
    endtask

    task axi_read(input [AW-1:0] addr, output [31:0] data);
        begin
            @(posedge clk); #1;
            araddr = addr; arvalid = 1'b1; rready = 1'b1;
            wait (arready);
            @(posedge clk); #1;
            arvalid = 1'b0;
            wait (rvalid);
            data = rdata;
            @(posedge clk); #1;
            rready = 1'b0;
        end
    endtask

    integer i, s, f, ng;
    reg [WBITS-1:0] wimg [0:WROM_N-1];
    reg [8:0]       pimg [0:`PPROMPT-1];
    reg [WBITS-1:0] wword;
    reg [31:0]      rv;

    initial begin
        aresetn = 1'b0;
        awaddr=0; awvalid=0; wdata=0; wvalid=0; bready=0;
        araddr=0; arvalid=0; rready=0;
        repeat (4) @(posedge clk); #1;
        aresetn = 1'b1;
        repeat (2) @(posedge clk); #1;

        // sanity: IDCODE
        axi_read(R_IDCODE, rv);
        $display("IDCODE=%08x", rv);

        // ---- 1. stream the transposed INT4 image through W_DATA (the load port) ----
        $readmemh("wrom.mem", wimg);
        $readmemh("prompt.mem", pimg);
        axi_write(R_CTRL, 32'h2);                  // wl_rst=1 (reset load assembler)
        for (i = 0; i < WROM_N; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1)
                axi_write(R_WDATA, wword[s*32 +: 32]);   // low chunk first
        end

        // ---- 2. write the prompt over PL_ADDR/PL_DATA ----
        for (i = 0; i < `PPROMPT; i = i + 1) begin
            axi_write(R_PLADDR, i);
            axi_write(R_PLDATA, pimg[i]);
        end

        // ---- 3. GO, poll done ----
        axi_write(R_CTRL, 32'h1);                  // go=1
        rv = 0;
        while (rv[0] !== 1'b1) axi_read(R_STATUS, rv);   // poll STATUS.done

        // ---- 4. read NGEN, drain the token stream ----
        axi_read(R_NGEN, rv);  ng = rv[8:0];
        f = $fopen("tokstream.out", "w");
        for (i = 0; i < ng; i = i + 1) begin
            axi_write(R_TSADDR, i);
            axi_read(R_TSDATA, rv);
            $fwrite(f, "%0d\n", rv[8:0]);
        end
        $fclose(f);

        axi_read(R_CYCLES, rv);
        $display("TB_DONE ngen=%0d cycles=%0d", ng, rv);
        $finish;
    end

    initial begin
        #4000000000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
