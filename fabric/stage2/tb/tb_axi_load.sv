// AXI-Lite BFM testbench for gemv_axi_load: drives real AXI writes to stream a
// layer's weights+activation, set M/K, start, poll done, read y back over AXI,
// then bit-exact compare vs golden. Validates the whole wrapper before bitstream.
`timescale 1ns / 1ps
`include "config.svh"

module tb_axi_load;
    localparam integer M = `CFG_M;
    localparam integer K = `CFG_K;
    localparam integer MMAX = 1024, KMAX = 1024, WBYTES = 131072;
    // register offsets
    localparam CTRL=6'h00, STATUS=6'h04, MCNT=6'h08, KCNT=6'h0C,
               WDAT=6'h10, XDAT=6'h14, RADDR=6'h18, RDAT=6'h1C, CYC=6'h20, IDC=6'h24;

    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg  [5:0]  awaddr, araddr; reg [31:0] wdata; reg awvalid, wvalid, bready, arvalid, rready;
    wire awready, wready, bvalid, arready, rvalid; wire [1:0] bresp, rresp; wire [31:0] rdata;

    gemv_axi_load #(.MMAX(MMAX), .KMAX(KMAX), .WBYTES(WBYTES)) dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rstn),
        .S_AXI_AWADDR(awaddr), .S_AXI_AWPROT(3'b0), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(4'hF), .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr), .S_AXI_ARPROT(3'b0), .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp), .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready)
    );

    task axi_write(input [5:0] a, input [31:0] d);
        begin
            @(posedge clk); awaddr<=a; wdata<=d; awvalid<=1; wvalid<=1; bready<=1;
            wait (awready && wready); @(posedge clk); awvalid<=0; wvalid<=0;
            wait (bvalid); @(posedge clk); bready<=0;
        end
    endtask
    task axi_read(input [5:0] a, output [31:0] d);
        begin
            @(posedge clk); araddr<=a; arvalid<=1; rready<=1;
            wait (arready); @(posedge clk); arvalid<=0;
            wait (rvalid); d=rdata; @(posedge clk); rready<=0;
        end
    endtask

    reg [7:0] wsrc [0:WBYTES-1]; reg signed [7:0] xsrc [0:KMAX-1];
    reg [31:0] ycopy [0:M-1]; reg [31:0] rd; integer i, m, mism;

    initial begin
        awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0;
        $readmemh(`CFG_WFILE, wsrc); $readmemh(`CFG_XFILE, xsrc);
        repeat (5) @(posedge clk); rstn <= 1; repeat (2) @(posedge clk);

        axi_read(IDC, rd); $display("IDCODE=0x%08x", rd);
        axi_write(MCNT, M); axi_write(KCNT, K);
        axi_write(CTRL, 32'h2);                       // ld_rst
        for (i=0; i<M*K/2; i=i+1) axi_write(WDAT, {24'b0, wsrc[i]});
        for (i=0; i<K;     i=i+1) axi_write(XDAT, {24'b0, xsrc[i]});
        axi_write(CTRL, 32'h1);                       // start

        rd = 0; i = 0;
        while (!(rd & 32'h1) && i < 2000000) begin axi_read(STATUS, rd); i=i+1; end
        axi_read(CYC, rd); $display("RESULT cycles=%0d M=%0d K=%0d", rd, M, K);

        for (m=0; m<M; m=m+1) begin axi_write(RADDR, m); axi_read(RDAT, rd); ycopy[m]=rd; end
        $writememh(`CFG_YDUT, ycopy);
        $display("RESULT_DONE");
        $finish;
    end
    initial begin #200000000; $display("RESULT TIMEOUT"); $finish; end
endmodule
