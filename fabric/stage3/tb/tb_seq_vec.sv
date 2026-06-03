// Testbench for sequencer_vec. Streams the qkv weight block into the resident GEMV, pulses
// go for token TOK at pos 0, waits for done, then reads back lnout (256 x Q.22 -> lnout.out,
// M1 gate) and qkv (768 x Q.16 -> qkv.out, M2 gate) for the Python compares vs
// seq_ref.block0_phase_signals.
`timescale 1ns / 1ps
`ifndef TOK
 `define TOK 48
`endif
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef GWQKV
 `define GWQKV 12288
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer D     = 256;
    localparam integer D3    = 768;
    localparam integer LANES = 16;
    localparam integer WBITS = LANES * 4;          // 64
    localparam integer SUBW  = WBITS / 32;         // 2

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [8:0] tok, pos;
    reg [10:0] rdaddr;
    wire done;
    wire signed [63:0] lnrd;
    wire signed [31:0] qkvrd, gemvyrd, ctxrd;
    reg wl_rst, wl_we; reg [31:0] wl_data;

    sequencer_vec #(.P(P), .LANES(LANES), .GBASE(0)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .rd_addr(rdaddr), .ln_rd(lnrd), .qkv_rd(qkvrd), .gemvy_rd(gemvyrd), .ctx_rd(ctxrd),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data));

    reg [WBITS-1:0] wimg [0:`GWQKV-1];
    reg [WBITS-1:0] wword;
    integer i, s, f;

    initial begin
        rst = 1'b1; go = 1'b0; tok = `TOK; pos = 9'd0; rdaddr = 0;
        wl_rst = 1'b0; wl_we = 1'b0; wl_data = 32'b0;
        $readmemh("wrom.mem", wimg);                 // fills first GWQKV words (block0 qkv)
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        // stream the qkv weights into the resident GEMV (w_base 0)
        wl_rst = 1'b1; @(posedge clk); #1; wl_rst = 1'b0;
        for (i = 0; i < `GWQKV; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                wl_we = 1'b1; wl_data = wword[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        wl_we = 1'b0; @(posedge clk); #1;
        // run
        go = 1'b1; @(posedge clk); #1; go = 1'b0;
        wait (done == 1'b1); @(posedge clk); #1;
        // dump lnout (M1) then qkv (M2)
        f = $fopen("lnout.out", "w");
        for (i = 0; i < D; i = i + 1) begin
            rdaddr = i[10:0]; @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%016x\n", lnrd);
        end
        $fclose(f);
        f = $fopen("qkv.out", "w");
        for (i = 0; i < D3; i = i + 1) begin
            rdaddr = i[10:0]; @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", qkvrd);
        end
        $fclose(f);
        f = $fopen("gemvy.out", "w");
        for (i = 0; i < D3; i = i + 1) begin
            rdaddr = i[10:0]; @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", gemvyrd);
        end
        $fclose(f);
        f = $fopen("ctx.out", "w");
        for (i = 0; i < D; i = i + 1) begin
            rdaddr = i[10:0]; @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", ctxrd);
        end
        $fclose(f);
        $display("TB_DONE");
        $finish;
    end
    initial begin #500000000; $display("TB_TIMEOUT"); $finish; end
endmodule
