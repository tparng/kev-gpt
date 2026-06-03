// Testbench for sequencer_vec FULL FORWARD (M5): streams the whole 4-block+head weight
// image into the resident GEMV, runs token TOK at pos 0, and dumps tok_out + the phase keys
// (x4 residual after the last block, LN_f out, head logits) vs seq_ref.full_forward_signals.
`timescale 1ns / 1ps
`ifndef TOK
 `define TOK 48
`endif
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef WROMN
 `define WROMN 199936
`endif
`ifndef LVAL
 `define LVAL 16
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = `LVAL;
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [8:0] tok, pos;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire [8:0] tok_out;
    wire signed [63:0] rdata;
    reg wl_rst, wl_we; reg [31:0] wl_data;

    sequencer_vec #(.P(P), .LANES(LANES)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .tok_out(tok_out), .rd_sel(rsel), .rd_addr(raddr), .rd_data(rdata),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data));

    reg [WBITS-1:0] wimg [0:`WROMN-1];
    reg [WBITS-1:0] wword;
    integer i, s, f, cyc0, cyc1;

    task dump(input [3:0] sel, input integer n, input [127:0] fname);
        integer k, ff;
        begin
            ff = $fopen(fname, "w");
            for (k = 0; k < n; k = k + 1) begin
                rsel = sel; raddr = k[10:0];
                @(posedge clk); @(posedge clk); #1;
                $fwrite(ff, "%016x\n", rdata);
            end
            $fclose(ff);
        end
    endtask

    initial begin
        rst = 1'b1; go = 1'b0; tok = `TOK; pos = 9'd0; rsel = 0; raddr = 0;
        wl_rst = 1'b0; wl_we = 1'b0; wl_data = 32'b0;
        $readmemh("wrom.mem", wimg);
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        wl_rst = 1'b1; @(posedge clk); #1; wl_rst = 1'b0;
        for (i = 0; i < `WROMN; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                wl_we = 1'b1; wl_data = wword[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        wl_we = 1'b0; @(posedge clk); #1;
        cyc0 = dbgcyc;
        go = 1'b1; @(posedge clk); #1; go = 1'b0;
        wait (done == 1'b1); cyc1 = dbgcyc; @(posedge clk); #1;
        $display("FWD_CYCLES=%0d", cyc1 - cyc0);
        f = $fopen("cyc.out", "w"); $fwrite(f, "%0d\n", cyc1 - cyc0); $fclose(f);
        f = $fopen("tok.out", "w"); $fwrite(f, "%0d\n", tok_out); $fclose(f);
        dump(4'd7, 256, "x4.out");
        dump(4'd0, 256, "lnf.out");
        dump(4'd8, 193, "head.out");
        $display("TB_DONE tok_out=%0d", tok_out);
        $finish;
    end
    // progress probe
    integer dbgcyc = 0;
    always @(posedge clk) begin
        dbgcyc = dbgcyc + 1;
        if (dbgcyc % 100000 == 0)
            $display("[cyc %0d] st=%0d blk=%0d ci=%0d done=%b", dbgcyc, dut.st, dut.blk, dut.ci, done);
    end
    initial begin #80000000; $display("TB_TIMEOUT cyc=%0d st=%0d blk=%0d", dbgcyc, dut.st, dut.blk); $finish; end
endmodule
