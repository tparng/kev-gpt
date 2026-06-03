// Testbench for sequencer_vec FULL block-0 (M4). Streams the whole block-0 weight image
// into the resident GEMV, pulses go for token TOK at pos 0, waits for done, then dumps every
// phase bank (rd_sel 0..7) for the Python compare vs seq_ref.block0_phase_signals. All values
// dumped as 64-bit hex (32-bit signed banks are sign-extended by the readback mux).
`timescale 1ns / 1ps
`ifndef TOK
 `define TOK 48
`endif
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef GWBLK
 `define GWBLK 49152
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = 16;
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [8:0] tok, pos;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire signed [63:0] rdata;
    reg wl_rst, wl_we; reg [31:0] wl_data;

    sequencer_vec #(.P(P), .LANES(LANES)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .rd_sel(rsel), .rd_addr(raddr), .rd_data(rdata),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data));

    reg [WBITS-1:0] wimg [0:`GWBLK-1];
    reg [WBITS-1:0] wword;
    integer i, s, f;

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
        for (i = 0; i < `GWBLK; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                wl_we = 1'b1; wl_data = wword[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        wl_we = 1'b0; @(posedge clk); #1;
        go = 1'b1; @(posedge clk); #1; go = 1'b0;
        wait (done == 1'b1); @(posedge clk); #1;
        dump(4'd0, 256,  "ln1.out");
        dump(4'd1, 768,  "qkv.out");
        dump(4'd2, 256,  "ctx.out");
        dump(4'd3, 256,  "attn.out");
        dump(4'd4, 256,  "ln2.out");
        dump(4'd5, 1024, "gelu.out");
        dump(4'd6, 256,  "mlp.out");
        dump(4'd7, 256,  "xout.out");
        $display("TB_DONE");
        $finish;
    end
    // progress probe: print the FSM state every 50k cycles so a stuck state is visible
    integer dbgcyc = 0;
    always @(posedge clk) begin
        dbgcyc = dbgcyc + 1;
        if (dbgcyc % 50000 == 0)
            $display("[cyc %0d] st=%0d ci=%0d hh=%0d fr=%0d dr=%0d gfr=%0d gor=%0d done=%b",
                     dbgcyc, dut.st, dut.ci, dut.hh, dut.fr, dut.dr, dut.gfr, dut.gor, done);
    end
    initial begin #60000000; $display("TB_TIMEOUT cyc=%0d st=%0d", dbgcyc, dut.st); $finish; end
endmodule
