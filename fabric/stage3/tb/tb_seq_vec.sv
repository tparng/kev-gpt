// Testbench for sequencer_vec MILESTONE 1 (embed -> LN1). Pulses go for token TOK at pos 0,
// waits for done, then reads back the 256 lnout (Q.22) values via rd_addr (2-cycle latency)
// and dumps them for the Python compare vs seq_ref.block0_phase_signals["ln1_out_q22"].
`timescale 1ns / 1ps
`ifndef TOK
 `define TOK 48
`endif
`ifndef PVAL
 `define PVAL 8
`endif
module tb;
    localparam integer P = `PVAL;
    localparam integer D = 256;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [8:0] tok, pos, rdaddr;
    wire done;
    wire signed [63:0] lnrd;

    sequencer_vec #(.P(P), .GBASE(0)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos),
        .done(done), .rd_addr(rdaddr), .ln_rd(lnrd));

    integer i, f;
    initial begin
        rst = 1'b1; go = 1'b0; tok = `TOK; pos = 9'd0; rdaddr = 0;
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        go = 1'b1; @(posedge clk); #1; go = 1'b0;
        wait (done == 1'b1); @(posedge clk); #1;
        f = $fopen("lnout.out", "w");
        for (i = 0; i < D; i = i + 1) begin
            rdaddr = i[8:0];
            @(posedge clk); @(posedge clk); #1;     // 2-cycle registered readback
            $fwrite(f, "%016x\n", lnrd);
        end
        $fclose(f);
        $display("TB_DONE");
        $finish;
    end
    initial begin #50000000; $display("TB_TIMEOUT"); $finish; end
endmodule
