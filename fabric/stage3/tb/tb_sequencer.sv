// Testbench for the Tier-3 sequencer: preloads all weight/table ROMs (the driver
// writes the .mem files), pulses `go` for one token at (tok_id, pos), waits for
// `done`, then dumps the 256-element Q6.25 residual output x_out to xout.out for the
// Python bit-exact compare (run_sequencer.py) against seq_ref.block0_signals.
`timescale 1ns / 1ps
`ifndef TOKID
 `define TOKID 0
`endif
`ifndef POS
 `define POS 0
`endif
`ifndef LANES
 `define LANES 16
`endif
`ifndef PNLAYER
 `define PNLAYER 1
`endif

module tb;
    localparam integer D = 256;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg               rst, go;
    reg  [7:0]        tok_id;
    reg  [8:0]        pos;
    wire              busy, done;
    reg  [8:0]        rd_addr;
    wire signed [31:0] x_out;
    wire [8:0]        tok_out;

    sequencer #(.LANES(`LANES), .NLAYER(`PNLAYER)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok_id), .pos(pos),
        .busy(busy), .done(done), .rd_addr(rd_addr), .x_out(x_out), .tok_out(tok_out)
    );

    integer i, f;

    initial begin
        rst = 1'b1; go = 1'b0; tok_id = `TOKID; pos = `POS; rd_addr = 0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        // run one token through block 0
        go = 1'b1; @(posedge clk); #1; go = 1'b0;

        // wait for done (with a generous timeout guard below)
        wait (done == 1'b1);
        @(posedge clk); #1;

        // dump x_out (Q6.25) — 2-cycle readback latency
        f = $fopen("xout.out", "w");
        for (i = 0; i < D; i = i + 1) begin
            rd_addr = i[8:0];
            @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", x_out);
        end
        $fclose(f);
        // dump the emitted token id (full-forward gate)
        f = $fopen("tokout.out", "w");
        $fwrite(f, "%0d\n", tok_out);
        $fclose(f);
        $display("TB_DONE tok=%0d pos=%0d tok_out=%0d", `TOKID, `POS, tok_out);
        $finish;
    end

    initial begin
        #300000000;           // generous: ~49k weight words stream serially per token
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
