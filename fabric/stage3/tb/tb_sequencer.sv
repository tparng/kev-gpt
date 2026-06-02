// Testbench for the Tier-3 sequencer. Preloads all weight/table/prompt ROMs (the
// driver writes the .mem files), pulses `go`, waits for `done`, then dumps:
//   xout.out    — the 256-element Q6.25 residual x_out (block-0 bit-exact gate)
//   tokout.out  — the final emitted token id (single-token full-forward gate)
//   tokstream.out — the NGEN-token greedy stream (multi-token gate)
// for the Python compare (run_sequencer.py) against seq_ref.IntSequencer.
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
`ifndef PPROMPT
 `define PPROMPT 1
`endif
`ifndef PNGEN
 `define PNGEN 1
`endif
`ifndef PKVMAX
 `define PKVMAX 32
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
    reg  [8:0]        ts_addr;
    wire [8:0]        ts_out;
    wire [8:0]        ngen_out;

    sequencer #(.LANES(`LANES), .NLAYER(`PNLAYER),
                .PROMPT_LEN(`PPROMPT), .NGEN(`PNGEN), .KVMAX(`PKVMAX)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok_id), .pos(pos),
        .busy(busy), .done(done), .rd_addr(rd_addr), .x_out(x_out), .tok_out(tok_out),
        .ts_addr(ts_addr), .ts_out(ts_out), .ngen_out(ngen_out)
    );

    integer i, f, ng;

    initial begin
        rst = 1'b1; go = 1'b0; tok_id = `TOKID; pos = `POS; rd_addr = 0; ts_addr = 0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        // start the (multi-token autoregressive) decode
        go = 1'b1; @(posedge clk); #1; go = 1'b0;

        // wait for done (with a generous timeout guard below)
        wait (done == 1'b1);
        @(posedge clk); #1;

        // dump x_out (Q6.25) — 2-cycle readback latency (single-token block gate)
        f = $fopen("xout.out", "w");
        for (i = 0; i < D; i = i + 1) begin
            rd_addr = i[8:0];
            @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", x_out);
        end
        $fclose(f);

        // dump the most-recently emitted token id (single-token full-forward gate)
        f = $fopen("tokout.out", "w");
        $fwrite(f, "%0d\n", tok_out);
        $fclose(f);

        // dump the full generated token stream (multi-token gate)
        ng = ngen_out;
        f = $fopen("tokstream.out", "w");
        for (i = 0; i < ng; i = i + 1) begin
            ts_addr = i[8:0];
            @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%0d\n", ts_out);
        end
        $fclose(f);

        $display("TB_DONE tok=%0d pos=%0d tok_out=%0d ngen=%0d", `TOKID, `POS, tok_out, ng);
        $finish;
    end

    initial begin
        #3000000000;          // generous: ~49k weight words stream serially per token * NGEN
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
