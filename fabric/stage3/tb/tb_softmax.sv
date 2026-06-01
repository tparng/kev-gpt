// Testbench for softmax: drive one or more causal rows from score.mem (concatenated
// scores) with per-row lengths from tlen.mem, capture every emitted prob (when
// out_valid) to prob.out, one hex word per line. Python (run_softmax.py) aligns the
// stream by row and does the bit-true compare. Latency is data dependent; the TB just
// waits for `done` per row and feeds the next.
`timescale 1ns / 1ps
`ifndef NROWS
 `define NROWS 1
`endif
`ifndef NSCORES
 `define NSCORES 256
`endif

module tb;
    localparam integer NROWS   = `NROWS;
    localparam integer NSCORES = `NSCORES;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                rst;
    reg                start;
    reg  [8:0]         t_count;
    reg                in_valid;
    reg  signed [15:0] score;
    wire               in_ready;
    wire               out_valid;
    wire [20:0]        prob;
    wire               done;

    softmax dut (
        .clk(clk), .rst(rst), .start(start), .t_count(t_count),
        .in_valid(in_valid), .score(score), .in_ready(in_ready),
        .out_valid(out_valid), .prob(prob), .done(done)
    );

    reg signed [15:0] scores [0:NSCORES-1];
    reg [8:0]         tlen   [0:NROWS-1];
    integer r, base, k, f;

    initial begin
        $readmemh("score.mem", scores);
        $readmemh("tlen.mem",  tlen);
        f = $fopen("prob.out", "w");

        rst = 1'b1; start = 1'b0; in_valid = 1'b0; score = 16'sd0; t_count = 9'd0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        base = 0;
        for (r = 0; r < NROWS; r = r + 1) begin
            // start pulse with this row's length
            t_count = tlen[r];
            start   = 1'b1;
            @(posedge clk); #1;
            start   = 1'b0;

            // stream the scores while in_ready
            k = 0;
            while (k < tlen[r]) begin
                if (in_ready) begin
                    in_valid = 1'b1;
                    score    = scores[base + k];
                    k        = k + 1;
                end else begin
                    in_valid = 1'b0;
                end
                @(posedge clk); #1;
            end
            in_valid = 1'b0;

            // wait for done, capturing probs as they appear
            while (!done) begin
                if (out_valid) $fwrite(f, "%06x\n", prob);
                @(posedge clk); #1;
            end
            // mark row boundary so Python can split unambiguously
            $fwrite(f, "ROW\n");
            base = base + tlen[r];
            @(posedge clk); #1;
        end

        $fclose(f);
        $display("TB_DONE NROWS=%0d", NROWS);
        $finish;
    end

    // safety timeout
    initial begin
        #2000000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
