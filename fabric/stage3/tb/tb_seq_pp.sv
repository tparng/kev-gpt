// Testbench for sequencer_pp: N-stream batched full forward. Streams TOK0..TOK3
// through 4 blocks + head + argmax; dumps per-stream tok_out and the phase keys
// (x4 / lnf / head) per stream vs seq_ref.full_forward_signals.
`timescale 1ns / 1ps
`ifndef TOK0
 `define TOK0 48
`endif
`ifndef TOK1
 `define TOK1 10
`endif
`ifndef TOK2
 `define TOK2 100
`endif
`ifndef TOK3
 `define TOK3 77
`endif
`ifndef TOK4
 `define TOK4 5
`endif
`ifndef TOK5
 `define TOK5 60
`endif
`ifndef TOK6
 `define TOK6 120
`endif
`ifndef TOK7
 `define TOK7 180
`endif
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef WROMN
 `define WROMN 199936
`endif
`ifndef LVAL
 `define LVAL 128
`endif
`ifndef TMAXVAL
 `define TMAXVAL 32
`endif
`ifndef NVAL
 `define NVAL 8
`endif
`ifndef DBGSTOP
 `define DBGSTOP 0
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = `LVAL;
    localparam integer TMAXP = `TMAXVAL;
    localparam integer N     = `NVAL;
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [N*9-1:0] toks;
    reg [8:0] pos;
    reg [2:0]  rstream;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire [N*9-1:0] tok_outs;
    wire signed [63:0] rdata;
    reg wl_rst, wl_we; reg [31:0] wl_data;

    sequencer_pp #(.P(P), .LANES(LANES), .N(N), .TMAX(TMAXP)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_ids(toks), .pos(pos), .done(done),
        .tok_outs(tok_outs), .rd_stream(rstream), .rd_sel(rsel), .rd_addr(raddr),
        .rd_data(rdata), .wl_rst(wl_rst), .wl_we(wl_we), .el_we(1'b0), .wl_data(wl_data),
        .dbg_stop(2'd`DBGSTOP));

    reg [WBITS-1:0] wimg [0:`WROMN-1];
    reg [WBITS-1:0] wword;
    integer i, s, f, cyc0, cyc1, b;

    task dump(input [2:0] strm, input [3:0] sel, input integer n, input [255:0] fname);
        integer k, ff;
        begin
            ff = $fopen(fname, "w");
            for (k = 0; k < n; k = k + 1) begin
                rstream = strm; rsel = sel; raddr = k[10:0];
                @(posedge clk); @(posedge clk); #1;
                $fwrite(ff, "%016x\n", rdata);
            end
            $fclose(ff);
        end
    endtask

    reg [255:0] fn;
    initial begin
        rst = 1'b1; go = 1'b0; pos = 9'd0; rstream = 0; rsel = 0; raddr = 0;
        toks = {9'd`TOK7, 9'd`TOK6, 9'd`TOK5, 9'd`TOK4, 9'd`TOK3, 9'd`TOK2, 9'd`TOK1, 9'd`TOK0};
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
        f = $fopen("tok.out", "w");
        for (b = 0; b < N; b = b + 1) $fwrite(f, "%0d\n", tok_outs[b*9 +: 9]);
        $fclose(f);
        for (b = 0; b < N; b = b + 1) begin
            $sformat(fn, "x4_%0d.out", b);   dump(b[2:0], 4'd7, 256, fn);
            $sformat(fn, "lnf_%0d.out", b);  dump(b[2:0], 4'd0, 256, fn);
            $sformat(fn, "head_%0d.out", b); dump(b[2:0], 4'd8, 193, fn);
        end
        $display("TB_DONE tok0=%0d tok1=%0d tok2=%0d tok3=%0d",
                 tok_outs[8:0], tok_outs[17:9], tok_outs[26:18], tok_outs[35:27]);
        $finish;
    end
    // progress probe + per-state cycle profile
    integer dbgcyc = 0;
    integer stcnt [0:31];
    integer running = 0, k2, fprof;
    initial for (k2 = 0; k2 < 32; k2 = k2 + 1) stcnt[k2] = 0;
    always @(posedge clk) begin
        dbgcyc = dbgcyc + 1;
        if (go) running = 1;
        if (done) begin
            if (running) begin
                fprof = $fopen("profile.out", "w");
                for (k2 = 0; k2 < 32; k2 = k2 + 1)
                    if (stcnt[k2] > 0) $fwrite(fprof, "%0d %0d\n", k2, stcnt[k2]);
                $fclose(fprof);
            end
            running = 0;
        end
        if (running) stcnt[dut.nl[dut.gc]] = stcnt[dut.nl[dut.gc]] + 1;
        if (dbgcyc % 100000 == 0)
            $display("[cyc %0d] nl0=%0d nl1=%0d ge=%0d gc=%0d bs=%0d ar=%0d arv=%b amv=%b req0=%b req1=%b done=%b", dbgcyc, dut.nl[0], dut.nl[1], dut.ge, dut.gc, dut.bs, dut.ar, dut.arv, dut.amv, dut.g_req[0], dut.g_req[1], done);
    end
    initial begin #120000000; $display("TB_TIMEOUT cyc=%0d nl0=%0d nl1=%0d ge=%0d", dbgcyc, dut.nl[0], dut.nl[1], dut.ge); $finish; end
endmodule
