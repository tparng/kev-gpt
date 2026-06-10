// Testbench for sequencer_vec MULTI-TOKEN KV-faithful decode (doc-7 R1): streams the
// weight image once, then drives PLEN prompt positions + NGEN greedy feedback positions
// through repeated GO pulses (pos advancing, kv_bank persisting). Dumps the generated
// token stream (the outputs of passes PLEN-1 .. PLEN+NGEN-2) and per-pass cycle counts.
// Gate: stream must equal IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True)
// .generate_greedy(prompt, NGEN)[PLEN:].
`timescale 1ns / 1ps
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef WROMN
 `define WROMN 199936
`endif
`ifndef LVAL
 `define LVAL 16
`endif
`ifndef TMAXVAL
 `define TMAXVAL 256
`endif
`ifndef PLEN
 `define PLEN 4
`endif
`ifndef NGEN
 `define NGEN 6
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = `LVAL;
    localparam integer TMAXP = `TMAXVAL;
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;
    localparam integer PLEN  = `PLEN;
    localparam integer NGEN  = `NGEN;
    localparam integer NPASS = PLEN + NGEN - 1;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [8:0] tok, pos;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire [8:0] tok_out;
    wire signed [63:0] rdata;
    reg wl_rst, wl_we; reg [31:0] wl_data;

`ifndef KVSTOP
 `define KVSTOP 0
`endif
    reg [1:0] dbgstop_r = 2'b0;
    sequencer_vec #(.P(P), .LANES(LANES), .TMAX(TMAXP)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .tok_out(tok_out), .rd_sel(rsel), .rd_addr(raddr), .rd_data(rdata),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data), .dbg_stop(dbgstop_r));

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

    reg [WBITS-1:0] wimg [0:`WROMN-1];
    reg [WBITS-1:0] wword;
    reg [8:0] prompt [0:PLEN-1];
    reg [8:0] stream [0:PLEN+NGEN-1];
    integer i, s, f, fs, fc, cyc0, pi;

    initial begin
        rst = 1'b1; go = 1'b0; tok = 9'd0; pos = 9'd0; rsel = 0; raddr = 0;
        wl_rst = 1'b0; wl_we = 1'b0; wl_data = 32'b0;
        $readmemh("wrom.mem", wimg);
        $readmemh("prompt.mem", prompt);
        for (i = 0; i < PLEN; i = i + 1) stream[i] = prompt[i];
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        wl_rst = 1'b1; @(posedge clk); #1; wl_rst = 1'b0;
        for (i = 0; i < `WROMN; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                wl_we = 1'b1; wl_data = wword[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        wl_we = 1'b0; @(posedge clk); #1;

        fc = $fopen("cycs.out", "w");
        for (pi = 0; pi < NPASS; pi = pi + 1) begin
            tok = stream[pi]; pos = pi[8:0];
            dbgstop_r = (pi == NPASS-1) ? `KVSTOP : 2'b0;
            cyc0 = dbgcyc;
            go = 1'b1; @(posedge clk); #1; go = 1'b0;
            wait (done == 1'b1); @(posedge clk); #1;
            $fwrite(fc, "%0d\n", dbgcyc - cyc0);
            if (pi + 1 >= PLEN) begin        // greedy append ONLY past the prompt
                stream[pi+1] = tok_out;
                $display("GEN pos=%0d tok=%0d", pi, tok_out);
            end
        end
        $fclose(fc);
        if (`KVSTOP != 0) dump(4'd7, 256, "x.out");   // xres snapshot at the halt point
        fs = $fopen("stream.out", "w");
        for (i = PLEN; i < PLEN + NGEN; i = i + 1) $fwrite(fs, "%0d\n", stream[i]);
        $fclose(fs);
        $display("TB_DONE");
        $finish;
    end

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
        end
        if (running) stcnt[dut.st] = stcnt[dut.st] + 1;
        if (dbgcyc % 500000 == 0)
            $display("[cyc %0d] st=%0d blk=%0d pos=%0d done=%b", dbgcyc, dut.st, dut.blk, pos, done);
    end
`ifdef KVDBG
    // KV-write header trace + dequant read-stream trace (lane 0 of each row)
    always @(posedge clk) begin
        if (dut.u_kvb.wq_done)
            $display("KVW blk=%0d kv=%0d h=%0d pos=%0d lo=%0d scale=%0d inv=%0d",
                     dut.blk, dut.kvw_kv, dut.kvw_h, pos,
                     $signed(dut.u_kvb.w_lo), dut.u_kvb.w_scale, dut.u_kvb.w_inv);
        if (dut.u_kvb.rd_valid)
            $display("KVR blk=%0d kv=%0d h=%0d l0=%0d hdr_lo=%0d hdr_sc=%0d",
                     dut.blk, dut.kb_rkv, dut.hh, $signed(dut.u_kvb.rd_data[31:0]),
                     $signed(dut.u_kvb.hdr_rd[31:0]), dut.u_kvb.hdr_rd[47:32]);
        if (dut.u_kvb.wq_done)
            $display("KVWD pbase=%0d", dut.u_kvb.w_pbase);
        if (dut.at_start)
            $display("AST blk=%0d h=%0d tcount=%0d ldcnt_prev=%0d", dut.blk, dut.hh,
                     dut.at_tcount, ldcnt);
        if (dut.at_ldv) ldcnt = ldcnt + 1;
        // first-X detectors at each datapath write point (print once each)
        if (dut.gv_xwe   && (^dut.gv_xdata === 1'bx) && !xs[0]) begin
            $display("X@AQ st=%0d blk=%0d asrc=%0d row=%0d cyc=%0d glcnt=%0d",
                     dut.st, dut.blk, dut.g_asrc, dut.cid2, dbgcyc, glcnt); xs[0]=1; end
        if (dut.gl_vout && (^dut.gsw === 1'bx) && !xs[6]) begin
            $display("X@GELUW gor=%0d blk=%0d cyc=%0d", dut.gor, dut.blk, dbgcyc); xs[6]=1; end
        if (dut.gl_vout) glcnt = glcnt + 1;
        if (dut.dq_vout  && (^dut.dword === 1'bx) && !xs[1]) begin
            $display("X@DQ dst=%0d blk=%0d cyc=%0d", dut.g_dst, dut.blk, dbgcyc); xs[1]=1; end
        if (dut.ln_yv    && (^dut.ln_y === 1'bx) && !xs[2]) begin
            $display("X@LN gbase=%0d blk=%0d cyc=%0d", dut.l_gbase, dut.blk, dbgcyc); xs[2]=1; end
        if (dut.at_ctxv  && (^dut.at_ctxdata === 1'bx) && !xs[3]) begin
            $display("X@CTX blk=%0d h=%0d cyc=%0d", dut.blk, dut.hh, dbgcyc); xs[3]=1; end
        if (dut.st==5'd14 && (^dut.sw === 1'bx) && !xs[4]) begin
            $display("X@RES1 blk=%0d cyc=%0d cid=%0d xres_x=%b attn_x=%b", dut.blk, dbgcyc,
                     dut.cid, (^dut.xres_r === 1'bx), (^dut.attn_r === 1'bx)); xs[4]=1; end
        if (dut.st==5'd20 && (^dut.sw === 1'bx) && !xs[5]) begin
            $display("X@RES2 blk=%0d cyc=%0d", dut.blk, dbgcyc); xs[5]=1; end
    end
    integer ldcnt = 0;
    integer glcnt = 0;
    reg [7:0] xs = 8'b0;
`endif
    initial begin #400000000; $display("TB_TIMEOUT cyc=%0d st=%0d blk=%0d", dbgcyc, dut.st, dut.blk); $finish; end
endmodule
