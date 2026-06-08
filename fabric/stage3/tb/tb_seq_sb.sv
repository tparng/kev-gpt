// Testbench for sequencer_sb (SPLIT-BRAIN): N=16 = two independent N=8 cohorts.
// Streams TOK0..TOK15 through 4 blocks + head + argmax; cohort 0 = streams 0..7,
// cohort 1 = streams 8..15. Dumps per-stream tok_out + phase keys (x4/lnf/head)
// vs seq_ref.full_forward_signals. Same load/dump shape as tb_seq_pp.
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
`ifndef TOK8
 `define TOK8 33
`endif
`ifndef TOK9
 `define TOK9 7
`endif
`ifndef TOK10
 `define TOK10 150
`endif
`ifndef TOK11
 `define TOK11 90
`endif
`ifndef TOK12
 `define TOK12 14
`endif
`ifndef TOK13
 `define TOK13 66
`endif
`ifndef TOK14
 `define TOK14 111
`endif
`ifndef TOK15
 `define TOK15 173
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
 `define NVAL 16
`endif
`ifndef NDVAL
 `define NDVAL 0
`endif
`ifndef DBGSTOP
 `define DBGSTOP 0
`endif
`ifndef ATT2VAL
 `define ATT2VAL 1
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = `LVAL;
    localparam integer TMAXP = `TMAXVAL;
    localparam integer N     = `NVAL;
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;
    // DOUBLE-PUMP-100K Stage 1: -DDPUMP runs the MAC at 2 K-steps/clk (DP=1).
`ifdef DPUMP
    localparam integer DPV = 1;
`else
    localparam integer DPV = 0;
`endif

    reg clk = 1'b0; always #5 clk = ~clk;     // 10 ns period, posedges at 5,15,...
    reg clk2x = 1'b0;
    initial begin #2.5 forever #2.5 clk2x = ~clk2x; end   // 2x, quarter-shifted (tb_macdp)
    reg rst, go;
    reg [N*9-1:0] toks;
    reg [8:0] pos;
    reg [3:0]  rstream;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire [N*9-1:0] tok_outs;
    wire signed [63:0] rdata;
    reg wl_rst, wl_we; reg [31:0] wl_data;

    sequencer_sb #(.P(P), .LANES(LANES), .N(N), .NC(N/2), .ND(`NDVAL),
                   .TMAX(TMAXP), .ATT2(`ATT2VAL), .DP(DPV)) dut (
        .clk(clk), .clk2x(clk2x), .rst(rst), .go(go), .tok_ids(toks), .pos(pos), .done(done),
        .tok_outs(tok_outs), .rd_stream(rstream[$clog2(N)-1:0]), .rd_sel(rsel),
        .rd_addr(raddr), .rd_data(rdata), .wl_rst(wl_rst), .wl_we(wl_we),
        .el_we(1'b0), .wl_data(wl_data), .dbg_stop(2'd`DBGSTOP));

    reg [WBITS-1:0] wimg [0:`WROMN-1];
    reg [WBITS-1:0] wword;
    integer i, s, f, cyc0, cyc1, b;

    task dump(input [3:0] strm, input [3:0] sel, input integer n, input [255:0] fname);
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
    integer tokv [0:15];
    initial begin
        tokv[0]=`TOK0;  tokv[1]=`TOK1;  tokv[2]=`TOK2;  tokv[3]=`TOK3;
        tokv[4]=`TOK4;  tokv[5]=`TOK5;  tokv[6]=`TOK6;  tokv[7]=`TOK7;
        tokv[8]=`TOK8;  tokv[9]=`TOK9;  tokv[10]=`TOK10; tokv[11]=`TOK11;
        tokv[12]=`TOK12; tokv[13]=`TOK13; tokv[14]=`TOK14; tokv[15]=`TOK15;
    end

    reg [255:0] fn;
    initial begin
        rst = 1'b1; go = 1'b0; pos = 9'd0; rstream = 0; rsel = 0; raddr = 0;
        #1; for (b = 0; b < N; b = b + 1) toks[b*9 +: 9] = tokv[b][8:0];
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
            $sformat(fn, "x4_%0d.out", b);   dump(b[3:0], 4'd7, 256, fn);
            $sformat(fn, "lnf_%0d.out", b);  dump(b[3:0], 4'd0, 256, fn);
            $sformat(fn, "head_%0d.out", b); dump(b[3:0], 4'd8, 193, fn);
        end
        prof_report;
        $display("TB_DONE tok0=%0d tok8=%0d tok15=%0d",
                 tok_outs[8:0], tok_outs[8*9 +: 9], tok_outs[15*9 +: 9]);
        $finish;
    end
    integer dbgcyc = 0;
    integer running = 0;
    // ---------------- profiling counters (observer only — no DUT timing change) --
    // GE-state cycle counts per cohort: index = ge state (0..7)
    //   0 GE_IDLE 1 GE_AQ 2 GE_AQN 3 GE_RUN 4 GE_WAIT 5 GE_RB 6 GE_RBN 7 GE_DQW
    integer ge_cnt0 [0:7];
    integer ge_cnt1 [0:7];
    // nl-state cycle counts per cohort (0..22)
    integer nl_cnt0 [0:22];
    integer nl_cnt1 [0:22];
    // grant-wait sub-counts (req asserted but not yet granted by the shared unit)
    integer lnw0, lnw1, atw0, atw1, dqw0, dqw1;
    // GE_IDLE split: idle WAITING for a descriptor (g_req low) vs idle truly done
    integer ge_idle_nordesc0, ge_idle_nordesc1;
    integer gi;
    initial begin
        for (gi=0; gi<8;  gi=gi+1) begin ge_cnt0[gi]=0; ge_cnt1[gi]=0; end
        for (gi=0; gi<23; gi=gi+1) begin nl_cnt0[gi]=0; nl_cnt1[gi]=0; end
        lnw0=0; lnw1=0; atw0=0; atw1=0; dqw0=0; dqw1=0;
        ge_idle_nordesc0=0; ge_idle_nordesc1=0;
    end
    always @(posedge clk) begin
        dbgcyc = dbgcyc + 1;
        if (go) running = 1;
        if (done) running = 0;
        if (running && !done) begin
            ge_cnt0[dut.coh0.ge] = ge_cnt0[dut.coh0.ge] + 1;
            ge_cnt1[dut.coh1.ge] = ge_cnt1[dut.coh1.ge] + 1;
            nl_cnt0[dut.coh0.eng.nl] = nl_cnt0[dut.coh0.eng.nl] + 1;
            nl_cnt1[dut.coh1.eng.nl] = nl_cnt1[dut.coh1.eng.nl] + 1;
            // GE_IDLE with no pending descriptor request (handoff latency / true end)
            if (dut.coh0.ge==0 && !dut.coh0.g_req) ge_idle_nordesc0 = ge_idle_nordesc0 + 1;
            if (dut.coh1.ge==0 && !dut.coh1.g_req) ge_idle_nordesc1 = ge_idle_nordesc1 + 1;
            // LN grant-wait: in NL_LGAM (2), req high, no grant
            if (dut.coh0.eng.nl==2 && !dut.ln_gnt0) lnw0 = lnw0 + 1;
            if (dut.coh1.eng.nl==2 && !dut.ln_gnt1) lnw1 = lnw1 + 1;
            // attn grant-wait: in NL_AST (7), no grant
            if (dut.coh0.eng.nl==7 && !dut.at_gnt0) atw0 = atw0 + 1;
            if (dut.coh1.eng.nl==7 && !dut.at_gnt1) atw1 = atw1 + 1;
            // dq grant-wait already captured by GE_DQW; track separately too
            if (dut.coh0.ge==7) dqw0 = dqw0 + 1;
            if (dut.coh1.ge==7) dqw1 = dqw1 + 1;
        end
        if (dbgcyc % 100000 == 0)
            $display("[cyc %0d] ge0=%0d ge1=%0d nl0=%0d nl1=%0d done=%b",
                     dbgcyc, dut.coh0.ge, dut.coh1.ge, dut.coh0.eng.nl, dut.coh1.eng.nl, done);
    end
    task prof_report;
        integer j;
        reg [8*8-1:0] gname [0:7];
        begin
            gname[0]="IDLE"; gname[1]="AQ"; gname[2]="AQN"; gname[3]="RUN";
            gname[4]="WAIT"; gname[5]="RB"; gname[6]="RBN"; gname[7]="DQW";
            $display("==== PROFILE (per-cohort GE-state cycles, running window) ====");
            for (j=0; j<8; j=j+1)
                $display("GE_%0s        c0=%8d  c1=%8d", gname[j], ge_cnt0[j], ge_cnt1[j]);
            $display("  (GE_IDLE no-descriptor: c0=%0d c1=%0d)", ge_idle_nordesc0, ge_idle_nordesc1);
            $display("---- nl_engine GEMM-wait + grant-wait ----");
            $display("NL_WQKV(6)   c0=%8d  c1=%8d", nl_cnt0[6],  nl_cnt1[6]);
            $display("NL_WPROJ(11) c0=%8d  c1=%8d", nl_cnt0[11], nl_cnt1[11]);
            $display("NL_WFC(14)   c0=%8d  c1=%8d", nl_cnt0[14], nl_cnt1[14]);
            $display("NL_WMP(16)   c0=%8d  c1=%8d", nl_cnt0[16], nl_cnt1[16]);
            $display("NL_WHEAD(19) c0=%8d  c1=%8d", nl_cnt0[19], nl_cnt1[19]);
            $display("LN grant-wait(NL_LGAM,!gnt) c0=%8d c1=%8d", lnw0, lnw1);
            $display("AT grant-wait(NL_AST,!gnt)  c0=%8d c1=%8d", atw0, atw1);
            $display("DQ grant-wait(GE_DQW)       c0=%8d c1=%8d", dqw0, dqw1);
            $display("---- nl_engine compute states (non-wait, sample) ----");
            $display("NL_EMB(1)    c0=%8d  c1=%8d", nl_cnt0[1],  nl_cnt1[1]);
            $display("NL_LGAM(2)   c0=%8d  c1=%8d", nl_cnt0[2],  nl_cnt1[2]);
            $display("NL_LFEED(3)  c0=%8d  c1=%8d", nl_cnt0[3],  nl_cnt1[3]);
            $display("NL_LCOLL(4)  c0=%8d  c1=%8d", nl_cnt0[4],  nl_cnt1[4]);
            $display("NL_AST(7)    c0=%8d  c1=%8d", nl_cnt0[7],  nl_cnt1[7]);
            $display("NL_ALD(8)    c0=%8d  c1=%8d", nl_cnt0[8],  nl_cnt1[8]);
            $display("NL_ACL(9)    c0=%8d  c1=%8d", nl_cnt0[9],  nl_cnt1[9]);
            $display("NL_RES1(12)  c0=%8d  c1=%8d", nl_cnt0[12], nl_cnt1[12]);
            $display("NL_RES2(17)  c0=%8d  c1=%8d", nl_cnt0[17], nl_cnt1[17]);
            $display("NL_ARG(20)   c0=%8d  c1=%8d", nl_cnt0[20], nl_cnt1[20]);
            $display("==== END PROFILE ====");
        end
    endtask
    initial begin #120000000; $display("TB_TIMEOUT cyc=%0d ge0=%0d ge1=%0d", dbgcyc, dut.coh0.ge, dut.coh1.ge); $finish; end
endmodule
