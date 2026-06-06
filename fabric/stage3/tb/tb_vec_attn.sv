// Testbench for vec_attn: drive NCASE attention-head cases. Each case has T, a
// q vector (HEAD_DIM words), a K cache (T*HEAD_DIM words) and a V cache
// (T*HEAD_DIM words), all concatenated across cases in q.mem / k.mem / v.mem.
// Per-case T comes from tlen.mem. Capture the HEAD_DIM ctx words per case to
// ctx.out (one hex word per line, signed 32-bit, ROW marker per case).
// Python (run_vec_attn.py) splits by ROW and does the bit-true compare.
`timescale 1ns / 1ps
`ifndef NCASE
 `define NCASE 1
`endif
`ifndef QWORDS
 `define QWORDS 64
`endif
`ifndef KVWORDS
 `define KVWORDS 64
`endif
`ifndef PVAL
 `define PVAL 8
`endif

module tb;
    localparam integer NCASE    = `NCASE;
    localparam integer QWORDS   = `QWORDS;    // total q words across all cases
    localparam integer KVWORDS  = `KVWORDS;   // total K (and V) words across all cases
    localparam integer HEAD_DIM = 64;
    localparam integer TMAX     = 32;
    localparam integer P        = `PVAL;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                rst;
    reg                start;
    reg  [8:0]         tcount;
    reg                ld_valid;
    reg  [P*32-1:0]    ld_data;
    wire               ld_ready;
    wire               ctx_valid;
    wire [6:0]         ctx_idx;          // dim-group index
    wire [P*32-1:0]    ctx_data;         // P ctx lanes per strobe
    wire               done;

    vec_attn #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(TMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .tcount(tcount),
        .ld_valid(ld_valid), .ld_data(ld_data), .ld_ready(ld_ready),
        .ctx_valid(ctx_valid), .ctx_idx(ctx_idx), .ctx_data(ctx_data),
        .done(done)
    );

    reg signed [31:0] qmem  [0:QWORDS-1];
    reg signed [31:0] kmem  [0:KVWORDS-1];
    reg signed [31:0] vmem  [0:KVWORDS-1];
    reg [8:0]         tlen  [0:NCASE-1];

    // capture ctx in index order (module emits in order 0..HEAD_DIM-1)
    reg signed [31:0] ctxcap [0:HEAD_DIM-1];

    integer c, t, kbase, qbase, n, k, f, i;

    initial begin
        $readmemh("q.mem",   qmem);
        $readmemh("k.mem",   kmem);
        $readmemh("v.mem",   vmem);
        $readmemh("tlen.mem", tlen);
        f = $fopen("ctx.out", "w");

        rst = 1'b1; start = 1'b0; ld_valid = 1'b0; ld_data = 32'sd0; tcount = 9'd0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        qbase = 0; kbase = 0;
        for (c = 0; c < NCASE; c = c + 1) begin
            n = tlen[c];
            // start pulse with this case's T
            tcount = n[8:0];
            start  = 1'b1;
            @(posedge clk); #1;
            start  = 1'b0;

            // stream q (HEAD_DIM/P wide words: P lanes per beat)
            k = 0;
            while (k < HEAD_DIM) begin
                if (ld_ready) begin
                    ld_valid = 1'b1;
                    for (i = 0; i < P; i = i + 1)
                        ld_data[i*32 +: 32] = qmem[qbase + k + i];
                    k = k + P;
                end else ld_valid = 1'b0;
                @(posedge clk); #1;
            end
            // stream K (n*HEAD_DIM/P wide words)
            k = 0;
            while (k < n*HEAD_DIM) begin
                if (ld_ready) begin
                    ld_valid = 1'b1;
                    for (i = 0; i < P; i = i + 1)
                        ld_data[i*32 +: 32] = kmem[kbase + k + i];
                    k = k + P;
                end else ld_valid = 1'b0;
                @(posedge clk); #1;
            end
            // stream V (n*HEAD_DIM/P wide words)
            k = 0;
            while (k < n*HEAD_DIM) begin
                if (ld_ready) begin
                    ld_valid = 1'b1;
                    for (i = 0; i < P; i = i + 1)
                        ld_data[i*32 +: 32] = vmem[kbase + k + i];
                    k = k + P;
                end else ld_valid = 1'b0;
                @(posedge clk); #1;
            end
            ld_valid = 1'b0;

            // capture ctx until done (P lanes per strobe)
            while (!done) begin
                if (ctx_valid)
                    for (i = 0; i < P; i = i + 1)
                        ctxcap[ctx_idx*P + i] = ctx_data[i*32 +: 32];
                @(posedge clk); #1;
            end

            // dump in index order
            for (i = 0; i < HEAD_DIM; i = i + 1)
                $fwrite(f, "%08x\n", ctxcap[i]);
            $fwrite(f, "ROW\n");

            qbase = qbase + HEAD_DIM;
            kbase = kbase + n*HEAD_DIM;
            @(posedge clk); #1;
        end

        $fclose(f);
        $display("TB_DONE NCASE=%0d", NCASE);
        $finish;
    end

    // safety timeout
    initial begin
        #5000000;
        $display("TB_TIMEOUT");
        $finish;
    end

`ifdef PROFILE
    // -------- per-state cycle profiler (counts cycles in each DUT FSM state) -----
    // Also profiles the softmax sub-FSM. Reports cumulative counts over the whole
    // run; for a single-case (NCASE=1, T=1) run these ARE the per-head budget.
    integer cnt_st [0:15];
    integer cnt_sm [0:7];
    integer tot_cyc;
    integer pidx;
    initial begin
        for (pidx = 0; pidx < 16; pidx = pidx + 1) cnt_st[pidx] = 0;
        for (pidx = 0; pidx < 8;  pidx = pidx + 1) cnt_sm[pidx] = 0;
        tot_cyc = 0;
    end
    always @(posedge clk) if (!rst) begin
        tot_cyc = tot_cyc + 1;
        cnt_st[dut.st]        = cnt_st[dut.st] + 1;
        cnt_sm[dut.u_sm.state]= cnt_sm[dut.u_sm.state] + 1;
    end
    final begin
        $display("PROF total=%0d", tot_cyc);
        $display("PROF attn IDLE=%0d LD_Q=%0d LD_K=%0d LD_V=%0d SCORE=%0d SCORE_EMIT=%0d SM_COLL=%0d CTX=%0d CTX_EMIT=%0d DONE=%0d",
            cnt_st[0],cnt_st[1],cnt_st[2],cnt_st[3],cnt_st[4],cnt_st[5],cnt_st[6],cnt_st[7],cnt_st[8],cnt_st[9]);
        $display("PROF smax IDLE=%0d LOAD=%0d EXP=%0d RECIP=%0d NORM=%0d DONE=%0d",
            cnt_sm[0],cnt_sm[1],cnt_sm[2],cnt_sm[3],cnt_sm[4],cnt_sm[5]);
    end
`endif
endmodule
