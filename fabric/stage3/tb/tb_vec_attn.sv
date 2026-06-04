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
endmodule
