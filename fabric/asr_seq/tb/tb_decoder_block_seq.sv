// tb_decoder_block_seq -- the first FUNCTIONAL (not just elaboration) gate
// for decoder_block_seq.sv: loads real moonshine-tiny layer-0 weights
// (INT8-quantized per pack_decoder_block.py's own documented scheme),
// gamma, bias, and cross-attention K/V (T2=6), writes the real initial
// residual (a real decoder token embedding), pulses `go` for ONE full
// decoder-layer forward pass (decode step 0), and checks the final
// residual stream against pack_decoder_block.py's own golden xres3 --
// bit-exact.
//
// Weight-load protocol: identical to fabric/stage3/run_resident_banked_vec.py's
// own established gemv_banked_resident_vec.sv gate -- one-time resident load,
// each LANES*WBW=1024-bit wide word streamed as SUBW=32 32-bit chunks
// (low-chunk-first) via w_we/w_data, after one ld_rst pulse.
`timescale 1ns / 1ps
`ifndef NWORDS
 `define NWORDS 13824
`endif

module tb;
    localparam integer P        = 8;
    localparam integer D        = 288;
    localparam integer FFN      = 1152;
    localparam integer DFFN2    = 2*FFN;
    localparam integer NHEAD    = 8;
    localparam integer HEAD_DIM = 36;
    localparam integer ATTN_P   = 4;
    localparam integer T2       = 6;
    localparam integer ROWS_D   = D/P;
    localparam integer ROWS_FFN2 = DFFN2/P;
    localparam integer LANES    = 128;
    localparam integer WBW      = 8;
    localparam integer WBITS    = LANES*WBW;
    localparam integer SUBW     = WBITS/32;
    localparam integer NWORDS   = `NWORDS;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    reg go; reg [3:0] blk; reg [8:0] step; wire done;
    reg xres_wr; reg [$clog2(D/P)-1:0] xres_waddr; reg [P*32-1:0] xres_wdata;
    wire [P*32-1:0] xres_rdata_dbg;

    reg gv_ld_rst, gv_ld_we; reg [31:0] gv_ld_data;
    reg gam_we; reg [1:0] gam_sel; reg [$clog2(D/P)-1:0] gam_waddr; reg [P*32-1:0] gam_wdata;
    reg bias_we, bias_sel; reg [$clog2(DFFN2/P)-1:0] bias_waddr; reg [P*32-1:0] bias_wdata;
    reg xkv_wstart, xkv_wkv; reg [$clog2(NHEAD)-1:0] xkv_whead; reg [8:0] xkv_wpos;
    reg xkv_wvalid; reg [ATTN_P*32-1:0] xkv_wdata; wire xkv_wdone;

    decoder_block_seq #(
        .P(P), .D(D), .FFN(FFN), .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .ATTN_P(ATTN_P),
        .ATTN_TMAX(8), .T2(T2),
        .POST_SCALE_Q16(75674),
        .ACT_Q(17), .ACT_K(17), .ACT_V(17), .ACT_O(15),
        .ACT_CQ(19), .ACT_OC(20), .ACT_FC1(17), .ACT_FC2(11),
        .GF_Q(-4), .GF_K(-4), .GF_V(-2), .GF_O(-9),
        .GF_CQ(-6), .GF_OC(-14), .GF_FC1(-1), .GF_FC2(-18),
        .WB_Q(0), .WB_K(864), .WB_V(1728), .WB_O(2592),
        .WB_CQ(3456), .WB_CO(4320), .WB_FC1(5184), .WB_FC2(10368)
    ) dut (
        .clk(clk), .rst(rst), .go(go), .blk(blk), .step(step), .done(done),
        .xres_wr(xres_wr), .xres_waddr(xres_waddr), .xres_wdata(xres_wdata),
        .xres_rdata_dbg(xres_rdata_dbg),
        .gv_ld_rst(gv_ld_rst), .gv_ld_we(gv_ld_we), .gv_ld_data(gv_ld_data),
        .gam_we(gam_we), .gam_sel(gam_sel), .gam_waddr(gam_waddr), .gam_wdata(gam_wdata),
        .bias_we(bias_we), .bias_sel(bias_sel), .bias_waddr(bias_waddr), .bias_wdata(bias_wdata),
        .xkv_wstart(xkv_wstart), .xkv_wkv(xkv_wkv), .xkv_whead(xkv_whead), .xkv_wpos(xkv_wpos),
        .xkv_wvalid(xkv_wvalid), .xkv_wdata(xkv_wdata), .xkv_wdone(xkv_wdone)
    );

    // ---- test vectors ----
    reg [WBITS-1:0] wload [0:NWORDS-1];
    reg [31:0] gamma1 [0:ROWS_D*P-1];
    reg [31:0] gamma2 [0:ROWS_D*P-1];
    reg [31:0] gamma3 [0:ROWS_D*P-1];
    reg [31:0] biasfc1 [0:ROWS_FFN2*P-1];
    reg [31:0] biasfc2 [0:ROWS_D*P-1];
    reg [31:0] xkvin [0:T2*NHEAD*HEAD_DIM*2-1];
    reg [31:0] xres0v [0:ROWS_D*P-1];
    reg [31:0] xres3ref [0:D-1];

    integer i, s, hcnt, pos, head, b, l, mism, checked;
    reg [HEAD_DIM*32-1:0] kvec, vvec;
    reg [WBITS-1:0] word_tmp;
    reg [P*32-1:0] rowbuf;

    // watchdog: force a diagnostic dump + finish if `done` hasn't fired
    // within a generous margin of one decoder-layer forward pass, so a
    // hang doesn't run out the full $finish-at-2e9 backstop below.
    reg dbg_armed;
    reg [31:0] watchdog_cnt;
    always @(posedge clk) begin
        if (dbg_armed && !done) watchdog_cnt <= watchdog_cnt + 1;
        else watchdog_cnt <= 0;
        if (dbg_armed && watchdog_cnt == 32'd60000) begin
            $display("TB_WATCHDOG_TIMEOUT,t=%0t,st=%0d,hh=%0d,wi=%0d", $time, dut.st, dut.hh, dut.wi);
            $finish;
        end
    end

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("gamma_ln1.mem", gamma1);
        $readmemh("gamma_ln2.mem", gamma2);
        $readmemh("gamma_ln3.mem", gamma3);
        $readmemh("bias_fc1.mem", biasfc1);
        $readmemh("bias_fc2.mem", biasfc2);
        $readmemh("xkv_in.mem", xkvin);
        $readmemh("xres0.mem", xres0v);
        $readmemh("xres3_ref.mem", xres3ref);

        go=0; blk=0; step=0; xres_wr=0; xres_waddr=0; xres_wdata=0;
        gv_ld_rst=0; gv_ld_we=0; gv_ld_data=0;
        gam_we=0; gam_sel=0; gam_waddr=0; gam_wdata=0;
        bias_we=0; bias_sel=0; bias_waddr=0; bias_wdata=0;
        xkv_wstart=0; xkv_wkv=0; xkv_whead=0; xkv_wpos=0; xkv_wvalid=0; xkv_wdata=0;

        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ---- 1. load GEMV resident weight image ----
        gv_ld_rst = 1; @(posedge clk); #1; gv_ld_rst = 0;
        for (i = 0; i < NWORDS; i = i + 1) begin
            word_tmp = wload[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                gv_ld_we = 1; gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        gv_ld_we = 0;
        $display("TB_WEIGHTS_LOADED,nwords=%0d", NWORDS);

        // ---- 2. load LN gamma tables ----
        for (i = 0; i < ROWS_D; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = gamma1[i*P+l];
            gam_sel = 0; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
            @(posedge clk); #1;
        end
        for (i = 0; i < ROWS_D; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = gamma2[i*P+l];
            gam_sel = 1; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
            @(posedge clk); #1;
        end
        for (i = 0; i < ROWS_D; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = gamma3[i*P+l];
            gam_sel = 2; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
            @(posedge clk); #1;
        end
        gam_we = 0;
        $display("TB_GAMMA_LOADED");

        // ---- 3. load bias tables ----
        for (i = 0; i < ROWS_FFN2; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = biasfc1[i*P+l];
            bias_sel = 0; bias_waddr = i[$clog2(DFFN2/P)-1:0]; bias_wdata = rowbuf; bias_we = 1;
            @(posedge clk); #1;
        end
        for (i = 0; i < ROWS_D; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = biasfc2[i*P+l];
            bias_sel = 1; bias_waddr = i[$clog2(DFFN2/P)-1:0]; bias_wdata = rowbuf; bias_we = 1;
            @(posedge clk); #1;
        end
        bias_we = 0;
        $display("TB_BIAS_LOADED");

        // ---- 4. preload cross-attn K/V (same protocol as
        // tb_decoder_cross_attn.sv's do_kv_write task) ----
        for (pos = 0; pos < T2; pos = pos + 1) begin
            for (head = 0; head < NHEAD; head = head + 1) begin
                hcnt = (pos * NHEAD + head) * HEAD_DIM * 2;
                for (i = 0; i < HEAD_DIM; i = i + 1) begin
                    kvec[i*32 +: 32] = xkvin[hcnt + i];
                    vvec[i*32 +: 32] = xkvin[hcnt + HEAD_DIM + i];
                end
                // write K
                xkv_wkv = 1'b0; xkv_whead = head[$clog2(NHEAD)-1:0]; xkv_wpos = pos[8:0];
                xkv_wstart = 1'b1; @(posedge clk); #1; xkv_wstart = 1'b0;
                for (b = 0; b < HEAD_DIM/ATTN_P; b = b + 1) begin
                    for (l = 0; l < ATTN_P; l = l + 1) xkv_wdata[l*32 +: 32] = kvec[(b*ATTN_P+l)*32 +: 32];
                    xkv_wvalid = 1'b1; @(posedge clk); #1;
                end
                xkv_wvalid = 1'b0;
                while (!xkv_wdone) @(posedge clk); #1;
                // write V
                xkv_wkv = 1'b1; xkv_whead = head[$clog2(NHEAD)-1:0]; xkv_wpos = pos[8:0];
                xkv_wstart = 1'b1; @(posedge clk); #1; xkv_wstart = 1'b0;
                for (b = 0; b < HEAD_DIM/ATTN_P; b = b + 1) begin
                    for (l = 0; l < ATTN_P; l = l + 1) xkv_wdata[l*32 +: 32] = vvec[(b*ATTN_P+l)*32 +: 32];
                    xkv_wvalid = 1'b1; @(posedge clk); #1;
                end
                xkv_wvalid = 1'b0;
                while (!xkv_wdone) @(posedge clk); #1;
            end
        end
        $display("TB_CROSS_KV_LOADED,t2=%0d,nhead=%0d", T2, NHEAD);

        // ---- 5. write initial residual (real decoder token embedding) ----
        for (i = 0; i < ROWS_D; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = xres0v[i*P+l];
            xres_waddr = i[$clog2(D/P)-1:0]; xres_wdata = rowbuf; xres_wr = 1;
            @(posedge clk); #1;
        end
        xres_wr = 0;
        $display("TB_XRES0_LOADED");

        // ---- 6. run one decoder-layer forward pass, decode step 0 ----
        blk = 0; step = 0;
        dbg_armed = 1'b1;
        go = 1; @(posedge clk); #1; go = 0;
        while (!done) @(posedge clk); #1;
        dbg_armed = 1'b0;
        $display("TB_LAYER_DONE");

        // ---- 7. check final residual vs golden ----
        mism = 0; checked = 0;
        for (i = 0; i < ROWS_D; i = i + 1) begin
            xres_waddr = i[$clog2(D/P)-1:0];
            #1;
            for (l = 0; l < P; l = l + 1) begin
                checked = checked + 1;
                if (xres_rdata_dbg[l*32 +: 32] !== xres3ref[i*P+l]) begin
                    mism = mism + 1;
                    if (mism <= 10)
                        $display("MISMATCH,row=%0d,lane=%0d,got=%0d,ref=%0d",
                                  i, l, $signed(xres_rdata_dbg[l*32 +: 32]), $signed(xres3ref[i*P+l]));
                end
            end
        end

        $display("TB_DONE,checked=%0d,mismatches=%0d", checked, mism);
        $display("DECODER_BLOCK_SEQ_VERDICT,bitexact=%0d,mismatches=%0d,checked=%0d",
                  (mism == 0), mism, checked);
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
