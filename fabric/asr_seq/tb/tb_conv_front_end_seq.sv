// Testbench for conv_front_end_seq.sv -- preloads all 3 convs' own weight/
// bias images, groupnorm1's own gamma/beta, and conv1's own real audio
// input, sets the 5 runtime format-glue shifts, pulses `go` ONCE (the DUT's
// own internal FSM sequences conv1->tanh->groupnorm1->conv2->gelu->conv3->
// gelu end to end), and dumps y.out (one P*32-bit hex word per y_valid
// cycle) for the Python bit-true compare (run_conv_front_end.py against
// pack_conv_front_end.py's own golden). Weight-load procedure mirrors
// tb_encoder_block_seq.sv's own (SUBW 32-bit chunks per wide word).
`timescale 1ns / 1ps
`ifndef TIN1VAL
 `define TIN1VAL 3000
`endif
`ifndef NWORDS1
 `define NWORDS1 3048
`endif
`ifndef NWORDS2
 `define NWORDS2 10080
`endif
`ifndef NWORDS3
 `define NWORDS3 5184
`endif
`ifndef DQ1
 `define DQ1 0
`endif
`ifndef GNSHIFT
 `define GNSHIFT 0
`endif
`ifndef DQ2
 `define DQ2 0
`endif
`ifndef GE1SHIFT
 `define GE1SHIFT 0
`endif
`ifndef DQ3
 `define DQ3 0
`endif

module tb;
    localparam integer P      = 8;
    localparam integer LANES  = 128;
    localparam integer WBW    = 8;
    localparam integer WBITS  = LANES*WBW;
    localparam integer SUBW   = WBITS/32;
    localparam integer TIN1   = `TIN1VAL;
    localparam integer NWORDS1 = `NWORDS1;
    localparam integer NWORDS2 = `NWORDS2;
    localparam integer NWORDS3 = `NWORDS3;

    // real conv shapes (must match conv_front_end_seq.sv's own localparams)
    localparam integer COUT1=288, KW1=127, STRIDE1=64;
    localparam integer TOUT1 = (TIN1-KW1)/STRIDE1 + 1;
    localparam integer XTROWS1 = TIN1;                    // CGRP1=1 (CIN1=P=8)
    localparam integer CROWS_GN = COUT1/P;
    localparam integer COUT2=576, KW2=7, STRIDE2=3;
    localparam integer TIN2 = TOUT1;
    localparam integer TOUT2 = (TIN2-KW2)/STRIDE2 + 1;
    localparam integer COUT3=288, KW3=3, STRIDE3=2;
    localparam integer TIN3 = TOUT2;
    localparam integer TOUT3 = (TIN3-KW3)/STRIDE3 + 1;
    localparam integer MROWS3 = COUT3/P;
    localparam integer ROWS_C3 = TOUT3*MROWS3;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst = 1'b1;

    reg c1_gv_ld_rst, c1_gv_ld_we; reg [31:0] c1_gv_ld_data;
    reg c1_xt_we; reg [P*8-1:0] c1_xt_data;
    reg signed [7:0] c1_dq_shift;

    reg gn_g_we, gn_b_we; reg [P*32-1:0] gn_g_data, gn_b_data;
    reg signed [7:0] gn_ashift;

    reg c2_gv_ld_rst, c2_gv_ld_we; reg [31:0] c2_gv_ld_data;
    reg c2_b_we; reg [P*32-1:0] c2_b_data;
    reg signed [7:0] c2_dq_shift;

    reg signed [7:0] ge1_ashift;

    reg c3_gv_ld_rst, c3_gv_ld_we; reg [31:0] c3_gv_ld_data;
    reg c3_b_we; reg [P*32-1:0] c3_b_data;
    reg signed [7:0] c3_dq_shift;

    reg go; wire done;
    wire yv; wire [P*32-1:0] ydata;

    conv_front_end_seq #(.P(P), .LANES(LANES), .WBW(WBW), .TIN1(TIN1)) dut (
        .clk(clk), .rst(rst),
        .c1_gv_ld_rst(c1_gv_ld_rst), .c1_gv_ld_we(c1_gv_ld_we), .c1_gv_ld_data(c1_gv_ld_data),
        .c1_xt_we(c1_xt_we), .c1_xt_data(c1_xt_data), .c1_dq_shift(c1_dq_shift),
        .gn_g_we(gn_g_we), .gn_g_data(gn_g_data), .gn_b_we(gn_b_we), .gn_b_data(gn_b_data),
        .gn_ashift(gn_ashift),
        .c2_gv_ld_rst(c2_gv_ld_rst), .c2_gv_ld_we(c2_gv_ld_we), .c2_gv_ld_data(c2_gv_ld_data),
        .c2_b_we(c2_b_we), .c2_b_data(c2_b_data), .c2_dq_shift(c2_dq_shift),
        .ge1_ashift(ge1_ashift),
        .c3_gv_ld_rst(c3_gv_ld_rst), .c3_gv_ld_we(c3_gv_ld_we), .c3_gv_ld_data(c3_gv_ld_data),
        .c3_b_we(c3_b_we), .c3_b_data(c3_b_data), .c3_dq_shift(c3_dq_shift),
        .go(go), .done(done), .y_valid(yv), .y_data(ydata)
    );

    reg [WBITS-1:0] w1load [0:NWORDS1-1];
    reg [WBITS-1:0] w2load [0:NWORDS2-1];
    reg [WBITS-1:0] w3load [0:NWORDS3-1];
    reg [P*8-1:0]   xt1load [0:XTROWS1-1];
    reg [P*32-1:0]  gload   [0:CROWS_GN-1];
    reg [P*32-1:0]  bload_gn[0:CROWS_GN-1];
    reg [P*32-1:0]  b2load  [0:COUT2/P-1];
    reg [P*32-1:0]  b3load  [0:COUT3/P-1];

    integer i, s, f;
    reg [WBITS-1:0] word_tmp;

    initial begin
        $readmemh("w1.mem", w1load);
        $readmemh("w2.mem", w2load);
        $readmemh("w3.mem", w3load);
        $readmemh("xt1.mem", xt1load);
        $readmemh("g.mem", gload);
        $readmemh("b.mem", bload_gn);
        $readmemh("b2.mem", b2load);
        $readmemh("b3.mem", b3load);

        c1_gv_ld_rst=0; c1_gv_ld_we=0; c1_gv_ld_data=0; c1_xt_we=0; c1_xt_data=0;
        c1_dq_shift = `DQ1;
        gn_g_we=0; gn_g_data=0; gn_b_we=0; gn_b_data=0; gn_ashift = `GNSHIFT;
        c2_gv_ld_rst=0; c2_gv_ld_we=0; c2_gv_ld_data=0; c2_b_we=0; c2_b_data=0;
        c2_dq_shift = `DQ2;
        ge1_ashift = `GE1SHIFT;
        c3_gv_ld_rst=0; c3_gv_ld_we=0; c3_gv_ld_data=0; c3_b_we=0; c3_b_data=0;
        c3_dq_shift = `DQ3;
        go = 0;

        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        c1_gv_ld_rst = 1; @(posedge clk); #1; c1_gv_ld_rst = 0;
        for (i = 0; i < NWORDS1; i = i + 1) begin
            word_tmp = w1load[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                c1_gv_ld_we = 1; c1_gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        c1_gv_ld_we = 0;

        c2_gv_ld_rst = 1; @(posedge clk); #1; c2_gv_ld_rst = 0;
        for (i = 0; i < NWORDS2; i = i + 1) begin
            word_tmp = w2load[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                c2_gv_ld_we = 1; c2_gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        c2_gv_ld_we = 0;

        c3_gv_ld_rst = 1; @(posedge clk); #1; c3_gv_ld_rst = 0;
        for (i = 0; i < NWORDS3; i = i + 1) begin
            word_tmp = w3load[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                c3_gv_ld_we = 1; c3_gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        c3_gv_ld_we = 0;

        for (i = 0; i < XTROWS1; i = i + 1) begin
            c1_xt_we = 1; c1_xt_data = xt1load[i]; @(posedge clk); #1;
        end
        c1_xt_we = 0;

        for (i = 0; i < CROWS_GN; i = i + 1) begin
            gn_g_we = 1; gn_g_data = gload[i]; gn_b_we = 1; gn_b_data = bload_gn[i];
            @(posedge clk); #1;
        end
        gn_g_we = 0; gn_b_we = 0;

        for (i = 0; i < COUT2/P; i = i + 1) begin
            c2_b_we = 1; c2_b_data = b2load[i]; @(posedge clk); #1;
        end
        c2_b_we = 0;

        for (i = 0; i < COUT3/P; i = i + 1) begin
            c3_b_we = 1; c3_b_data = b3load[i]; @(posedge clk); #1;
        end
        c3_b_we = 0;

        $display("TB_LOADED");

        f = $fopen("y.out", "w");
        go = 1; @(posedge clk); #1; go = 0;

        i = 0;
        while (i < ROWS_C3) begin
            if (yv) begin
                $fwrite(f, "%0x\n", ydata);
                i = i + 1;
            end
            @(posedge clk); #1;
        end
        $fclose(f);

        $display("TB_DONE");
        $finish;
    end

    initial begin #2000000000; $display("TB_TIMEOUT"); $finish; end
endmodule
