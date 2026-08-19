// tb_gemv_i4i8 — T calls through gemv_i4i8. Files: gv_cfg.mem (T),
// gv_w.mem (ROWS*WPR x 32b), gv_x.mem (T*D_IN x 8b). Out: gv_acc.out.
`timescale 1ns/1ps

module tb_gemv_i4i8;
    localparam int ROWS = 1160, D_IN = 256, WPR = D_IN/8, TMAXV = 8;

    reg clk = 0, rst = 1, start = 0;
    wire done;
    reg wr_w = 0, wr_x = 0;
    reg [$clog2(ROWS*WPR)-1:0] wr_w_addr;
    reg [$clog2(D_IN)-1:0] wr_x_addr;
    reg [31:0] wr_w_data;
    reg signed [7:0] wr_x_data;
    reg [$clog2(ROWS)-1:0] rd_acc_addr;
    wire signed [31:0] rd_acc_data;
    localparam int WMEM = 262144;
    reg [$clog2(WMEM)-1:0] base = 0;
    reg [$clog2(ROWS+1)-1:0] rows = ROWS;
    reg [$clog2(WPR+1)-1:0] wpr = WPR;

    gemv_i4i8 #(.ROWS(ROWS), .D_IN(D_IN*2), .WMEM(WMEM)) dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .base(base), .rows(rows), .wpr(wpr),
        .wr_w(wr_w), .wr_w_addr(wr_w_addr2), .wr_w_data(wr_w_data),
        .wr_x(wr_x), .wr_x_addr(wr_x_addr2), .wr_x_data(wr_x_data),
        .rd_acc_addr(rd_acc_addr), .rd_acc_data(rd_acc_data));
    wire [$clog2(WMEM)-1:0] wr_w_addr2 = wr_w_addr;
    wire [$clog2(D_IN*2)-1:0] wr_x_addr2 = wr_x_addr;
    always #2 clk = ~clk;

    reg [31:0] wmem [0:ROWS*WPR-1];
    reg [7:0]  xmem [0:TMAXV*D_IN-1];
    reg [31:0] cfg [0:0];
    integer T, t, i, fd;

    initial begin
        $readmemh("gv_cfg.mem", cfg); T = cfg[0];
        $readmemh("gv_w.mem", wmem);
        $readmemh("gv_x.mem", xmem, 0, T*D_IN-1);
        fd = $fopen("gv_acc.out", "w");
        repeat (4) @(posedge clk);
        rst <= 0; @(posedge clk);
        for (i = 0; i < ROWS*WPR; i = i + 1) begin
            @(posedge clk); wr_w <= 1; wr_w_addr <= i; wr_w_data <= wmem[i];
        end
        @(posedge clk); wr_w <= 0;
        for (t = 0; t < T; t = t + 1) begin
            for (i = 0; i < D_IN; i = i + 1) begin
                @(posedge clk); wr_x <= 1; wr_x_addr <= i; wr_x_data <= xmem[t*D_IN+i];
            end
            @(posedge clk); wr_x <= 0;
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (done);
            @(posedge clk);
            for (i = 0; i < ROWS; i = i + 1) begin
                rd_acc_addr <= i;
                @(posedge clk);
                $fdisplay(fd, "%08x", rd_acc_data);
            end
        end
        $fclose(fd);
        $display("TB_GV_DONE T=%0d", T);
        $finish;
    end
endmodule
