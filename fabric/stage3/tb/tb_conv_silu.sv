// tb_conv_silu — T tokens through conv_silu from .mem stimulus.
// Files: cs_cfg.mem (T), cs_w.mem (CH*K), cs_b.mem (CH), cs_lut.mem (256),
//        cs_x.mem (T*CH). Output: cs_y.out (T*CH hex INT16).
`timescale 1ns/1ps

module tb_conv_silu;
    localparam int CH = 640, K = 4, L = 7, TMAXV = 64;

    reg clk = 0, rst = 1, start = 0;
    reg [$clog2(L)-1:0] layer = 0;   // single-layer test drives bank 0
    wire done;
    reg wr_w = 0, wr_b = 0, wr_x = 0, wr_lut = 0;
    reg [$clog2(CH*K)-1:0] wr_w_addr;
    reg [$clog2(CH)-1:0]   wr_b_addr, wr_x_addr, rd_y_addr;
    reg [7:0]              wr_lut_addr;
    reg signed [15:0] wr_w_data, wr_b_data, wr_x_data, wr_lut_data;
    wire signed [15:0] rd_y_data;

    conv_silu #(.CH(CH), .K(K), .L(L)) dut (.*);
    always #2 clk = ~clk;

    reg [15:0] wmem [0:CH*K-1];
    reg [15:0] bmem [0:CH-1];
    reg [15:0] lmem [0:255];
    reg [15:0] xmem [0:TMAXV*CH-1];
    reg [15:0] lymem[0:TMAXV-1];      // per-token history bank select
    reg [31:0] cfg  [0:0];
    integer T, t, i, fd;

    initial begin
        $readmemh("cs_cfg.mem", cfg);  T = cfg[0];
        $readmemh("cs_w.mem", wmem);
        $readmemh("cs_b.mem", bmem);
        $readmemh("cs_lut.mem", lmem);
        $readmemh("cs_x.mem", xmem, 0, T*CH-1);
        for (i = 0; i < TMAXV; i = i + 1) lymem[i] = 0;   // default bank 0
        $readmemh("cs_layer.mem", lymem, 0, T-1);
        fd = $fopen("cs_y.out", "w");

        repeat (4) @(posedge clk);
        rst <= 0;
        repeat (L*CH + 4) @(posedge clk);   // history clear sweep (all L banks)

        for (i = 0; i < CH*K; i = i + 1) begin
            @(posedge clk); wr_w <= 1; wr_w_addr <= i; wr_w_data <= wmem[i];
        end
        @(posedge clk); wr_w <= 0;
        for (i = 0; i < CH; i = i + 1) begin
            @(posedge clk); wr_b <= 1; wr_b_addr <= i; wr_b_data <= bmem[i];
        end
        @(posedge clk); wr_b <= 0;
        for (i = 0; i < 256; i = i + 1) begin
            @(posedge clk); wr_lut <= 1; wr_lut_addr <= i[7:0]; wr_lut_data <= lmem[i];
        end
        @(posedge clk); wr_lut <= 0;

        for (t = 0; t < T; t = t + 1) begin
            layer <= lymem[t][$clog2(L)-1:0];
            for (i = 0; i < CH; i = i + 1) begin
                @(posedge clk); wr_x <= 1; wr_x_addr <= i; wr_x_data <= xmem[t*CH + i];
            end
            @(posedge clk); wr_x <= 0;
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (done);
            @(posedge clk);
            for (i = 0; i < CH; i = i + 1) begin
                rd_y_addr <= i;
                @(posedge clk);
                $fdisplay(fd, "%04x", rd_y_data & 16'hffff);
            end
        end
        $fclose(fd);
        $display("TB_CS_DONE T=%0d", T);
        $finish;
    end
endmodule
