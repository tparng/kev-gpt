// tb_rmsnorm_gated — T tokens through rmsnorm_gated from .mem stimulus.
// Files: rg_cfg.mem (T), rg_mode.mem (T: 1=gated 0=ungated), rg_lut.mem (256),
//        rg_gamma.mem (D), rg_y.mem (T*D), rg_z.mem (T*D), seed.mem (DUT ROM).
// Output: rg_o.out (T*D hex INT16).
`timescale 1ns/1ps

module tb_rmsnorm_gated;
    localparam int D = 512, TMAXV = 64;

    reg clk = 0, rst = 1, start = 0, gated = 1;
    wire done;
    reg wr_y = 0, wr_z = 0, wr_g = 0, wr_lut = 0;
    reg [$clog2(D)-1:0] wr_y_addr, wr_z_addr, wr_g_addr, rd_o_addr;
    reg [7:0]           wr_lut_addr;
    reg signed [15:0] wr_y_data, wr_z_data, wr_g_data, wr_lut_data;
    wire signed [15:0] rd_o_data;

    wire short_len = 1'b0;
    rmsnorm_gated #(.D(D)) dut (
        .short_len(short_len),
        .clk(clk), .rst(rst), .start(start), .gated(gated), .done(done),
        .wr_y(wr_y), .wr_y_addr(wr_y_addr), .wr_y_data(wr_y_data),
        .wr_z(wr_z), .wr_z_addr(wr_z_addr), .wr_z_data(wr_z_data),
        .wr_g(wr_g), .wr_g_addr(wr_g_addr), .wr_g_data(wr_g_data),
        .wr_lut(wr_lut), .wr_lut_addr(wr_lut_addr), .wr_lut_data(wr_lut_data),
        .rd_o_addr(rd_o_addr), .rd_o_data(rd_o_data)
    );
    always #2 clk = ~clk;

    reg [15:0] lmem [0:255];
    reg [15:0] gmem [0:D-1];
    reg [15:0] ymem [0:TMAXV*D-1];
    reg [15:0] zmem [0:TMAXV*D-1];
    reg [31:0] cfg  [0:0];
    reg [3:0]  mmem [0:TMAXV-1];
    integer T, t, i, fd;

    initial begin
        $readmemh("rg_cfg.mem", cfg);  T = cfg[0];
        $readmemh("rg_mode.mem", mmem, 0, T-1);
        $readmemh("rg_lut.mem", lmem);
        $readmemh("rg_gamma.mem", gmem);
        $readmemh("rg_y.mem", ymem, 0, T*D-1);
        $readmemh("rg_z.mem", zmem, 0, T*D-1);
        fd = $fopen("rg_o.out", "w");

        repeat (4) @(posedge clk);
        rst <= 0;
        repeat (4) @(posedge clk);

        for (i = 0; i < 256; i = i + 1) begin
            @(posedge clk); wr_lut <= 1; wr_lut_addr <= i[7:0]; wr_lut_data <= lmem[i];
        end
        @(posedge clk); wr_lut <= 0;
        for (i = 0; i < D; i = i + 1) begin
            @(posedge clk); wr_g <= 1; wr_g_addr <= i; wr_g_data <= gmem[i];
        end
        @(posedge clk); wr_g <= 0;

        for (t = 0; t < T; t = t + 1) begin
            for (i = 0; i < D; i = i + 1) begin
                @(posedge clk);
                wr_y <= 1; wr_y_addr <= i; wr_y_data <= ymem[t*D + i];
                wr_z <= 1; wr_z_addr <= i; wr_z_data <= zmem[t*D + i];
            end
            @(posedge clk); wr_y <= 0; wr_z <= 0;
            gated <= mmem[t][0];
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (done);
            @(posedge clk);
            for (i = 0; i < D; i = i + 1) begin
                rd_o_addr <= i;
                @(posedge clk);
                $fdisplay(fd, "%04x", rd_o_data & 16'hffff);
            end
        end
        $fclose(fd);
        $display("TB_RG_DONE T=%0d", T);
        $finish;
    end
endmodule
