// tb_ssm_scan.sv — drives ssm_scan through T token steps from .mem files,
// dumps y per step to ssm_y.out (one hex INT16 per line, T*P lines).
// Files (written by run_ssm_scan.py in the sim dir):
//   ssm_a.mem   T      x 16-bit hex   (a_q per step)
//   ssm_dtx.mem T*P    x 16-bit hex
//   ssm_b.mem   T*N    x  8-bit hex
//   ssm_c.mem   T*N    x  8-bit hex
//   ssm_cfg.mem 1      x 32-bit hex   (T)

`timescale 1ns/1ps

module tb_ssm_scan;
    localparam int P = 64, N = 64;
    localparam int TMAXV = 4096;

    reg clk = 0, rst = 1, start = 0;
    wire done, ready;
    reg  [15:0] a_q;
    reg         wr_dtx = 0, wr_b = 0, wr_c = 0;
    reg  [5:0]  wr_dtx_addr, wr_b_addr, wr_c_addr;
    reg  signed [15:0] wr_dtx_data;
    reg  signed [7:0]  wr_b_data, wr_c_data;
    reg  [5:0]  rd_y_addr;
    wire signed [15:0] rd_y_data;

    ssm_scan #(.P(P), .N(N)) dut (.*);

    always #2 clk = ~clk;   // 250 MHz nominal, irrelevant in sim

    reg [15:0] a_mem   [0:TMAXV-1];
    reg [15:0] dtx_mem [0:TMAXV*P-1];
    reg [7:0]  b_mem   [0:TMAXV*N-1];
    reg [7:0]  c_mem   [0:TMAXV*N-1];
    reg [31:0] cfg     [0:0];

    integer T, t, i, fd;

    initial begin
        $readmemh("ssm_cfg.mem", cfg);  T = cfg[0];
        $readmemh("ssm_a.mem",   a_mem,   0, T-1);
        $readmemh("ssm_dtx.mem", dtx_mem, 0, T*P-1);
        $readmemh("ssm_b.mem",   b_mem,   0, T*N-1);
        $readmemh("ssm_c.mem",   c_mem,   0, T*N-1);
        fd = $fopen("ssm_y.out", "w");

        repeat (4) @(posedge clk);
        rst <= 0;
        wait (ready);            // clear-FSM sweeps the state BRAM first
        @(posedge clk);

        for (t = 0; t < T; t = t + 1) begin
            // load this step's vectors
            for (i = 0; i < P; i = i + 1) begin
                @(posedge clk);
                wr_dtx <= 1; wr_dtx_addr <= i[5:0];
                wr_dtx_data <= dtx_mem[t*P + i];
            end
            @(posedge clk); wr_dtx <= 0;
            for (i = 0; i < N; i = i + 1) begin
                @(posedge clk);
                wr_b <= 1; wr_b_addr <= i[5:0]; wr_b_data <= b_mem[t*N + i];
                wr_c <= 1; wr_c_addr <= i[5:0]; wr_c_data <= c_mem[t*N + i];
            end
            @(posedge clk); wr_b <= 0; wr_c <= 0;
            a_q <= a_mem[t];
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (done);
            @(posedge clk);
            for (i = 0; i < P; i = i + 1) begin
                rd_y_addr <= i[5:0];
                @(posedge clk);
                $fdisplay(fd, "%04x", rd_y_data & 16'hffff);
            end
        end
        $fclose(fd);
        $display("TB_SSM_DONE T=%0d", T);
        $finish;
    end
endmodule
