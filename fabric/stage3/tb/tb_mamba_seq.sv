// tb_mamba_seq — drives the full mamba engine: loads every table from .mem
// files (ms_t<sel>.mem, sizes from ms_cfg.mem), then for each token in
// ms_tok.mem runs a step and dumps logits + final residual to ms_out.
// ms_cfg.mem: word0=T, word1..15 = element count for wsel 1..15.
`timescale 1ns/1ps

module tb_mamba_seq;
    localparam int D = 256, DIN = 512, V = 1024, INROWS = 1160;

    reg clk = 0, rst = 1, start = 0;
    wire ready, done;
    reg  [9:0] tok_in;
    wire [9:0] tok_out;
    reg        wr_en = 0;
    reg  [3:0] wr_sel;
    reg  [18:0] wr_addr;
    reg  [31:0] wr_data;
    reg  [3:0] dbg_sel = 0;
    reg  [10:0] dbg_addr = 0;
    wire signed [31:0] dbg_data;

    mamba_seq dut (.*);
    always #2 clk = ~clk;

    // one big stimulus memory per selector, loaded lazily
    reg [31:0] tbl [0:524287];
    reg [31:0] cfg [0:15];
    reg [15:0] toks [0:255];
    integer T, t, i, s, n, fd, fdx;

    task load_sel(input integer sel, input integer count, input [1023:0] fname);
        begin
            $readmemh(fname, tbl, 0, count-1);
            for (i = 0; i < count; i = i + 1) begin
                @(posedge clk);
                wr_en <= 1; wr_sel <= sel[3:0]; wr_addr <= i[18:0];
                wr_data <= tbl[i];
            end
            @(posedge clk); wr_en <= 0;
        end
    endtask

    initial begin
        $readmemh("ms_cfg.mem", cfg);
        T = cfg[0];
        $readmemh("ms_tok.mem", toks, 0, T-1);
        fd  = $fopen("ms_logit.out", "w");
        fdx = $fopen("ms_x.out", "w");

        repeat (4) @(posedge clk);
        rst <= 0;
        wait (ready);

        load_sel(0,  cfg[15], "ms_t0.mem");   // gemv weights (count in cfg[15])
        load_sel(1,  cfg[1],  "ms_t1.mem");
        load_sel(2,  cfg[2],  "ms_t2.mem");
        load_sel(3,  cfg[3],  "ms_t3.mem");
        load_sel(4,  cfg[4],  "ms_t4.mem");
        load_sel(5,  cfg[5],  "ms_t5.mem");
        load_sel(6,  cfg[6],  "ms_t6.mem");
        load_sel(7,  cfg[7],  "ms_t7.mem");
        load_sel(8,  cfg[8],  "ms_t8.mem");
        load_sel(9,  cfg[9],  "ms_t9.mem");
        load_sel(10, cfg[10], "ms_t10.mem");
        load_sel(11, cfg[11], "ms_t11.mem");
        load_sel(12, cfg[12], "ms_t12.mem");
        load_sel(13, cfg[13], "ms_t13.mem");
        load_sel(14, cfg[14], "ms_t14.mem");

        for (t = 0; t < T; t = t + 1) begin
            @(posedge clk); tok_in <= toks[t][9:0];
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (done);
            @(posedge clk);
            $display("TOK %0d -> %0d", toks[t], tok_out);
            dbg_sel <= 6;                       // logits
            for (i = 0; i < V; i = i + 1) begin
                dbg_addr <= i[10:0];
                @(posedge clk); @(posedge clk);
                $fdisplay(fd, "%04x", dbg_data[15:0]);
            end
            dbg_sel <= 0;                       // final residual
            for (i = 0; i < D; i = i + 1) begin
                dbg_addr <= i[10:0];
                @(posedge clk); @(posedge clk);
                $fdisplay(fdx, "%04x", dbg_data[15:0]);
            end
        end
        $fclose(fd); $fclose(fdx);
        $display("TB_MS_DONE T=%0d", T);
        $finish;
    end
endmodule
