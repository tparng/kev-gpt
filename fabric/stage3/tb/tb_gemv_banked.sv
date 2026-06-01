// Testbench for gemv_banked. Loads w.mem/x.mem (transposed layout from
// pack_banked), runs the GEMV, dumps y.out for a Python bit-exact compare.
// Dims arrive as +define macros from the run driver (iverilog -D...).
`timescale 1ns / 1ps

`ifndef LANES
 `define LANES 16
`endif
`ifndef MMAX
 `define MMAX 1024
`endif
`ifndef KMAX
 `define KMAX 1024
`endif
`ifndef MVAL
 `define MVAL 64
`endif
`ifndef KVAL
 `define KVAL 128
`endif

module tb;
    localparam integer LANES  = `LANES;
    localparam integer MMAX   = `MMAX;
    localparam integer KMAX   = `KMAX;
    localparam integer MVAL   = `MVAL;
    localparam integer KVAL   = `KVAL;
    localparam integer GROUPS = (MVAL + LANES - 1) / LANES;
    localparam integer NW     = GROUPS * KVAL;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                   rst, ld_rst, w_we, x_we, start;
    reg [LANES*4-1:0]     w_data;
    reg signed [7:0]      x_data;
    reg [15:0]            m_count, k_count;
    reg [$clog2(MMAX)-1:0] rd_addr;
    wire signed [31:0]    y_out;
    wire                  done;

    reg [LANES*4-1:0]     wload [0:NW-1];
    reg [7:0]             xload [0:KVAL-1];
    integer i, f;

    gemv_banked #(.LANES(LANES), .MMAX(MMAX), .KMAX(KMAX), .RLAT(2)) dut (
        .clk(clk), .rst(rst),
        .m_count(m_count[$clog2(MMAX+1)-1:0]), .k_count(k_count[$clog2(KMAX+1)-1:0]),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data), .x_we(x_we), .x_data(x_data),
        .start(start), .done(done), .rd_addr(rd_addr), .y_out(y_out)
    );

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("x.mem", xload);
        rst = 1; ld_rst = 1; w_we = 0; x_we = 0; start = 0;
        w_data = 0; x_data = 0; rd_addr = 0;
        m_count = MVAL; k_count = KVAL;
        @(posedge clk); #1;
        rst = 0; ld_rst = 1; @(posedge clk); #1; ld_rst = 0;

        for (i = 0; i < NW; i = i + 1) begin
            w_we = 1; w_data = wload[i]; @(posedge clk); #1;
        end
        w_we = 0;
        for (i = 0; i < KVAL; i = i + 1) begin
            x_we = 1; x_data = xload[i]; @(posedge clk); #1;
        end
        x_we = 0;

        start = 1; @(posedge clk); #1; start = 0;
        wait (done == 1'b1);
        @(posedge clk); #1;

        f = $fopen("y.out", "w");
        for (i = 0; i < MVAL; i = i + 1) begin
            rd_addr = i;
            @(posedge clk); @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", y_out);
        end
        $fclose(f);
        $display("TB_DONE words=%0d groups=%0d", NW, GROUPS);
        $finish;
    end

    initial begin
        #2000000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
