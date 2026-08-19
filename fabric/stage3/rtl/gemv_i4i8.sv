// gemv_i4i8.sv — bring-up INT4xINT8 GEMV for the mamba engine (doc 9 §7).
//
// One packed weight word (8 x INT4) per cycle: out[r] = sum_c w[r][c]*x[c],
// INT32 accumulator, raw accumulator output (dequant happens downstream in
// the sequencer's scale unit — this core is pure integer, gated bit-exact
// against mamba2_fixed.gemv_i4i8's acc).
//
// Weight image: row-major, D_IN/8 words per row (LSB-first nibbles,
// two's-complement INT4), loaded once via wr_w. ROWS x D_IN up to
// 1160 x 256 (in_proj). Sequential: ROWS * D_IN/8 cycles per call.

`default_nettype none

module gemv_i4i8 #(
    parameter int ROWS  = 1160,             // MAX geometry (buffer sizing)
    parameter int D_IN  = 512,
    parameter int WPR   = D_IN / 8,
    parameter int WMEM  = 262144            // weight words (URAM budget)
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    output reg                      done,

    // runtime geometry: this call reads `rows` rows of `wpr` words starting
    // at weight word `base` (one banked image serves every projection)
    input  wire [$clog2(WMEM)-1:0]  base,
    input  wire [$clog2(ROWS+1)-1:0] rows,
    input  wire [$clog2(WPR+1)-1:0]  wpr,

    input  wire                     wr_w,
    input  wire [$clog2(WMEM)-1:0]  wr_w_addr,
    input  wire [31:0]              wr_w_data,

    input  wire                     wr_x,
    input  wire [$clog2(D_IN)-1:0]  wr_x_addr,
    input  wire signed [7:0]        wr_x_data,

    input  wire [$clog2(ROWS)-1:0]  rd_acc_addr,
    output wire signed [31:0]       rd_acc_data
);
    localparam int RW = $clog2(ROWS);
    localparam int WW = $clog2(WPR);

    (* ram_style = "ultra" *)
    reg [31:0] wrom [0:WMEM-1];
    reg [$clog2(WMEM)-1:0] wptr;            // running word pointer
    reg [63:0] xin  [0:WPR-1];              // x packed 8 lanes/word (INT8)
    reg signed [31:0] accs [0:ROWS-1];

    assign rd_acc_data = accs[rd_acc_addr];

    always @(posedge clk) begin
        if (wr_w) wrom[wr_w_addr] <= wr_w_data;
        if (wr_x) xin[wr_x_addr[$clog2(D_IN)-1:3]][(wr_x_addr[2:0])*8 +: 8]
                      <= wr_x_data;
    end

    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, DRAIN = 2'd2;
    reg [1:0]  st;
    reg [RW-1:0] ri;
    reg [WW-1:0] wi;
    wire issue = (st == RUN);

    // S0: read weight word + x word
    reg          v1, last1;
    reg [RW-1:0] r1;
    reg [31:0]   w1;
    reg [63:0]   x1;
    reg [$clog2(ROWS*WPR)-1:0] waddr_r;

    // S1: 8 nibble MACs (comb) -> registered partial
    reg          v2, last2;
    reg [RW-1:0] r2;
    reg signed [15:0] p2 [0:7];

    // S2: sum 8 partials into the row accumulator
    reg signed [31:0] racc;

    integer k;
    // nibble decode: two's complement INT4
    function automatic signed [4:0] nib(input [3:0] n);
        nib = {n[3], n};                     // sign-extend 4 -> 5
    endfunction

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            st <= IDLE; v1 <= 0; v2 <= 0;
        end else begin
            case (st)
              IDLE: if (start) begin
                st <= RUN; ri <= '0; wi <= '0; wptr <= base;
              end
              RUN: begin
                wptr <= wptr + 1'b1;
                if (wi == wpr-1) begin
                    wi <= '0;
                    if (ri == rows-1) st <= DRAIN; else ri <= ri + 1'b1;
                end else wi <= wi + 1'b1;
              end
              DRAIN: if (!v1 && !v2) begin st <= IDLE; done <= 1'b1; end
            endcase

            // S0 -> S1
            v1 <= issue; r1 <= ri; last1 <= (wi == wpr-1);
            w1 <= wrom[wptr];
            x1 <= xin[wi];

            // S1 -> S2: 8 nibble products
            v2 <= v1; r2 <= r1; last2 <= last1;
            for (k = 0; k < 8; k = k + 1)
                p2[k] <= $signed(nib(w1[k*4 +: 4])) * $signed(x1[k*8 +: 8]);

            // S2: accumulate into racc; on row end, store and clear
            if (v2) begin
                if (last2) begin
                    accs[r2] <= racc + p2[0] + p2[1] + p2[2] + p2[3]
                                     + p2[4] + p2[5] + p2[6] + p2[7];
                    racc <= 32'sd0;
                end else
                    racc <= racc + p2[0] + p2[1] + p2[2] + p2[3]
                                 + p2[4] + p2[5] + p2[6] + p2[7];
            end else
                racc <= 32'sd0;
        end
    end

endmodule

`default_nettype wire
