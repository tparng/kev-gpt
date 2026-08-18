// ssm_scan_row.sv — wide Mamba-2 scan core: one full state row (N lanes) per
// cycle (doc 9 §4/§5, the throughput version of the silicon-proven ssm_scan).
//
// Same arithmetic as ssm_scan v5 / mamba2_fixed_scan.scan_step_fixed2:
//   h[p][n] <- rnd(( a*h + (dtx[p]*B[n]) << sh_i ) >>> 16)   [INT16, sat]
//   y[p]     = sat(( sum_n h'[p][n]*C[n] ) >>[rnd] sh_y)     [INT16]
//
// Layout: the doc-6 wide-word rule — state is ONE N*16-bit word per channel
// (reg [N*16-1:0] h [0:P-1]); the p index is a memory address, never a mux.
// Per token per head: P + pipe-depth cycles (64+8 at P=N=64) vs the
// sequential core's P*N (4096).
//
// Pipeline: S0 read row p | S1 lane products (a*h, dtx*B) | S2 fused update
// (single rounding) + row writeback | S3.. y adder tree (6 levels, 3 stages)
// | S_y shift+sat+write. Row sequence strictly increasing => the 2-cycle
// read-to-write lag never collides.

`default_nettype none

module ssm_scan_row #(
    parameter int P  = 64,
    parameter int N  = 64,
    parameter int QA = 16
) (
    input  wire                 clk,
    input  wire                 rst,
    output reg                  ready,
    input  wire                 start,
    output reg                  done,

    input  wire [15:0]          a_q,
    input  wire signed [5:0]    sh_i,
    input  wire signed [5:0]    sh_y,

    input  wire                 wr_dtx,
    input  wire [$clog2(P)-1:0] wr_dtx_addr,
    input  wire signed [15:0]   wr_dtx_data,
    input  wire                 wr_b,
    input  wire [$clog2(N)-1:0] wr_b_addr,
    input  wire signed [7:0]    wr_b_data,
    input  wire                 wr_c,
    input  wire [$clog2(N)-1:0] wr_c_addr,
    input  wire signed [7:0]    wr_c_data,

    input  wire [$clog2(P)-1:0] rd_y_addr,
    output wire signed [15:0]   rd_y_data
);
    localparam int PW = $clog2(P);

    // state: one wide word per channel (lane n in bits [n*16 +: 16])
    (* ram_style = "block" *)
    reg [N*16-1:0] h [0:P-1];
    reg [PW-1:0]   raddr;
    reg [N*16-1:0] hq;
    reg            we;
    reg [PW-1:0]   waddr;
    reg [N*16-1:0] wdata;

    always @(posedge clk) begin
        hq <= h[raddr];
        if (we) h[waddr] <= wdata;
    end

    reg signed [15:0] dtx [0:P-1];
    reg signed [7:0]  bvec[0:N-1];
    reg signed [7:0]  cvec[0:N-1];
    reg signed [15:0] yout[0:P-1];
    assign rd_y_data = yout[rd_y_addr];

    always @(posedge clk) begin
        if (wr_dtx) dtx [wr_dtx_addr] <= wr_dtx_data;
        if (wr_b)   bvec[wr_b_addr]   <= wr_b_data;
        if (wr_c)   cvec[wr_c_addr]   <= wr_c_data;
    end

    // ---- control ----
    localparam [1:0] CLEARING = 2'd0, IDLE = 2'd1, RUN = 2'd2, DRAIN = 2'd3;
    reg [1:0]    st;
    reg [PW-1:0] pi, clr_addr;
    wire issue = (st == RUN);

    // ---- pipeline ----
    reg              v1, v2;
    reg [PW-1:0]     p1, p2;
    reg signed [15:0] dtx1;
    reg signed [23:0] injl [0:N-1];       // dtx*B per lane (S1)
    reg signed [32:0] mull [0:N-1];       // a*h per lane (S1)

    wire signed [16:0] a_s = {1'b0, a_q};

    // S2 combinational per lane: fused single rounding + sat
    wire [N*16-1:0] hnew_w;
    wire signed [23:0] yprod [0:N-1];
    genvar g;
    generate for (g = 0; g < N; g = g + 1) begin : lane
        wire signed [41:0] inj_x = injl[g];
        wire signed [41:0] inj_sh = (sh_i >= 0) ? (inj_x <<< sh_i)
                     : ((inj_x + ($signed(42'sd1) <<< (-sh_i - 1))) >>> -sh_i);
        wire signed [42:0] acc  = mull[g] + inj_sh;
        wire signed [42:0] accr = acc + $signed(43'sd1 <<< (QA - 1));
        wire signed [26:0] shft = accr >>> QA;
        wire signed [15:0] hn = (shft > $signed(27'sd32767))  ? 16'sd32767 :
                                (shft < -$signed(27'sd32768)) ? -16'sd32768 :
                                shft[15:0];
        assign hnew_w[g*16 +: 16] = hn;
        assign yprod[g] = hn * cvec[g];
    end endgenerate

    // y adder tree: 64 -> 16 -> 4 -> 1, registered every level-pair
    reg               v3, v4, v5, v6;
    reg [PW-1:0]      p3, p4, p5, p6;
    reg signed [27:0] t16 [0:15];
    reg signed [29:0] t4  [0:3];
    reg signed [31:0] ysum5;
    reg signed [15:0] yfin6;

    wire signed [31:0] ysx   = ysum5;
    wire signed [31:0] yrnd  = (sh_y > 0)
                             ? ysx + ($signed(32'sd1) <<< (sh_y - 1)) : ysx;
    wire signed [31:0] yshft = (sh_y >= 0) ? (yrnd >>> sh_y)
                                           : (ysx <<< -sh_y);
    wire signed [15:0] yfin  = (yshft > $signed(32'sd32767))  ? 16'sd32767 :
                               (yshft < -$signed(32'sd32768)) ? -16'sd32768 :
                               yshft[15:0];

    integer k;
    always @(posedge clk) begin
        done <= 1'b0;
        we   <= 1'b0;
        if (rst) begin
            st <= CLEARING; ready <= 1'b0; clr_addr <= '0;
            v1 <= 0; v2 <= 0; v3 <= 0; v4 <= 0; v5 <= 0; v6 <= 0;
        end else begin
            case (st)
              CLEARING: begin
                we <= 1'b1; waddr <= clr_addr; wdata <= '0;
                if (clr_addr == {PW{1'b1}}) begin st <= IDLE; ready <= 1'b1; end
                clr_addr <= clr_addr + 1'b1;
              end
              IDLE: if (start) begin st <= RUN; pi <= '0; raddr <= '0; end
              RUN: begin
                raddr <= pi + 1'b1;
                if (pi == P-1) st <= DRAIN; else pi <= pi + 1'b1;
              end
              DRAIN: if (!v1 && !v2 && !v3 && !v4 && !v5 && !v6) begin
                st <= IDLE; done <= 1'b1;
              end
            endcase

            // S0 -> S1: row read lands in hq; register per-lane products
            v1 <= issue; p1 <= pi; dtx1 <= dtx[pi];
            // S1 -> S2
            v2 <= v1; p2 <= p1;
            for (k = 0; k < N; k = k + 1) begin
                mull[k] <= a_s * $signed(hq[k*16 +: 16]);
                injl[k] <= dtx1 * bvec[k];
            end
            // S2: writeback + first tree level pair (64 products -> 16)
            v3 <= v2; p3 <= p2;
            if (v2) begin
                we <= 1'b1; waddr <= p2; wdata <= hnew_w;
                for (k = 0; k < 16; k = k + 1)
                    t16[k] <= yprod[4*k] + yprod[4*k+1]
                            + yprod[4*k+2] + yprod[4*k+3];
            end
            // S3: 16 -> 4
            v4 <= v3; p4 <= p3;
            if (v3)
                for (k = 0; k < 4; k = k + 1)
                    t4[k] <= t16[4*k] + t16[4*k+1] + t16[4*k+2] + t16[4*k+3];
            // S4: 4 -> 1
            v5 <= v4; p5 <= p4;
            if (v4) ysum5 <= t4[0] + t4[1] + t4[2] + t4[3];
            // S5: shift + sat; S6 write
            v6 <= v5; p6 <= p5;
            if (v5) yfin6 <= yfin;
            if (v6) yout[p6] <= yfin6;
        end
    end

endmodule

`default_nettype wire
