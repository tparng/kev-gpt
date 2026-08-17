// ssm_scan.sv — Mamba-2 recurrent scan core, one head-slice (doc 9 §4).
//
// Bit-true implementation of model/mamba2_fixed_scan.py:
//   h[p][n] <- rnd(( a*h + (dtx[p]*B[n]) << (QA-QACT) ) >>> QA)   [Q3.13, sat16]
//   y[p]     = rnd(( sum_n h[p][n]*C[n] ) >>> QACT)               [Q3.13, sat16]
//
// This is the sequential gate version: one (p,n) MAC per cycle, P*N cycles
// per token per head (4096 @ P=N=64). It exists to be provably bit-exact
// against the Python reference; the wide (N-lane) version is a later
// optimization with the same arithmetic. Interface mirrors the stage-3
// resident cores: load vectors, pulse start, poll done, read y.
//
// iverilog -g2012 clean; no variable part-selects on unpacked arrays
// (state is a plain memory indexed by a computed address).

`default_nettype none

module ssm_scan #(
    parameter int P      = 64,   // head channels
    parameter int N      = 64,   // state dim
    parameter int QA     = 16,   // decay fraction bits
    parameter int QACT   = 7     // activation fraction bits
) (
    input  wire                 clk,
    input  wire                 rst,        // sync: clears state
    input  wire                 start,      // pulse: run one token step
    output reg                  done,

    input  wire [15:0]          a_q,        // UINT Q0.16 decay, this head/token
    // input vectors, written before start (simple write ports)
    input  wire                 wr_dtx,
    input  wire [$clog2(P)-1:0] wr_dtx_addr,
    input  wire signed [15:0]   wr_dtx_data,  // Q3.13
    input  wire                 wr_b,
    input  wire [$clog2(N)-1:0] wr_b_addr,
    input  wire signed [7:0]    wr_b_data,
    input  wire                 wr_c,
    input  wire [$clog2(N)-1:0] wr_c_addr,
    input  wire signed [7:0]    wr_c_data,

    // y readback
    input  wire [$clog2(P)-1:0] rd_y_addr,
    output wire signed [15:0]   rd_y_data
);
    localparam int AW = $clog2(P*N);

    reg signed [15:0] h   [0:P*N-1];   // state memory (BRAM), Q3.13
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

    // ---- datapath ----
    // acc = a*h + (dtx*B << (QA-QACT)); h' = sat16(rnd(acc >>> QA))
    reg  [$clog2(P)-1:0] pi;
    reg  [$clog2(N)-1:0] ni;
    reg                  running;
    reg  signed [29:0]   yacc;        // sum of 64 x (16x8) products: 24+6 bits

    wire [AW-1:0] haddr = {pi, ni};   // p*N + n (N power of 2)

    wire signed [16:0] a_s   = {1'b0, a_q};                  // unsigned -> signed
    wire signed [33:0] decay = a_s * h[haddr];               // Q0.16*Q3.13
    wire signed [23:0] inj   = dtx[pi] * bvec[ni];           // Q3.13*Q0.7
    // sign-extend via signed-to-signed assignment: a {..} concat is ALWAYS
    // unsigned and poisons the addition to unsigned (soak caught acc
    // wrapping to 2^34 + correct_negative on the first negative state)
    wire signed [34:0] inj_x = inj;
    wire signed [34:0] acc   = decay + (inj_x <<< (QA - QACT));
    wire signed [34:0] rnd   = acc + $signed(35'd1 << (QA - 1));
    wire signed [18:0] shft  = rnd >>> QA;
    wire signed [15:0] hnew  = (shft > $signed(19'd32767))  ? 16'sd32767 :
                               (shft < -$signed(19'd32768)) ? -16'sd32768 :
                               shft[15:0];

    wire signed [23:0] yprod = hnew * cvec[ni];              // uses h' (post-update)

    // y finalize: rnd(yacc >>> QACT), sat16
    wire signed [29:0] yrnd  = yacc + yprod + $signed(30'd1 << (QACT - 1));
    wire signed [22:0] yshft = yrnd >>> QACT;
    wire signed [15:0] yfin  = (yshft > $signed(23'd32767))  ? 16'sd32767 :
                               (yshft < -$signed(23'd32768)) ? -16'sd32768 :
                               yshft[15:0];

    integer k;
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            running <= 1'b0;
            for (k = 0; k < P*N; k = k + 1) h[k] <= 16'sd0;
        end else if (start && !running) begin
            running <= 1'b1;
            pi <= '0; ni <= '0; yacc <= 30'sd0;
        end else if (running) begin
            h[haddr] <= hnew;
            if (ni == N-1) begin
                yout[pi] <= yfin;                 // last n: fold + finalize
                yacc     <= 30'sd0;
                ni       <= '0;
                if (pi == P-1) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end else pi <= pi + 1'b1;
            end else begin
                yacc <= yacc + yprod;
                ni   <= ni + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
