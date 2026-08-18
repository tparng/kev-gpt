// ssm_scan.sv — Mamba-2 recurrent scan core, one head-slice (doc 9 §4).
//
// Bit-true implementation of model/mamba2_fixed_scan.py:
//   h[p][n] <- rnd(( a*h + (dtx[p]*B[n]) << (QA-QACT) ) >>> QA)   [Q3.13, sat16]
//   y[p]     = rnd(( sum_n h[p][n]*C[n] ) >>> QACT)               [Q3.13, sat16]
//
// v2, after the first OOC pass (v1 lessons, both silicon-fatal):
//  * v1 cleared all P*N state cells in the reset branch and read h[addr]
//    asynchronously — Vivado inferred 65,934 FFs + 8.7k F7 muxes (the doc-6
//    mux blow-up) and WNS was -9 ns at a 4 ns clock. BRAM needs a SYNCHRONOUS
//    read and no whole-array reset: state now lives in a true BRAM with a
//    registered read, and reset runs a clear-FSM that walks the write port
//    (ready deasserts for P*N cycles after rst).
//  * The single-cycle read-multiply-add-saturate cloud is now a 4-stage
//    pipeline (S0 read / S1 products / S2 update+write / S3 y-accumulate).
//    Element sequence is strictly increasing, so the 2-cycle read-to-write
//    lag can never collide on an address (k vs k+2), including across token
//    boundaries (same address only revisited a full P*N later).
//
// Still the sequential gate core: II=1, one (p,n) element per cycle,
// P*N + pipeline-depth cycles per token per head. Wide-N is a later rung
// with the same arithmetic.

`default_nettype none

module ssm_scan #(
    parameter int P      = 64,   // head channels
    parameter int N      = 64,   // state dim
    parameter int QA     = 16,   // decay fraction bits
    parameter int QACT   = 7     // activation fraction bits
) (
    input  wire                 clk,
    input  wire                 rst,        // sync: triggers state-clear sweep
    output reg                  ready,      // high once the clear sweep is done
    input  wire                 start,      // pulse: run one token step
    output reg                  done,

    input  wire [15:0]          a_q,        // UINT Q0.16 decay, this head/token
    input  wire                 wr_dtx,
    input  wire [$clog2(P)-1:0] wr_dtx_addr,
    input  wire signed [15:0]   wr_dtx_data,  // Q3.13
    input  wire                 wr_b,
    input  wire [$clog2(N)-1:0] wr_b_addr,
    input  wire signed [7:0]    wr_b_data,
    input  wire                 wr_c,
    input  wire [$clog2(N)-1:0] wr_c_addr,
    input  wire signed [7:0]    wr_c_data,

    input  wire [$clog2(P)-1:0] rd_y_addr,
    output wire signed [15:0]   rd_y_data
);
    localparam int AW = $clog2(P*N);

    // ---- state memory: true BRAM (sync read, no array reset) ----
    (* ram_style = "block" *)
    reg signed [15:0] h [0:P*N-1];
    reg  [AW-1:0]      raddr, waddr;
    reg  signed [15:0] hq;            // BRAM output register
    reg                we;
    reg  signed [15:0] wdata;

    always @(posedge clk) begin
        hq <= h[raddr];
        if (we) h[waddr] <= wdata;
    end

    // small operand stores (LUTRAM is fine at depth 64)
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
    reg [1:0] st;
    reg [$clog2(P)-1:0] pi;
    reg [$clog2(N)-1:0] ni;
    reg [AW-1:0]        clr_addr;

    wire issue = (st == RUN);        // one element enters the pipe per cycle

    // ---- pipeline ----
    // S0 (issue): sync-read h at {pi,ni}; register inj product + metadata
    reg                 v1;
    reg  [AW-1:0]       addr1;
    reg  signed [23:0]  inj1;        // dtx*B, frac 20
    reg  signed [7:0]   c1;
    reg                 lastn1;

    // S1: decay product a*h (hq is the BRAM output reg), align inject
    reg                 v2;
    reg  [AW-1:0]       addr2;
    reg  signed [33:0]  mul2;        // a*h, frac 29
    reg  signed [34:0]  inj2;        // (dtx*B) << (QA-QACT), frac 29
    reg  signed [7:0]   c2;
    reg                 lastn2;

    // S2: update + saturate + write back; register hnew for the y multiply
    reg                 v3;
    reg  signed [15:0]  hnew3;
    reg  signed [7:0]   c3;
    reg                 lastn3;

    wire signed [16:0] a_s = {1'b0, a_q};

    wire signed [34:0] acc  = mul2 + inj2;
    wire signed [34:0] accr = acc + $signed(35'd1 << (QA - 1));
    wire signed [18:0] shft = accr >>> QA;
    wire signed [15:0] hnew = (shft > $signed(19'd32767))  ? 16'sd32767 :
                              (shft < -$signed(19'd32768)) ? -16'sd32768 :
                              shft[15:0];

    // S3: y product (registered — the mult+accumulate+round+sat cloud was
    // the -0.4ns path at 4ns); S4: accumulate / finalize
    reg                 v4;
    reg                 lastn4;
    reg signed [23:0]   yprod4;
    reg signed [29:0]   yacc;
    reg [$clog2(P)-1:0] yp;          // which p the S4 stream is folding into
    wire signed [29:0] ysum  = yacc + yprod4;
    wire signed [29:0] yrnd  = ysum + $signed(30'd1 << (QACT - 1));
    wire signed [22:0] yshft = yrnd >>> QACT;
    wire signed [15:0] yfin  = (yshft > $signed(23'd32767))  ? 16'sd32767 :
                               (yshft < -$signed(23'd32768)) ? -16'sd32768 :
                               yshft[15:0];

    always @(posedge clk) begin
        done <= 1'b0;
        we   <= 1'b0;

        if (rst) begin
            st <= CLEARING; ready <= 1'b0; clr_addr <= '0;
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else begin
            case (st)
              CLEARING: begin
                we <= 1'b1; waddr <= clr_addr; wdata <= 16'sd0;
                if (clr_addr == {AW{1'b1}}) begin
                    st <= IDLE; ready <= 1'b1;
                end
                clr_addr <= clr_addr + 1'b1;
              end
              IDLE: if (start) begin
                st <= RUN; pi <= '0; ni <= '0;
                raddr <= '0; yacc <= 30'sd0; yp <= '0;
              end
              RUN: begin
                raddr <= {pi, ni} + 1'b1;    // read-ahead for the next element
                if (ni == N-1) begin
                    ni <= '0;
                    if (pi == P-1) st <= DRAIN; else pi <= pi + 1'b1;
                end else ni <= ni + 1'b1;
              end
              DRAIN: if (!v3 && !v4) begin st <= IDLE; done <= 1'b1; end
            endcase

            // S0 -> S1
            v1     <= issue;
            addr1  <= {pi, ni};
            inj1   <= dtx[pi] * bvec[ni];
            c1     <= cvec[ni];
            lastn1 <= (ni == N-1);

            // S1 -> S2  (hq now holds h[{pi,ni}] from the sync read)
            v2     <= v1;
            addr2  <= addr1;
            mul2   <= a_s * hq;
            inj2   <= $signed({{11{inj1[23]}}, inj1}) <<< (QA - QACT);
            c2     <= c1;
            lastn2 <= lastn1;

            // S2 -> S3: write back + hand hnew to the y stage
            v3     <= v2;
            hnew3  <= hnew;
            c3     <= c2;
            lastn3 <= lastn2;
            if (v2) begin
                we <= 1'b1; waddr <= addr2; wdata <= hnew;
            end

            // S3 -> S4: register the y product
            v4     <= v3;
            lastn4 <= lastn3;
            yprod4 <= hnew3 * c3;

            // S4: accumulate y, fold at each row end
            if (v4) begin
                if (lastn4) begin
                    yout[yp] <= yfin;
                    yacc     <= 30'sd0;
                    yp       <= yp + 1'b1;
                end else
                    yacc <= ysum;
            end
        end
    end

endmodule

`default_nettype wire
