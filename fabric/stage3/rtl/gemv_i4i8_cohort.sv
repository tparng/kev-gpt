// gemv_i4i8_cohort.sv — WEIGHT-STATIONARY N-stream cohort GEMV for the mamba
// engine (Phase-2 Rung B: 100k tok/s campaign).
//
// The single-stream gemv_i4i8 reads PE 32-bit weight words/cycle and folds
// PE*8 INT4xINT8 nibble-MACs into ONE row accumulator.  This cohort variant
// reads each weight word ONCE (the shared `wq`) and folds it into N per-stream
// row accumulators simultaneously — so a full GEMV over N streams costs the
// SAME cycles as one stream (weight-read bound), i.e. ~25.5k cyc/token ÷ N.
// This is the gemm_cohort_vec.sv pattern ported to the mamba INT4 GEMV: shared
// weight bank, N MAC banks, per-stream activation memories and accumulators.
//
// Bit-identical arithmetic to gemv_i4i8 per stream (integer add is associative;
// the MAC/fold code is the single-stream code replicated by genvar).  Only the
// weight pipeline is shared.  Gated by run_gemv_cohort.py: each stream's INT32
// accumulators must equal the single-stream gemv (== numpy w@x[stream]).
//
// Weight image + geometry contract is UNCHANGED from gemv_i4i8 (row-major,
// D_IN/8 words/row, base/rows/wpr runtime geometry, PE|wpr, PE|base, PE|WMEM).

`default_nettype none

module gemv_i4i8_cohort #(
    parameter int N     = 8,                // streams in the cohort
    parameter int PE    = 16,               // weight WORDS read per cycle (pow2)
    parameter int ROWS  = 1160,             // MAX geometry (buffer sizing)
    parameter int D_IN  = 512,
    parameter int WPR   = D_IN / 8,
    parameter int WMEM  = 262144            // weight words (URAM budget)
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    output reg                      done,

    input  wire [$clog2(WMEM)-1:0]  base,
    input  wire [$clog2(ROWS+1)-1:0] rows,
    input  wire [$clog2(WPR+1)-1:0]  wpr,

    // shared weight write (identical assembly to gemv_i4i8)
    input  wire                     wr_w,
    input  wire [$clog2(WMEM)-1:0]  wr_w_addr,
    input  wire [31:0]              wr_w_data,

    // per-stream activation write: wr_x picks the stream via wr_x_stream
    input  wire                     wr_x,
    input  wire [$clog2(N)-1:0]     wr_x_stream,
    input  wire [$clog2(D_IN)-1:0]  wr_x_addr,
    input  wire signed [7:0]        wr_x_data,

    // per-stream readback
    input  wire [$clog2(N)-1:0]     rd_stream,
    input  wire [$clog2(ROWS)-1:0]  rd_acc_addr,
    output wire signed [31:0]       rd_acc_data
);
    localparam int LP    = $clog2(PE);
    localparam int WBITS = PE * 32;
    localparam int WWW   = WMEM / PE;
    localparam int AWM   = $clog2(WMEM);
    localparam int AWW   = $clog2(WWW);
    localparam int RW    = $clog2(ROWS);
    localparam int WSW   = $clog2(WPR/PE + 1);

    // -------- shared weight image (identical to gemv_i4i8) --------------------
    (* ram_style = "ultra" *)
    reg [WBITS-1:0] wrom [0:WWW-1];
    reg [WBITS-1:0] wbuf;
    reg [WBITS-1:0] wq;

    wire [LP-1:0]  wsub  = wr_w_addr[LP-1:0];
    wire [AWW-1:0] wwide = wr_w_addr[AWM-1:LP];
    reg  [AWW-1:0] wptr;

    always @(posedge clk) begin
        if (wr_w) begin
            wbuf[wsub*32 +: 32] <= wr_w_data;
            if (wsub == PE-1)
                wrom[wwide] <= {wr_w_data, wbuf[(PE-1)*32-1:0]};
        end
        wq <= wrom[wptr];
    end

    // -------- shared control FSM (row x wide-step iteration) ------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, DRAIN = 2'd2;
    reg [1:0]  st;
    reg [RW-1:0] ri;
    reg [WSW-1:0] wi;
    wire [WSW-1:0] wprw = wpr >> LP;
    wire issue = (st == RUN);

    // shared pipeline tags (weight read is stream-independent)
    reg           v1, last1, v2, last2;
    reg [RW-1:0]  r1, r2;
    reg [WSW-1:0] wi_s;                 // wide-step for the x fetch, staged

    function automatic signed [4:0] nib(input [3:0] n);
        nib = {n[3], n};
    endfunction

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            st <= IDLE; v1 <= 0; v2 <= 0;
        end else begin
            case (st)
              IDLE: if (start) begin
                st <= RUN; ri <= '0; wi <= '0; wptr <= base >> LP;
              end
              RUN: begin
                wptr <= wptr + 1'b1;
                if (wi == wprw-1) begin
                    wi <= '0;
                    if (ri == rows-1) st <= DRAIN; else ri <= ri + 1'b1;
                end else wi <= wi + 1'b1;
              end
              DRAIN: if (!v1 && !v2) begin st <= IDLE; done <= 1'b1; end
            endcase

            v1 <= issue; r1 <= ri; last1 <= (wi == wprw-1); wi_s <= wi;
            v2 <= v1; r2 <= r1; last2 <= last1;
        end
    end

    // -------- per-stream datapath (shared wq; own xin/pipe/acc) ---------------
    wire signed [31:0] rd_bus [0:N-1];
    assign rd_acc_data = rd_bus[rd_stream];

    genvar sgen;
    generate
    for (sgen = 0; sgen < N; sgen = sgen + 1) begin : g_stream
        // per-stream activation memory (8 INT8 lanes / 64-bit word)
        reg [63:0] xin [0:WPR-1];
        always @(posedge clk)
            if (wr_x && wr_x_stream == sgen[$clog2(N)-1:0])
                xin[wr_x_addr[$clog2(D_IN)-1:3]][(wr_x_addr[2:0])*8 +: 8]
                    <= wr_x_data;

        // build the PE-wide x word for the CURRENT wide-step `wi`
        integer xk;
        reg [PE*64-1:0] xrow;
        always @(*) begin
            xrow = {(PE*64){1'b0}};
            for (xk = 0; xk < PE; xk = xk + 1)
                xrow[xk*64 +: 64] = xin[wi*PE + xk];
        end

        // S0->S1 latch x wide word (aligned to the shared wq read)
        reg [PE*64-1:0] x1;
        // S1->S2 PE partial sums
        reg signed [15:0] ps [0:PE-1];
        // S2 row accumulator + stored accs
        reg signed [31:0] racc;
        reg signed [31:0] accs [0:ROWS-1];

        function automatic signed [31:0] sum_ps(input signed [31:0] a);
            integer j;
            begin
                sum_ps = a;
                for (j = 0; j < PE; j = j + 1) sum_ps = sum_ps + ps[j];
            end
        endfunction

        integer p, k;
        reg signed [15:0] acc8;
        always @(posedge clk) begin
            if (rst) begin
                // accs/racc left uninitialised as in gemv_i4i8 (cleared by run)
                racc <= 32'sd0;
            end else begin
                x1 <= xrow;
                for (p = 0; p < PE; p = p + 1) begin
                    acc8 = 16'sd0;
                    for (k = 0; k < 8; k = k + 1)
                        acc8 = acc8 + $signed(nib(wq[p*32 + k*4 +: 4]))
                                    * $signed(x1[p*64 + k*8 +: 8]);
                    ps[p] <= acc8;
                end
                if (v2) begin
                    if (last2) begin
                        accs[r2] <= sum_ps(racc);
                        racc <= 32'sd0;
                    end else
                        racc <= sum_ps(racc);
                end else
                    racc <= 32'sd0;
            end
        end
        assign rd_bus[sgen] = accs[rd_acc_addr];
    end
    endgenerate

endmodule

`default_nettype wire
