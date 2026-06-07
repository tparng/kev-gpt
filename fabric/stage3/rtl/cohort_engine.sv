// -----------------------------------------------------------------------------
// cohort_engine — ONE split-brain cohort: a complete N=8 full-forward pipeline
// (1 nl_engine owning all G=8 streams + 1 GE FSM + dequant + gelu + 1
// gemm_cohort_vec datapath), exposing ONLY a shared weight-bank read port and
// the shared LN / attention / embed ports. Two of these run fully independently,
// sharing only weight_bank_tdp (one TDP read port each) and the shared LN/attn/
// embed units (arbitrated in sequencer_sb).
//
// Derived MECHANICALLY from sequencer_pp by collapsing its two stream-groups to
// ONE engine and DELETING the merge / GWAIT / aq_eng machinery (a cohort never
// shares a weight pass — that whole desync problem dissolves). The GE FSM here
// always serves its single engine's 8 streams: gmerge≡1, glim≡N-1, no GE_AQW.
//
// Arithmetic is BIT-IDENTICAL per stream to sequencer_pp (same dequant/AQ/GELU/
// readback algebra). iverilog-2012 traps honoured.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module cohort_engine #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 128,
    parameter integer N     = 8,        // this cohort's streams (one engine owns all)
    parameter integer ND    = 0,        // DSP-packed GEMM streams (of the N)
    parameter integer VOCAB = 193,
    parameter integer TMAX  = 32,
    parameter integer GAMMA_N = 9,
    parameter integer DQ_N  = 9409,
    parameter integer NSACT = 17,
    parameter integer WWORDS = 25600,
    parameter integer NLAYER = 4,
    parameter integer NHEAD = 4,
    parameter integer HEAD_DIM = 64,
    parameter integer RESID_FRAC  = 25,
    parameter integer LN_OUT_FRAC = 22,
    parameter integer VFRAC       = 16,
    parameter integer GELU_FRAC   = 12,
    parameter integer ISH         = 40,
    parameter integer DBG         = 1    // 0 = tie off the board-debug readback mux
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    input  wire [N*9-1:0] tok_ids,
    input  wire [8:0]  pos,
    output wire        done_o,             // level: 1 when this cohort finished all N
    output wire [N*9-1:0] tok_outs,
    // host-debug readback (LOCAL stream 0..N-1)
    input  wire [$clog2(N)-1:0] rd_stream,
    input  wire [3:0]  rd_sel,
    input  wire [10:0] rd_addr,
    output reg signed [63:0] rd_data,
    // ---- shared weight bank read port (to weight_bank_tdp) ----
    output wire [$clog2(WWORDS)-1:0] waddr,
    input  wire [LANES*4-1:0]        wword_rd,
    // ---- shared embed read port (arbitrated in sequencer_sb) ----
    output wire        emb_req,
    output wire [$clog2(VOCAB*(D/P))-1:0] emb_addr,
    input  wire        emb_gnt,
    input  wire [P*32-1:0] emb_q,
    // pos_emb runtime write (broadcast from sequencer_sb embed loader)
    input  wire        pw_we,
    input  wire [$clog2(TMAX*(D/P))-1:0] pw_addr,
    input  wire [P*32-1:0] pw_data,
    // ---- shared LayerNorm port (arbitrated in sequencer_sb) ----
    output wire        ln_req,
    output wire        ln_start_o,
    output wire        ln_vin_o,
    output wire [P*32-1:0] ln_x_o,
    output wire [P*32-1:0] ln_g_o,
    input  wire        ln_gnt,
    input  wire        ln_yv_i,
    input  wire [P*64-1:0] ln_y_i,
    input  wire        ln_done_i,
    // ---- shared attention port (arbitrated in sequencer_sb) ----
    output wire        at_req,
    output wire        at_start_o,
    output wire [8:0]  at_tcount_o,
    output wire        at_ldv_o,
    output wire [P*32-1:0] at_ld_data_o,
    input  wire        at_gnt,
    input  wire        at_ldready_i,
    input  wire        at_ctxv_i,
    input  wire [6:0]  at_ctxidx_i,
    input  wire [P*32-1:0] at_ctxdata_i,
    input  wire        at_done_i,
    // ---- shared dequant+gelu readback channel (arbitrated in sequencer_sb) ----
    // req/gnt span one whole GEMM call's readback (GE_DQW..GE_RB..GE_IDLE);
    // the ROM fetch + dq/gelu beats keep EXACTLY the per-cohort pipeline timing.
    output wire        dq_req,
    input  wire        dq_gnt,
    output wire [11:0] dq_raddr_o,         // dqm/dqe ROM row = a_dqrow + rb1
    output reg         dq_vin_o,
    output reg signed [6:0] dq_frac_o,
    output reg  [P*32-1:0] dq_gemvy_o,
    input  wire        dq_vout_i,
    input  wire [P*32-1:0] dq_out_i,
    output reg         gl_vin_o,
    output reg  [P*16-1:0] gl_x_o,
    input  wire        gl_vout_i,
    input  wire [P*16-1:0] gl_y_i
);
    localparam integer ROWS  = D    / P;
    localparam integer ROWS3 = D3   / P;
    localparam integer ROWSM = D_MLP/ P;
    localparam integer LSH   = $clog2(P);
    localparam integer DQROWS = (DQ_N + P - 1) / P;
    localparam integer ARROWS  = (VOCAB + P - 1)/P;

    // ---- per-cohort inv_sact ROM (tiny). The dequant mant/exp ROMs + the
    // vec_dequant/vec_gelu engines are SHARED (hoisted to sequencer_sb behind
    // the dq-channel arbiter) — replicating them cost ~12k LUT at N=16.
    reg signed [63:0] inv_sact [0:NSACT-1];
    initial begin
        $readmemh("inv_sact.mem", inv_sact);
    end

    // ================================================================
    //              NL  ENGINE  (one, owns all N streams)
    // ================================================================
    wire [19:0] d_wbase; wire [10:0] d_m, d_k; wire [1:0] d_asrc; wire [5:0] d_asel;
    wire signed [6:0] d_frac; wire [11:0] d_dqrow; wire [2:0] d_dst; wire g_req;
    wire [P*32-1:0] lnout1_r, lnout2_r, ctxv_g; wire [P*16-1:0] mlpbuf_r;
    wire [P*32-1:0] xres_r, qkv_r, attn_r, mlp_r, head_r;

    wire        g_done_p;            // pulse: current GEMM call fully drained
    reg         dwr_we;  reg [2:0] dwr_dst; reg [10:0] dwr_addr; reg [P*32-1:0] dwr_data;
    reg         dwmr_we; reg [10:0] dwmr_addr; reg [P*16-1:0] dwmr_data;
    reg  [10:0] ge_ra_local;

    nl_engine #(.D(D), .D3(D3), .D_MLP(D_MLP), .P(P), .LANES(LANES), .G(N),
                .VOCAB(VOCAB), .TMAX(TMAX), .GAMMA_N(GAMMA_N), .NLAYER(NLAYER),
                .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .RESID_FRAC(RESID_FRAC),
                .LN_OUT_FRAC(LN_OUT_FRAC), .VFRAC(VFRAC), .GELU_FRAC(GELU_FRAC),
                .ISH(ISH)) eng (
        .clk(clk), .rst(rst), .go(go),
        .tok_ids(tok_ids), .pos(pos),
        .served(g_done_p),
        .s_done(g_sdone), .s_done_idx(g_sdone_idx),
        .emb_gnt(emb_gnt), .emb_q(emb_q), .emb_req(emb_req), .emb_addr(emb_addr),
        .dw_we(dwr_we), .dw_dst(dwr_dst), .dw_addr(dwr_addr), .dw_data(dwr_data),
        .dwm_we(dwmr_we), .dwm_addr(dwmr_addr), .dwm_data(dwmr_data),
        .ge_ra(ge_ra_local),
        .dbg_stream(rd_stream), .dbg_addr(rd_addr),
        .pw_we(pw_we), .pw_addr(pw_addr), .pw_data(pw_data),
        .ln_req(ln_req), .ln_start_o(ln_start_o), .ln_vin_o(ln_vin_o),
        .ln_x_o(ln_x_o), .ln_g_o(ln_g_o),
        .ln_gnt(ln_gnt), .ln_yv_i(ln_yv_i), .ln_y_i(ln_y_i), .ln_done_i(ln_done_i),
        .at_req(at_req), .at_start_o(at_start_o), .at_tcount_o(at_tcount_o),
        .at_ldv_o(at_ldv_o), .at_ld_data_o(at_ld_data_o),
        .at_gnt(at_gnt), .at_ldready_i(at_ldready_i), .at_ctxv_i(at_ctxv_i),
        .at_ctxidx_i(at_ctxidx_i), .at_ctxdata_i(at_ctxdata_i), .at_done_i(at_done_i),
        .req(g_req), .d_wbase(d_wbase), .d_m(d_m), .d_k(d_k),
        .d_asrc(d_asrc), .d_asel(d_asel), .d_frac(d_frac),
        .d_dqrow(d_dqrow), .d_dst(d_dst), .done_o(done_o),
        .tok_outs_g(tok_outs),
        .lnout1_r(lnout1_r), .lnout2_r(lnout2_r), .ctxv_g(ctxv_g),
        .mlpbuf_r(mlpbuf_r),
        .xres_r(xres_r), .qkv_r(qkv_r), .attn_r(attn_r),
        .mlp_r(mlp_r), .head_r(head_r));

    // ================================================================
    //                       GEMM  DATAPATH  (this cohort)
    // ================================================================
    reg                gv_xrst, gv_xwe, gv_start, gv_rowcommit;
    reg  [$clog2(N)-1:0] gv_xstream, gv_rdstream;
    reg  [$clog2(1024/P)-1:0] gv_xrow;
    reg  [P*8-1:0]     gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    reg [10:0]         gv_rdaddr;
    wire [P*32-1:0]    gv_yout;
    (* keep_hierarchy = "yes" *)
    gemm_cohort_vec #(.LANES(LANES), .N(N), .ND(ND), .P(P), .MMAX(1024),
                  .KMAX(1024), .RLAT(2), .WWORDS(WWORDS)) u_gemm (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .x_rst(gv_xrst), .x_we(gv_xwe), .x_stream(gv_xstream),
        .x_row(gv_xrow), .x_data(gv_xdata),
        .ovl_en(1'b1), .x_rowcommit(gv_rowcommit),
        .start(gv_start), .done(gv_done),
        .rd_stream(gv_rdstream), .rd_addr(gv_rdaddr[$clog2(1024/P)-1:0]), .y_out(gv_yout),
        .waddr(waddr), .wword_rd(wword_rd));

    // dq/gelu live in sequencer_sb (shared channel); this cohort's beats are
    // gated on its grant so the other cohort's traffic is invisible here.
    wire              dq_vout = dq_vout_i && dq_gnt;
    wire [P*32-1:0]   dq_out  = dq_out_i;
    wire              gl_vout = gl_vout_i && dq_gnt;
    wire [P*16-1:0]   gl_y    = gl_y_i;

    // active call context (no merge: one engine, all N streams)
    reg [19:0] a_wbase;
    reg [10:0] a_m, a_k;
    reg [1:0]  a_asrc;
    reg [5:0]  a_asel;
    reg signed [6:0] a_frac;
    reg [11:0] a_dqrow;
    reg [2:0]  a_dst;
    localparam [$clog2(N)-1:0] glim = N-1;

    reg [$clog2(N)-1:0] gbs;            // stream (inner loop, row-major AQ)
    reg [10:0] grow;                    // row (outer loop, row-major AQ)
    reg [10:0] gci;                     // GE_RB readback row walk (also reused as ge_ra row)
    reg        gciv, gciv1, gciv2;
    // pipeline tags for the row-major AQ: each beat's (stream,row,row-last) carried
    // 3 deep so the x-write at the gciv2 stage targets the right (stream,row) and
    // the row-commit pulse fires when the LAST stream of a row reaches the write.
    // 4-deep tag pipeline: the AQ shift/round/saturate cone (aq_prod_r -> gv_xdata)
    // was THE 5ns-critical family (OOC: -0.366 @5ns, all 128 endpoints). Split it
    // with an operand register: stage gciv2 does the 96b add + >>>62 shift into
    // aq_sh_r; stage gciv3 saturates+packs+writes. Zero-stall, +1 AQ latency.
    reg [$clog2(N)-1:0] gbsd, gbsd1, gbsd2, gbsd3;
    reg [10:0] growd, growd1, growd2, growd3;
    reg        rowl, rowl1, rowl2, rowl3; // this beat is the last stream of its row
    reg        gciv3;
    reg signed [31:0] aq_sh_r [0:P-1];    // registered post-shift value (feeds sat)
    reg        gv_started;              // gv_start already pulsed for this call
    reg [2:0]  grbn;
    reg signed [63:0] lntmp_g;
    reg signed [95:0] aq_prod, aq_sh;
    reg signed [31:0] aq_int;
    reg signed [95:0] aq_prod_r [0:P-1];
    reg                aq_neg_r [0:P-1];
    // AQ-multiply operand widths NARROWED from 64x64 to 32x48 (research/aq_range_proof.py):
    //   lnt_r : signed [31:0]  — sources 0/2 ARE 32-bit Q.22 LN/lnf regs (cbg slice);
    //           source 3 (GELU 16b sat <<<10) <= 2^25; source 1 (ctx>>>3) observed 24b.
    //           PROVEN LN container |lnt|<2^29 (|n|<sqrt(D)=16 * |gamma|<8) -> 31b, so 32b
    //           is exact-by-construction for all four sources. lntmp_g still computes in 64b
    //           (no overflow possible) and is truncated to the proven-32b register here.
    //   inv_sact: sliced [47:0] at use — PROVEN ROM max = 137979319167552 ~ 2^47 (48b signed).
    reg signed [31:0] lnt_r [0:P-1];
    reg signed [31:0] cbg, dqv;
    reg signed [15:0] mbg;
    reg [P*8-1:0]   aqw;
    reg [P*16-1:0]  mword;
    reg [P*32-1:0]  dword;
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    reg [$clog2(ROWSM+1)-1:0] gdor, gor;
    integer gp;

    localparam [2:0] GE_IDLE=0, GE_AQ=1, GE_AQN=2, GE_RUN=3, GE_WAIT=4, GE_RB=5, GE_RBN=6,
                     GE_DQW=7;   // waiting for the shared dq channel grant
    reg [2:0] ge;
    // rows of cushion committed before the overlapped run is launched. The N=P=8
    // produce/consume rates tie exactly, so the stall guard in gemm_cohort_vec is
    // what guarantees correctness; this just keeps the run a couple rows behind AQ
    // so it rarely stalls. (GE_AQN/GE_RUN states retired — AQ now starts the run.)
    localparam integer AQ_START_MARGIN = 2;

    // dq channel request: from the cycle gv_done arrives (zero-cycle grant when
    // the channel is free) until the call's last readback beat returns to IDLE.
    assign dq_req = (ge == GE_WAIT && gv_done) || (ge == GE_DQW) ||
                    (ge == GE_RB) || (ge == GE_RBN);
    assign dq_raddr_o = a_dqrow + rb1;

    // current stream addressing (LOCAL = global; one engine)
    wire [$clog2(N)-1:0] sid = gbs;
    wire [10:0] sGl  = sid * ROWS;
    wire [10:0] sG3l = sid * ROWS3;
    wire [10:0] sGMl = sid * ROWSM;
    wire [10:0] sGAl = sid * ARROWS;

    // GE-side read address (LOCAL): board readback reuses this port while idle
    wire [10:0] rbr   = rd_stream*ROWS   + (rd_addr >> LSH);
    wire [10:0] rbrM  = rd_stream*ROWSM  + (rd_addr >> LSH);
    // AQ beat reads (stream gbs, row grow); GE_RB reuses (gbs, gci) for its walk.
    wire [10:0] ge_ra_row = (ge == GE_AQ) ? grow : gci;
    always @* begin
        if (ge == GE_IDLE)
            ge_ra_local = (rd_sel == 4'd5) ? rbrM : rbr;
        else
            ge_ra_local = (a_asrc == 2'd3) ? (sGMl + ge_ra_row) : (sGl + ge_ra_row);
    end

    wire rb_last = (a_dst == 3'd2) ? (gl_vout && gor == ROWSM-1 && gbs == glim)
                                   : (dq_vout && gdor == ((a_m + P-1) >> LSH) - 1 && gbs == glim);
    assign g_done_p = (ge == GE_RB) && rb_last;

    // STREAM-GRANULAR completion: pulse when ANY stream gbs's last readback row
    // commits (dst 0/1/3/4 — the dq path that writes qkv/attn/mlp/head banks; FC's
    // dst=2 mlpbuf path has NO nl post-processing so it is intentionally excluded).
    // Same alignment as g_done_p: combinational on the last-row commit of stream gbs.
    wire g_sdone = (ge == GE_RB) && (a_dst != 3'd2) && dq_vout &&
                   (gdor == ((a_m + P-1) >> LSH) - 1);
    wire [$clog2(N)-1:0] g_sdone_idx = gbs;

    always @* begin
        dword = {(P*32){1'b0}}; mword = {(P*16){1'b0}};
        for (gp=0; gp<P; gp=gp+1) begin
            dqv = dq_out[gp*32 +: 32];
            dword[gp*32 +: 32] = dqv;
            if      ($signed(dqv) >  32'sd32767)  mword[gp*16 +: 16] = 16'sd32767;
            else if ($signed(dqv) < -32'sd32768)  mword[gp*16 +: 16] = -16'sd32768;
            else    mword[gp*16 +: 16] = dqv[15:0];
        end
    end

    always @* begin
        dwr_we = 1'b0; dwr_dst = 3'd0; dwr_addr = 11'd0; dwr_data = dword;
        dwmr_we = 1'b0; dwmr_addr = 11'd0; dwmr_data = gl_y;
        if (ge == GE_RB) begin
            if (dq_vout) begin
                case (a_dst)
                    3'd0: begin dwr_we = 1'b1; dwr_dst = 3'd0; dwr_addr = sG3l + gdor; end
                    3'd1: begin dwr_we = 1'b1; dwr_dst = 3'd1; dwr_addr = sGl  + gdor; end
                    3'd2: ;
                    3'd3: begin dwr_we = 1'b1; dwr_dst = 3'd3; dwr_addr = sGl  + gdor; end
                    default: begin dwr_we = 1'b1; dwr_dst = 3'd4; dwr_addr = sGAl + gdor; end
                endcase
            end
            if (gl_vout) begin dwmr_we = 1'b1; dwmr_addr = sGMl + gor; end
        end
    end

    always @(posedge clk) begin
        gv_xrst<=0; gv_xwe<=0; gv_start<=0; gv_rowcommit<=0; dq_vin_o<=0; gl_vin_o<=0;
        if (rst) begin
            ge<=GE_IDLE; gbs<=0; grow<=0; gci<=0; gciv<=0; gciv1<=0; gciv2<=0; gciv3<=0;
            gbsd<=0; gbsd1<=0; gbsd2<=0; gbsd3<=0;
            growd<=0; growd1<=0; growd2<=0; growd3<=0;
            rowl<=0; rowl1<=0; rowl2<=0; rowl3<=0; gv_started<=0;
            rv0<=0; rv1<=0; rv2<=0; gdor<=0; gor<=0;
        end else begin
            case (ge)
                GE_IDLE: begin
                    if (g_req) begin
                        a_wbase <= d_wbase;
                        a_m     <= d_m;     a_k     <= d_k;
                        a_asrc  <= d_asrc;  a_asel  <= d_asel;
                        a_frac  <= d_frac;  a_dqrow <= d_dqrow;
                        a_dst   <= d_dst;
                        gbs<=0; grow<=0; gciv<=0; gciv1<=0; gciv2<=0; gciv3<=0;
                        rowl<=0; rowl1<=0; rowl2<=0; rowl3<=0; gv_started<=0;
                        gv_xrst<=1'b1; ge<=GE_AQ;
                    end
                end
                // ---- ROW-MAJOR act-quantize, OVERLAPPED with the gemm run ----
                // For each row r (0..K/P-1) walk all N streams (inner). The gemm's
                // RUN reads row r of EVERY stream at the SAME xrd_row, so a row
                // must be committed (all N streams written) before RUN's kc reaches
                // r*P. We start the run after MARGIN rows commit and rely on
                // gemm_cohort_vec's stall guard (issue gated on xrd_row <
                // rows_committed) for the zero-margin N=P=8 rate tie. Continuous
                // 4-deep pipeline (issue->mult->shift->write); no inter-stream gaps.
                GE_AQ: begin
                    // ---- stage 0: ISSUE (read the (gbs,grow) AQ source) ----------
                    // a_k captured at GE_IDLE; the descriptor is held by nl_engine
                    // through the whole call so d_k would also be stable, but a_k
                    // decouples the AQ loop from the live bus.
                    gciv  <= (grow != (a_k >> LSH));
                    gbsd  <= gbs;   growd <= grow;   rowl  <= (gbs == glim);
                    // advance the (row,stream) cursor
                    if (grow != (a_k >> LSH)) begin
                        if (gbs == glim) begin gbs<=0; grow<=grow+1'b1; end
                        else gbs<=gbs+1'b1;
                    end
                    // ---- pipeline tags (4 deep) ---------------------------------
                    gciv1 <= gciv;  gbsd1 <= gbsd;  growd1 <= growd;  rowl1 <= rowl;
                    gciv2 <= gciv1; gbsd2 <= gbsd1; growd2 <= growd1; rowl2 <= rowl1;
                    gciv3 <= gciv2; gbsd3 <= gbsd2; growd3 <= growd2; rowl3 <= rowl2;
                    if (gciv) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            case (d_asrc)
                                2'd0: begin
                                    cbg = lnout1_r[gp*32 +: 32];
                                    lntmp_g = {{32{cbg[31]}}, cbg};
                                end
                                2'd1: begin
                                    cbg = ctxv_g[gp*32 +: 32];
                                    if ($signed({{32{cbg[31]}}, cbg}) >= 0)
                                        lntmp_g = ($signed({{32{cbg[31]}}, cbg}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC);
                                    else
                                        lntmp_g = -((-$signed({{32{cbg[31]}}, cbg}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC));
                                end
                                2'd2: begin
                                    cbg = lnout2_r[gp*32 +: 32];
                                    lntmp_g = {{32{cbg[31]}}, cbg};
                                end
                                default: begin
                                    mbg = mlpbuf_r[gp*16 +: 16];
                                    lntmp_g = $signed({{48{mbg[15]}}, mbg}) <<< (LN_OUT_FRAC - GELU_FRAC);
                                end
                            endcase
                            lnt_r[gp] <= lntmp_g;
                        end
                    end
                    if (gciv1) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            // 32x48 signed multiply (was 64x64). Operand widths PROVEN in
                            // research/aq_range_proof.py: lnt_r is signed[31:0]; inv_sact is
                            // a fixed positive ROM with max ~2^47 < 2^47, so bit[47]=0 and the
                            // signed[47:0] slice is always non-negative -> bit-identical product.
                            aq_prod_r[gp] <= $signed(lnt_r[gp]) * $signed(inv_sact[a_asel][47:0]);
                            aq_neg_r[gp]  <= (lnt_r[gp] < 0);
                        end
                    end
                    // ---- stage 2: round + shift -> register (the retimed cone) ---
                    if (gciv2) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            aq_prod = aq_prod_r[gp];
                            if (!aq_neg_r[gp])
                                aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                            else
                                aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                            aq_sh_r[gp] <= aq_sh[31:0];   // only [31:0] feeds saturation
                        end
                    end
                    // ---- stage 3: saturate + pack -> WRITE x row + commit --------
                    if (gciv3) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            aq_int = aq_sh_r[gp];
                            if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                            aqw[gp*8 +: 8] = aq_int[7:0];
                        end
                        gv_xwe<=1'b1; gv_xdata<=aqw;
                        gv_xstream<=gbsd3; gv_xrow<=growd3[$clog2(1024/P)-1:0];
                        // a row is fully written when its LAST stream lands -> commit
                        if (rowl3) begin
                            gv_rowcommit<=1'b1;
                            // start the run once MARGIN rows have committed (this
                            // commit makes rows_committed = growd3+1)
                            if (!gv_started && growd3 >= AQ_START_MARGIN-1) begin
                                gv_started<=1'b1;
                                gv_m<=a_m; gv_k<=a_k; gv_wbase<=a_wbase[$clog2(WWORDS)-1:0];
                                gv_start<=1'b1;
                            end
                        end
                    end
                    // ---- AQ done: last x-write committed -> wait for the gemm ----
                    if (gciv3 && rowl3 && growd3==(a_k >> LSH)-1) begin
                        gbs<=0; grow<=0; gciv<=0; gciv1<=0; gciv2<=0; gciv3<=0;
                        // tiny calls (K/P <= MARGIN) never hit the early-start branch
                        // above; issue the run now so the call still completes.
                        if (!gv_started) begin
                            gv_started<=1'b1;
                            gv_m<=a_m; gv_k<=a_k; gv_wbase<=a_wbase[$clog2(WWORDS)-1:0];
                            gv_start<=1'b1;
                        end
                        ge<=GE_WAIT;
                    end
                end
                GE_WAIT: if (gv_done) begin
                    gci<=0; gv_rdaddr<=0; rv0<=0; rv1<=0; rv2<=0;
                    gv_rdstream <= {($clog2(N)){1'b0}};
                    gdor<=0; gor<=0;
                    // zero-cycle grant when the shared dq channel is free;
                    // otherwise park in GE_DQW (req held) until the other
                    // cohort's readback completes.
                    ge <= dq_gnt ? GE_RB : GE_DQW;
                end
                GE_DQW: if (dq_gnt) ge<=GE_RB;
                GE_RB: begin
                    if (gci < ((a_m + P-1) >> LSH)) begin
                        gv_rdaddr<=gci; rb0<=gci; gci<=gci+1'b1;
                    end else gv_rdaddr <= ((a_m + P-1) >> LSH) - 1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(gci < ((a_m + P-1) >> LSH)); rv1<=rv0; rv2<=rv1;
                    if (rv2) begin
                        dq_vin_o<=1'b1; dq_frac_o<=a_frac;
                        dq_gemvy_o <= gv_yout;
                        // mant/exp ride the shared side's free-running 2-stage
                        // ROM fetch at dq_raddr_o — same alignment as the old
                        // local mwr/ewr -> dq_mant/dq_exp registers.
                    end
                    if (dq_vout) begin
                        if (a_dst == 3'd2) begin gl_vin_o<=1'b1; gl_x_o<=mword; end
                        if (a_dst != 3'd2 && gdor==((a_m + P-1) >> LSH)-1) begin
                            gci<=0; gdor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (gbs == glim) begin gbs<=0; ge<=GE_IDLE; end
                            // advance the stream pointer + gemm read-port AT RBN ENTRY so
                            // gv_rdstream is stable through the inter-stream gap (the gemm
                            // readback's 3-stage pipeline then refills during the rv0/1/2
                            // re-arm in GE_RB). The write side (dwr) is gated on dq_vout,
                            // which stays low until RB re-arms, so the early gbs bump is
                            // hazard-free. Gap shortened 8->4 (the readback latency).
                            else begin grbn<=0; gbs<=gbs+1'b1; gv_rdstream<=gbs+1'b1; ge<=GE_RBN; end
                        end else gdor<=gdor+1'b1;
                    end
                    if (gl_vout) begin
                        if (gor==ROWSM-1) begin
                            gci<=0; gor<=0; gdor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (gbs == glim) begin gbs<=0; ge<=GE_IDLE; end
                            else begin grbn<=0; gbs<=gbs+1'b1; gv_rdstream<=gbs+1'b1; ge<=GE_RBN; end
                        end else gor<=gor+1'b1;
                    end
                end
                GE_RBN: begin
                    grbn <= grbn + 1'b1;
                    if (grbn == 3'd1) ge<=GE_RB;   // gbs/gv_rdstream already advanced at entry;
                                                   // the gemm-readback pipeline (3 deep) refills
                                                   // during the rv0/1/2 re-arm in GE_RB, so a
                                                   // 2-cycle gap is sufficient (was 8).
                end
                default: ge<=GE_IDLE;
            endcase
        end
    end

    // ---- board readback --------------------------------------------------------
    // DBG=1 (default, ALL sim gates): the full per-phase debug readback the
    // run_sb_* harnesses compare against seq_ref — this is the bit-honest path.
    // DBG=0 (bitstream builds only, set in build_bd_seq_sb.tcl): the mux tree is
    // tied off (~1.5-2k LUT/device back). The record protocol reads ONLY
    // tok_outs + CYCLES, never rd_data; the compute datapath is identical.
    generate if (DBG) begin : g_dbg
    reg [LSH-1:0] rd_lane;
    reg [P*64-1:0] rw64; reg [P*32-1:0] rw32; reg is64;
    integer pp;
    always @* begin
        is64 = 1'b0; rw64 = {(P*64){1'b0}}; rw32 = {(P*32){1'b0}};
        case (rd_sel)
            4'd0: begin
                for (pp = 0; pp < P; pp = pp + 1)
                    rw64[pp*64 +: 64] = {{32{lnout1_r[pp*32 + 31]}}, lnout1_r[pp*32 +: 32]};
                is64 = 1'b1;
            end
            4'd1: rw32 = qkv_r;
            4'd2: rw32 = ctxv_g;
            4'd3: rw32 = attn_r;
            4'd4: begin
                for (pp = 0; pp < P; pp = pp + 1)
                    rw64[pp*64 +: 64] = {{32{lnout2_r[pp*32 + 31]}}, lnout2_r[pp*32 +: 32]};
                is64 = 1'b1;
            end
            4'd5: begin
                for (pp = 0; pp < P; pp = pp + 1)
                    rw64[pp*64 +: 64] = $signed({{48{mlpbuf_r[pp*16 + 15]}}, mlpbuf_r[pp*16 +: 16]})
                                        <<< (LN_OUT_FRAC - GELU_FRAC);
                is64 = 1'b1;
            end
            4'd6: rw32 = mlp_r;
            4'd7: rw32 = xres_r;
            default: rw32 = head_r;
        endcase
    end
    always @(posedge clk) begin
        rd_lane <= rd_addr[LSH-1:0];
        if (is64) rd_data <= $signed(rw64[rd_lane*64 +: 64]);
        else      rd_data <= {{32{rw32[rd_lane*32 + 31]}}, rw32[rd_lane*32 +: 32]};
    end
    end else begin : g_nodbg
        always @(posedge clk) rd_data <= 64'd0;
    end endgenerate
endmodule
