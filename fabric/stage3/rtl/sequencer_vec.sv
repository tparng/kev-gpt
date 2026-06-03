// -----------------------------------------------------------------------------
// sequencer_vec — the P-WIDE vector-datapath sequencer (the road to ~10k tok/s).
// Built up PHASE BY PHASE, each gated bit-exact vs seq_ref.block0_phase_signals.
//
//   M1 (done): embed -> LN1 -> lnout                       gate ln1_out_q22
//   M2 (this): + act-quant -> qkv GEMV -> P-wide dequant   gate qkv_q16  (q|k|v, Q.16)
//
// The banking scheme (element e -> bank e%P, row e/P; one banked row feeds P lanes/cycle)
// is proven in M1 and reused everywhere. The GEMV (gemv_banked_resident, LANES-wide MAC,
// loaded once via wl_*) keeps its inherent 1-col/cyc stream; M2 widens the DEQUANT after it
// (vec_dequant, P lanes) reading a banked gemvy. act-feed/readback stay 1/cyc for now (a
// later optimization widens them at the bank boundary).
//
// iverilog-2012 safe: copy unpacked-array elements to plain vectors before any indexed use;
// packed-vector +: part-selects only.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_vec #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer P     = 8,            // vector lanes (D, D3 % P == 0)
    parameter integer LANES = 16,           // GEMV MAC width (PE), independent of P
    parameter integer VOCAB = 193,
    parameter integer TMAX  = 256,
    parameter integer GAMMA_N = 9,          // (ln1|ln2)*NLAYER + ln_f rows
    parameter integer GBASE = 0,            // gamma row-base for LN1 (block0 = 0)
    parameter integer DQ_N  = 9409,         // dequant ROM length (per-channel, all layers)
    parameter integer NSACT = 17,           // inv_sact entries (4*NLAYER+1)
    parameter integer WWORDS = 32768,       // GEMV resident weight URAM capacity (words)
    parameter integer RESID_FRAC  = 25,
    parameter integer LN_OUT_FRAC = 22,
    parameter integer VFRAC       = 16,
    parameter integer ISH         = 40      // inv_sact fixed-point shift
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    input  wire [8:0]  tok_id,
    input  wire [8:0]  pos,
    output reg         done,
    // readback (shared rd_addr): ln_rd = lnout Q.22 (addr 0..D-1, M1 gate);
    //                            qkv_rd = q|k|v dequant Q.16 (addr 0..D3-1, M2 gate)
    input  wire [10:0] rd_addr,
    output reg signed [63:0] ln_rd,
    output reg signed [31:0] qkv_rd,
    output reg signed [31:0] gemvy_rd,   // raw GEMV INT32 (debug: vs seq_ref qkv_yint)
    // runtime weight load (stream the transposed INT4 image once)
    input  wire        wl_rst,
    input  wire        wl_we,
    input  wire [31:0] wl_data
);
    localparam integer ROWS  = D  / P;
    localparam integer ROWS3 = D3 / P;
    localparam integer LSH   = $clog2(P);
    localparam integer DQROWS = (DQ_N + P - 1) / P;   // banked dequant ROM depth

    // ---- small ROMs ($readmemh) ------------------------------------------------
    (* rom_style = "block" *) reg signed [31:0] tok_emb   [0:VOCAB*D-1];
    (* rom_style = "block" *) reg signed [31:0] pos_emb   [0:TMAX*D-1];
    (* rom_style = "block" *) reg signed [31:0] gamma_rom [0:GAMMA_N*D-1];
    (* rom_style = "block" *) reg signed [63:0] inv_sact  [0:NSACT-1];
    // dequant per-channel mant/exp, P-banked into ONE flat ROM each (iverilog $readmemh
    // can't target a 2D row): bank b occupies [b*DQROWS : (b+1)*DQROWS-1], and
    // dqm_flat[p*DQROWS + r] = channel r*P+p (bank b row r = vals[b + r*P]).
    (* rom_style = "block" *) reg signed [31:0] dqm_flat [0:P*DQROWS-1];
    (* rom_style = "block" *) reg signed [7:0]  dqe_flat [0:P*DQROWS-1];
    initial begin
        $readmemh("tok_emb.mem",  tok_emb);
        $readmemh("pos_emb.mem",  pos_emb);
        $readmemh("gamma.mem",    gamma_rom);
        $readmemh("inv_sact.mem", inv_sact);
    end
    genvar gb;
    generate for (gb = 0; gb < P; gb = gb + 1) begin: dqload
        initial begin
            $readmemh($sformatf("dq_mant.b%0d.mem", gb), dqm_flat, gb*DQROWS, (gb+1)*DQROWS-1);
            $readmemh($sformatf("dq_exp.b%0d.mem",  gb), dqe_flat, gb*DQROWS, (gb+1)*DQROWS-1);
        end
    end endgenerate

    // ---- P-banked scratch ------------------------------------------------------
    reg signed [31:0] xres_bank  [0:P-1][0:ROWS-1];    // residual Q6.25
    reg signed [31:0] gamma_bank [0:P-1][0:ROWS-1];    // gamma Q4.20
    reg signed [63:0] lnout_bank [0:P-1][0:ROWS-1];    // LN out Q.22
    reg signed [31:0] gemvy_bank [0:P-1][0:ROWS3-1];   // GEMV INT32 outputs (D3 wide)
    reg signed [31:0] qkv_bank   [0:P-1][0:ROWS3-1];   // q|k|v dequant Q.16

    // ---- layernorm_vec ---------------------------------------------------------
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    // ---- GEMV (resident; weights streamed once via wl_*) -----------------------
    reg                gv_ldrst, gv_xwe, gv_start;
    reg signed [7:0]   gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    reg [9:0]          gv_rdaddr;
    wire signed [31:0] gv_yout;
    gemv_banked_resident #(.LANES(LANES), .MMAX(1024), .KMAX(1024), .RLAT(2),
                  .WWORDS(WWORDS)) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ldrst | wl_rst), .w_we(wl_we), .w_data(wl_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .rd_addr(gv_rdaddr), .y_out(gv_yout));

    // ---- vec_dequant (P lanes, FRAC = VFRAC for qkv) ---------------------------
    reg               dq_vin;
    reg  [P*32-1:0]   dq_gemvy;
    reg  [P*24-1:0]   dq_mant;
    reg  [P*8-1:0]    dq_exp;
    wire              dq_vout;
    wire [P*32-1:0]   dq_out;
    vec_dequant #(.P(P), .FRAC(VFRAC)) u_dq (
        .clk(clk), .rst(rst), .in_valid(dq_vin),
        .gemvy(dq_gemvy), .mant(dq_mant), .exp(dq_exp),
        .out_valid(dq_vout), .dq_out(dq_out));

    // ---- act-quant temporaries (mirror sequencer_fast G_ACT) -------------------
    reg signed [63:0]  lntmp;
    reg signed [95:0]  aq_prod, aq_sh;
    reg signed [31:0]  aq_int;

    // ---- FSM -------------------------------------------------------------------
    localparam [3:0] S_IDLE=0, S_EMB=1, S_FEED=2, S_COLL=3,
                     S_AQ=4, S_GRUN=5, S_GWAIT=6, S_RB=7, S_DQ=8, S_FIN=9;
    reg [3:0] st;
    reg [10:0] ci;                          // element counter (to D3-1)
    reg [$clog2(ROWS3+1)-1:0] fr, orow, dr, dor;
    integer pp;
    reg signed [31:0] e_tok, e_pos, xb, gb_, gy;
    reg signed [31:0] mant_v; reg signed [7:0] exp_v;
    // GEMV readback 3-deep address shift (2-cyc gemv latency)
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;

    always @(posedge clk) begin
        ln_start <= 1'b0; ln_vin <= 1'b0; gv_ldrst <= 1'b0; gv_xwe <= 1'b0;
        gv_start <= 1'b0; dq_vin <= 1'b0; done <= 1'b0;
        if (rst) begin
            st <= S_IDLE; ci <= 0; fr <= 0; orow <= 0; dr <= 0; dor <= 0;
            rv0 <= 0; rv1 <= 0; rv2 <= 0;
        end else begin
            case (st)
                S_IDLE: if (go) begin ci <= 0; st <= S_EMB; end
                // ---- embed -> banks --------------------------------------------
                S_EMB: begin
                    e_tok = tok_emb[tok_id*D + ci];
                    e_pos = pos_emb[pos*D + ci];
                    xres_bank [ci[LSH-1:0]][ci >> LSH] <= e_tok + e_pos;
                    gamma_bank[ci[LSH-1:0]][ci >> LSH] <= gamma_rom[GBASE*D + ci];
                    if (ci == D-1) begin ci <= 0; fr <= 0; ln_start <= 1'b1; st <= S_FEED; end
                    else ci <= ci + 1'b1;
                end
                // ---- feed LN P-wide --------------------------------------------
                S_FEED: begin
                    ln_vin <= 1'b1;
                    for (pp = 0; pp < P; pp = pp + 1) begin
                        xb  = xres_bank [pp][fr];
                        gb_ = gamma_bank[pp][fr];
                        ln_x[pp*32 +: 32] <= xb;
                        ln_g[pp*32 +: 32] <= gb_;
                    end
                    if (fr == ROWS-1) begin orow <= 0; st <= S_COLL; end
                    else fr <= fr + 1'b1;
                end
                // ---- collect LN P-wide -> lnout banks --------------------------
                S_COLL: begin
                    if (ln_yv) begin
                        for (pp = 0; pp < P; pp = pp + 1)
                            lnout_bank[pp][orow] <= ln_y[pp*64 +: 64];
                        if (orow == ROWS-1) begin ci <= 0; gv_ldrst <= 1'b1; st <= S_AQ; end
                        else orow <= orow + 1'b1;
                    end
                end
                // ---- act-quant lnout (Q.22) -> INT8, stream into GEMV (1/cyc) ---
                S_AQ: begin
                    lntmp   = lnout_bank[ci[LSH-1:0]][ci >> LSH];
                    aq_prod = $signed(lntmp) * $signed(inv_sact[0]);          // qkv s_act = idx 0
                    if (lntmp >= 0)
                        aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                    else
                        aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                    aq_int = aq_sh[31:0];
                    if (aq_int > 127)  aq_int = 127;
                    if (aq_int < -128) aq_int = -128;
                    gv_xwe <= 1'b1; gv_xdata <= aq_int[7:0];
                    if (ci == D-1) begin ci <= 0; st <= S_GRUN; end
                    else ci <= ci + 1'b1;
                end
                S_GRUN: begin
                    gv_m <= D3[10:0]; gv_k <= D[10:0]; gv_wbase <= 0; gv_start <= 1'b1;
                    st <= S_GWAIT;
                end
                S_GWAIT: if (gv_done) begin
                    ci <= 0; gv_rdaddr <= 0; rv0 <= 0; rv1 <= 0; rv2 <= 0; st <= S_RB;
                end
                // ---- readback gemvy (1/cyc) -> banked gemvy (2-cyc latency) -----
                S_RB: begin
                    if (ci < D3) begin gv_rdaddr <= ci[9:0]; rb0 <= ci; ci <= ci + 1'b1; end
                    else gv_rdaddr <= D3[9:0]-1'b1;
                    rb1 <= rb0; rb2 <= rb1;
                    rv0 <= (ci < D3); rv1 <= rv0; rv2 <= rv1;
                    if (rv2) begin
                        gemvy_bank[rb2[LSH-1:0]][rb2 >> LSH] <= gv_yout;
                        if (rb2 == D3-1) begin dr <= 0; dor <= 0; st <= S_DQ; end
                    end
                end
                // ---- P-wide dequant: feed banked gemvy+mant+exp, collect qkv ----
                S_DQ: begin
                    if (dr < ROWS3) begin
                        dq_vin <= 1'b1;
                        for (pp = 0; pp < P; pp = pp + 1) begin
                            gy     = gemvy_bank[pp][dr];
                            mant_v = dqm_flat[pp*DQROWS + dr];
                            exp_v  = dqe_flat[pp*DQROWS + dr];
                            dq_gemvy[pp*32 +: 32] <= gy;
                            dq_mant [pp*24 +: 24] <= mant_v[23:0];
                            dq_exp  [pp*8  +: 8]  <= exp_v;
                        end
                        dr <= dr + 1'b1;
                    end else dq_vin <= 1'b0;
                    // vec_dequant has 1-cycle latency: collect dq_out the cycle after feed
                    if (dq_vout) begin
                        for (pp = 0; pp < P; pp = pp + 1)
                            qkv_bank[pp][dor] <= dq_out[pp*32 +: 32];
                        if (dor == ROWS3-1) st <= S_FIN;
                        else dor <= dor + 1'b1;
                    end
                end
                S_FIN: begin done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

    // ---- readback (2-cyc): lnout (Q.22) and qkv (Q.16), both indexed by rd_addr ---
    reg signed [63:0] ln_rd0;
    reg signed [31:0] qkv_rd0, gemvy_rd0;
    always @(posedge clk) begin
        ln_rd0    <= lnout_bank[rd_addr[LSH-1:0]][rd_addr >> LSH];
        ln_rd     <= ln_rd0;
        qkv_rd0   <= qkv_bank [rd_addr[LSH-1:0]][rd_addr >> LSH];
        qkv_rd    <= qkv_rd0;
        gemvy_rd0 <= gemvy_bank[rd_addr[LSH-1:0]][rd_addr >> LSH];
        gemvy_rd  <= gemvy_rd0;
    end
endmodule
