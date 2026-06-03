// -----------------------------------------------------------------------------
// sequencer_vec — the P-WIDE vector-datapath sequencer (the road to ~10k tok/s).
// MILESTONE 4: the FULL block-0 forward, P-wide, each phase bit-exact vs
// seq_ref.block0_phase_signals:
//
//   embed -> LN1 -> qkv GEMV -> attention(x4 heads) -> proj GEMV -> +res1
//         -> LN2 -> mlp_fc GEMV -> GELU -> mlp_proj GEMV -> +res2 -> x_out
//
// The GEMV and LN paths are CALLABLE micro-sequences (parameter registers + a return
// state), reused for all 4 GEMVs / 2 LNs. Each gateable intermediate has its own banked
// buffer so every phase key is readable at the end. The banking scheme (element e ->
// bank e%P, row e/P) is the same one proven in M1.
//
// iverilog-2012 safe: copy unpacked-array elements to plain vectors before any indexed
// use; packed-vector +: part-selects only; flat ROMs for $readmemh (no 2D-row init).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_vec #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 16,
    parameter integer VOCAB = 193,
    parameter integer TMAX  = 256,
    parameter integer GAMMA_N = 9,
    parameter integer DQ_N  = 9409,
    parameter integer NSACT = 17,
    parameter integer WWORDS = 65536,
    parameter integer NHEAD = 4,
    parameter integer HEAD_DIM = 64,
    parameter integer RESID_FRAC  = 25,
    parameter integer LN_OUT_FRAC = 22,
    parameter integer VFRAC       = 16,
    parameter integer GELU_FRAC   = 12,
    parameter integer ISH         = 40
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    input  wire [8:0]  tok_id,
    input  wire [8:0]  pos,
    output reg         done,
    // readback: rd_sel picks the bank, rd_addr the element (2-cyc registered). 64-bit so
    // the Q.22 LN/gelu values fit; 32-bit values are sign-extended in their bank.
    input  wire [3:0]  rd_sel,
    input  wire [10:0] rd_addr,
    output reg signed [63:0] rd_data,
    // runtime weight load (stream the whole block-0 transposed image once)
    input  wire        wl_rst,
    input  wire        wl_we,
    input  wire [31:0] wl_data
);
    localparam integer ROWS  = D    / P;
    localparam integer ROWS3 = D3   / P;
    localparam integer ROWSM = D_MLP/ P;
    localparam integer LSH   = $clog2(P);
    localparam integer DQROWS = (DQ_N + P - 1) / P;
    // weight bases (LANES=16) and dequant channel-row bases (/P)
    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;   // 12288
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;   // 4096
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;   // 16384
    localparam integer WB_QKV = 0, WB_PROJ = GW_QKV,
                       WB_FC = GW_QKV+GW_PROJ, WB_MP = GW_QKV+GW_PROJ+GW_FC;
    localparam integer DR_QKV = 0, DR_PROJ = D3/P, DR_FC = (D3+D)/P, DR_MP = (D3+D+D_MLP)/P;

    // ---- small ROMs ($readmemh) ------------------------------------------------
    (* rom_style = "block" *) reg signed [31:0] tok_emb   [0:VOCAB*D-1];
    (* rom_style = "block" *) reg signed [31:0] pos_emb   [0:TMAX*D-1];
    (* rom_style = "block" *) reg signed [31:0] gamma_rom [0:GAMMA_N*D-1];
    (* rom_style = "block" *) reg signed [63:0] inv_sact  [0:NSACT-1];
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

    // ---- P-banked scratch (one bank per gateable intermediate) -----------------
    reg signed [31:0] xres_bank  [0:P-1][0:ROWS-1];    // residual Q6.25
    reg signed [31:0] gamma_bank [0:P-1][0:ROWS-1];    // current LN gamma Q4.20
    reg signed [63:0] lnout1_bank[0:P-1][0:ROWS-1];    // LN1 out Q.22
    reg signed [63:0] lnout2_bank[0:P-1][0:ROWS-1];    // LN2 out Q.22
    reg signed [31:0] qkv_bank   [0:P-1][0:ROWS3-1];   // q|k|v Q.16
    reg signed [31:0] ctxv_bank  [0:P-1][0:ROWS-1];    // attention ctx Q6.25
    reg signed [31:0] attn_bank  [0:P-1][0:ROWS-1];    // proj out (attn_out) Q6.25
    reg signed [63:0] mlpbuf_bank[0:P-1][0:ROWSM-1];   // mlp hidden Q4.12 then Q.22
    reg signed [31:0] mlp_bank   [0:P-1][0:ROWS-1];    // mlp_proj out Q6.25
    reg signed [31:0] gemvy_bank [0:P-1][0:ROWSM-1];   // GEMV INT32 (up to D_MLP wide)

    // ---- layernorm_vec ---------------------------------------------------------
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    // ---- GEMV (resident) -------------------------------------------------------
    reg                gv_ldrst, gv_xwe, gv_start;
    reg signed [7:0]   gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    reg [10:0]         gv_rdaddr;
    wire signed [31:0] gv_yout;
    gemv_banked_resident #(.LANES(LANES), .MMAX(1024), .KMAX(1024), .RLAT(2),
                  .WWORDS(WWORDS)) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ldrst | wl_rst), .w_we(wl_we), .w_data(wl_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .rd_addr(gv_rdaddr[9:0]), .y_out(gv_yout));

    // ---- vec_dequant (P lanes, runtime frac) -----------------------------------
    reg               dq_vin;
    reg signed [6:0]  dq_frac;
    reg  [P*32-1:0]   dq_gemvy;
    reg  [P*24-1:0]   dq_mant;
    reg  [P*8-1:0]    dq_exp;
    wire              dq_vout;
    wire [P*32-1:0]   dq_out;
    vec_dequant #(.P(P)) u_dq (
        .clk(clk), .rst(rst), .in_valid(dq_vin), .frac(dq_frac),
        .gemvy(dq_gemvy), .mant(dq_mant), .exp(dq_exp),
        .out_valid(dq_vout), .dq_out(dq_out));

    // ---- vec_gelu (P lanes, Q4.12 in/out, 3-cyc latency) -----------------------
    reg             gl_vin;
    reg [P*16-1:0]  gl_x;
    wire            gl_vout;
    wire [P*16-1:0] gl_y;
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(gl_vin), .x(gl_x),
        .out_valid(gl_vout), .y(gl_y));

    // ---- vec_attn (one head at a time; T=1 single-token block0 gate) -----------
    reg               at_start, at_ldv;
    reg [8:0]         at_tcount;
    wire              at_ldready, at_ctxv, at_done;
    wire [6:0]        at_ctxidx;
    wire signed [31:0] at_ctxdata;
    reg [1:0]  hh;
    reg [8:0]  wi;
    wire [10:0] aw_src = (wi < 64)  ? (hh*HEAD_DIM + wi) :
                         (wi < 128) ? (11'd256 + hh*HEAD_DIM + (wi - 64)) :
                                      (11'd512 + hh*HEAD_DIM + (wi - 128));
    wire signed [31:0] aw_data = qkv_bank[aw_src[LSH-1:0]][aw_src >> LSH];
    vec_attn #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(32)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .ld_valid(at_ldv), .ld_data(aw_data), .ld_ready(at_ldready),
        .ctx_valid(at_ctxv), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata), .done(at_done));

    // ---- callable GEMV / LN parameter registers --------------------------------
    reg [19:0] g_wbase;            // weight base
    reg [10:0] g_m, g_k;           // dims
    reg [1:0]  g_asrc;             // act source: 0 lnout1, 1 ctxv(>>3), 2 lnout2, 3 mlpbuf
    reg [5:0]  g_asel;             // inv_sact index
    reg signed [6:0] g_frac;       // dequant frac
    reg [8:0]  g_dqrow;            // dequant channel-row base
    reg [1:0]  g_dst;              // dequant dest: 0 qkv, 1 attn, 2 mlpbuf(sat16), 3 mlp
    reg [4:0]  g_ret;             // return state after the GEMV
    reg [3:0]  l_gbase;            // LN gamma row
    reg        l_dst;              // LN dest: 0 lnout1, 1 lnout2
    reg [4:0]  l_ret;             // return state after the LN

    // ---- act-quant + dequant temporaries ---------------------------------------
    reg signed [63:0]  lntmp;
    reg signed [95:0]  aq_prod, aq_sh;
    reg signed [31:0]  aq_int;
    reg signed [31:0]  cb;
    reg signed [31:0]  e_tok, e_pos, xb, gb_, gy, om;
    reg signed [63:0]  lo;
    reg signed [31:0]  mant_v; reg signed [7:0] exp_v;
    reg signed [31:0]  dqv;
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    // gelu address pipeline (3-cyc latency)
    reg [$clog2(ROWSM+1)-1:0] gfr, gor;
    reg signed [63:0] gl_sh;
    integer pp;
    reg [8:0] ce;

    // ---- FSM -------------------------------------------------------------------
    localparam [4:0]
      S_IDLE=0, S_EMB=1,
      L_GAM=2, L_FEED=3, L_COLL=4,                  // callable LN
      G_AQ=5, G_RUN=6, G_WAIT=7, G_RB=8, G_DQ=9,    // callable GEMV
      S_QKVRET=10, S_AST=11, S_ALD=12, S_ACL=13,    // attention
      S_RES1=14, S_LN2=15, S_FCRET=16,              // proj/res1/LN2-call
      S_GELU=17, S_GELUC=18, S_MPRET=19, S_RES2=20, S_FIN=21,
      S_LN1=22;                                     // entry to LN1 call
    reg [4:0] st;
    reg [10:0] ci;
    reg [$clog2(ROWSM+1)-1:0] fr, orow, dr, dor;

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; gv_ldrst<=1'b0; gv_xwe<=1'b0; gv_start<=1'b0;
        dq_vin<=1'b0; gl_vin<=1'b0; at_start<=1'b0; at_ldv<=1'b0; done<=1'b0;
        if (rst) begin
            st<=S_IDLE; ci<=0; fr<=0; orow<=0; dr<=0; dor<=0; gfr<=0; gor<=0;
            rv0<=0; rv1<=0; rv2<=0;
        end else begin
            case (st)
                S_IDLE: if (go) begin ci<=0; st<=S_EMB; end
                // ---- embed -> xres banks; then call LN1 ------------------------
                S_EMB: begin
                    e_tok = tok_emb[tok_id*D + ci];
                    e_pos = pos_emb[pos*D + ci];
                    xres_bank[ci[LSH-1:0]][ci >> LSH] <= e_tok + e_pos;
                    if (ci == D-1) begin
                        ci<=0; l_gbase<=4'd0; l_dst<=1'b0; l_ret<=S_QKVRET; st<=L_GAM;
                    end else ci<=ci+1'b1;
                end
                // ================= callable LayerNorm =========================
                L_GAM: begin                              // load gamma row -> gamma_bank
                    gamma_bank[ci[LSH-1:0]][ci >> LSH] <= gamma_rom[l_gbase*D + ci];
                    if (ci==D-1) begin ci<=0; fr<=0; ln_start<=1'b1; st<=L_FEED; end
                    else ci<=ci+1'b1;
                end
                L_FEED: begin
                    ln_vin<=1'b1;
                    for (pp=0; pp<P; pp=pp+1) begin
                        xb = xres_bank[pp][fr]; gb_ = gamma_bank[pp][fr];
                        ln_x[pp*32 +: 32] <= xb; ln_g[pp*32 +: 32] <= gb_;
                    end
                    if (fr==ROWS-1) begin orow<=0; st<=L_COLL; end else fr<=fr+1'b1;
                end
                L_COLL: begin
                    if (ln_yv) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            if (l_dst==1'b0) lnout1_bank[pp][orow] <= ln_y[pp*64 +: 64];
                            else             lnout2_bank[pp][orow] <= ln_y[pp*64 +: 64];
                        end
                        if (orow==ROWS-1) st<=l_ret; else orow<=orow+1'b1;
                    end
                end
                // qkv GEMV setup (after LN1) and the GEMV call ------------------
                S_QKVRET: begin
                    g_wbase<=WB_QKV[19:0]; g_m<=D3[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=6'd0; g_frac<=7'd16; g_dqrow<=DR_QKV[8:0]; g_dst<=2'd0;
                    g_ret<=S_AST; ci<=0; hh<=2'd0; gv_ldrst<=1'b1; st<=G_AQ;  // init head counter
                end
                // ================= callable GEMV ==============================
                G_AQ: begin                               // act-quant from the selected source
                    case (g_asrc)
                        2'd0: lntmp = lnout1_bank[ci[LSH-1:0]][ci >> LSH];
                        2'd1: begin cb = ctxv_bank[ci[LSH-1:0]][ci >> LSH];
                                    // ctx Q6.25 -> Q.22 with rsh_round (round-half-away-from-0,
                                    // shift 3) — matches seq_ref._proj_after_attn exactly.
                                    if ($signed({{32{cb[31]}}, cb}) >= 0)
                                        lntmp = ($signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC);
                                    else
                                        lntmp = -((-$signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC));
                              end
                        2'd2: lntmp = lnout2_bank[ci[LSH-1:0]][ci >> LSH];
                        default: lntmp = mlpbuf_bank[ci[LSH-1:0]][ci >> LSH];
                    endcase
                    aq_prod = $signed(lntmp) * $signed(inv_sact[g_asel]);
                    if (lntmp >= 0)
                        aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                    else
                        aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                    aq_int = aq_sh[31:0];
                    if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                    gv_xwe<=1'b1; gv_xdata<=aq_int[7:0];
                    if (ci==g_k-1) begin ci<=0; st<=G_RUN; end else ci<=ci+1'b1;
                end
                G_RUN: begin
                    gv_m<=g_m; gv_k<=g_k; gv_wbase<=g_wbase[$clog2(WWORDS)-1:0];
                    gv_start<=1'b1; st<=G_WAIT;
                end
                G_WAIT: if (gv_done) begin
                    ci<=0; gv_rdaddr<=0; rv0<=0; rv1<=0; rv2<=0; st<=G_RB;
                end
                G_RB: begin                               // readback g_m outputs -> gemvy banks
                    if (ci < g_m) begin gv_rdaddr<=ci; rb0<=ci; ci<=ci+1'b1; end
                    else gv_rdaddr<=g_m-1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(ci<g_m); rv1<=rv0; rv2<=rv1;
                    if (rv2) begin
                        gemvy_bank[rb2[LSH-1:0]][rb2 >> LSH] <= gv_yout;
                        if (rb2==g_m-1) begin dr<=0; dor<=0; st<=G_DQ; end
                    end
                end
                G_DQ: begin                               // P-wide dequant -> selected dest
                    if (dr < (g_m >> LSH)) begin
                        dq_vin<=1'b1; dq_frac<=g_frac;
                        for (pp=0; pp<P; pp=pp+1) begin
                            gy = gemvy_bank[pp][dr];
                            mant_v = dqm_flat[pp*DQROWS + g_dqrow + dr];
                            exp_v  = dqe_flat[pp*DQROWS + g_dqrow + dr];
                            dq_gemvy[pp*32 +: 32] <= gy;
                            dq_mant [pp*24 +: 24] <= mant_v[23:0];
                            dq_exp  [pp*8  +: 8 ] <= exp_v;
                        end
                        dr<=dr+1'b1;
                    end else dq_vin<=1'b0;
                    if (dq_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            dqv = dq_out[pp*32 +: 32];
                            case (g_dst)
                                2'd0: qkv_bank [pp][dor] <= dqv;
                                2'd1: attn_bank[pp][dor] <= dqv;
                                2'd2: begin                       // mlpbuf: sat16 to Q4.12, sign-ext
                                    if      ($signed(dqv) >  32'sd32767)  mlpbuf_bank[pp][dor] <= 64'sd32767;
                                    else if ($signed(dqv) < -32'sd32768)  mlpbuf_bank[pp][dor] <= -64'sd32768;
                                    else    mlpbuf_bank[pp][dor] <= {{32{dqv[31]}}, dqv};
                                end
                                default: mlp_bank[pp][dor] <= dqv;
                            endcase
                        end
                        if (dor==(g_m >> LSH)-1) st<=g_ret; else dor<=dor+1'b1;
                    end
                end
                // ================= attention ==================================
                S_AST: begin at_start<=1'b1; at_tcount<=9'd1; wi<=9'd0; st<=S_ALD; end
                S_ALD: begin
                    at_ldv<=1'b1;
                    if (at_ldready) begin
                        if (wi==9'd191) st<=S_ACL; else wi<=wi+1'b1;
                    end
                end
                S_ACL: begin
                    if (at_ctxv) begin
                        ce = {2'd0, hh}*HEAD_DIM + {2'd0, at_ctxidx};
                        ctxv_bank[ce[LSH-1:0]][ce >> LSH] <= at_ctxdata;
                    end
                    if (at_done) begin
                        if (hh==NHEAD-1) begin                    // -> proj GEMV
                            g_wbase<=WB_PROJ[19:0]; g_m<=D[10:0]; g_k<=D[10:0]; g_asrc<=2'd1;
                            g_asel<=6'd1; g_frac<=7'd25; g_dqrow<=DR_PROJ[8:0]; g_dst<=2'd1;
                            g_ret<=S_RES1; ci<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else begin hh<=hh+2'd1; st<=S_AST; end
                    end
                end
                // ---- res1: xres += attn_out ; then LN2 -----------------------
                S_RES1: begin
                    for (pp=0; pp<P; pp=pp+1) begin
                        xb = xres_bank[pp][ci]; om = attn_bank[pp][ci];
                        xres_bank[pp][ci] <= xb + om;
                    end
                    if (ci==ROWS-1) begin
                        ci<=0; l_gbase<=4'd1; l_dst<=1'b1; l_ret<=S_FCRET; st<=L_GAM;
                    end else ci<=ci+1'b1;
                end
                // mlp_fc GEMV setup (after LN2) --------------------------------
                S_FCRET: begin
                    g_wbase<=WB_FC[19:0]; g_m<=D_MLP[10:0]; g_k<=D[10:0]; g_asrc<=2'd2;
                    g_asel<=6'd2; g_frac<=7'd12; g_dqrow<=DR_FC[8:0]; g_dst<=2'd2;
                    g_ret<=S_GELU; ci<=0; gfr<=0; gor<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ================= GELU (vec_gelu over mlpbuf) ================
                S_GELU: begin                              // drive P Q4.12, capture P Q.22
                    // feed
                    if (gfr < ROWSM) begin
                        gl_vin<=1'b1;
                        for (pp=0; pp<P; pp=pp+1) begin
                            lo = mlpbuf_bank[pp][gfr];
                            gl_x[pp*16 +: 16] <= lo[15:0];
                        end
                        gfr<=gfr+1'b1;
                    end else gl_vin<=1'b0;
                    // capture: gl_y (Q4.12) << (LN_OUT_FRAC-GELU_FRAC) -> Q.22
                    if (gl_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            gl_sh = $signed(gl_y[pp*16 +: 16]) <<< (LN_OUT_FRAC-GELU_FRAC);
                            mlpbuf_bank[pp][gor] <= gl_sh;
                        end
                        if (gor==ROWSM-1) begin                   // -> mlp_proj GEMV
                            g_wbase<=WB_MP[19:0]; g_m<=D[10:0]; g_k<=D_MLP[10:0]; g_asrc<=2'd3;
                            g_asel<=6'd3; g_frac<=7'd25; g_dqrow<=DR_MP[8:0]; g_dst<=2'd3;
                            g_ret<=S_RES2; ci<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else gor<=gor+1'b1;
                    end
                end
                // ---- res2: xres += mlp_out -> x_out --------------------------
                S_RES2: begin
                    for (pp=0; pp<P; pp=pp+1) begin
                        xb = xres_bank[pp][ci]; om = mlp_bank[pp][ci];
                        xres_bank[pp][ci] <= xb + om;
                    end
                    if (ci==ROWS-1) st<=S_FIN; else ci<=ci+1'b1;
                end
                S_FIN: begin done<=1'b1; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end

    // ---- readback mux (2-cyc registered). 32-bit signed banks auto sign-extend to
    //      the 64-bit signed rd0; 64-bit banks (ln1/ln2/gelu) pass through. ---------
    reg signed [63:0] rd0;
    reg [LSH-1:0] rba; reg [10:0] rbr;
    always @(posedge clk) begin
        rba = rd_addr[LSH-1:0]; rbr = rd_addr >> LSH;
        case (rd_sel)
            4'd0: rd0 <= lnout1_bank[rba][rbr];   // ln1   (Q.22)
            4'd1: rd0 <= qkv_bank   [rba][rbr];   // qkv   (Q.16, sign-ext)
            4'd2: rd0 <= ctxv_bank  [rba][rbr];   // ctx   (Q6.25)
            4'd3: rd0 <= attn_bank  [rba][rbr];   // attn_out (Q6.25)
            4'd4: rd0 <= lnout2_bank[rba][rbr];   // ln2   (Q.22)
            4'd5: rd0 <= mlpbuf_bank[rba][rbr];   // gelu  (Q.22)
            4'd6: rd0 <= mlp_bank   [rba][rbr];   // mlp_out (Q6.25)
            default: rd0 <= xres_bank[rba][rbr];  // x_out (Q6.25)
        endcase
        rd_data <= rd0;
    end
endmodule
