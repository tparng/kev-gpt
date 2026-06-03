// -----------------------------------------------------------------------------
// sequencer_vec — the P-WIDE vector-datapath sequencer (the road to ~10k tok/s).
// WIDE-WORD banking edition. Functionally identical (bit-exact) to the proven
// [P][rows] version, but every scratch buffer + every P-wide-read ROM is ONE
// row-addressed memory holding P elements per word (lane l in bits [l*W +: W]):
//
//   reg [P*W-1:0] buf [0:ROWS-1]      instead of    reg [W] buf [0:P-1][0:ROWS-1]
//
// WHY: the [P][rows] layout, read/written at a VARIABLE row, synthesises to giant
// per-lane row-multiplexers (the 88k MUXF7 / 216k-LUT blow-up that overflowed the
// KV260). With wide words the variable row is just a memory ADDRESS (free) and only
// a small P:1 lane-select remains. P-wide accesses touch the whole word (constant
// offsets, no mux); the two inherently-1/cycle producers (GEMV readback, attention
// ctx) accumulate into a plain-reg staging word and write one wide word every P.
//
// Bonus cycle win: embeddings/gamma are wide ROMs, so embed is ROWS (not D) cycles
// and the separate gamma-load phase disappears.
//
// iverilog-2012 safe: VARIABLE +: part-selects only ever index PLAIN REGS (never an
// unpacked-array element); unpacked arrays are only ever read/written by element
// index; flat/packed ROMs for $readmemh.
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
    parameter integer WWORDS = 262144,
    parameter integer NLAYER = 4,
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
    output reg [8:0]   tok_out,     // argmax token id (after the full forward + head)
    // readback: rd_sel picks the bank, rd_addr the element (2-cyc registered). 64-bit so
    // the Q.22 LN/gelu values fit; 32-bit values are sign-extended in their bank.
    input  wire [3:0]  rd_sel,
    input  wire [10:0] rd_addr,
    output reg signed [63:0] rd_data,
    // runtime weight load (stream the whole transposed image once)
    input  wire        wl_rst,
    input  wire        wl_we,
    input  wire [31:0] wl_data,
    // DEBUG: when high, halt after block 0's residual (banks hold block-0 ln1/qkv/ctx/attn/
    // ln2/gelu/mlp/x) so a board readback can be compared to seq_ref.block0_phase_signals.
    input  wire        dbg_stop
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
    localparam integer GW_MP   = ((D + LANES-1)/LANES) * D_MLP;          // 16384
    localparam integer GW_HEAD = ((VOCAB + LANES-1)/LANES) * D;          // 3328
    localparam integer GW_BLK  = GW_QKV+GW_PROJ+GW_FC+GW_MP;             // 49152 (words/block)
    localparam integer DQ_BLK  = D3+D+D_MLP+D;                           // 2304 (channels/block)
    localparam integer DQB_P   = DQ_BLK/P;                               // 288 (rows/block, /P)
    localparam integer WB_HEAD = NLAYER*GW_BLK;                          // head weight base
    localparam integer DR_HEAD = (NLAYER*DQ_BLK)/P;                      // head dequant row base
    localparam integer ARROWS  = (VOCAB + P - 1)/P;                      // argmax rows
    localparam integer EROWS   = D / P;                                  // emb/gamma rows per set

    // ---- wide-word ROMs ($readmemh: one P-packed word per line) -----------------
    (* rom_style = "block" *) reg [P*32-1:0] tok_emb_w [0:VOCAB*EROWS-1];  // Q6.25
    (* rom_style = "block" *) reg [P*32-1:0] pos_emb_w [0:TMAX*EROWS-1];   // Q6.25
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w   [0:GAMMA_N*EROWS-1];// Q4.20
    (* rom_style = "block" *) reg signed [63:0] inv_sact [0:NSACT-1];
    (* rom_style = "block" *) reg [P*24-1:0] dqm_w [0:DQROWS-1];           // P mant / word
    (* rom_style = "block" *) reg [P*8-1:0]  dqe_w [0:DQROWS-1];           // P exp  / word
    initial begin
        $readmemh("tok_emb_w.mem", tok_emb_w);
        $readmemh("pos_emb_w.mem", pos_emb_w);
        $readmemh("gamma_w.mem",   gamma_w);
        $readmemh("inv_sact.mem",  inv_sact);
        $readmemh("dqm_w.mem",     dqm_w);
        $readmemh("dqe_w.mem",     dqe_w);
    end

    // ---- wide-word scratch (one row-addressed memory per gateable intermediate) -
    reg [P*32-1:0] xres_bank  [0:ROWS-1];    // residual Q6.25
    reg [P*64-1:0] lnout1_bank[0:ROWS-1];    // LN1 / LN_f out Q.22
    reg [P*64-1:0] lnout2_bank[0:ROWS-1];    // LN2 out Q.22
    reg [P*32-1:0] qkv_bank   [0:ROWS3-1];   // q|k|v Q.16
    reg [P*32-1:0] ctxv_bank  [0:ROWS-1];    // attention ctx Q6.25
    reg [P*32-1:0] attn_bank  [0:ROWS-1];    // proj out (attn_out) Q6.25
    reg [P*64-1:0] mlpbuf_bank[0:ROWSM-1];   // mlp hidden Q4.12 then Q.22
    reg [P*32-1:0] mlp_bank   [0:ROWS-1];    // mlp_proj out Q6.25
    reg [P*32-1:0] gemvy_bank [0:ROWSM-1];   // GEMV INT32 (up to D_MLP wide)
    reg [P*32-1:0] head_bank  [0:ARROWS-1];  // head logits (real, Q6.25)
    reg [P*32-1:0] gystage, cstage;          // P-deep staging words for 1/cyc producers

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
    // wide-word qkv read: element-select the word (plain reg), then plain-reg lane part-sel
    reg  [P*32-1:0] qw_r;
    always @* qw_r = qkv_bank[aw_src >> LSH];
    wire signed [31:0] aw_data = qw_r[(aw_src[LSH-1:0])*32 +: 32];
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
    reg [11:0] g_dqrow;            // dequant channel-row base (up to NLAYER*DQ_BLK/P = 1152)
    reg [2:0]  g_dst;              // dest: 0 qkv,1 attn,2 mlpbuf(sat16),3 mlp,4 head
    reg [4:0]  g_ret;             // return state after the GEMV
    reg [3:0]  l_gbase;            // LN gamma set (0=ln1.0,1=ln2.0,...,2*NLAYER=ln_f)
    reg        l_dst;              // LN dest: 0 lnout1, 1 lnout2
    reg [4:0]  l_ret;             // return state after the LN

    // ---- temporaries (PLAIN regs — safe targets for variable part-selects) -----
    reg signed [63:0]  lntmp;
    reg signed [95:0]  aq_prod, aq_sh;
    reg signed [31:0]  aq_int;
    reg signed [31:0]  cb, dqv, gy_v, hv;
    reg [P*32-1:0]  tw, pw, ww, xw, ow, sw, dword, gyr, hw, gyw, cw;
    reg [P*64-1:0]  w64, mw64, mword, gsw;
    reg [P*32-1:0]  w32;
    reg [P*24-1:0]  mwr;
    reg [P*8-1:0]   ewr;
    reg signed [63:0] gl_sh;
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    reg dq_rd_v;                             // G_DQ read-pipeline valid (BRAM 1-cyc latency)
    reg [$clog2(ROWSM+1)-1:0] gfr, gor;
    integer pp;
    reg [8:0] ce;
    reg [3:0] blk;                           // transformer block 0..NLAYER-1
    reg signed [31:0] best_val; reg [8:0] best_idx, hidx;
    reg [$clog2(ARROWS+1)-1:0] ar;           // argmax row counter

    // ---- FSM -------------------------------------------------------------------
    localparam [4:0]
      S_IDLE=0, S_EMB=1,
      L_GAM=2, L_FEED=3, L_COLL=4,                  // callable LN (L_GAM now = start-only)
      G_AQ=5, G_RUN=6, G_WAIT=7, G_RB=8, G_DQ=9,    // callable GEMV
      S_QKVRET=10, S_AST=11, S_ALD=12, S_ACL=13,    // attention
      S_RES1=14, S_LN2=15, S_FCRET=16,              // proj/res1/LN2-call
      S_GELU=17, S_GELUC=18, S_MPRET=19, S_RES2=20, S_FIN=21,
      S_HEADSET=22, S_ARGMAX=23;                    // final LN_f -> head -> argmax
    reg [4:0] st;
    reg [10:0] ci;
    reg [$clog2(ROWSM+1)-1:0] fr, orow, dr, dor;

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; gv_ldrst<=1'b0; gv_xwe<=1'b0; gv_start<=1'b0;
        dq_vin<=1'b0; gl_vin<=1'b0; at_start<=1'b0; at_ldv<=1'b0; done<=1'b0;
        if (rst) begin
            st<=S_IDLE; ci<=0; fr<=0; orow<=0; dr<=0; dor<=0; gfr<=0; gor<=0;
            rv0<=0; rv1<=0; rv2<=0; gystage<=0; cstage<=0; dq_rd_v<=0;
        end else begin
            case (st)
                S_IDLE: if (go) begin fr<=0; blk<=4'd0; gystage<=0; cstage<=0; st<=S_EMB; end
                // ---- embed -> xres (wide ROM, P/cyc); then call LN1 of block 0 --
                S_EMB: begin
                    tw = tok_emb_w[tok_id*EROWS + fr];
                    pw = pos_emb_w[pos*EROWS + fr];
                    for (pp=0; pp<P; pp=pp+1)
                        ww[pp*32 +: 32] = $signed(tw[pp*32 +: 32]) + $signed(pw[pp*32 +: 32]);
                    xres_bank[fr] <= ww;
                    if (fr==ROWS-1) begin
                        fr<=0; l_gbase<=4'd0; l_dst<=1'b0; l_ret<=S_QKVRET; st<=L_GAM;
                    end else fr<=fr+1'b1;
                end
                // ================= callable LayerNorm =========================
                L_GAM: begin ln_start<=1'b1; fr<=0; st<=L_FEED; end   // gamma is a wide ROM now
                L_FEED: begin
                    ln_vin<=1'b1;
                    ln_x <= xres_bank[fr];
                    ln_g <= gamma_w[l_gbase*EROWS + fr];
                    if (fr==ROWS-1) begin orow<=0; st<=L_COLL; end else fr<=fr+1'b1;
                end
                L_COLL: begin
                    if (ln_yv) begin
                        if (l_dst==1'b0) lnout1_bank[orow] <= ln_y;
                        else             lnout2_bank[orow] <= ln_y;
                        if (orow==ROWS-1) st<=l_ret; else orow<=orow+1'b1;
                    end
                end
                // qkv GEMV setup (after LN1) and the GEMV call ------------------
                S_QKVRET: begin
                    g_wbase<=blk*GW_BLK + WB_QKV; g_m<=D3[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=blk*4 + 6'd0; g_frac<=7'd16; g_dqrow<=blk*DQB_P + DR_QKV; g_dst<=3'd0;
                    g_ret<=S_AST; ci<=0; hh<=2'd0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ================= callable GEMV ==============================
                G_AQ: begin                               // act-quant from the selected source
                    case (g_asrc)
                        2'd0: begin w64 = lnout1_bank[ci >> LSH];
                                    lntmp = w64[(ci[LSH-1:0])*64 +: 64]; end
                        2'd1: begin w32 = ctxv_bank[ci >> LSH];
                                    cb = w32[(ci[LSH-1:0])*32 +: 32];
                                    // ctx Q6.25 -> Q.22 rsh_round (matches _proj_after_attn)
                                    if ($signed({{32{cb[31]}}, cb}) >= 0)
                                        lntmp = ($signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC);
                                    else
                                        lntmp = -((-$signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC));
                              end
                        2'd2: begin w64 = lnout2_bank[ci >> LSH];
                                    lntmp = w64[(ci[LSH-1:0])*64 +: 64]; end
                        default: begin w64 = mlpbuf_bank[ci >> LSH];
                                    lntmp = w64[(ci[LSH-1:0])*64 +: 64]; end
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
                    ci<=0; gv_rdaddr<=0; rv0<=0; rv1<=0; rv2<=0; gystage<=0; dq_rd_v<=0; st<=G_RB;
                end
                G_RB: begin                               // readback g_m outputs -> gemvy (staged)
                    if (ci < g_m) begin gv_rdaddr<=ci; rb0<=ci; ci<=ci+1'b1; end
                    else gv_rdaddr<=g_m-1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(ci<g_m); rv1<=rv0; rv2<=rv1;
                    if (rv2) begin
                        gyw = gystage;
                        gyw[(rb2[LSH-1:0])*32 +: 32] = gv_yout;
                        if (rb2[LSH-1:0]==P-1 || rb2==g_m-1) begin
                            gemvy_bank[rb2 >> LSH] <= gyw; gystage <= 0;
                        end else gystage <= gyw;
                        if (rb2==g_m-1) begin dr<=0; dor<=0; st<=G_DQ; end
                    end
                end
                G_DQ: begin                               // P-wide dequant -> selected dest
                    // gemvy/dqm_w/dqe_w map to BLOCK RAM (1-cyc registered read on silicon).
                    // Stage 1: REGISTERED read (nonblocking) — iverilog + BRAM both latch the
                    // address now, data valid next cycle, so sim == hardware. Stage 2: one cycle
                    // later, the row's data is valid in gyr/mwr/ewr -> drive the dequant lanes.
                    if (dr < ((g_m + P-1) >> LSH)) begin   // ceil rows (head VOCAB not P-mult)
                        gyr <= gemvy_bank[dr];
                        mwr <= dqm_w[g_dqrow + dr];
                        ewr <= dqe_w[g_dqrow + dr];
                        dq_rd_v <= 1'b1;
                        dr<=dr+1'b1;
                    end else dq_rd_v <= 1'b0;
                    if (dq_rd_v) begin                      // BRAM outputs now valid
                        dq_vin<=1'b1; dq_frac<=g_frac;
                        for (pp=0; pp<P; pp=pp+1) begin
                            dq_gemvy[pp*32 +: 32] <= gyr[pp*32 +: 32];
                            dq_mant [pp*24 +: 24] <= mwr[pp*24 +: 24];
                            dq_exp  [pp*8  +: 8 ] <= ewr[pp*8  +: 8 ];
                        end
                    end
                    if (dq_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            dqv = dq_out[pp*32 +: 32];
                            dword[pp*32 +: 32] = dqv;
                            if      ($signed(dqv) >  32'sd32767)  mword[pp*64 +: 64] = 64'sd32767;
                            else if ($signed(dqv) < -32'sd32768)  mword[pp*64 +: 64] = -64'sd32768;
                            else    mword[pp*64 +: 64] = {{32{dqv[31]}}, dqv};
                        end
                        case (g_dst)
                            3'd0: qkv_bank [dor] <= dword;
                            3'd1: attn_bank[dor] <= dword;
                            3'd2: mlpbuf_bank[dor] <= mword;   // Q4.12 sat16, sign-ext
                            3'd3: mlp_bank [dor] <= dword;
                            default: head_bank[dor] <= dword;  // 4: head logits Q6.25
                        endcase
                        if (dor==((g_m + P-1) >> LSH)-1) st<=g_ret; else dor<=dor+1'b1;
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
                        cw = cstage;
                        cw[(ce[LSH-1:0])*32 +: 32] = at_ctxdata;
                        if (ce[LSH-1:0]==P-1) begin ctxv_bank[ce >> LSH] <= cw; cstage <= 0; end
                        else cstage <= cw;
                    end
                    if (at_done) begin
                        if (hh==NHEAD-1) begin                    // -> proj GEMV
                            g_wbase<=blk*GW_BLK + WB_PROJ; g_m<=D[10:0]; g_k<=D[10:0]; g_asrc<=2'd1;
                            g_asel<=blk*4 + 6'd1; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_PROJ; g_dst<=3'd1;
                            g_ret<=S_RES1; ci<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else begin hh<=hh+2'd1; st<=S_AST; end
                    end
                end
                // ---- res1: xres += attn_out ; then LN2 -----------------------
                S_RES1: begin
                    xw = xres_bank[ci]; ow = attn_bank[ci];
                    for (pp=0; pp<P; pp=pp+1)
                        sw[pp*32 +: 32] = $signed(xw[pp*32 +: 32]) + $signed(ow[pp*32 +: 32]);
                    xres_bank[ci] <= sw;
                    if (ci==ROWS-1) begin
                        ci<=0; l_gbase<=blk*2 + 4'd1; l_dst<=1'b1; l_ret<=S_FCRET; st<=L_GAM;
                    end else ci<=ci+1'b1;
                end
                // mlp_fc GEMV setup (after LN2) --------------------------------
                S_FCRET: begin
                    g_wbase<=blk*GW_BLK + WB_FC; g_m<=D_MLP[10:0]; g_k<=D[10:0]; g_asrc<=2'd2;
                    g_asel<=blk*4 + 6'd2; g_frac<=7'd12; g_dqrow<=blk*DQB_P + DR_FC; g_dst<=3'd2;
                    g_ret<=S_GELU; ci<=0; gfr<=0; gor<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ================= GELU (vec_gelu over mlpbuf) ================
                S_GELU: begin                              // drive P Q4.12, capture P Q.22
                    if (gfr < ROWSM) begin
                        gl_vin<=1'b1;
                        mw64 = mlpbuf_bank[gfr];
                        for (pp=0; pp<P; pp=pp+1)
                            gl_x[pp*16 +: 16] <= mw64[pp*64 +: 16];   // low 16 of each Q4.12 lane
                        gfr<=gfr+1'b1;
                    end else gl_vin<=1'b0;
                    if (gl_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            gl_sh = $signed(gl_y[pp*16 +: 16]) <<< (LN_OUT_FRAC-GELU_FRAC);
                            gsw[pp*64 +: 64] = gl_sh;
                        end
                        mlpbuf_bank[gor] <= gsw;
                        if (gor==ROWSM-1) begin                   // -> mlp_proj GEMV
                            g_wbase<=blk*GW_BLK + WB_MP; g_m<=D[10:0]; g_k<=D_MLP[10:0]; g_asrc<=2'd3;
                            g_asel<=blk*4 + 6'd3; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_MP; g_dst<=3'd3;
                            g_ret<=S_RES2; ci<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else gor<=gor+1'b1;
                    end
                end
                // ---- res2: xres += mlp_out ; next block or final LN_f --------
                S_RES2: begin
                    xw = xres_bank[ci]; ow = mlp_bank[ci];
                    for (pp=0; pp<P; pp=pp+1)
                        sw[pp*32 +: 32] = $signed(xw[pp*32 +: 32]) + $signed(ow[pp*32 +: 32]);
                    xres_bank[ci] <= sw;
                    if (ci==ROWS-1) begin
                        ci<=0;
                        if (dbg_stop && blk==4'd0) st<=S_FIN;      // DEBUG: stop after block 0
                        else if (blk == NLAYER-1) begin            // -> final LN_f -> head
                            l_gbase<=NLAYER*2; l_dst<=1'b0; l_ret<=S_HEADSET; st<=L_GAM;
                        end else begin                             // -> next block LN1
                            blk<=blk+1'b1; l_gbase<=(blk+1'b1)*2; l_dst<=1'b0;
                            l_ret<=S_QKVRET; st<=L_GAM;
                        end
                    end else ci<=ci+1'b1;
                end
                // ---- head GEMV (act = LN_f out in lnout1) -> head_bank -------
                S_HEADSET: begin
                    g_wbase<=WB_HEAD; g_m<=VOCAB[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=4*NLAYER; g_frac<=7'd25; g_dqrow<=DR_HEAD[11:0]; g_dst<=3'd4;
                    g_ret<=S_ARGMAX; ci<=0; gv_ldrst<=1'b1;
                    best_val<=32'sh80000000; best_idx<=9'd0; ar<=0; st<=G_AQ;
                end
                // ---- P-wide argmax over the VOCAB logits (Q6.25) ------------
                S_ARGMAX: begin
                    hw = head_bank[ar];
                    for (pp=0; pp<P; pp=pp+1) begin
                        hv   = hw[pp*32 +: 32];
                        hidx = ar*P + pp;
                        if (hidx < VOCAB && $signed(hv) > best_val) begin
                            best_val = $signed(hv); best_idx = hidx;
                        end
                    end
                    if (ar==ARROWS-1) begin tok_out<=best_idx; st<=S_FIN; end
                    else ar<=ar+1'b1;
                end
                S_FIN: begin done<=1'b1; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end

    // ---- readback mux (2-cyc registered): pick the bank word, extract the lane ----
    reg signed [63:0] rd0;
    reg [LSH-1:0] rba; reg [10:0] rbr;
    reg [P*64-1:0] rw64; reg [P*32-1:0] rw32; reg is64;
    always @(posedge clk) begin
        rba = rd_addr[LSH-1:0]; rbr = rd_addr >> LSH;
        is64 = 1'b0; rw64 = {(P*64){1'b0}}; rw32 = {(P*32){1'b0}};
        case (rd_sel)
            4'd0: begin rw64 = lnout1_bank[rbr]; is64 = 1'b1; end   // ln1   (Q.22)
            4'd1: rw32 = qkv_bank   [rbr];                          // qkv   (Q.16)
            4'd2: rw32 = ctxv_bank  [rbr];                          // ctx   (Q6.25)
            4'd3: rw32 = attn_bank  [rbr];                          // attn_out (Q6.25)
            4'd4: begin rw64 = lnout2_bank[rbr]; is64 = 1'b1; end   // ln2   (Q.22)
            4'd5: begin rw64 = mlpbuf_bank[rbr]; is64 = 1'b1; end   // gelu  (Q.22)
            4'd6: rw32 = mlp_bank   [rbr];                          // mlp_out (Q6.25)
            4'd7: rw32 = xres_bank  [rbr];                          // x4 residual (Q6.25)
            default: rw32 = head_bank[rbr];                         // 8: head logits (Q6.25)
        endcase
        if (is64) rd0 <= $signed(rw64[rba*64 +: 64]);
        else      rd0 <= {{32{rw32[rba*32 + 31]}}, rw32[rba*32 +: 32]};  // sign-ext 32->64
        rd_data <= rd0;
    end
endmodule
