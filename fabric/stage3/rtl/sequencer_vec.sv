// -----------------------------------------------------------------------------
// sequencer_vec — the P-WIDE vector-datapath sequencer (the road to ~10k tok/s).
// WIDE-WORD banking + SYNCHRONOUS-READ (BRAM) edition.
//
// Every scratch buffer and every big ROM is ONE row-addressed memory holding P
// elements per word (lane l in bits [l*W +: W]) — and every read is REGISTERED:
// address presented on cycle k, data consumed on k+1. One read register per
// memory, addressed via a small state mux, so Vivado infers BLOCK RAM instead of
// LUTRAM. Why: the P=8 LUT budget was 109% (~31k distributed-RAM LUTs); BRAM
// frees this. Bonus: iverilog sees the same 1-cycle read latency as silicon, so
// async-read divergences (the §6 board bug class) are out by construction.
//
// FSM loops pipeline (one cycle skew, not 2x):
//   fr/ci/gfr/ar  : read-address counter (held at limit when done)
//   frd/cid/...   : address delayed 1 cycle  -> consume stage
//   frv/civ/...   : valid bit (entered loop, address in range)
// Producer states (G_RB, S_ACL) keep their plain-reg staging words.
//
// iverilog-2012 safe: variable +: part-selects only index PLAIN REGS; flat ROMs
// for $readmemh; reads of unpacked arrays only by element index.
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
    // DEBUG halt points (banks then hold a block-0 snapshot for board readback vs
    // seq_ref.block0_phase_signals): 1=after embed (xres=x_in), 2=after LN2 (xres=x_res1,
    // lnout2=ln2), 3=after block0 (xres=x_out). 0 = no stop (normal forward).
    input  wire [1:0]  dbg_stop
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
    // All sync-read (registered) so they infer BLOCK RAM, not LUTs.
    (* rom_style = "block" *) reg [P*32-1:0] tok_emb_w [0:VOCAB*EROWS-1];  // Q6.25
    // BRAM budget: both embeds in BRAM at TMAX=256 hit 174/144 tiles. URAM has no init.
    // Build at TMAX=64 (Kevin context) -> ~131/144 BRAM.
    (* rom_style = "block" *) reg [P*32-1:0] pos_emb_w [0:TMAX*EROWS-1];   // Q6.25
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w   [0:GAMMA_N*EROWS-1];// Q4.20
    reg signed [63:0] inv_sact [0:NSACT-1];                       // 17-deep: stays LUT
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
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank  [0:ROWS-1];    // residual Q6.25
    (* ram_style = "block" *) reg [P*64-1:0] lnout1_bank[0:ROWS-1];    // LN1 / LN_f out Q.22
    (* ram_style = "block" *) reg [P*64-1:0] lnout2_bank[0:ROWS-1];    // LN2 out Q.22
    (* ram_style = "block" *) reg [P*32-1:0] qkv_bank   [0:ROWS3-1];   // q|k|v Q.16
    (* ram_style = "block" *) reg [P*32-1:0] ctxv_bank  [0:ROWS-1];    // attention ctx Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] attn_bank  [0:ROWS-1];    // proj out Q6.25
    (* ram_style = "block" *) reg [P*64-1:0] mlpbuf_bank[0:ROWSM-1];   // mlp hidden Q4.12 / Q.22
    (* ram_style = "block" *) reg [P*32-1:0] mlp_bank   [0:ROWS-1];    // mlp_proj out Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] gemvy_bank [0:ROWSM-1];   // GEMV INT32
    (* ram_style = "block" *) reg [P*32-1:0] head_bank  [0:ARROWS-1];  // head logits Q6.25
    reg [P*32-1:0] gystage, cstage;          // P-deep staging words for 1/cyc producers

    // ---- synchronous-read registers (BRAM output stage; declared before use) ---
    reg [P*32-1:0] xres_r, qkv_r, ctxv_r, attn_r, mlp_r, head_r;
    reg [P*64-1:0] lnout1_r, lnout2_r, mlpbuf_r;
    reg [P*32-1:0] temb_r, pemb_r, gam_r;

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
    reg [8:0]  wi;                          // load-address counter (runs ahead)
    reg [8:0]  wic;                         // accepted-word counter (consume stage)
    reg        wiv;                         // load-data valid (qkv read pipeline)
    reg [LSH-1:0] awl;                      // delayed lane of the prefetched word
    wire [10:0] aw_src = (wi < 64)  ? (hh*HEAD_DIM + wi) :
                         (wi < 128) ? (11'd256 + hh*HEAD_DIM + (wi - 64)) :
                                      (11'd512 + hh*HEAD_DIM + (wi - 128));
    // sync-read prefetch: address on cycle k -> qkv_r/awl on k+1 (ld_ready stays high
    // through the q/K/V stream, so the 1-deep prefetch keeps 1 word/cycle).
    wire signed [31:0] aw_data = qkv_r[awl*32 +: 32];
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
    reg signed [31:0]  cb, dqv, hv;
    reg [P*32-1:0]  ww, sw, dword, gyr, hw, gyw, cw;
    reg [P*64-1:0]  mword, gsw;
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
      L_GAM=2, L_FEED=3, L_COLL=4,                  // callable LN (L_GAM = start-only)
      G_AQ=5, G_RUN=6, G_WAIT=7, G_RB=8, G_DQ=9,    // callable GEMV
      S_QKVRET=10, S_AST=11, S_ALD=12, S_ACL=13,    // attention
      S_RES1=14, S_LN2=15, S_FCRET=16,              // proj/res1/LN2-call
      S_GELU=17, S_GELUC=18, S_MPRET=19, S_RES2=20, S_FIN=21,
      S_HEADSET=22, S_ARGMAX=23;                    // final LN_f -> head -> argmax
    reg [4:0] st;
    reg [10:0] ci;
    reg [$clog2(ROWSM+1)-1:0] fr, orow, dr, dor;
    // read-pipeline delayed addresses + valids (consume stage of each FSM loop)
    reg [10:0] cid;  reg civ;
    reg [$clog2(ROWSM+1)-1:0] frd, gfrd;  reg frv, gfrv;
    reg [$clog2(ARROWS+1)-1:0] ard;  reg arv;

    // ---- synchronous reads (one read register per memory, address muxed) -------
    wire [10:0] rbr = rd_addr >> LSH;            // board readback row (idle only)
    wire [10:0] xres_ra   = (st==L_FEED) ? {{(11-$clog2(ROWSM+1)){1'b0}}, fr} :
                            (st==S_RES1 || st==S_RES2) ? ci : rbr;
    wire [10:0] lnout1_ra = (st==G_AQ) ? (ci >> LSH) : rbr;
    wire [10:0] lnout2_ra = (st==G_AQ) ? (ci >> LSH) : rbr;
    wire [10:0] ctxv_ra   = (st==G_AQ) ? (ci >> LSH) : rbr;
    wire [10:0] mlpbuf_ra = (st==S_GELU) ? {{(11-$clog2(ROWSM+1)){1'b0}}, gfr} :
                            (st==G_AQ) ? (ci >> LSH) : rbr;
    wire [10:0] qkv_ra    = (st==S_ALD) ? (aw_src >> LSH) : rbr;
    wire [10:0] attn_ra   = (st==S_RES1) ? ci : rbr;
    wire [10:0] mlp_ra    = (st==S_RES2) ? ci : rbr;
    wire [10:0] head_ra   = (st==S_ARGMAX) ? {{(11-$clog2(ARROWS+1)){1'b0}}, ar} : rbr;

    always @(posedge clk) begin
        xres_r   <= xres_bank  [xres_ra];
        lnout1_r <= lnout1_bank[lnout1_ra];
        lnout2_r <= lnout2_bank[lnout2_ra];
        qkv_r    <= qkv_bank   [qkv_ra];
        ctxv_r   <= ctxv_bank  [ctxv_ra];
        attn_r   <= attn_bank  [attn_ra];
        mlpbuf_r <= mlpbuf_bank[mlpbuf_ra];
        mlp_r    <= mlp_bank   [mlp_ra];
        head_r   <= head_bank  [head_ra];
        temb_r   <= tok_emb_w[tok_id*EROWS + fr];
        pemb_r   <= pos_emb_w[pos*EROWS + fr];
        gam_r    <= gamma_w  [l_gbase*EROWS + fr];
        awl      <= aw_src[LSH-1:0];
    end

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; gv_ldrst<=1'b0; gv_xwe<=1'b0; gv_start<=1'b0;
        dq_vin<=1'b0; gl_vin<=1'b0; at_start<=1'b0; at_ldv<=1'b0; done<=1'b0;
        if (rst) begin
            st<=S_IDLE; ci<=0; fr<=0; orow<=0; dr<=0; dor<=0; gfr<=0; gor<=0;
            rv0<=0; rv1<=0; rv2<=0; gystage<=0; cstage<=0; dq_rd_v<=0;
            civ<=0; frv<=0; gfrv<=0; arv<=0; wiv<=0;
        end else begin
            case (st)
                S_IDLE: if (go) begin
                    fr<=0; frv<=0; blk<=4'd0; gystage<=0; cstage<=0; st<=S_EMB;
                end
                // ---- embed -> xres (wide ROM, sync read, P/cyc); then LN1 of block 0
                S_EMB: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    if (frv) begin
                        for (pp=0; pp<P; pp=pp+1)
                            ww[pp*32 +: 32] = $signed(temb_r[pp*32 +: 32])
                                            + $signed(pemb_r[pp*32 +: 32]);
                        xres_bank[frd] <= ww;
                        if (frd==ROWS-1) begin
                            fr<=0; frv<=0;
                            if (dbg_stop==2'd1) st<=S_FIN;       // DEBUG: stop after embed
                            else begin l_gbase<=4'd0; l_dst<=1'b0; l_ret<=S_QKVRET; st<=L_GAM; end
                        end
                    end
                end
                // ================= callable LayerNorm =========================
                L_GAM: begin ln_start<=1'b1; fr<=0; frv<=0; st<=L_FEED; end
                L_FEED: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    ln_vin <= frv;
                    ln_x   <= xres_r;
                    ln_g   <= gam_r;
                    if (frv && frd==ROWS-1) begin orow<=0; fr<=0; frv<=0; st<=L_COLL; end
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
                    g_ret<=S_AST; ci<=0; civ<=0; hh<=2'd0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ================= callable GEMV ==============================
                G_AQ: begin            // act-quant from the selected source (sync read)
                    cid <= ci; civ <= (ci != g_k);
                    if (ci != g_k) ci <= ci + 1'b1;
                    if (civ) begin
                        case (g_asrc)
                            2'd0: lntmp = $signed(lnout1_r[(cid[LSH-1:0])*64 +: 64]);
                            2'd1: begin
                                cb = ctxv_r[(cid[LSH-1:0])*32 +: 32];
                                // ctx Q6.25 -> Q.22 rsh_round (matches _proj_after_attn)
                                if ($signed({{32{cb[31]}}, cb}) >= 0)
                                    lntmp = ($signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC);
                                else
                                    lntmp = -((-$signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC));
                            end
                            2'd2: lntmp = $signed(lnout2_r[(cid[LSH-1:0])*64 +: 64]);
                            default: lntmp = $signed(mlpbuf_r[(cid[LSH-1:0])*64 +: 64]);
                        endcase
                        aq_prod = $signed(lntmp) * $signed(inv_sact[g_asel]);
                        if (lntmp >= 0)
                            aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                        else
                            aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                        aq_int = aq_sh[31:0];
                        if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                        gv_xwe<=1'b1; gv_xdata<=aq_int[7:0];
                        if (cid==g_k-1) begin ci<=0; civ<=0; st<=G_RUN; end
                    end
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
                    // gemvy/dqm_w/dqe_w are BLOCK RAM (1-cyc registered read). Stage 1:
                    // registered read; stage 2: drive the dequant lanes.
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
                        // ci is left at g_m by G_RB; reset it so the g_ret consumer (S_RES1/
                        // S_RES2 stream ci=0..ROWS-1) starts clean. Without this, ci enters res
                        // as 256: iverilog drops the out-of-range xres_bank[256] (gate passes by
                        // luck) but on silicon the 6-bit address WRAPS and the residual adds attn
                        // ~29x as ci counts 256..2047..63 -> x_res1 blows up. The real bug.
                        if (dor==((g_m + P-1) >> LSH)-1) begin ci<=0; civ<=0; st<=g_ret; end
                        else dor<=dor+1'b1;
                    end
                end
                // ================= attention ==================================
                S_AST: begin
                    at_start<=1'b1; at_tcount<=9'd1; wi<=9'd0; wic<=9'd0; wiv<=1'b0; st<=S_ALD;
                end
                S_ALD: begin
                    // 1-deep prefetch: qkv_r/awl trail wi by one cycle, and at_ldv is one
                    // register behind wi — both land in the same cycle at vec_attn.
                    at_ldv <= (wi != 9'd192);
                    if (wi != 9'd192) wi <= wi + 1'b1;
                    if (at_ldv) begin
                        if (wic == 9'd191) st <= S_ACL;
                        else wic <= wic + 1'b1;
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
                            g_ret<=S_RES1; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else begin hh<=hh+2'd1; st<=S_AST; end
                    end
                end
                // ---- res1: xres += attn_out (sync read, 1-cyc skew) ; then LN2 ----
                S_RES1: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(attn_r[pp*32 +: 32]);
                        xres_bank[cid] <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            l_gbase<=blk*2 + 4'd1; l_dst<=1'b1; l_ret<=S_FCRET; st<=L_GAM;
                        end
                    end
                end
                // mlp_fc GEMV setup (after LN2) --------------------------------
                S_FCRET: if (dbg_stop==2'd2 && blk==4'd0) st<=S_FIN;  // DEBUG: stop after LN2
                  else begin
                    g_wbase<=blk*GW_BLK + WB_FC; g_m<=D_MLP[10:0]; g_k<=D[10:0]; g_asrc<=2'd2;
                    g_asel<=blk*4 + 6'd2; g_frac<=7'd12; g_dqrow<=blk*DQB_P + DR_FC; g_dst<=3'd2;
                    g_ret<=S_GELU; ci<=0; civ<=0; gfr<=0; gfrv<=0; gor<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ================= GELU (vec_gelu over mlpbuf, sync read) =====
                S_GELU: begin
                    gfrd <= gfr; gfrv <= (gfr != ROWSM[$clog2(ROWSM+1)-1:0]);
                    if (gfr != ROWSM[$clog2(ROWSM+1)-1:0]) gfr <= gfr + 1'b1;
                    gl_vin <= gfrv;
                    if (gfrv)
                        for (pp=0; pp<P; pp=pp+1)
                            gl_x[pp*16 +: 16] <= mlpbuf_r[pp*64 +: 16];  // low 16 of each Q4.12 lane
                    if (gl_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            gl_sh = $signed(gl_y[pp*16 +: 16]) <<< (LN_OUT_FRAC-GELU_FRAC);
                            gsw[pp*64 +: 64] = gl_sh;
                        end
                        mlpbuf_bank[gor] <= gsw;
                        if (gor==ROWSM-1) begin                   // -> mlp_proj GEMV
                            g_wbase<=blk*GW_BLK + WB_MP; g_m<=D[10:0]; g_k<=D_MLP[10:0]; g_asrc<=2'd3;
                            g_asel<=blk*4 + 6'd3; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_MP; g_dst<=3'd3;
                            g_ret<=S_RES2; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else gor<=gor+1'b1;
                    end
                end
                // ---- res2: xres += mlp_out ; next block or final LN_f --------
                S_RES2: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(mlp_r[pp*32 +: 32]);
                        xres_bank[cid] <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (dbg_stop==2'd3 && blk==4'd0) st<=S_FIN; // DEBUG: stop after block 0
                            else if (blk == NLAYER-1) begin            // -> final LN_f -> head
                                l_gbase<=NLAYER*2; l_dst<=1'b0; l_ret<=S_HEADSET; st<=L_GAM;
                            end else begin                             // -> next block LN1
                                blk<=blk+1'b1; l_gbase<=(blk+1'b1)*2; l_dst<=1'b0;
                                l_ret<=S_QKVRET; st<=L_GAM;
                            end
                        end
                    end
                end
                // ---- head GEMV (act = LN_f out in lnout1) -> head_bank -------
                S_HEADSET: begin
                    g_wbase<=WB_HEAD; g_m<=VOCAB[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=4*NLAYER; g_frac<=7'd25; g_dqrow<=DR_HEAD[11:0]; g_dst<=3'd4;
                    g_ret<=S_ARGMAX; ci<=0; civ<=0; gv_ldrst<=1'b1;
                    best_val<=32'sh80000000; best_idx<=9'd0; ar<=0; arv<=0; st<=G_AQ;
                end
                // ---- P-wide argmax over the VOCAB logits (Q6.25, sync read) --
                S_ARGMAX: begin
                    ard <= ar; arv <= (ar != ARROWS[$clog2(ARROWS+1)-1:0]);
                    if (ar != ARROWS[$clog2(ARROWS+1)-1:0]) ar <= ar + 1'b1;
                    if (arv) begin
                        hw = head_r;
                        for (pp=0; pp<P; pp=pp+1) begin
                            hv   = hw[pp*32 +: 32];
                            hidx = ard*P + pp;
                            if (hidx < VOCAB && $signed(hv) > best_val) begin
                                best_val = $signed(hv); best_idx = hidx;
                            end
                        end
                        if (ard==ARROWS-1) begin tok_out<=best_idx; st<=S_FIN; end
                    end
                end
                S_FIN: begin done<=1'b1; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end

    // ---- readback (2-cyc): bank read registers above are stage 1 (address = rbr
    // while idle); stage 2 muxes the selected word + extracts the lane.
    reg [LSH-1:0] rd_lane;
    reg [P*64-1:0] rw64; reg [P*32-1:0] rw32; reg is64;
    always @* begin
        is64 = 1'b0; rw64 = {(P*64){1'b0}}; rw32 = {(P*32){1'b0}};
        case (rd_sel)
            4'd0: begin rw64 = lnout1_r; is64 = 1'b1; end   // ln1   (Q.22)
            4'd1: rw32 = qkv_r;                             // qkv   (Q.16)
            4'd2: rw32 = ctxv_r;                            // ctx   (Q6.25)
            4'd3: rw32 = attn_r;                            // attn_out (Q6.25)
            4'd4: begin rw64 = lnout2_r; is64 = 1'b1; end   // ln2   (Q.22)
            4'd5: begin rw64 = mlpbuf_r; is64 = 1'b1; end   // gelu  (Q.22)
            4'd6: rw32 = mlp_r;                             // mlp_out (Q6.25)
            4'd7: rw32 = xres_r;                            // x4 residual (Q6.25)
            default: rw32 = head_r;                         // 8: head logits (Q6.25)
        endcase
    end
    always @(posedge clk) begin
        rd_lane <= rd_addr[LSH-1:0];
        if (is64) rd_data <= $signed(rw64[rd_lane*64 +: 64]);
        else      rd_data <= {{32{rw32[rd_lane*32 + 31]}}, rw32[rd_lane*32 +: 32]};
    end
endmodule
