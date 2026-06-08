// -----------------------------------------------------------------------------
// sequencer_gemm — N-STREAM batched sequencer (keystroke speculative decoding).
//
// sequencer_vec with every scratch bank stream-flattened (row = stream*ROWS + r)
// and every boundary phase looped over streams. The GEMV becomes a GEMM
// (gemm_banked_resident_vec): ONE weight pass serves N tokens — the URAM read
// bandwidth (12.8k words/token) is shared, so the floor is 25.6k cycles per
// N=4 tokens at LANES=128 (~17k tok/s aggregate at 200 MHz). The non-linears
// run round-robin per stream — overlap with the GEMM run is the next lever.
//
// Bit-exact per stream vs seq_ref full_forward (same per-stream arithmetic;
// streams only share weights, never intermediates).
//
// iverilog-2012 safe: same conventions as sequencer_vec.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_gemm #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 128,
    parameter integer N     = 4,
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
    parameter integer ISH         = 40
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    input  wire [N*9-1:0] tok_ids,   // stream b in bits [b*9 +: 9]
    input  wire [8:0]  pos,
    output reg         done,
    output wire [N*9-1:0] tok_outs,
    // board readback (stream select in rd_stream, bank in rd_sel)
    input  wire [$clog2(N)-1:0] rd_stream,
    input  wire [3:0]  rd_sel,
    input  wire [10:0] rd_addr,
    output reg signed [63:0] rd_data,
    input  wire        wl_rst,
    input  wire        wl_we,
    input  wire        el_we,        // embed-table load strobe (same wl_data path)
    input  wire [31:0] wl_data,
    input  wire [1:0]  dbg_stop
);
    localparam integer ROWS  = D    / P;
    localparam integer ROWS3 = D3   / P;
    localparam integer ROWSM = D_MLP/ P;
    localparam integer LSH   = $clog2(P);
    localparam integer NSH   = (N > 1) ? $clog2(N) : 1;
    localparam integer DQROWS = (DQ_N + P - 1) / P;
    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;
    localparam integer WB_QKV = 0, WB_PROJ = GW_QKV,
                       WB_FC = GW_QKV+GW_PROJ, WB_MP = GW_QKV+GW_PROJ+GW_FC;
    localparam integer DR_QKV = 0, DR_PROJ = D3/P, DR_FC = (D3+D)/P, DR_MP = (D3+D+D_MLP)/P;
    localparam integer GW_MP   = ((D + LANES-1)/LANES) * D_MLP;
    localparam integer GW_HEAD = ((VOCAB + LANES-1)/LANES) * D;
    localparam integer GW_BLK  = GW_QKV+GW_PROJ+GW_FC+GW_MP;
    localparam integer DQ_BLK  = D3+D+D_MLP+D;
    localparam integer DQB_P   = DQ_BLK/P;
    localparam integer WB_HEAD = NLAYER*GW_BLK;
    localparam integer DR_HEAD = (NLAYER*DQ_BLK)/P;
    localparam integer ARROWS  = (VOCAB + P - 1)/P;
    localparam integer EROWS   = D / P;
    localparam integer HR      = HEAD_DIM / P;

    // ---- shared ROMs (one copy serves all streams round-robin) -----------------
    // The embeds are the BRAM budget: ~52 tiles at N=4 (the scratch needs them).
    // They live in URAM instead (no bitstream init!) and are STREAMED at boot
    // through the wl port after the weight image (el_we strobes, same 32-bit
    // chunks, EROWS-aligned). In sim, $readmemh preloads them directly.
    // tok table SDP URAM (write port = boot stream, read port = embed) — TDP URAM
    // never maps (NO_CHANGE rule), so pos lives in its own small BRAM.
    localparam integer EMB_ROWS = (VOCAB + TMAX) * EROWS;
    (* ram_style = "ultra" *) reg [P*32-1:0] emb_w [0:VOCAB*EROWS-1];
    (* ram_style = "block" *) reg [P*32-1:0] pos_w [0:TMAX*EROWS-1];
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w   [0:GAMMA_N*EROWS-1];
    reg signed [63:0] inv_sact [0:NSACT-1];
    // distributed: BRAM budget is the binding constraint at N=4; ~5k LUT buys 15 tiles
    (* rom_style = "distributed" *) reg [P*24-1:0] dqm_w [0:DQROWS-1];
    (* rom_style = "distributed" *) reg [P*8-1:0]  dqe_w [0:DQROWS-1];
`ifndef SYNTHESIS
    initial begin
        $readmemh("tok_emb_w.mem", emb_w);
        $readmemh("pos_emb_w.mem", pos_w, 0, TMAX*EROWS-1);
    end
`endif
    initial begin
        $readmemh("gamma_w.mem",   gamma_w);
        $readmemh("inv_sact.mem",  inv_sact);
        $readmemh("dqm_w.mem",     dqm_w);
        $readmemh("dqe_w.mem",     dqe_w);
    end

    // ---- embed loader: 32-bit chunks -> P*32 words, tok_emb then pos_emb -------
    localparam integer ESUB = (P*32)/32;
    reg [$clog2(EMB_ROWS):0] el_word;
    reg [$clog2(ESUB)-1:0]      el_sub;
    reg [P*32-1:0]              el_buf;
    wire [P*32-1:0] el_next = el_buf | ({{(P*32-32){1'b0}}, wl_data} << (el_sub*32));
    always @(posedge clk) begin
        if (wl_rst) begin el_word <= 0; el_sub <= 0; el_buf <= 0; end
        else if (el_we) begin
            if (el_sub == ESUB-1) begin
                el_word <= el_word + 1'b1; el_sub <= 0; el_buf <= 0;
            end else begin
                el_buf <= el_next; el_sub <= el_sub + 1'b1;
            end
        end
    end
    wire el_commit = el_we && (el_sub == ESUB-1);

    // tok URAM write port (boot stream); pos table is its own BRAM (write + read)
    always @(posedge clk) begin
        if (el_commit && el_word < VOCAB*EROWS)
            emb_w[el_word] <= el_next;
    end
    always @(posedge clk) begin
        if (el_commit && el_word >= VOCAB*EROWS)
            pos_w[el_word - VOCAB*EROWS] <= el_next;
        else
            pemb_r <= pos_w[pos*EROWS + fr];
    end

    // ---- stream-flattened scratch (row = stream*ROWS + r). Depths pad to >=512:
    // BRAM needs >=512 rows, shallower arrays fall back to LUTRAM (~kLUTs each).
    localparam integer BD = 512;
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank  [0:BD-1];
    (* ram_style = "block" *) reg [P*64-1:0] lnout1_bank[0:BD-1];
    (* ram_style = "block" *) reg [P*64-1:0] lnout2_bank[0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] qkv_bank   [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] ctxv_bank  [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] attn_bank  [0:BD-1];
    (* ram_style = "block" *) reg [P*64-1:0] mlpbuf_bank[0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] mlp_bank   [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] head_bank  [0:BD-1];

    reg [P*32-1:0] xres_r, qkv_r, ctxv_r, attn_r, mlp_r, head_r;
    reg [P*64-1:0] lnout1_r, lnout2_r, mlpbuf_r;
    reg [P*32-1:0] temb_r, pemb_r, gam_r;

    // ---- layernorm_vec (round-robin over streams) -------------------------------
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    // ---- GEMM (batched resident core) -------------------------------------------
    reg                gv_ldrst, gv_xrst, gv_xwe, gv_start;
    reg  [NSH-1:0]     gv_xstream, gv_rdstream;
    reg  [P*8-1:0]     gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    reg [10:0]         gv_rdaddr;
    wire [P*32-1:0]    gv_yout;
    gemm_banked_resident_vec #(.LANES(LANES), .N(N), .P(P), .MMAX(1024), .KMAX(1024),
                  .RLAT(2), .WWORDS(WWORDS)) u_gemm (
        .clk(clk), .clk2x(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ldrst | wl_rst), .w_we(wl_we), .w_data(wl_data),
        .x_rst(gv_xrst), .x_we(gv_xwe), .x_stream(gv_xstream), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done),
        .rd_stream(gv_rdstream), .rd_addr(gv_rdaddr[$clog2(1024/P)-1:0]), .y_out(gv_yout));

    // ---- vec_dequant -------------------------------------------------------------
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

    // ---- vec_gelu ----------------------------------------------------------------
    reg             gl_vin;
    reg [P*16-1:0]  gl_x;
    wire            gl_vout;
    wire [P*16-1:0] gl_y;
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(gl_vin), .x(gl_x),
        .out_valid(gl_vout), .y(gl_y));

    // ---- vec_attn (one head x one stream at a time) ------------------------------
    reg               at_start, at_ldv;
    reg [8:0]         at_tcount;
    wire              at_ldready, at_ctxv, at_done;
    wire [6:0]        at_ctxidx;
    wire [P*32-1:0]   at_ctxdata;
    reg [1:0]  hh;
    reg [8:0]  wi, wic;
    reg        wiv;
    wire [10:0] aw_src = (wi < HR)   ? (hh*HR + wi) :
                         (wi < 2*HR) ? (D/P   + hh*HR + (wi - HR)) :
                                       (2*D/P + hh*HR + (wi - 2*HR));
    wire [P*32-1:0] aw_data = qkv_r;
    vec_attn #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(32)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .ld_valid(at_ldv), .ld_data(aw_data), .ld_ready(at_ldready),
        .ctx_valid(at_ctxv), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata), .done(at_done));

    // ---- GEMM call parameters ----------------------------------------------------
    reg [19:0] g_wbase;
    reg [10:0] g_m, g_k;
    reg [1:0]  g_asrc;
    reg [5:0]  g_asel;
    reg signed [6:0] g_frac;
    reg [11:0] g_dqrow;
    reg [2:0]  g_dst;
    reg [4:0]  g_ret;
    reg [3:0]  l_gbase;
    reg        l_dst;
    reg [4:0]  l_ret;

    // ---- temporaries ---------------------------------------------------------------
    reg signed [63:0]  lntmp;
    reg signed [95:0]  aq_prod, aq_sh;
    reg signed [31:0]  aq_int;
    reg signed [95:0]  aq_prod_r [0:P-1];
    reg                aq_neg_r  [0:P-1];
    reg signed [63:0]  lnt_r     [0:P-1];
    reg [10:0]         cid, cid1, cid2;  reg civ, civ1, civ2;
    reg signed [31:0]  cb, dqv, hv;
    reg [P*32-1:0]  ww, sw, dword, hw;
    reg [P*8-1:0]   aqw;
    reg [P*16-1:0]  mword;
    reg [P*64-1:0]  gsw;
    reg [P*24-1:0]  mwr;
    reg [P*8-1:0]   ewr;
    reg signed [63:0] gl_sh;
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    reg [$clog2(ROWSM+1)-1:0] gor;
    integer pp;
    reg [3:0] blk;
    reg signed [31:0] best_val; reg [8:0] best_idx, hidx;
    reg [$clog2(ARROWS+1)-1:0] ar;
    reg signed [31:0] wm_val;  reg [8:0] wm_idx;
    reg signed [31:0] am_val;  reg [8:0] am_idx;
    reg [$clog2(ARROWS+1)-1:0] amd;  reg amv;
    reg [P*32-1:0] hw_r;
    reg signed [31:0] pv0, pv1, pv2, pv3;
    reg [8:0]         pi0, pi1, pi2, pi3;
    reg [$clog2(ARROWS+1)-1:0] ad1;  reg av1;
    reg signed [31:0] va, vb;
    reg [8:0]         ia, ib;

    // ---- batch stream counter ------------------------------------------------------
    reg [NSH-1:0] bs;                        // current stream in round-robin phases

    // per-stream argmax results (variable part-select on a packed output is an
    // iverilog NBA trap — the base is evaluated at update time, all writes land
    // in the LAST stream's slot)
    reg [8:0] tok_out_b [0:N-1];
    genvar gtb;
    generate
        for (gtb = 0; gtb < N; gtb = gtb + 1) begin : g_tok
            assign tok_outs[gtb*9 +: 9] = tok_out_b[gtb];
        end
    endgenerate

    // ---- FSM -------------------------------------------------------------------------
    localparam [4:0]
      S_IDLE=0, S_EMB=1,
      L_GAM=2, L_FEED=3, L_COLL=4,
      G_AQ=5, G_RUN=6, G_WAIT=7, G_RB=8,
      S_QKVRET=10, S_AST=11, S_ALD=12, S_ACL=13,
      S_RES1=14, S_FCRET=16,
      S_MPSET=17, S_RES2=20, S_FIN=21,
      S_HEADSET=22, S_ARGMAX=23, S_ARGNEXT=24, G_RBN=25, G_AQN=26;
    reg [4:0] st;
    reg [2:0] rbn;                         // dequant-tail drain counter
    reg [10:0] ci;
    reg [$clog2(ROWSM+1)-1:0] fr, orow, dor;
    reg [$clog2(ROWSM+1)-1:0] frd;  reg frv;
    reg [$clog2(ARROWS+1)-1:0] ard;  reg arv;

    wire [8:0] tok_id_b = tok_ids[bs*9 +: 9];

    // ---- synchronous reads (stream-flattened addressing) ---------------------------
    wire [10:0] rbr  = rd_stream*ROWS   + (rd_addr >> LSH);
    wire [10:0] rbr3 = rd_stream*ROWS3  + (rd_addr >> LSH);
    wire [10:0] rbrM = rd_stream*ROWSM  + (rd_addr >> LSH);
    wire [10:0] rbrA = rd_stream*ARROWS + (rd_addr >> LSH);
    wire [10:0] bR  = bs*ROWS;
    wire [10:0] xres_ra   = (st==L_FEED) ? bR + {{(11-$clog2(ROWSM+1)){1'b0}}, fr} :
                            (st==S_RES1 || st==S_RES2) ? bR + ci : rbr;
    wire [10:0] lnout1_ra = (st==G_AQ) ? bR + ci : rbr;
    wire [10:0] lnout2_ra = (st==G_AQ) ? bR + ci : rbr;
    wire [10:0] ctxv_ra   = (st==G_AQ) ? bR + ci : rbr;
    wire [10:0] mlpbuf_ra = (st==G_AQ) ? bs*ROWSM + ci : rbrM;
    wire [10:0] qkv_ra    = (st==S_ALD) ? bs*ROWS3 + aw_src : rbr3;
    wire [10:0] attn_ra   = (st==S_RES1) ? bR + ci : rbr;
    wire [10:0] head_ra   = (st==S_ARGMAX)
                          ? bs*ARROWS + {{(11-$clog2(ARROWS+1)){1'b0}}, ar} : rbrA;
    wire [10:0] mlp_ra    = (st==S_RES2) ? bR + ci : rbr;

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
        temb_r   <= emb_w[tok_id_b*EROWS + fr];   // emb_w port A: tok read
        gam_r    <= gamma_w  [l_gbase*EROWS + fr];
    end

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; gv_ldrst<=1'b0; gv_xrst<=1'b0; gv_xwe<=1'b0;
        gv_start<=1'b0; dq_vin<=1'b0; gl_vin<=1'b0; at_start<=1'b0; at_ldv<=1'b0; done<=1'b0;
        if (rst) begin
            st<=S_IDLE; ci<=0; fr<=0; orow<=0; dor<=0; gor<=0; bs<=0;
            rv0<=0; rv1<=0; rv2<=0;
            civ<=0; civ1<=0; civ2<=0; frv<=0; arv<=0; wiv<=0;
        end else begin
            case (st)
                S_IDLE: if (go) begin
                    fr<=0; frv<=0; blk<=4'd0; bs<=0; st<=S_EMB;
                end
                // ---- embed for stream bs; loop streams, then LN1 of block 0 ----
                S_EMB: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    if (frv) begin
                        for (pp=0; pp<P; pp=pp+1)
                            ww[pp*32 +: 32] = $signed(temb_r[pp*32 +: 32])
                                            + $signed(pemb_r[pp*32 +: 32]);
                        xres_bank[bs*ROWS + frd] <= ww;
                        if (frd==ROWS-1) begin
                            fr<=0; frv<=0;
                            if (bs == N-1) begin
                                bs<=0;
                                if (dbg_stop==2'd1) st<=S_FIN;
                                else begin l_gbase<=4'd0; l_dst<=1'b0; l_ret<=S_QKVRET; st<=L_GAM; end
                            end else bs <= bs + 1'b1;
                        end
                    end
                end
                // ============ callable LayerNorm (round-robin streams) =========
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
                        if (l_dst==1'b0) lnout1_bank[bs*ROWS + orow] <= ln_y;
                        else             lnout2_bank[bs*ROWS + orow] <= ln_y;
                        if (orow==ROWS-1) begin
                            if (bs == N-1) begin bs<=0; st<=l_ret; end
                            else begin bs<=bs+1'b1; st<=L_GAM; end
                        end else orow<=orow+1'b1;
                    end
                end
                // qkv GEMM setup (after LN1) ------------------------------------
                S_QKVRET: begin
                    g_wbase<=blk*GW_BLK + WB_QKV; g_m<=D3[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=blk*4 + 6'd0; g_frac<=7'd16; g_dqrow<=blk*DQB_P + DR_QKV; g_dst<=3'd0;
                    g_ret<=S_AST; ci<=0; civ<=0; hh<=2'd0; bs<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ============ callable GEMM: act-quant N streams round-robin ===
                G_AQ: begin
                    cid <= ci; civ <= (ci != (g_k >> LSH));
                    cid1 <= cid; civ1 <= civ;
                    cid2 <= cid1; civ2 <= civ1;
                    if (ci != (g_k >> LSH)) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            case (g_asrc)
                                2'd0: lntmp = $signed(lnout1_r[pp*64 +: 64]);
                                2'd1: begin
                                    cb = ctxv_r[pp*32 +: 32];
                                    if ($signed({{32{cb[31]}}, cb}) >= 0)
                                        lntmp = ($signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC);
                                    else
                                        lntmp = -((-$signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC));
                                end
                                2'd2: lntmp = $signed(lnout2_r[pp*64 +: 64]);
                                default: lntmp = $signed(mlpbuf_r[pp*64 +: 64]);
                            endcase
                            lnt_r[pp] <= lntmp;
                        end
                    end
                    if (civ1) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_prod_r[pp] <= $signed(lnt_r[pp]) * $signed(inv_sact[g_asel]);
                            aq_neg_r[pp]  <= (lnt_r[pp] < 0);
                        end
                    end
                    if (civ2) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_prod = aq_prod_r[pp];
                            if (!aq_neg_r[pp])
                                aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                            else
                                aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                            aq_int = aq_sh[31:0];
                            if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                            aqw[pp*8 +: 8] = aq_int[7:0];
                        end
                        gv_xwe<=1'b1; gv_xdata<=aqw; gv_xstream<=bs;
                        if (cid2==(g_k >> LSH)-1) begin
                            ci<=0; civ<=0; civ1<=0; civ2<=0;
                            if (bs == N-1) begin bs<=0; st<=G_RUN; end
                            else begin rbn<=0; st<=G_AQN; end
                        end
                    end
                end
                G_AQN: begin
                    // drain the act-quant pipeline before switching streams — the
                    // in-flight xwe of the OLD stream must commit before x_rst
                    // rewinds the row pointer (else the tail lands at row 0 of the
                    // NEXT stream and every following row shifts by one).
                    rbn <= rbn + 1'b1;
                    if (rbn == 3'd3) begin
                        bs<=bs+1'b1; gv_xrst<=1'b1; st<=G_AQ;
                    end
                end
                G_RUN: begin
                    gv_m<=g_m; gv_k<=g_k; gv_wbase<=g_wbase[$clog2(WWORDS)-1:0];
                    gv_start<=1'b1; st<=G_WAIT;
                end
                G_WAIT: if (gv_done) begin
                    ci<=0; gv_rdaddr<=0; gv_rdstream<=0; rv0<=0; rv1<=0; rv2<=0; dor<=0; st<=G_RB;
                end
                G_RB: begin
                    // fused readback->dequant->dest for stream bs; loop streams
                    if (ci < ((g_m + P-1) >> LSH)) begin
                        gv_rdaddr<=ci; rb0<=ci; ci<=ci+1'b1;
                    end else gv_rdaddr <= ((g_m + P-1) >> LSH) - 1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(ci < ((g_m + P-1) >> LSH)); rv1<=rv0; rv2<=rv1;
                    mwr <= dqm_w[g_dqrow + rb1];
                    ewr <= dqe_w[g_dqrow + rb1];
                    if (rv2) begin
                        dq_vin<=1'b1; dq_frac<=g_frac;
                        dq_gemvy <= gv_yout;
                        dq_mant  <= mwr;
                        dq_exp   <= ewr;
                    end
                    if (dq_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            dqv = dq_out[pp*32 +: 32];
                            dword[pp*32 +: 32] = dqv;
                            if      ($signed(dqv) >  32'sd32767)  mword[pp*16 +: 16] = 16'sd32767;
                            else if ($signed(dqv) < -32'sd32768)  mword[pp*16 +: 16] = -16'sd32768;
                            else    mword[pp*16 +: 16] = dqv[15:0];
                        end
                        case (g_dst)
                            3'd0: qkv_bank [bs*ROWS3 + dor] <= dword;
                            3'd1: attn_bank[bs*ROWS  + dor] <= dword;
                            3'd2: begin gl_vin<=1'b1; gl_x<=mword; end
                            3'd3: mlp_bank [bs*ROWS  + dor] <= dword;
                            default: head_bank[bs*ARROWS + dor] <= dword;
                        endcase
                        if (g_dst != 3'd2 && dor==((g_m + P-1) >> LSH)-1) begin
                            ci<=0; civ<=0; dor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (bs == N-1) begin bs<=0; st<=g_ret; end
                            else begin rbn<=0; st<=G_RBN; end
                        end else dor<=dor+1'b1;
                    end
                    if (gl_vout) begin                     // GELU collect (g_dst==2)
                        for (pp=0; pp<P; pp=pp+1) begin
                            gl_sh = $signed(gl_y[pp*16 +: 16]) <<< (LN_OUT_FRAC-GELU_FRAC);
                            gsw[pp*64 +: 64] = gl_sh;
                        end
                        mlpbuf_bank[bs*ROWSM + gor] <= gsw;
                        if (gor==ROWSM-1) begin
                            ci<=0; civ<=0; gor<=0; dor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (bs == N-1) begin bs<=0; st<=g_ret; end
                            else begin rbn<=0; st<=G_RBN; end
                        end else gor<=gor+1'b1;
                    end
                end
                G_RBN: begin
                    // drain the dequant/GELU tail before switching streams — an
                    // in-flight dq_vout after the switch would write rows 0..1 of
                    // the NEXT stream's bank (the X corruption gate-caught above).
                    rbn <= rbn + 1'b1;
                    if (rbn == 3'd7) begin
                        bs<=bs+1'b1; gv_rdstream<=bs+1'b1; st<=G_RB;
                    end
                end
                // ============ attention: heads x streams ======================
                S_AST: begin
                    at_start<=1'b1; at_tcount<=9'd1; wi<=9'd0; wic<=9'd0; wiv<=1'b0; st<=S_ALD;
                end
                S_ALD: begin
                    at_ldv <= (wi != 3*HR);
                    if (wi != 3*HR) wi <= wi + 1'b1;
                    if (at_ldv) begin
                        if (wic == 3*HR-1) st <= S_ACL;
                        else wic <= wic + 1'b1;
                    end
                end
                S_ACL: begin
                    if (at_ctxv)
                        ctxv_bank[bs*ROWS + hh*HR + at_ctxidx] <= at_ctxdata;
                    if (at_done) begin
                        if (hh==NHEAD-1) begin
                            if (bs == N-1) begin
                                bs<=0; hh<=2'd0;
                                g_wbase<=blk*GW_BLK + WB_PROJ; g_m<=D[10:0]; g_k<=D[10:0]; g_asrc<=2'd1;
                                g_asel<=blk*4 + 6'd1; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_PROJ; g_dst<=3'd1;
                                g_ret<=S_RES1; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                            end else begin bs<=bs+1'b1; hh<=2'd0; st<=S_AST; end
                        end else begin hh<=hh+2'd1; st<=S_AST; end
                    end
                end
                // ---- res1 (per stream) ; then LN2 -----------------------------
                S_RES1: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(attn_r[pp*32 +: 32]);
                        xres_bank[bs*ROWS + cid] <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (bs == N-1) begin
                                bs<=0;
                                l_gbase<=blk*2 + 4'd1; l_dst<=1'b1; l_ret<=S_FCRET; st<=L_GAM;
                            end else bs<=bs+1'b1;
                        end
                    end
                end
                S_FCRET: if (dbg_stop==2'd2 && blk==4'd0) st<=S_FIN;
                  else begin
                    g_wbase<=blk*GW_BLK + WB_FC; g_m<=D_MLP[10:0]; g_k<=D[10:0]; g_asrc<=2'd2;
                    g_asel<=blk*4 + 6'd2; g_frac<=7'd12; g_dqrow<=blk*DQB_P + DR_FC; g_dst<=3'd2;
                    g_ret<=S_MPSET; ci<=0; civ<=0; gor<=0; bs<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                S_MPSET: begin
                    g_wbase<=blk*GW_BLK + WB_MP; g_m<=D[10:0]; g_k<=D_MLP[10:0]; g_asrc<=2'd3;
                    g_asel<=blk*4 + 6'd3; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_MP; g_dst<=3'd3;
                    g_ret<=S_RES2; ci<=0; civ<=0; bs<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ---- res2 (per stream) ; next block / LN_f --------------------
                S_RES2: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(mlp_r[pp*32 +: 32]);
                        xres_bank[bs*ROWS + cid] <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (bs == N-1) begin
                                bs<=0;
                                if (dbg_stop==2'd3 && blk==4'd0) st<=S_FIN;
                                else if (blk == NLAYER-1) begin
                                    l_gbase<=NLAYER*2; l_dst<=1'b0; l_ret<=S_HEADSET; st<=L_GAM;
                                end else begin
                                    blk<=blk+1'b1; l_gbase<=(blk+1'b1)*2; l_dst<=1'b0;
                                    l_ret<=S_QKVRET; st<=L_GAM;
                                end
                            end else bs<=bs+1'b1;
                        end
                    end
                end
                // ---- head GEMM -> per-stream argmax ---------------------------
                S_HEADSET: begin
                    g_wbase<=WB_HEAD; g_m<=VOCAB[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=4*NLAYER; g_frac<=7'd25; g_dqrow<=DR_HEAD[11:0]; g_dst<=3'd4;
                    g_ret<=S_ARGMAX; ci<=0; civ<=0; bs<=0; gv_ldrst<=1'b1;
                    best_val<=32'sh80000000; best_idx<=9'd0; ar<=0; arv<=0; av1<=0; amv<=0; st<=G_AQ;
                end
                S_ARGMAX: begin
                    ard <= ar; arv <= (ar != ARROWS[$clog2(ARROWS+1)-1:0]);
                    if (ar != ARROWS[$clog2(ARROWS+1)-1:0]) ar <= ar + 1'b1;
                    hw_r <= head_r; ad1 <= ard; av1 <= arv;
                    if (av1) begin
                        for (pp=0; pp<4; pp=pp+1) begin
                            va = $signed(hw_r[pp*32 +: 32]);
                            vb = $signed(hw_r[(pp+4)*32 +: 32]);
                            ia = ad1*P + pp;  ib = ad1*P + pp + 4;
                            if (ia >= VOCAB) va = 32'sh80000000;
                            if (ib >= VOCAB) vb = 32'sh80000000;
                            case (pp)
                                0: begin pv0 <= (vb > va) ? vb : va; pi0 <= (vb > va) ? ib : ia; end
                                1: begin pv1 <= (vb > va) ? vb : va; pi1 <= (vb > va) ? ib : ia; end
                                2: begin pv2 <= (vb > va) ? vb : va; pi2 <= (vb > va) ? ib : ia; end
                                default: begin pv3 <= (vb > va) ? vb : va; pi3 <= (vb > va) ? ib : ia; end
                            endcase
                        end
                    end
                    amv <= av1; amd <= ad1;
                    if (amv) begin
                        wm_val = pv0; wm_idx = pi0;
                        if (pv1 > wm_val) begin wm_val = pv1; wm_idx = pi1; end
                        if (pv2 > wm_val) begin wm_val = pv2; wm_idx = pi2; end
                        if (pv3 > wm_val) begin wm_val = pv3; wm_idx = pi3; end
                        if (wm_val > best_val) begin
                            best_val <= wm_val; best_idx <= wm_idx;
                        end
                        if (amd==ARROWS-1) begin
                            tok_out_b[bs] <= (wm_val > best_val) ? wm_idx : best_idx;
                            st <= S_ARGNEXT;
                        end
                    end
                end
                S_ARGNEXT: begin
                    if (bs == N-1) begin bs<=0; st<=S_FIN; end
                    else begin
                        bs<=bs+1'b1;
                        best_val<=32'sh80000000; best_idx<=9'd0;
                        ar<=0; arv<=0; av1<=0; amv<=0; st<=S_ARGMAX;
                    end
                end
                S_FIN: begin done<=1'b1; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end

    // ---- board readback (2-cyc): stream picked by rd_stream -------------------
    reg [LSH-1:0] rd_lane;
    reg [P*64-1:0] rw64; reg [P*32-1:0] rw32; reg is64;
    always @* begin
        is64 = 1'b0; rw64 = {(P*64){1'b0}}; rw32 = {(P*32){1'b0}};
        case (rd_sel)
            4'd0: begin rw64 = lnout1_r; is64 = 1'b1; end
            4'd1: rw32 = qkv_r;
            4'd2: rw32 = ctxv_r;
            4'd3: rw32 = attn_r;
            4'd4: begin rw64 = lnout2_r; is64 = 1'b1; end
            4'd5: begin rw64 = mlpbuf_r; is64 = 1'b1; end
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
endmodule
