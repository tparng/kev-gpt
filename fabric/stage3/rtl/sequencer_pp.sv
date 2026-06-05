// -----------------------------------------------------------------------------
// sequencer_pp — PING-PONG batch sequencer: N=8 streams, two groups of 4.
//
// While the GEMM engine streams the resident URAM weight image for group A,
// the NL engine runs the non-linears (embed/LN/attention/residual/argmax) for
// group B, then they swap. One weight pass feeds 4 token streams; the URAM
// read port never idles. ~32.7k cycles per 8 tokens = ~4.1k cyc/token.
//
// Engines:
//   GEMM: act-quant feed (4 streams) -> run -> fused readback/dequant/GELU.
//   NL:   embed / LayerNorm / attention / residual / argmax, group-blocked
//         on the GEMM via a request/done handshake (one in-flight call/group).
//
// Bank discipline: every bank has exactly one writing engine and one reading
// engine, on disjoint groups - no port conflicts (TDP BRAM, one port each).
//
// Bit-exact per stream vs seq_ref.full_forward_signals (same arithmetic).
// iverilog-2012 traps honoured (plain-reg part-selects, generate MAC banks).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_pp #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 128,
    parameter integer N     = 8,        // total streams (= 2 groups of G)
    parameter integer G     = 4,        // streams per group
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
    input  wire [N*9-1:0] tok_ids,
    input  wire [8:0]  pos,
    output reg         done,
    output wire [N*9-1:0] tok_outs,
    input  wire [$clog2(N)-1:0] rd_stream,
    input  wire [3:0]  rd_sel,
    input  wire [10:0] rd_addr,
    output reg signed [63:0] rd_data,
    input  wire        wl_rst,
    input  wire        wl_we,
    input  wire        el_we,
    input  wire [31:0] wl_data,
    input  wire [1:0]  dbg_stop
);
    localparam integer ROWS  = D    / P;
    localparam integer ROWS3 = D3   / P;
    localparam integer ROWSM = D_MLP/ P;
    localparam integer LSH   = $clog2(P);
    localparam integer GSH   = $clog2(G);
    localparam integer DQROWS = (DQ_N + P - 1) / P;
    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;
    localparam integer WB_QKV = 0, WB_PROJ = GW_QKV,
                       WB_FC = GW_QKV+GW_PROJ, WB_MP = GW_QKV+GW_PROJ+GW_FC;
    localparam integer DR_QKV = 0, DR_PROJ = D3/P, DR_FC = (D3+D)/P, DR_MP = (D3+D+D_MLP)/P;
    localparam integer GW_MP   = ((D + LANES-1)/LANES) * D_MLP;
    localparam integer GW_BLK  = GW_QKV+GW_PROJ+GW_FC+GW_MP;
    localparam integer DQ_BLK  = D3+D+D_MLP+D;
    localparam integer DQB_P   = DQ_BLK/P;
    localparam integer WB_HEAD = NLAYER*GW_BLK;
    localparam integer DR_HEAD = (NLAYER*DQ_BLK)/P;
    localparam integer ARROWS  = (VOCAB + P - 1)/P;
    localparam integer EROWS   = D / P;
    localparam integer HR      = HEAD_DIM / P;
    localparam integer BD      = 1024;      // bank depth (N * ROWSM rows worst case)

    // ---- shared ROMs ------------------------------------------------------------
    localparam integer EMB_ROWS = (VOCAB + TMAX) * EROWS;
    (* ram_style = "ultra" *) reg [P*32-1:0] emb_w [0:VOCAB*EROWS-1];
    (* ram_style = "block" *) reg [P*32-1:0] pos_w [0:TMAX*EROWS-1];
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w [0:GAMMA_N*EROWS-1];
    reg signed [63:0] inv_sact [0:NSACT-1];
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

    // ---- embed loader (URAM tok + BRAM pos, streamed at boot) --------------------
    localparam integer ESUB = (P*32)/32;
    reg [$clog2(EMB_ROWS):0] el_word;
    reg [$clog2(ESUB)-1:0]   el_sub;
    reg [P*32-1:0]           el_buf;
    wire [P*32-1:0] el_next = el_buf | ({{(P*32-32){1'b0}}, wl_data} << (el_sub*32));
    always @(posedge clk) begin
        if (wl_rst) begin el_word <= 0; el_sub <= 0; el_buf <= 0; end
        else if (el_we) begin
            if (el_sub == ESUB-1) begin el_word <= el_word + 1'b1; el_sub <= 0; el_buf <= 0; end
            else begin el_buf <= el_next; el_sub <= el_sub + 1'b1; end
        end
    end
    wire el_commit = el_we && (el_sub == ESUB-1);
    always @(posedge clk)
        if (el_commit && el_word < VOCAB*EROWS) emb_w[el_word] <= el_next;

    // ---- stream-flattened scratch (row = stream*ROWS + r), depth 1024 ------------
    // One writing engine + one reading engine each; groups are disjoint.
    // LN outputs are Q.22 with |y| < 2^26 -> 32-bit lanes; GELU hidden is Q4.12
    // sat16 -> 16-bit lanes. Sign-extend on the GEMM side to Q.22.
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank  [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] lnout1_bank[0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] lnout2_bank[0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] qkv_bank   [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] ctxv_bank  [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] attn_bank  [0:BD-1];
    (* ram_style = "block" *) reg [P*16-1:0] mlpbuf_bank[0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] mlp_bank   [0:BD-1];
    (* ram_style = "block" *) reg [P*32-1:0] head_bank  [0:BD-1];

    reg [P*32-1:0] xres_r, qkv_r, attn_r, mlp_r, head_r;          // NL-side reads
    reg [P*32-1:0] lnout1_r, lnout2_r;                            // GEMM-side reads
    reg [P*16-1:0] mlpbuf_r;
    reg [P*32-1:0] ctxv_g;
    reg [P*32-1:0] temb_r, pemb_r, gam_r;

    // ================================================================
    //                       NL  ENGINE
    // ================================================================
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    reg               at_start, at_ldv;
    reg [8:0]         at_tcount;
    wire              at_ldready, at_ctxv, at_done;
    wire [6:0]        at_ctxidx;
    wire [P*32-1:0]   at_ctxdata;
    reg [1:0]  hh;
    reg [8:0]  wi, wic;
    wire [10:0] aw_src = (wi < HR)   ? (hh*HR + wi) :
                         (wi < 2*HR) ? (D/P   + hh*HR + (wi - HR)) :
                                       (2*D/P + hh*HR + (wi - 2*HR));
    wire [P*32-1:0] aw_data = qkv_r;
    vec_attn #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(32)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .ld_valid(at_ldv), .ld_data(aw_data), .ld_ready(at_ldready),
        .ctx_valid(at_ctxv), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata), .done(at_done));

    // per-group GEMM-call descriptor + request flag (cleared by the GEMM engine)
    reg  [19:0] d_wbase [0:1];
    reg  [10:0] d_m [0:1], d_k [0:1];
    reg  [1:0]  d_asrc [0:1];
    reg  [5:0]  d_asel [0:1];
    reg signed [6:0] d_frac [0:1];
    reg  [11:0] d_dqrow [0:1];
    reg  [2:0]  d_dst [0:1];
    reg         g_req [0:1];
    wire        g_done_p;            // pulse: current GEMM call fully drained
    wire        rb_last;

    localparam [4:0]
      NL_IDLE=0, NL_EMB=1, NL_LGAM=2, NL_LFEED=3, NL_LCOLL=4,
      NL_QKV=5, NL_WQKV=6, NL_AST=7, NL_ALD=8, NL_ACL=9,
      NL_PROJ=10, NL_WPROJ=11, NL_RES1=12,
      NL_FC=13, NL_WFC=14, NL_MP=15, NL_WMP=16, NL_RES2=17,
      NL_HEAD=18, NL_WHEAD=19, NL_ARG=20, NL_ARG2=21, NL_DONE=22;
    reg [4:0]  nl [0:1];
    reg        gc;                  // group the NL engine is working on
    reg [3:0]  blkg [0:1];
    reg [1:0]  lnphase [0:1];       // 0 = LN1->QKV, 1 = LN2->FC, 2 = LNF->HEAD
    reg        donef [0:1];

    reg [GSH-1:0] bsg [0:1];           // PER-GROUP stream counter (argmax of group A
                                        // and LN of group B run interleaved)
    reg [10:0] ci, cid;  reg civ;
    reg [$clog2(ROWSM+1)-1:0] fr, frd, orow;  reg frv;
    reg [$clog2(ARROWS+1)-1:0] ar, ard, amd, ad1;
    reg arv, av1, amv;
    reg [8:0] wm_idx, am_idx, best_idx;
    reg signed [31:0] wm_val, am_val, best_val;
    reg [P*32-1:0] hw_r;
    reg signed [31:0] pv0, pv1, pv2, pv3;
    reg [8:0]         pi0, pi1, pi2, pi3;
    reg signed [31:0] va, vb;
    reg [8:0]         ia, ib;
    reg [P*32-1:0] ww, sw;
    reg [3:0] lgam;
    integer pp;

    wire [GSH-1:0] bs = bsg[gc];
    wire [8:0] tok_id_b = tok_ids[({gc, bs})*9 +: 9];
    reg [8:0] tok_out_b [0:N-1];
    genvar gtb;
    generate
        for (gtb = 0; gtb < N; gtb = gtb + 1) begin : g_tok
            assign tok_outs[gtb*9 +: 9] = tok_out_b[gtb];
        end
    endgenerate

    // NL-side addressing: stream = {gc, bs}
    wire [10:0] sN  = {gc, bs} * ROWS;
    wire [10:0] sN3 = {gc, bs} * ROWS3;
    wire [10:0] sNA = {gc, bs} * ARROWS;
    wire [10:0] rbr  = rd_stream*ROWS   + (rd_addr >> LSH);
    wire [10:0] rbr3 = rd_stream*ROWS3  + (rd_addr >> LSH);
    wire [10:0] rbrA = rd_stream*ARROWS + (rd_addr >> LSH);

    wire [10:0] xres_ra = (nl[gc]==NL_LFEED) ? sN + {{(11-$clog2(ROWSM+1)){1'b0}}, fr} :
                          (nl[gc]==NL_RES1 || nl[gc]==NL_RES2) ? sN + ci : rbr;
    wire [10:0] qkv_ra  = (nl[gc]==NL_ALD) ? sN3 + aw_src : rbr3;
    wire [10:0] attn_ra = (nl[gc]==NL_RES1) ? sN + ci : rbr;
    wire [10:0] mlp_ra  = (nl[gc]==NL_RES2) ? sN + ci : rbr;
    wire [10:0] head_ra = (nl[gc]==NL_ARG) ? sNA + {{(11-$clog2(ARROWS+1)){1'b0}}, ar} : rbrA;

    always @(posedge clk) begin
        xres_r <= xres_bank[xres_ra];
        qkv_r  <= qkv_bank [qkv_ra];
        attn_r <= attn_bank[attn_ra];
        mlp_r  <= mlp_bank [mlp_ra];
        head_r <= head_bank[head_ra];
        temb_r <= emb_w[tok_id_b*EROWS + fr];
        gam_r  <= gamma_w[lgam*EROWS + fr];
    end
    always @(posedge clk) begin
        if (el_commit && el_word >= VOCAB*EROWS)
            pos_w[el_word - VOCAB*EROWS] <= el_next;
        else
            pemb_r <= pos_w[pos*EROWS + fr];
    end

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; at_start<=1'b0; at_ldv<=1'b0; done<=1'b0;
        if (rst) begin
            nl[0]<=NL_IDLE; nl[1]<=NL_IDLE; gc<=0; bsg[0]<=0; bsg[1]<=0;
            g_req[0]<=0; g_req[1]<=0; donef[0]<=0; donef[1]<=0;
            lnphase[0]<=0; lnphase[1]<=0;
            fr<=0; frv<=0; civ<=0; arv<=0; av1<=0; amv<=0;
        end else begin
            if (go) begin
                nl[0]<=NL_EMB; nl[1]<=NL_EMB; blkg[0]<=0; blkg[1]<=0;
                lnphase[0]<=0; lnphase[1]<=0; donef[0]<=0; donef[1]<=0;
                gc<=0; bsg[0]<=0; bsg[1]<=0; fr<=0; frv<=0;
            end
            if (g_done_p) begin
                if (gmerge) begin g_req[0] <= 1'b0; g_req[1] <= 1'b0; end
                else g_req[g_served] <= 1'b0;
            end

            case (nl[gc])
                NL_IDLE: ;
                NL_EMB: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    if (frv) begin
                        for (pp=0; pp<P; pp=pp+1)
                            ww[pp*32 +: 32] = $signed(temb_r[pp*32 +: 32])
                                            + $signed(pemb_r[pp*32 +: 32]);
                        xres_bank[sN + frd] <= ww;
                        if (frd==ROWS-1) begin
                            fr<=0; frv<=0;
                            if (bs == G-1) begin
                                bsg[gc]<=0; lgam<=blkg[gc]*2; nl[gc]<=NL_LGAM;
                            end else bsg[gc] <= bs + 1'b1;
                        end
                    end
                end
                NL_LGAM: begin ln_start<=1'b1; fr<=0; frv<=0; nl[gc]<=NL_LFEED; end
                NL_LFEED: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    ln_vin <= frv;
                    ln_x   <= xres_r;
                    ln_g   <= gam_r;
                    if (frv && frd==ROWS-1) begin orow<=0; fr<=0; frv<=0; nl[gc]<=NL_LCOLL; end
                end
                NL_LCOLL: begin
                    if (ln_yv) begin
                        for (pp=0; pp<P; pp=pp+1)
                            ww[pp*32 +: 32] = ln_y[pp*64 +: 32];   // Q.22 fits 32b
                        if (lnphase[gc]==2'd1) lnout2_bank[sN + orow] <= ww;
                        else                   lnout1_bank[sN + orow] <= ww;
                        if (orow==ROWS-1) begin
                            if (bs == G-1) begin
                                bsg[gc]<=0;
                                case (lnphase[gc])
                                    2'd0: nl[gc] <= NL_QKV;
                                    2'd1: nl[gc] <= NL_FC;
                                    default: nl[gc] <= NL_HEAD;
                                endcase
                            end else begin bsg[gc]<=bs+1'b1; nl[gc]<=NL_LGAM; end
                        end else orow<=orow+1'b1;
                    end
                end
                // ---- GEMM requests: queue + switch group while waiting --------
                NL_QKV: begin
                    d_wbase[gc]<=blkg[gc]*GW_BLK + WB_QKV; d_m[gc]<=D3[10:0]; d_k[gc]<=D[10:0];
                    d_asrc[gc]<=2'd0; d_asel[gc]<=blkg[gc]*4 + 6'd0; d_frac[gc]<=7'd16;
                    d_dqrow[gc]<=blkg[gc]*DQB_P + DR_QKV[11:0]; d_dst[gc]<=3'd0;
                    g_req[gc]<=1'b1; nl[gc]<=NL_WQKV; gc<=~gc;
                end
                NL_WQKV: if (!g_req[gc]) begin hh<=0; bsg[gc]<=0; nl[gc] <= NL_AST; end
                         else gc <= ~gc;
                NL_AST: begin
                    at_start<=1'b1; at_tcount<=9'd1; wi<=9'd0; wic<=9'd0; nl[gc]<=NL_ALD;
                end
                NL_ALD: begin
                    at_ldv <= (wi != 3*HR);
                    if (wi != 3*HR) wi <= wi + 1'b1;
                    if (at_ldv) begin
                        if (wic == 3*HR-1) nl[gc] <= NL_ACL;
                        else wic <= wic + 1'b1;
                    end
                end
                NL_ACL: begin
                    if (at_ctxv)
                        ctxv_bank[sN + hh*HR + at_ctxidx] <= at_ctxdata;
                    if (at_done) begin
                        if (hh==NHEAD-1) begin
                            if (bs == G-1) begin bsg[gc]<=0; hh<=0; nl[gc]<=NL_PROJ; end
                            else begin bsg[gc]<=bs+1'b1; hh<=0; nl[gc]<=NL_AST; end
                        end else begin hh<=hh+2'd1; nl[gc]<=NL_AST; end
                    end
                end
                NL_PROJ: begin
                    d_wbase[gc]<=blkg[gc]*GW_BLK + WB_PROJ; d_m[gc]<=D[10:0]; d_k[gc]<=D[10:0];
                    d_asrc[gc]<=2'd1; d_asel[gc]<=blkg[gc]*4 + 6'd1; d_frac[gc]<=7'd25;
                    d_dqrow[gc]<=blkg[gc]*DQB_P + DR_PROJ[11:0]; d_dst[gc]<=3'd1;
                    g_req[gc]<=1'b1; nl[gc]<=NL_WPROJ; gc<=~gc;
                end
                NL_WPROJ: if (!g_req[gc]) begin ci<=0; civ<=0; nl[gc] <= NL_RES1; end
                          else gc <= ~gc;
                NL_RES1: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(attn_r[pp*32 +: 32]);
                        xres_bank[sN + cid] <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (bs == G-1) begin
                                bsg[gc]<=0; lnphase[gc]<=2'd1; lgam<=blkg[gc]*2 + 4'd1; nl[gc]<=NL_LGAM;
                            end else bsg[gc]<=bs+1'b1;
                        end
                    end
                end
                NL_FC: begin
                    d_wbase[gc]<=blkg[gc]*GW_BLK + WB_FC; d_m[gc]<=D_MLP[10:0]; d_k[gc]<=D[10:0];
                    d_asrc[gc]<=2'd2; d_asel[gc]<=blkg[gc]*4 + 6'd2; d_frac[gc]<=7'd12;
                    d_dqrow[gc]<=blkg[gc]*DQB_P + DR_FC[11:0]; d_dst[gc]<=3'd2;
                    g_req[gc]<=1'b1; nl[gc]<=NL_WFC; gc<=~gc;
                end
                NL_WFC: if (!g_req[gc]) nl[gc] <= NL_MP;
                        else gc <= ~gc;
                NL_MP: begin
                    d_wbase[gc]<=blkg[gc]*GW_BLK + WB_MP; d_m[gc]<=D[10:0]; d_k[gc]<=D_MLP[10:0];
                    d_asrc[gc]<=2'd3; d_asel[gc]<=blkg[gc]*4 + 6'd3; d_frac[gc]<=7'd25;
                    d_dqrow[gc]<=blkg[gc]*DQB_P + DR_MP[11:0]; d_dst[gc]<=3'd3;
                    g_req[gc]<=1'b1; nl[gc]<=NL_WMP; gc<=~gc;
                end
                NL_WMP: if (!g_req[gc]) begin ci<=0; civ<=0; nl[gc] <= NL_RES2; end
                        else gc <= ~gc;
                NL_RES2: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(mlp_r[pp*32 +: 32]);
                        xres_bank[sN + cid] <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (bs == G-1) begin
                                bsg[gc]<=0;
                                if (blkg[gc] == NLAYER-1) begin
                                    lnphase[gc]<=2'd2; lgam<=NLAYER*2; nl[gc]<=NL_LGAM;
                                end else begin
                                    blkg[gc]<=blkg[gc]+1'b1; lnphase[gc]<=2'd0;
                                    lgam<=(blkg[gc]+1'b1)*2; nl[gc]<=NL_LGAM;
                                end
                            end else bsg[gc]<=bs+1'b1;
                        end
                    end
                end
                NL_HEAD: begin
                    d_wbase[gc]<=WB_HEAD[19:0]; d_m[gc]<=VOCAB[10:0]; d_k[gc]<=D[10:0];
                    d_asrc[gc]<=2'd0; d_asel[gc]<=4*NLAYER; d_frac[gc]<=7'd25;
                    d_dqrow[gc]<=DR_HEAD[11:0]; d_dst[gc]<=3'd4;
                    g_req[gc]<=1'b1;
                    best_val<=32'sh80000000; best_idx<=0; ar<=0; arv<=0; av1<=0; amv<=0;
                    nl[gc]<=NL_WHEAD; gc<=~gc;
                end
                NL_WHEAD: if (!g_req[gc]) begin
                    // argmax scratch is SHARED between groups — re-arm on entry
                    best_val<=32'sh80000000; best_idx<=0;
                    ar<=0; arv<=0; av1<=0; amv<=0;
                    nl[gc] <= NL_ARG;
                end else gc <= ~gc;
                NL_ARG: begin
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
                            tok_out_b[{gc, bs}] <= (wm_val > best_val) ? wm_idx : best_idx;
                            nl[gc] <= NL_ARG2;
                        end
                    end
                end
                NL_ARG2: begin
                    if (bs == G-1) begin
                        bsg[gc]<=0; donef[gc]<=1'b1; nl[gc]<=NL_DONE;
                        if (donef[~gc]) done<=1'b1;
                    end else begin
                        bsg[gc]<=bs+1'b1;
                        best_val<=32'sh80000000; best_idx<=0;
                        ar<=0; arv<=0; av1<=0; amv<=0; nl[gc]<=NL_ARG;
                    end
                end
                NL_DONE: begin
                    if (!donef[~gc]) gc <= ~gc;
                end
                default: nl[gc] <= NL_IDLE;
            endcase
        end
    end

    // ================================================================
    //                       GEMM ENGINE
    // ================================================================
    reg                gv_ldrst, gv_xrst, gv_xwe, gv_start;
    reg  [$clog2(N)-1:0] gv_xstream, gv_rdstream;
    reg  [P*8-1:0]     gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    reg [10:0]         gv_rdaddr;
    wire [P*32-1:0]    gv_yout;
    gemm_banked_resident_vec #(.LANES(LANES), .N(N), .P(P), .MMAX(1024), .KMAX(1024),
                  .RLAT(2), .WWORDS(WWORDS)) u_gemm (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ldrst | wl_rst), .w_we(wl_we), .w_data(wl_data),
        .x_rst(gv_xrst), .x_we(gv_xwe), .x_stream(gv_xstream), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done),
        .rd_stream(gv_rdstream), .rd_addr(gv_rdaddr[$clog2(1024/P)-1:0]), .y_out(gv_yout));

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

    reg             gl_vin;
    reg [P*16-1:0]  gl_x;
    wire            gl_vout;
    wire [P*16-1:0] gl_y;
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(gl_vin), .x(gl_x),
        .out_valid(gl_vout), .y(gl_y));

    // active call context
    reg        ggrp;
    reg        gmerge;                  // both groups share this pass
    reg [11:0] gwait;                   // solo-serve patience: wait for the partner
    reg [19:0] a_wbase;
    reg [10:0] a_m, a_k;
    reg [1:0]  a_asrc;
    reg [5:0]  a_asel;
    reg signed [6:0] a_frac;
    reg [11:0] a_dqrow;
    reg [2:0]  a_dst;
    wire g_served = ggrp;
    wire [$clog2(N)-1:0] glim = gmerge ? N-1 : G-1;

    reg [$clog2(N)-1:0] gbs;            // stream over ALL N (single-pass GEMM)
    reg [10:0] gci, gcid, gcid1, gcid2;
    reg        gciv, gciv1, gciv2;
    reg [2:0]  grbn;
    reg signed [63:0] lntmp_g;
    reg signed [95:0] aq_prod, aq_sh;
    reg signed [31:0] aq_int;
    reg signed [95:0] aq_prod_r [0:P-1];
    reg                aq_neg_r [0:P-1];
    reg signed [63:0] lnt_r [0:P-1];
    reg signed [31:0] cbg, dqv;
    reg signed [15:0] mbg;
    reg [P*8-1:0]   aqw;
    reg [P*16-1:0]  mword;
    reg [P*32-1:0]  dword;
    reg [P*24-1:0]  mwr;
    reg [P*8-1:0]   ewr;
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    reg [$clog2(ROWSM+1)-1:0] gdor, gor;
    integer gp;

    // merged: gbs covers all N streams; single: ggrp's G streams
    wire [$clog2(N)-1:0] sid = gmerge ? gbs : {ggrp, gbs[GSH-1:0]};
    wire [10:0] sG  = sid * ROWS;
    wire [10:0] sG3 = sid * ROWS3;
    wire [10:0] sGM = sid * ROWSM;
    wire [10:0] sGA = sid * ARROWS;

    localparam [2:0] GE_IDLE=0, GE_AQ=1, GE_AQN=2, GE_RUN=3, GE_WAIT=4, GE_RB=5, GE_RBN=6;
    reg [2:0] ge;

    // GEMM-side reads; board readback reuses this port while the engine idles
    wire [10:0] rbrM = rd_stream*ROWSM + (rd_addr >> LSH);
    wire [10:0] gemm_ra = (ge == GE_IDLE) ? ((rd_sel == 4'd5) ? rbrM : rbr)
                        : (a_asrc == 2'd3) ? (sGM + gci) : (sG + gci);
    always @(posedge clk) begin
        lnout1_r <= lnout1_bank[gemm_ra];
        lnout2_r <= lnout2_bank[gemm_ra];
        ctxv_g   <= ctxv_bank  [gemm_ra];
        mlpbuf_r <= mlpbuf_bank[gemm_ra];
        mwr      <= dqm_w[a_dqrow + rb1];
        ewr      <= dqe_w[a_dqrow + rb1];
    end

    assign rb_last = (a_dst == 3'd2) ? (gl_vout && gor == ROWSM-1 && gbs == glim)
                                     : (dq_vout && gdor == ((a_m + P-1) >> LSH) - 1 && gbs == glim);
    assign g_done_p = (ge == GE_RB) && rb_last;

    always @(posedge clk) begin
        gv_ldrst<=0; gv_xrst<=0; gv_xwe<=0; gv_start<=0; dq_vin<=0; gl_vin<=0;
        if (rst) begin
            ge<=GE_IDLE; gbs<=0; gci<=0; gciv<=0; gciv1<=0; gciv2<=0;
            rv0<=0; rv1<=0; rv2<=0; gdor<=0; gor<=0;
        end else begin
            case (ge)
                GE_IDLE: begin
                    // SINGLE PASS: wait for BOTH groups to request the SAME call
                    // (descriptors must match — NL groups run in near-lockstep but
                    // attention serialization can put A a phase ahead of B), then
                    // serve all 8 streams from one weight pass.
    if (g_req[0] && g_req[1] && d_wbase[0] == d_wbase[1]) begin
                        gwait   <= 0;
                        gmerge  <= 1'b1;  ggrp <= 1'b0;
                        a_wbase <= d_wbase[0]; a_m <= d_m[0]; a_k <= d_k[0];
                        a_asrc  <= d_asrc[0];  a_asel <= d_asel[0];
                        a_frac  <= d_frac[0];  a_dqrow <= d_dqrow[0];
                        a_dst   <= d_dst[0];
                        gbs<=0; gci<=0; gciv<=0; gciv1<=0; gciv2<=0;
                        gv_ldrst<=1'b1; ge<=GE_AQ;
                    end else if ((g_req[0] || g_req[1]) && gwait < 12'd2048) begin
                        // wait for the partner group — merged passes serve all 8
                        // streams from ONE weight pass (the 21k -> 40k lever)
                        gwait <= gwait + 1'b1;
                    end else if (g_req[0] || g_req[1]) begin
                        // partner did not arrive: serve one group alone
                        gwait   <= 0;
                        gmerge  <= 1'b0;
                        ggrp    <= g_req[0] ? 1'b0 : 1'b1;
                        // (no-request idle resets the patience timer below)
                        a_wbase <= g_req[0] ? d_wbase[0] : d_wbase[1];
                        a_m     <= g_req[0] ? d_m[0]     : d_m[1];
                        a_k     <= g_req[0] ? d_k[0]     : d_k[1];
                        a_asrc  <= g_req[0] ? d_asrc[0]  : d_asrc[1];
                        a_asel  <= g_req[0] ? d_asel[0]  : d_asel[1];
                        a_frac  <= g_req[0] ? d_frac[0]  : d_frac[1];
                        a_dqrow <= g_req[0] ? d_dqrow[0] : d_dqrow[1];
                        a_dst   <= g_req[0] ? d_dst[0]   : d_dst[1];
                        gbs<=0; gci<=0; gciv<=0; gciv1<=0; gciv2<=0;
                        gv_ldrst<=1'b1; ge<=GE_AQ;
                    end else gwait <= 0;
                end
                GE_AQ: begin
                    gcid <= gci; gciv <= (gci != (a_k >> LSH));
                    gcid1 <= gcid; gciv1 <= gciv;
                    gcid2 <= gcid1; gciv2 <= gciv1;
                    if (gci != (a_k >> LSH)) gci <= gci + 1'b1;
                    if (gciv) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            case (a_asrc)
                                2'd0: begin
                                    cbg = lnout1_r[gp*32 +: 32];               // Q.22, 32b lane
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
                                    // mlp hidden Q4.12 sat16 -> Q.22 (<< 10)
                                    mbg = mlpbuf_r[gp*16 +: 16];
                                    lntmp_g = $signed({{48{mbg[15]}}, mbg}) <<< (LN_OUT_FRAC - GELU_FRAC);
                                end
                            endcase
                            lnt_r[gp] <= lntmp_g;
                        end
                    end
                    if (gciv1) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            aq_prod_r[gp] <= $signed(lnt_r[gp]) * $signed(inv_sact[a_asel]);
                            aq_neg_r[gp]  <= (lnt_r[gp] < 0);
                        end
                    end
                    if (gciv2) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            aq_prod = aq_prod_r[gp];
                            if (!aq_neg_r[gp])
                                aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                            else
                                aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                            aq_int = aq_sh[31:0];
                            if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                            aqw[gp*8 +: 8] = aq_int[7:0];
                        end
                        gv_xwe<=1'b1; gv_xdata<=aqw; gv_xstream<=sid;
                        if (gcid2==(a_k >> LSH)-1) begin
                            gci<=0; gciv<=0; gciv1<=0; gciv2<=0;
                            if (gbs == glim) begin gbs<=0; ge<=GE_RUN; end
                            else begin grbn<=0; ge<=GE_AQN; end
                        end
                    end
                end
                GE_AQN: begin
                    grbn <= grbn + 1'b1;
                    if (grbn == 3'd3) begin
                        gbs<=gbs+1'b1; gv_xrst<=1'b1; ge<=GE_AQ;
                    end
                end
                GE_RUN: begin
                    gv_m<=a_m; gv_k<=a_k; gv_wbase<=a_wbase[$clog2(WWORDS)-1:0];
                    gv_start<=1'b1; ge<=GE_WAIT;
                end
                GE_WAIT: if (gv_done) begin
                    gci<=0; gv_rdaddr<=0; rv0<=0; rv1<=0; rv2<=0;
                    // readback starts at THIS group's first stream (solo passes!)
                    gv_rdstream <= gmerge ? {($clog2(N)){1'b0}} : {ggrp, {GSH{1'b0}}};
                    gdor<=0; gor<=0; ge<=GE_RB;
                end
                GE_RB: begin
                    if (gci < ((a_m + P-1) >> LSH)) begin
                        gv_rdaddr<=gci; rb0<=gci; gci<=gci+1'b1;
                    end else gv_rdaddr <= ((a_m + P-1) >> LSH) - 1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(gci < ((a_m + P-1) >> LSH)); rv1<=rv0; rv2<=rv1;
                    if (rv2) begin
                        dq_vin<=1'b1; dq_frac<=a_frac;
                        dq_gemvy <= gv_yout;
                        dq_mant  <= mwr;
                        dq_exp   <= ewr;
                    end
                    if (dq_vout) begin
                        for (gp=0; gp<P; gp=gp+1) begin
                            dqv = dq_out[gp*32 +: 32];
                            dword[gp*32 +: 32] = dqv;
                            if      ($signed(dqv) >  32'sd32767)  mword[gp*16 +: 16] = 16'sd32767;
                            else if ($signed(dqv) < -32'sd32768)  mword[gp*16 +: 16] = -16'sd32768;
                            else    mword[gp*16 +: 16] = dqv[15:0];
                        end
                        case (a_dst)
                            3'd0: qkv_bank [sG3 + gdor] <= dword;
                            3'd1: attn_bank[sG  + gdor] <= dword;
                            3'd2: begin gl_vin<=1'b1; gl_x<=mword; end
                            3'd3: mlp_bank [sG  + gdor] <= dword;
                            default: head_bank[sGA + gdor] <= dword;
                        endcase
                        if (a_dst != 3'd2 && gdor==((a_m + P-1) >> LSH)-1) begin
                            gci<=0; gdor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (gbs == glim) begin gbs<=0; ge<=GE_IDLE; end
                            else begin grbn<=0; ge<=GE_RBN; end
                        end else gdor<=gdor+1'b1;
                    end
                    if (gl_vout) begin
                        mlpbuf_bank[sGM + gor] <= gl_y;   // Q4.12 sat16: 16b lanes

                        if (gor==ROWSM-1) begin
                            gci<=0; gor<=0; gdor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (gbs == glim) begin gbs<=0; ge<=GE_IDLE; end
                            else begin grbn<=0; ge<=GE_RBN; end
                        end else gor<=gor+1'b1;
                    end
                end
                GE_RBN: begin
                    grbn <= grbn + 1'b1;
                    if (grbn == 3'd7) begin
                        gbs<=gbs+1'b1; gv_rdstream<=gmerge ? gbs+1'b1 : {ggrp, gbs[GSH-1:0]+1'b1}; ge<=GE_RB;
                    end
                end
                default: ge<=GE_IDLE;
            endcase
        end
    end

    // ---- board readback ----------------------------------------------------------
    reg [LSH-1:0] rd_lane;
    reg [P*64-1:0] rw64; reg [P*32-1:0] rw32; reg is64;
    always @* begin
        is64 = 1'b0; rw64 = {(P*64){1'b0}}; rw32 = {(P*32){1'b0}};
        case (rd_sel)
            4'd0: begin                                     // Q.22 stored in 32b lanes
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
            4'd5: begin                                     // gelu Q4.12 -> Q.22 sign-ext
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
endmodule
