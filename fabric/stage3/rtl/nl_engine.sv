// -----------------------------------------------------------------------------
// nl_engine — per-group NON-LINEAR engine for the ping-pong sequencer.
//
// Split out of sequencer_pp so the two stream-groups run their LayerNorm /
// attention / residual / argmax in PARALLEL on two independent copies of this
// module, instead of being serialised through one shared NL FSM (the old
// gc<=~gc group-toggle). The GEMM engine, weight/embed loaders, vec_dequant
// and vec_gelu stay SHARED in sequencer_pp.
//
// Each engine owns G = N/2 streams (LOCAL stream index bs in 0..G-1). Its scratch
// banks are sized at G-stream depth. GEMM calls are requested via the descriptor
// outputs (req/d_*) and acknowledged by the shared GE FSM driving `served`. Bank
// fills produced by the GE drain are pushed back in via dw_*/dwm_* write buses
// (one write SITE per bank — multi-writer banks LUTRAM-replicate in Vivado).
//
// Arithmetic is BIT-IDENTICAL to the pre-split sequencer_pp; this is a mechanical
// restructuring (group arrays collapsed to scalars, gc<=~gc removed: a W-state
// simply holds until `served`; g_done_p request-clear becomes `if (served) req<=0`).
//
// iverilog-2012 traps honoured (plain-reg part-selects, single-write-site banks).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module nl_engine #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 128,
    parameter integer G     = 4,        // streams owned by THIS engine
    parameter integer VOCAB = 193,
    parameter integer TMAX  = 32,
    parameter integer GAMMA_N = 9,
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
    input  wire [G*9-1:0] tok_ids,
    input  wire [8:0]  pos,
    input  wire        served,        // GE acknowledges THIS engine's in-flight call
    // STREAM-GRANULAR readback completion: s_done pulses when stream s_done_idx's
    // readback fully commits into this engine's banks (qkv/attn/mlp/head, NOT FC->
    // mlpbuf which has no nl post-processing). Lets the per-stream post-call loops
    // (attn / RES1 / RES2 / argmax) begin for stream s while the GE engine is still
    // draining streams s+1.. — the stream-granular NL/GEMM overlap. Arithmetic is
    // UNCHANGED; only the start-gate of each per-stream loop moves from `!req`
    // (whole-call done) to rb_rdy[bs] (this stream's readback done).
    input  wire        s_done,
    input  wire [$clog2(G)-1:0] s_done_idx,
    // shared embed read (sequencer_pp owns the URAM emb_w + arbiter)
    input  wire        emb_gnt,       // 1 when this engine holds the emb read port
    input  wire [P*32-1:0] emb_q,     // registered emb_w[emb_addr], 1-cycle latency
    output wire        emb_req,
    output wire [$clog2(VOCAB*(D/P))-1:0] emb_addr,
    // GE drain writes into this engine's banks
    input  wire        dw_we,
    input  wire [2:0]  dw_dst,        // 0 qkv,1 attn,3 mlp,4 head  (2 -> mlpbuf via dwm)
    input  wire [10:0] dw_addr,       // LOCAL row address (already local)
    input  wire [P*32-1:0] dw_data,
    input  wire        dwm_we,
    input  wire [10:0] dwm_addr,      // LOCAL mlpbuf row address
    input  wire [P*16-1:0] dwm_data,
    // GE-side AQ feed reads (registered, 1-cycle): ge_ra is LOCAL + final
    input  wire [10:0] ge_ra,
    // host-debug read path: dbg_stream LOCAL (0..G-1), dbg_addr already row
    input  wire [$clog2(G)-1:0] dbg_stream,
    input  wire [10:0] dbg_addr,
    // pos_w runtime write (broadcast from sequencer_pp embed loader)
    input  wire        pw_we,
    input  wire [$clog2(TMAX*(D/P))-1:0] pw_addr,
    input  wire [P*32-1:0] pw_data,
    // ---- SHARED LayerNorm port (hoisted into sequencer_pp) ----
    output reg         ln_req,        // 1 while this engine wants the LN unit
    output reg         ln_start_o,
    output reg         ln_vin_o,
    output reg  [P*32-1:0] ln_x_o,
    output reg  [P*32-1:0] ln_g_o,
    input  wire        ln_gnt,        // 1 when this engine holds the LN unit
    input  wire        ln_yv_i,
    input  wire [P*64-1:0] ln_y_i,
    input  wire        ln_done_i,     // (unused by FSM; orow drives collect end)
    // ---- SHARED attention port (hoisted into sequencer_pp) ----
    output reg         at_req,        // 1 while this engine wants the attn unit
    output reg         at_start_o,
    output reg  [8:0]  at_tcount_o,
    output reg         at_ldv_o,
    output wire [P*32-1:0] at_ld_data_o,
    input  wire        at_gnt,        // 1 when this engine holds the attn unit
    input  wire        at_ldready_i,
    input  wire        at_ctxv_i,
    input  wire [6:0]  at_ctxidx_i,
    input  wire [P*32-1:0] at_ctxdata_i,
    input  wire        at_done_i,
    // GEMM-call descriptor + request
    output reg         req,
    output reg  [19:0] d_wbase,
    output reg  [10:0] d_m,
    output reg  [10:0] d_k,
    output reg  [1:0]  d_asrc,
    output reg  [5:0]  d_asel,
    output reg signed [6:0] d_frac,
    output reg  [11:0] d_dqrow,
    output reg  [2:0]  d_dst,
    output reg         done_o,        // level: 1 when this engine has finished all G
    output wire [G*9-1:0] tok_outs_g,
    // GE-side registered bank reads at ge_ra
    output reg  [P*32-1:0] lnout1_r,
    output reg  [P*32-1:0] lnout2_r,
    output reg  [P*32-1:0] ctxv_g,
    output reg  [P*16-1:0] mlpbuf_r,
    // NL-side registered bank reads (for host debug muxing in sequencer_pp)
    output reg  [P*32-1:0] xres_r,
    output reg  [P*32-1:0] qkv_r,
    output reg  [P*32-1:0] attn_r,
    output reg  [P*32-1:0] mlp_r,
    output reg  [P*32-1:0] head_r
);
    localparam integer ROWS  = D    / P;
    localparam integer ROWS3 = D3   / P;
    localparam integer ROWSM = D_MLP/ P;
    localparam integer LSH   = $clog2(P);
    localparam integer GSH   = $clog2(G);
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
    // per-bank depths at THIS engine's G streams (stride-tight)
    localparam integer BD_S = G * ROWS;     // xres/lnout/attn/mlp/ctxv (32/stream)
    localparam integer BD_H = G * ARROWS;   // head logits (25/stream)
    localparam integer BD_Q = G * ROWS3;    // qkv (96/stream)
    localparam integer BD_M = G * ROWSM;    // mlpbuf (128/stream)

    // ---- per-engine embed-position + gamma ROMs ---------------------------------
    (* ram_style = "block" *) reg [P*32-1:0] pos_w [0:TMAX*EROWS-1];
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w [0:GAMMA_N*EROWS-1];
`ifndef SYNTHESIS
    initial begin
        $readmemh("pos_emb_w.mem", pos_w, 0, TMAX*EROWS-1);
    end
`endif
    initial begin
        $readmemh("gamma_w.mem",   gamma_w);
    end

    // ---- stream-flattened scratch (row = stream*ROWS + r) at G-stream depth ------
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank  [0:BD_S-1];
    (* ram_style = "block" *) reg [P*32-1:0] lnout1_bank[0:BD_S-1];
    (* ram_style = "block" *) reg [P*32-1:0] lnout2_bank[0:BD_S-1];
    (* ram_style = "block" *) reg [P*32-1:0] qkv_bank   [0:BD_Q-1];
    (* ram_style = "block" *) reg [P*32-1:0] ctxv_bank  [0:BD_S-1];
    (* ram_style = "block" *) reg [P*32-1:0] attn_bank  [0:BD_S-1];
    (* ram_style = "block" *) reg [P*16-1:0] mlpbuf_bank[0:BD_M-1];
    (* ram_style = "block" *) reg [P*32-1:0] mlp_bank   [0:BD_S-1];
    (* ram_style = "block" *) reg [P*32-1:0] head_bank  [0:BD_H-1];

    reg [P*32-1:0] pemb_r, gam_r;

    // ================================================================
    //                       NL  ENGINE
    // ================================================================
    // LN/attn units are now SHARED in sequencer_pp. The signals that used to
    // drive/return from the local u_ln / u_attn are mapped to the new ports:
    //   ln_start -> ln_start_o, ln_vin -> ln_vin_o, ln_x -> ln_x_o, ln_g -> ln_g_o,
    //   ln_yv    -> ln_yv_i,    ln_done -> ln_done_i, ln_y -> ln_y_i.
    //   at_start -> at_start_o, at_ldv -> at_ldv_o, at_tcount -> at_tcount_o,
    //   at_ldready->at_ldready_i, at_ctxv->at_ctxv_i, at_ctxidx->at_ctxidx_i,
    //   at_ctxdata->at_ctxdata_i, at_done->at_done_i, aw_data -> at_ld_data_o.
    // ARBITRATION (see header): ln_req asserted on entering NL_LGAM and held
    // through LFEED/LCOLL, dropped after the final collect row of the LAST stream;
    // at_req asserted at NL_AST entry of head 0 and held across the whole per-head
    // loop, dropped after the last head's at_done. Engine ln_start_o/at_start_o are
    // gated by ln_gnt/at_gnt so the FSM idle-waits in NL_LGAM/NL_AST until granted.
    reg [1:0]  hh;
    reg [8:0]  wi, wic;
    wire [10:0] aw_src = (wi < HR)   ? (hh*HR + wi) :
                         (wi < 2*HR) ? (D/P   + hh*HR + (wi - HR)) :
                                       (2*D/P + hh*HR + (wi - 2*HR));
    wire [P*32-1:0] aw_data = qkv_r;
    assign at_ld_data_o = aw_data;          // combinational, aligned with at_ldv_o

    localparam [4:0]
      NL_IDLE=0, NL_EMB=1, NL_LGAM=2, NL_LFEED=3, NL_LCOLL=4,
      NL_QKV=5, NL_WQKV=6, NL_AST=7, NL_ALD=8, NL_ACL=9,
      NL_PROJ=10, NL_WPROJ=11, NL_RES1=12,
      NL_FC=13, NL_WFC=14, NL_MP=15, NL_WMP=16, NL_RES2=17,
      NL_HEAD=18, NL_WHEAD=19, NL_ARG=20, NL_ARG2=21, NL_DONE=22;
    reg [4:0]  nl;
    reg [3:0]  blkg;
    reg [1:0]  lnphase;             // 0 = LN1->QKV, 1 = LN2->FC, 2 = LNF->HEAD

    reg [GSH-1:0] bs;               // LOCAL stream counter 0..G-1
    // STREAM-GRANULAR overlap: rb_rdy[s]=1 once stream s's readback for the CURRENT
    // GEMM call has committed into the banks. Set by s_done; cleared en-masse when a
    // NEW descriptor is issued (the call boundary). The per-stream post loops gate
    // their start on rb_rdy[bs], so nl works stream s while GE drains s+1..G-1.
    reg [G-1:0] rb_rdy;
    reg         rb_clr;                  // pulse: clear rb_rdy at a new call boundary
    wire        sready = rb_rdy[bs];
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
    reg              xr_we;
    reg [10:0]       xr_wa;
    reg [P*32-1:0]   xr_wd;
    // 250 GATE: pipeline register splitting the embed->residual cone (OOC worst
    // path: u_embank URAM emb_q -> (emb_q+pemb_r) adder -> coh*/eng/xres_bank).
    // The NL_EMB residual add (tok_emb + pos_emb) is registered here so the long
    // URAM-read -> add -> BRAM-write cone becomes two short paths: URAM->FF and
    // FF->BRAM. Embed is consumed ONCE/token/stream (NL_EMB only), so this adds
    // ~1 cyc/stream/pass (a few cyc per 16-token pass), not a per-cycle loop tax.
    // Arithmetic is BIT-IDENTICAL: the same emb_q+pemb_r reaches xres_bank, one
    // cycle later, with its write-address/enable delayed to match.
    reg              emb_we_p;     // delayed-by-1 embed xres write enable
    reg [10:0]       emb_wa_p;     // delayed-by-1 embed xres write address
    reg [P*32-1:0]   emb_wd_p;     // registered embed residual add (emb_q+pemb_r)
    // single-write-port fold (keeps xres_bank a clean BRAM): embed pipe + residual
    reg              x_we_w;
    reg [10:0]       x_wa_w;
    reg [P*32-1:0]   x_wd_w;
    reg [3:0] lgam;
    integer pp;

    wire [8:0] tok_id_b = tok_ids[bs*9 +: 9];
    reg [8:0] tok_out_b [0:G-1];
    genvar gtb;
    generate
        for (gtb = 0; gtb < G; gtb = gtb + 1) begin : g_tok
            assign tok_outs_g[gtb*9 +: 9] = tok_out_b[gtb];
        end
    endgenerate

    // NL-side addressing: LOCAL stream = bs (no group bit)
    wire [10:0] sN  = bs * ROWS;
    wire [10:0] sN3 = bs * ROWS3;
    wire [10:0] sNA = bs * ARROWS;

    // host-debug NL-side row addresses (sequencer_pp drives dbg_stream/dbg_addr local)
    wire [10:0] rbr  = dbg_stream*ROWS   + (dbg_addr >> LSH);
    wire [10:0] rbr3 = dbg_stream*ROWS3  + (dbg_addr >> LSH);
    wire [10:0] rbrA = dbg_stream*ARROWS + (dbg_addr >> LSH);

    wire [10:0] xres_ra = (nl==NL_LFEED) ? sN + {{(11-$clog2(ROWSM+1)){1'b0}}, fr} :
                          (nl==NL_RES1 || nl==NL_RES2) ? sN + ci : rbr;
    wire [10:0] qkv_ra  = (nl==NL_ALD) ? sN3 + aw_src : rbr3;
    wire [10:0] attn_ra = (nl==NL_RES1) ? sN + ci : rbr;
    wire [10:0] mlp_ra  = (nl==NL_RES2) ? sN + ci : rbr;
    wire [10:0] head_ra = (nl==NL_ARG) ? sNA + {{(11-$clog2(ARROWS+1)){1'b0}}, ar} : rbrA;

    // embed read: engine drives addr from its own state; valid only in NL_EMB
    assign emb_req  = (nl == NL_EMB);
    assign emb_addr = tok_id_b*EROWS + fr;

    always @(posedge clk) begin
        xres_r <= xres_bank[xres_ra];
        qkv_r  <= qkv_bank [qkv_ra];
        attn_r <= attn_bank[attn_ra];
        mlp_r  <= mlp_bank [mlp_ra];
        head_r <= head_bank[head_ra];
        gam_r  <= gamma_w[lgam*EROWS + fr];
    end
    // pos_w: runtime write from the embed loader (broadcast) + engine-local read
    always @(posedge clk) begin
        if (pw_we) pos_w[pw_addr] <= pw_data;
        else       pemb_r <= pos_w[pos*EROWS + fr];
    end

    // GE-side registered reads at ge_ra (mlpbuf only meaningful for FC dst)
    always @(posedge clk) begin
        lnout1_r <= lnout1_bank[ge_ra];
        lnout2_r <= lnout2_bank[ge_ra];
        ctxv_g   <= ctxv_bank  [ge_ra];
        mlpbuf_r <= mlpbuf_bank[ge_ra];
    end

    // GE drain writes (one write site per bank)
    always @(posedge clk) begin
        if (dw_we) begin
            case (dw_dst)
                3'd0: qkv_bank [dw_addr] <= dw_data;
                3'd1: attn_bank[dw_addr] <= dw_data;
                3'd3: mlp_bank [dw_addr] <= dw_data;
                default: head_bank[dw_addr] <= dw_data;
            endcase
        end
        if (dwm_we) mlpbuf_bank[dwm_addr] <= dwm_data;
    end

    always @(posedge clk) begin
        ln_start_o<=1'b0; ln_vin_o<=1'b0; at_start_o<=1'b0; at_ldv_o<=1'b0;
        emb_we_p <= 1'b0;                       // default: embed-write pipe reg idle
        rb_clr = 1'b0;                          // default: keep accumulating rb_rdy
        if (rst) begin
            nl<=NL_IDLE; bs<=0; req<=0; done_o<=0; lnphase<=0;
            fr<=0; frv<=0; civ<=0; arv<=0; av1<=0; amv<=0;
            ln_req<=0; at_req<=0; rb_rdy<=0;
        end else begin
            if (go) begin
                nl<=NL_EMB; blkg<=0; lnphase<=0; done_o<=0;
                bs<=0; fr<=0; frv<=0;
            end
            if (served) req <= 1'b0;

            xr_we = 1'b0;                       // default: no xres write this cycle
            case (nl)
                NL_IDLE: ;
                NL_EMB: if (emb_gnt) begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    if (frv) begin
                        // 250 GATE: register the residual add (emb_q+pemb_r) into
                        // the embed-write pipe instead of writing xres_bank this
                        // cycle. The add output is now a flop fed straight by the
                        // URAM read; the FF->xres_bank write happens next cycle.
                        for (pp=0; pp<P; pp=pp+1)
                            emb_wd_p[pp*32 +: 32] <= $signed(emb_q[pp*32 +: 32])
                                                   + $signed(pemb_r[pp*32 +: 32]);
                        emb_we_p <= 1'b1; emb_wa_p <= sN + frd;
                        if (frd==ROWS-1) begin
                            fr<=0; frv<=0;
                            if (bs == G-1) begin
                                bs<=0; lgam<=blkg*2; nl<=NL_LGAM;
                            end else bs <= bs + 1'b1;
                        end
                    end
                end
                NL_LGAM: begin
                    ln_req <= 1'b1;                 // request the shared LN unit (held)
                    if (ln_gnt) begin              // idle-wait here until granted
                        ln_start_o<=1'b1; fr<=0; frv<=0; nl<=NL_LFEED;
                    end
                end
                NL_LFEED: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    ln_vin_o <= frv;
                    ln_x_o   <= xres_r;
                    ln_g_o   <= gam_r;
                    if (frv && frd==ROWS-1) begin orow<=0; fr<=0; frv<=0; nl<=NL_LCOLL; end
                end
                NL_LCOLL: begin
                    if (ln_yv_i) begin
                        for (pp=0; pp<P; pp=pp+1)
                            ww[pp*32 +: 32] = ln_y_i[pp*64 +: 32];   // Q.22 fits 32b
                        if (lnphase==2'd1) lnout2_bank[sN + orow] <= ww;
                        else               lnout1_bank[sN + orow] <= ww;
                        if (orow==ROWS-1) begin
                            if (bs == G-1) begin
                                bs<=0; ln_req <= 1'b0;     // drop: last stream's last row
                                case (lnphase)
                                    2'd0: nl <= NL_QKV;
                                    2'd1: nl <= NL_FC;
                                    default: nl <= NL_HEAD;
                                endcase
                            end else begin bs<=bs+1'b1; nl<=NL_LGAM; end
                        end else orow<=orow+1'b1;
                    end
                end
                // ---- GEMM requests: queue + wait (no group toggle) -----------
                NL_QKV: begin
                    d_wbase<=blkg*GW_BLK + WB_QKV; d_m<=D3[10:0]; d_k<=D[10:0];
                    d_asrc<=2'd0; d_asel<=blkg*4 + 6'd0; d_frac<=7'd16;
                    d_dqrow<=blkg*DQB_P + DR_QKV[11:0]; d_dst<=3'd0;
                    req<=1'b1; rb_clr=1'b1; hh<=0; bs<=0; nl<=NL_WQKV;
                end
                // overlap: enter the attention loop as soon as stream 0's qkv
                // readback commits (rb_rdy[0]) — GE keeps draining streams 1..G-1
                // while attention for stream 0 runs. NL_AST is per-stream; its
                // own rb_rdy[bs] gate (via at_req) stalls if a later stream isn't
                // committed yet. (req is cleared call-granularly by `served`.)
                NL_WQKV: if (rb_rdy[0]) begin hh<=0; bs<=0; nl <= NL_AST; end
                NL_AST: begin
                    at_req <= sready;              // only grab the shared attn unit
                                                  // once THIS stream's qkv is ready
                    if (sready && at_gnt) begin   // idle-wait until ready AND granted
                        at_start_o<=1'b1; at_tcount_o<=9'd1; wi<=9'd0; wic<=9'd0; nl<=NL_ALD;
                    end
                end
                NL_ALD: begin
                    at_ldv_o <= (wi != 3*HR);
                    if (wi != 3*HR) wi <= wi + 1'b1;
                    if (at_ldv_o) begin
                        if (wic == 3*HR-1) nl <= NL_ACL;
                        else wic <= wic + 1'b1;
                    end
                end
                NL_ACL: begin
                    if (at_ctxv_i)
                        ctxv_bank[sN + hh*HR + at_ctxidx_i] <= at_ctxdata_i;
                    if (at_done_i) begin
                        if (hh==NHEAD-1) begin
                            if (bs == G-1) begin bs<=0; hh<=0; at_req<=1'b0; nl<=NL_PROJ; end
                            else begin bs<=bs+1'b1; hh<=0; nl<=NL_AST; end
                        end else begin hh<=hh+2'd1; nl<=NL_AST; end
                    end
                end
                NL_PROJ: begin
                    d_wbase<=blkg*GW_BLK + WB_PROJ; d_m<=D[10:0]; d_k<=D[10:0];
                    d_asrc<=2'd1; d_asel<=blkg*4 + 6'd1; d_frac<=7'd25;
                    d_dqrow<=blkg*DQB_P + DR_PROJ[11:0]; d_dst<=3'd1;
                    req<=1'b1; rb_clr=1'b1; nl<=NL_WPROJ;
                end
                // overlap: start residual-1 for stream 0 as soon as its proj (attn
                // bank) readback commits; per-stream sready gate stalls later streams.
                NL_WPROJ: if (rb_rdy[0]) begin ci<=0; civ<=0; bs<=0; nl <= NL_RES1; end
                NL_RES1: if (sready) begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(attn_r[pp*32 +: 32]);
                        xr_we = 1'b1; xr_wa = sN + cid; xr_wd = sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (bs == G-1) begin
                                bs<=0; lnphase<=2'd1; lgam<=blkg*2 + 4'd1; nl<=NL_LGAM;
                            end else bs<=bs+1'b1;
                        end
                    end
                end
                NL_FC: begin
                    d_wbase<=blkg*GW_BLK + WB_FC; d_m<=D_MLP[10:0]; d_k<=D[10:0];
                    d_asrc<=2'd2; d_asel<=blkg*4 + 6'd2; d_frac<=7'd12;
                    d_dqrow<=blkg*DQB_P + DR_FC[11:0]; d_dst<=3'd2;
                    req<=1'b1; rb_clr=1'b1; nl<=NL_WFC;
                end
                NL_WFC: if (!req) nl <= NL_MP;
                NL_MP: begin
                    d_wbase<=blkg*GW_BLK + WB_MP; d_m<=D[10:0]; d_k<=D_MLP[10:0];
                    d_asrc<=2'd3; d_asel<=blkg*4 + 6'd3; d_frac<=7'd25;
                    d_dqrow<=blkg*DQB_P + DR_MP[11:0]; d_dst<=3'd3;
                    req<=1'b1; rb_clr=1'b1; nl<=NL_WMP;
                end
                // overlap: start residual-2 for stream 0 as soon as its mlp readback
                // commits; per-stream sready gate stalls later streams.
                NL_WMP: if (rb_rdy[0]) begin ci<=0; civ<=0; bs<=0; nl <= NL_RES2; end
                NL_RES2: if (sready) begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(mlp_r[pp*32 +: 32]);
                        xr_we = 1'b1; xr_wa = sN + cid; xr_wd = sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (bs == G-1) begin
                                bs<=0;
                                if (blkg == NLAYER-1) begin
                                    lnphase<=2'd2; lgam<=NLAYER*2; nl<=NL_LGAM;
                                end else begin
                                    blkg<=blkg+1'b1; lnphase<=2'd0;
                                    lgam<=(blkg+1'b1)*2; nl<=NL_LGAM;
                                end
                            end else bs<=bs+1'b1;
                        end
                    end
                end
                NL_HEAD: begin
                    d_wbase<=WB_HEAD[19:0]; d_m<=VOCAB[10:0]; d_k<=D[10:0];
                    d_asrc<=2'd0; d_asel<=4*NLAYER; d_frac<=7'd25;
                    d_dqrow<=DR_HEAD[11:0]; d_dst<=3'd4;
                    req<=1'b1; rb_clr=1'b1; bs<=0;
                    best_val<=32'sh80000000; best_idx<=0; ar<=0; arv<=0; av1<=0; amv<=0;
                    nl<=NL_WHEAD;
                end
                // overlap: start argmax for stream 0 as soon as its head readback
                // commits; per-stream sready gate stalls later streams.
                NL_WHEAD: if (rb_rdy[0]) begin
                    // re-arm argmax scratch on entry (matches pre-split behaviour)
                    best_val<=32'sh80000000; best_idx<=0; bs<=0;
                    ar<=0; arv<=0; av1<=0; amv<=0;
                    nl <= NL_ARG;
                end
                NL_ARG: if (sready) begin
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
                            nl <= NL_ARG2;
                        end
                    end
                end
                NL_ARG2: begin
                    if (bs == G-1) begin
                        bs<=0; done_o<=1'b1; nl<=NL_DONE;
                    end else begin
                        bs<=bs+1'b1;
                        best_val<=32'sh80000000; best_idx<=0;
                        ar<=0; arv<=0; av1<=0; amv<=0; nl<=NL_ARG;
                    end
                end
                NL_DONE: ;       // hold (done_o stays high until next go)
                default: nl <= NL_IDLE;
            endcase
            // xres_bank's ONE write port. Two producers fold into a single
            // enable/address/data BEFORE the write so the bank stays a clean
            // single-write-port BRAM (a two-branch if/else-if write makes
            // ram_style="block" "infeasible" and Vivado spills it to LUTRAM):
            //  - the embed residual (NL_EMB) arrives via the registered emb_*_p
            //    pipe one cycle late (the 250-gate URAM->add->BRAM cone split);
            //  - the residual adds (NL_RES1/RES2) drive xr_* combinationally.
            // They never coincide (different FSM states; the trailing embed write
            // lands during NL_LGAM's idle-wait, well before NL_LFEED reads back),
            // so embed-priority is exact and bit-identical.
            x_we_w = emb_we_p | xr_we;
            x_wa_w = emb_we_p ? emb_wa_p : xr_wa;
            x_wd_w = emb_we_p ? emb_wd_p : xr_wd;
            if (x_we_w) xres_bank[x_wa_w] <= x_wd_w;     // the bank's ONLY write site
            // ---- stream-granular readback-ready tracker ----------------------
            // Clear at a fresh call boundary (descriptor just issued); otherwise
            // latch each stream's completion pulse. Clear dominates set: a new
            // descriptor's issue cannot coincide with a prior call's last commit
            // (the loop that consumes the prior call must finish first).
            if (rb_clr)        rb_rdy <= {G{1'b0}};
            else if (s_done)   rb_rdy[s_done_idx] <= 1'b1;
        end
    end
endmodule
