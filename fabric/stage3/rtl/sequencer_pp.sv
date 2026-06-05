// -----------------------------------------------------------------------------
// sequencer_pp — PING-PONG batch sequencer: N=8 streams, two groups of G=N/2.
//
// While the GEMM engine streams the resident URAM weight image, two PER-GROUP
// nl_engine instances run their non-linears (embed / LN / attention / residual /
// argmax) IN PARALLEL — one engine per stream-group — instead of the old single
// shared NL FSM that toggled gc<=~gc. The GEMM engine, weight/embed loaders,
// vec_dequant and vec_gelu stay SHARED here. One weight pass can feed all N
// streams (merged) or one group's G streams (solo); the URAM read port never idles.
//
// Engines:
//   GEMM (shared): act-quant feed -> run -> fused readback/dequant/GELU, draining
//        results back into the selected engine's banks via dw_*/dwm_* buses.
//   NL (x2):       embed / LayerNorm / attention / residual / argmax, each requesting
//        GEMM calls via req/d_* and acknowledged by `served`.
//
// Bit-exact per stream vs seq_ref.full_forward_signals (same arithmetic).
// iverilog-2012 traps honoured (plain-reg part-selects, single-write-site banks).
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
    parameter integer ND    = 0,        // DSP-packed GEMM streams (of the N)
    parameter integer GWAIT = 2048,     // merge patience: cycles to hold a lone
                                        // request hoping the partner group asks
                                        // for the same pass (then serve solo)
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

    // ---- shared ROMs ------------------------------------------------------------
    (* ram_style = "ultra" *) reg [P*32-1:0] emb_w [0:VOCAB*EROWS-1];
    reg signed [63:0] inv_sact [0:NSACT-1];
    (* rom_style = "distributed" *) reg [P*24-1:0] dqm_w [0:DQROWS-1];
    (* rom_style = "distributed" *) reg [P*8-1:0]  dqe_w [0:DQROWS-1];
`ifndef SYNTHESIS
    initial begin
        $readmemh("tok_emb_w.mem", emb_w);
    end
`endif
    initial begin
        $readmemh("inv_sact.mem",  inv_sact);
        $readmemh("dqm_w.mem",     dqm_w);
        $readmemh("dqe_w.mem",     dqe_w);
    end

    // ---- embed loader (URAM tok + per-engine BRAM pos, streamed at boot) ---------
    localparam integer EMB_ROWS = (VOCAB + TMAX) * EROWS;
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

    // pos_emb writes are broadcast to BOTH engines' local pos_w copies
    wire        pw_we   = el_commit && (el_word >= VOCAB*EROWS);
    wire [$clog2(TMAX*EROWS)-1:0] pw_addr = el_word - VOCAB*EROWS;
    wire [P*32-1:0] pw_data = el_next;

    // ================================================================
    //              EMBED ARBITER (shared URAM read port)
    // ================================================================
    wire        emb_req0, emb_req1;
    wire [$clog2(VOCAB*EROWS)-1:0] emb_addr0, emb_addr1;
    wire        emb_gnt0 = emb_req0;            // group 0 has priority
    wire        emb_gnt1 = emb_req1 && !emb_req0;
    reg  [P*32-1:0] emb_q;
    always @(posedge clk)
        emb_q <= emb_w[emb_gnt0 ? emb_addr0 : emb_addr1];

    // ================================================================
    //              GE <-> NL engine interconnect
    // ================================================================
    // per-engine descriptor + request (wires from the two nl_engine instances)
    wire [19:0] d_wbase [0:1];
    wire [10:0] d_m [0:1], d_k [0:1];
    wire [1:0]  d_asrc [0:1];
    wire [5:0]  d_asel [0:1];
    wire signed [6:0] d_frac [0:1];
    wire [11:0] d_dqrow [0:1];
    wire [2:0]  d_dst [0:1];
    wire        g_req [0:1];
    wire        done_o0, done_o1;
    wire [G*9-1:0] tok_outs_g0, tok_outs_g1;

    // GE-side registered reads from each engine (muxed by delayed group bit)
    wire [P*32-1:0] e0_lnout1_r, e0_lnout2_r, e0_ctxv_g; wire [P*16-1:0] e0_mlpbuf_r;
    wire [P*32-1:0] e1_lnout1_r, e1_lnout2_r, e1_ctxv_g; wire [P*16-1:0] e1_mlpbuf_r;
    // NL-side registered reads from each engine (host debug)
    wire [P*32-1:0] e0_xres_r, e0_qkv_r, e0_attn_r, e0_mlp_r, e0_head_r;
    wire [P*32-1:0] e1_xres_r, e1_qkv_r, e1_attn_r, e1_mlp_r, e1_head_r;

    wire        g_done_p;            // pulse: current GEMM call fully drained

    // drain-write + AQ-read buses driven by the GE (combinational from GE state)
    reg         dwr_we;            // generic bank drain
    reg  [2:0]  dwr_dst;
    reg  [10:0] dwr_addr;          // LOCAL
    reg  [P*32-1:0] dwr_data;
    reg         dwmr_we;           // mlpbuf drain
    reg  [10:0] dwmr_addr;         // LOCAL
    reg  [P*16-1:0] dwmr_data;
    reg         dwr_eng;           // engine select for both drain buses
    reg  [10:0] ge_ra_local;       // LOCAL AQ/host read address (both engines)

    // GE-side read mux: read regs are 1 cycle behind ge_ra -> delay group bit by 1
    reg         ge_grp_d;
    wire [P*32-1:0] lnout1_r = ge_grp_d ? e1_lnout1_r : e0_lnout1_r;
    wire [P*32-1:0] lnout2_r = ge_grp_d ? e1_lnout2_r : e0_lnout2_r;
    wire [P*32-1:0] ctxv_g   = ge_grp_d ? e1_ctxv_g   : e0_ctxv_g;
    wire [P*16-1:0] mlpbuf_r = ge_grp_d ? e1_mlpbuf_r : e0_mlpbuf_r;

    // NL-side host-debug read mux (rd_stream held constant across a dump)
    wire [P*32-1:0] xres_r = rd_stream[GSH] ? e1_xres_r : e0_xres_r;
    wire [P*32-1:0] qkv_r  = rd_stream[GSH] ? e1_qkv_r  : e0_qkv_r;
    wire [P*32-1:0] attn_r = rd_stream[GSH] ? e1_attn_r : e0_attn_r;
    wire [P*32-1:0] mlp_r  = rd_stream[GSH] ? e1_mlp_r  : e0_mlp_r;
    wire [P*32-1:0] head_r = rd_stream[GSH] ? e1_head_r : e0_head_r;

    // ================================================================
    //         SHARED LayerNorm + Attention (one each, arbitrated)
    // ================================================================
    // Each engine drives ln_*_o / at_*_o request-side signals and qualifies its
    // captures by its own FSM state + its gnt. Fixed priority eng0 > eng1 with HOLD:
    // once granted, the grant is kept until the holder drops its req (req fall edge
    // releases). LN and attn holds within one engine never overlap/nest (see
    // nl_engine header), so the two independent fixed-priority arbiters cannot
    // deadlock.

    // ---- per-engine LN request-side signals ----
    wire        ln_req0, ln_start0, ln_vin0;  wire [P*32-1:0] ln_x0, ln_g0;
    wire        ln_req1, ln_start1, ln_vin1;  wire [P*32-1:0] ln_x1, ln_g1;
    // ---- per-engine attn request-side signals ----
    wire        at_req0, at_start0, at_ldv0;  wire [8:0] at_tcount0;  wire [P*32-1:0] at_lddat0;
    wire        at_req1, at_start1, at_ldv1;  wire [8:0] at_tcount1;  wire [P*32-1:0] at_lddat1;

    // ---- LN arbiter (fixed priority eng0 > eng1, hold until req falls) ----
    reg  ln_owner;        // 0 = eng0 holds, 1 = eng1 holds
    reg  ln_busy;         // a grant is currently held
    always @(posedge clk) begin
        if (rst) begin ln_busy <= 1'b0; ln_owner <= 1'b0; end
        else if (!ln_busy) begin
            if (ln_req0)      begin ln_busy <= 1'b1; ln_owner <= 1'b0; end
            else if (ln_req1) begin ln_busy <= 1'b1; ln_owner <= 1'b1; end
        end else begin
            // release when the current owner drops its request
            if ((ln_owner == 1'b0 && !ln_req0) || (ln_owner == 1'b1 && !ln_req1))
                ln_busy <= 1'b0;
        end
    end
    wire ln_gnt0 = ln_busy && (ln_owner == 1'b0);
    wire ln_gnt1 = ln_busy && (ln_owner == 1'b1);

    // ---- attn arbiter (same shape) ----
    reg  at_owner;
    reg  at_busy;
    always @(posedge clk) begin
        if (rst) begin at_busy <= 1'b0; at_owner <= 1'b0; end
        else if (!at_busy) begin
            if (at_req0)      begin at_busy <= 1'b1; at_owner <= 1'b0; end
            else if (at_req1) begin at_busy <= 1'b1; at_owner <= 1'b1; end
        end else begin
            if ((at_owner == 1'b0 && !at_req0) || (at_owner == 1'b1 && !at_req1))
                at_busy <= 1'b0;
        end
    end
    wire at_gnt0 = at_busy && (at_owner == 1'b0);
    wire at_gnt1 = at_busy && (at_owner == 1'b1);

    // ---- request-side muxes into the shared units (granted engine) ----
    wire           sh_ln_start = ln_gnt1 ? ln_start1 : ln_start0;
    wire           sh_ln_vin   = ln_gnt1 ? ln_vin1   : ln_vin0;
    wire [P*32-1:0] sh_ln_x    = ln_gnt1 ? ln_x1     : ln_x0;
    wire [P*32-1:0] sh_ln_g    = ln_gnt1 ? ln_g1     : ln_g0;
    wire           sh_ln_yv;
    wire [P*64-1:0] sh_ln_y;
    wire           sh_ln_done;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(sh_ln_start), .valid_in(sh_ln_vin),
        .x_in(sh_ln_x), .gamma_in(sh_ln_g),
        .y_valid(sh_ln_yv), .y_out(sh_ln_y), .done(sh_ln_done));

    wire           sh_at_start = at_gnt1 ? at_start1 : at_start0;
    wire [8:0]     sh_at_tcount= at_gnt1 ? at_tcount1: at_tcount0;
    wire           sh_at_ldv   = at_gnt1 ? at_ldv1   : at_ldv0;
    wire [P*32-1:0] sh_at_lddat= at_gnt1 ? at_lddat1 : at_lddat0;
    wire           sh_at_ldready;
    wire           sh_at_ctxv;
    wire [6:0]     sh_at_ctxidx;
    wire [P*32-1:0] sh_at_ctxdata;
    wire           sh_at_done;
    vec_attn #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(32)) u_attn (
        .clk(clk), .rst(rst), .start(sh_at_start), .tcount(sh_at_tcount),
        .ld_valid(sh_at_ldv), .ld_data(sh_at_lddat), .ld_ready(sh_at_ldready),
        .ctx_valid(sh_at_ctxv), .ctx_idx(sh_at_ctxidx), .ctx_data(sh_at_ctxdata),
        .done(sh_at_done));

    nl_engine #(.D(D), .D3(D3), .D_MLP(D_MLP), .P(P), .LANES(LANES), .G(G),
                .VOCAB(VOCAB), .TMAX(TMAX), .GAMMA_N(GAMMA_N), .NLAYER(NLAYER),
                .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .RESID_FRAC(RESID_FRAC),
                .LN_OUT_FRAC(LN_OUT_FRAC), .VFRAC(VFRAC), .GELU_FRAC(GELU_FRAC),
                .ISH(ISH)) eng0 (
        .clk(clk), .rst(rst), .go(go),
        .tok_ids(tok_ids[G*9-1:0]), .pos(pos),
        .served(g_done_p && (gmerge || ggrp == 1'b0)),
        .emb_gnt(emb_gnt0), .emb_q(emb_q), .emb_req(emb_req0), .emb_addr(emb_addr0),
        .dw_we(dwr_we && (dwr_eng == 1'b0)), .dw_dst(dwr_dst), .dw_addr(dwr_addr), .dw_data(dwr_data),
        .dwm_we(dwmr_we && (dwr_eng == 1'b0)), .dwm_addr(dwmr_addr), .dwm_data(dwmr_data),
        .ge_ra(ge_ra_local),
        .dbg_stream(rd_stream[GSH-1:0]), .dbg_addr(rd_addr),
        .pw_we(pw_we), .pw_addr(pw_addr), .pw_data(pw_data),
        .ln_req(ln_req0), .ln_start_o(ln_start0), .ln_vin_o(ln_vin0),
        .ln_x_o(ln_x0), .ln_g_o(ln_g0),
        .ln_gnt(ln_gnt0), .ln_yv_i(sh_ln_yv), .ln_y_i(sh_ln_y), .ln_done_i(sh_ln_done),
        .at_req(at_req0), .at_start_o(at_start0), .at_tcount_o(at_tcount0),
        .at_ldv_o(at_ldv0), .at_ld_data_o(at_lddat0),
        .at_gnt(at_gnt0), .at_ldready_i(sh_at_ldready), .at_ctxv_i(sh_at_ctxv),
        .at_ctxidx_i(sh_at_ctxidx), .at_ctxdata_i(sh_at_ctxdata), .at_done_i(sh_at_done),
        .req(g_req[0]), .d_wbase(d_wbase[0]), .d_m(d_m[0]), .d_k(d_k[0]),
        .d_asrc(d_asrc[0]), .d_asel(d_asel[0]), .d_frac(d_frac[0]),
        .d_dqrow(d_dqrow[0]), .d_dst(d_dst[0]), .done_o(done_o0),
        .tok_outs_g(tok_outs_g0),
        .lnout1_r(e0_lnout1_r), .lnout2_r(e0_lnout2_r), .ctxv_g(e0_ctxv_g),
        .mlpbuf_r(e0_mlpbuf_r),
        .xres_r(e0_xres_r), .qkv_r(e0_qkv_r), .attn_r(e0_attn_r),
        .mlp_r(e0_mlp_r), .head_r(e0_head_r));

    nl_engine #(.D(D), .D3(D3), .D_MLP(D_MLP), .P(P), .LANES(LANES), .G(G),
                .VOCAB(VOCAB), .TMAX(TMAX), .GAMMA_N(GAMMA_N), .NLAYER(NLAYER),
                .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .RESID_FRAC(RESID_FRAC),
                .LN_OUT_FRAC(LN_OUT_FRAC), .VFRAC(VFRAC), .GELU_FRAC(GELU_FRAC),
                .ISH(ISH)) eng1 (
        .clk(clk), .rst(rst), .go(go),
        .tok_ids(tok_ids[N*9-1:G*9]), .pos(pos),
        .served(g_done_p && (gmerge || ggrp == 1'b1)),
        .emb_gnt(emb_gnt1), .emb_q(emb_q), .emb_req(emb_req1), .emb_addr(emb_addr1),
        .dw_we(dwr_we && (dwr_eng == 1'b1)), .dw_dst(dwr_dst), .dw_addr(dwr_addr), .dw_data(dwr_data),
        .dwm_we(dwmr_we && (dwr_eng == 1'b1)), .dwm_addr(dwmr_addr), .dwm_data(dwmr_data),
        .ge_ra(ge_ra_local),
        .dbg_stream(rd_stream[GSH-1:0]), .dbg_addr(rd_addr),
        .pw_we(pw_we), .pw_addr(pw_addr), .pw_data(pw_data),
        .ln_req(ln_req1), .ln_start_o(ln_start1), .ln_vin_o(ln_vin1),
        .ln_x_o(ln_x1), .ln_g_o(ln_g1),
        .ln_gnt(ln_gnt1), .ln_yv_i(sh_ln_yv), .ln_y_i(sh_ln_y), .ln_done_i(sh_ln_done),
        .at_req(at_req1), .at_start_o(at_start1), .at_tcount_o(at_tcount1),
        .at_ldv_o(at_ldv1), .at_ld_data_o(at_lddat1),
        .at_gnt(at_gnt1), .at_ldready_i(sh_at_ldready), .at_ctxv_i(sh_at_ctxv),
        .at_ctxidx_i(sh_at_ctxidx), .at_ctxdata_i(sh_at_ctxdata), .at_done_i(sh_at_done),
        .req(g_req[1]), .d_wbase(d_wbase[1]), .d_m(d_m[1]), .d_k(d_k[1]),
        .d_asrc(d_asrc[1]), .d_asel(d_asel[1]), .d_frac(d_frac[1]),
        .d_dqrow(d_dqrow[1]), .d_dst(d_dst[1]), .done_o(done_o1),
        .tok_outs_g(tok_outs_g1),
        .lnout1_r(e1_lnout1_r), .lnout2_r(e1_lnout2_r), .ctxv_g(e1_ctxv_g),
        .mlpbuf_r(e1_mlpbuf_r),
        .xres_r(e1_xres_r), .qkv_r(e1_qkv_r), .attn_r(e1_attn_r),
        .mlp_r(e1_mlp_r), .head_r(e1_head_r));

    assign tok_outs = {tok_outs_g1, tok_outs_g0};

    // done: pulse on rising edge of (both engines done)
    wire both_done = done_o0 && done_o1;
    reg  both_done_d;
    always @(posedge clk) begin
        if (rst) begin done <= 1'b0; both_done_d <= 1'b0; end
        else begin
            done <= both_done && !both_done_d;
            both_done_d <= both_done;
            if (go) both_done_d <= 1'b0;
        end
    end

    // ================================================================
    //                       GEMM ENGINE  (shared)
    // ================================================================
    reg                gv_ldrst, gv_xrst, gv_xwe, gv_start;
    reg  [$clog2(N)-1:0] gv_xstream, gv_rdstream;
    reg  [P*8-1:0]     gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    reg [10:0]         gv_rdaddr;
    wire [P*32-1:0]    gv_yout;
    (* keep_hierarchy = "yes" *)
    gemm_banked_resident_vec #(.LANES(LANES), .N(N), .ND(ND), .P(P), .MMAX(1024),
                  .KMAX(1024), .RLAT(2), .WWORDS(WWORDS)) u_gemm (
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
    reg [17:0] gwait;                   // solo-serve patience: wait for the partner
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
    wire             sid_eng = sid[GSH];          // group (engine) of current stream
    wire [GSH-1:0]   sidl    = sid[GSH-1:0];      // LOCAL stream within its engine
    wire [10:0] sGl  = sidl * ROWS;
    wire [10:0] sG3l = sidl * ROWS3;
    wire [10:0] sGMl = sidl * ROWSM;
    wire [10:0] sGAl = sidl * ARROWS;

    localparam [2:0] GE_IDLE=0, GE_AQ=1, GE_AQN=2, GE_RUN=3, GE_WAIT=4, GE_RB=5, GE_RBN=6;
    reg [2:0] ge;

    // GE-side read address (LOCAL): board readback reuses this port while idle
    wire [10:0] rbr   = rd_stream[GSH-1:0]*ROWS   + (rd_addr >> LSH);
    wire [10:0] rbrM  = rd_stream[GSH-1:0]*ROWSM  + (rd_addr >> LSH);
    wire        ge_grp = (ge == GE_IDLE) ? rd_stream[GSH] : sid_eng;
    always @(posedge clk) ge_grp_d <= ge_grp;
    always @* begin
        if (ge == GE_IDLE)
            ge_ra_local = (rd_sel == 4'd5) ? rbrM : rbr;
        else
            ge_ra_local = (a_asrc == 2'd3) ? (sGMl + gci) : (sGl + gci);
    end

    always @(posedge clk) begin
        mwr      <= dqm_w[a_dqrow + rb1];
        ewr      <= dqe_w[a_dqrow + rb1];
    end

    wire rb_last = (a_dst == 3'd2) ? (gl_vout && gor == ROWSM-1 && gbs == glim)
                                   : (dq_vout && gdor == ((a_m + P-1) >> LSH) - 1 && gbs == glim);
    assign g_done_p = (ge == GE_RB) && rb_last;

    // combinational dequant/saturate words used by the drain bus + gelu feed
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

    // drain-write buses (combinational): mirror the cycle the old bank write fired
    always @* begin
        dwr_we = 1'b0; dwr_dst = 3'd0; dwr_addr = 11'd0; dwr_data = dword;
        dwmr_we = 1'b0; dwmr_addr = 11'd0; dwmr_data = gl_y;
        dwr_eng = sid_eng;
        if (ge == GE_RB) begin
            if (dq_vout) begin
                case (a_dst)
                    3'd0: begin dwr_we = 1'b1; dwr_dst = 3'd0; dwr_addr = sG3l + gdor; end
                    3'd1: begin dwr_we = 1'b1; dwr_dst = 3'd1; dwr_addr = sGl  + gdor; end
                    3'd2: ;   // gelu hidden: routed through dwm on gl_vout
                    3'd3: begin dwr_we = 1'b1; dwr_dst = 3'd3; dwr_addr = sGl  + gdor; end
                    default: begin dwr_we = 1'b1; dwr_dst = 3'd4; dwr_addr = sGAl + gdor; end
                endcase
            end
            if (gl_vout) begin dwmr_we = 1'b1; dwmr_addr = sGMl + gor; end
        end
    end

    always @(posedge clk) begin
        gv_ldrst<=0; gv_xrst<=0; gv_xwe<=0; gv_start<=0; dq_vin<=0; gl_vin<=0;
        if (rst) begin
            ge<=GE_IDLE; gbs<=0; gci<=0; gciv<=0; gciv1<=0; gciv2<=0;
            rv0<=0; rv1<=0; rv2<=0; gdor<=0; gor<=0;
        end else begin
            case (ge)
                GE_IDLE: begin
                    // SINGLE PASS: wait for BOTH groups to request the SAME call
                    if (g_req[0] && g_req[1] && d_wbase[0] == d_wbase[1]) begin
                        gwait   <= 0;
                        gmerge  <= 1'b1;  ggrp <= 1'b0;
                        a_wbase <= d_wbase[0]; a_m <= d_m[0]; a_k <= d_k[0];
                        a_asrc  <= d_asrc[0];  a_asel <= d_asel[0];
                        a_frac  <= d_frac[0];  a_dqrow <= d_dqrow[0];
                        a_dst   <= d_dst[0];
                        gbs<=0; gci<=0; gciv<=0; gciv1<=0; gciv2<=0;
                        gv_ldrst<=1'b1; ge<=GE_AQ;
                    end else if ((g_req[0] || g_req[1]) && gwait < GWAIT[17:0]) begin
                        gwait <= gwait + 1'b1;
                    end else if (g_req[0] || g_req[1]) begin
                        gwait   <= 0;
                        gmerge  <= 1'b0;
                        ggrp    <= g_req[0] ? 1'b0 : 1'b1;
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
                        // bank write now via dwr_* bus (combinational, this cycle)
                        if (a_dst == 3'd2) begin gl_vin<=1'b1; gl_x<=mword; end
                        if (a_dst != 3'd2 && gdor==((a_m + P-1) >> LSH)-1) begin
                            gci<=0; gdor<=0; rv0<=0; rv1<=0; rv2<=0;
                            if (gbs == glim) begin gbs<=0; ge<=GE_IDLE; end
                            else begin grbn<=0; ge<=GE_RBN; end
                        end else gdor<=gdor+1'b1;
                    end
                    if (gl_vout) begin
                        // mlpbuf write via dwmr_* bus (combinational, this cycle)
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
    integer pp;
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
