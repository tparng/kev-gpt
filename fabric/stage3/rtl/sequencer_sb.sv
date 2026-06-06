// -----------------------------------------------------------------------------
// sequencer_sb — SPLIT-BRAIN sequencer: two fully independent N=8 cohorts
// (cohort_engine) sharing ONLY the resident weight image (weight_bank_tdp, read
// through both TDP ports) and the shared embed / LayerNorm / attention units
// (arbitrated 2-way, exactly as sequencer_pp shared them between its groups).
//
// 16 tokens / pass: cohort 0 = streams 0..7, cohort 1 = streams 8..15. There is
// NO merge and NO GWAIT — the cohorts never share a weight pass; each streams the
// URAM through its own read port at its own layer/call. The 73k cycles that used
// to queue behind one URAM read port in the N=16 single engine now run in
// parallel: target ~63k cyc / 16 tok (vs 99,828) -> ~42.3k tok/s @166.7.
//
// Loaders: wl_* writes the shared weight image (port A of weight_bank_tdp at
// boot); el_we writes the shared embed/pos image (tok_emb URAM + broadcast pos).
// Top-level interface matches sequencer_pp so tb_seq_sb mirrors tb_seq_pp.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_sb #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 128,
    parameter integer N     = 16,       // total streams (two cohorts of NC)
    parameter integer NC    = 8,        // streams per cohort
    parameter integer ND    = 0,        // DSP-packed GEMM streams per cohort
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
    localparam integer EROWS  = D / P;
    localparam integer CSH    = $clog2(NC);    // bit selecting cohort within stream

    // ================================================================
    //         SHARED weight image (dual-port URAM, weight_bank_tdp)
    // ================================================================
    wire [$clog2(WWORDS)-1:0] waddr0, waddr1;
    wire [LANES*4-1:0]        wword0, wword1;
    weight_bank_tdp #(.LANES(LANES), .WWORDS(WWORDS)) u_wbank (
        .clk(clk),
        .ld_rst(wl_rst), .w_we(wl_we), .w_data(wl_data),
        .raddr_b(waddr0), .rword_b(wword0),     // cohort 0 -> port B
        .raddr_a(waddr1), .rword_a(wword1));    // cohort 1 -> port A

    // ================================================================
    //         SHARED embed/pos image + loader + read arbiter
    // ================================================================
    (* ram_style = "ultra" *) reg [P*32-1:0] emb_w [0:VOCAB*EROWS-1];
`ifndef SYNTHESIS
    initial begin $readmemh("tok_emb_w.mem", emb_w); end
`endif
    localparam integer ESUB = (P*32)/32;
    reg [$clog2((VOCAB+TMAX)*EROWS):0] el_word;
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

    wire        pw_we   = el_commit && (el_word >= VOCAB*EROWS);
    wire [$clog2(TMAX*EROWS)-1:0] pw_addr = el_word - VOCAB*EROWS;
    wire [P*32-1:0] pw_data = el_next;

    // embed read arbiter (cohort 0 priority)
    wire        emb_req0, emb_req1;
    wire [$clog2(VOCAB*EROWS)-1:0] emb_addr0, emb_addr1;
    wire        emb_gnt0 = emb_req0;
    wire        emb_gnt1 = emb_req1 && !emb_req0;
    reg  [P*32-1:0] emb_q;
    always @(posedge clk)
        emb_q <= emb_w[emb_gnt0 ? emb_addr0 : emb_addr1];

    // ================================================================
    //         SHARED LayerNorm + Attention (arbitrated 2-way)
    // ================================================================
    wire        ln_req0, ln_start0, ln_vin0;  wire [P*32-1:0] ln_x0, ln_g0;
    wire        ln_req1, ln_start1, ln_vin1;  wire [P*32-1:0] ln_x1, ln_g1;
    wire        at_req0, at_start0, at_ldv0;  wire [8:0] at_tcount0;  wire [P*32-1:0] at_lddat0;
    wire        at_req1, at_start1, at_ldv1;  wire [8:0] at_tcount1;  wire [P*32-1:0] at_lddat1;

    reg  ln_owner, ln_busy;
    always @(posedge clk) begin
        if (rst) begin ln_busy <= 1'b0; ln_owner <= 1'b0; end
        else if (!ln_busy) begin
            if (ln_req0)      begin ln_busy <= 1'b1; ln_owner <= 1'b0; end
            else if (ln_req1) begin ln_busy <= 1'b1; ln_owner <= 1'b1; end
        end else begin
            if ((ln_owner == 1'b0 && !ln_req0) || (ln_owner == 1'b1 && !ln_req1))
                ln_busy <= 1'b0;
        end
    end
    wire ln_gnt0 = ln_busy && (ln_owner == 1'b0);
    wire ln_gnt1 = ln_busy && (ln_owner == 1'b1);

    reg  at_owner, at_busy;
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

    // ================================================================
    //   SHARED dequant+gelu readback channel (arbitrated 2-way)
    //   One vec_dequant + one vec_gelu + ONE copy of the dqm/dqe ROMs
    //   serve both cohorts; a cohort holds the channel for one whole
    //   GEMM call's readback (req spans GE_DQW..GE_RB..IDLE), so beats
    //   never interleave. Replicating all this cost ~12k LUT at N=16.
    // ================================================================
    localparam integer DQROWS = (DQ_N + P - 1) / P;
    (* rom_style = "distributed" *) reg [P*24-1:0] dqm_w [0:DQROWS-1];
    (* rom_style = "distributed" *) reg [P*8-1:0]  dqe_w [0:DQROWS-1];
    initial begin
        $readmemh("dqm_w.mem", dqm_w);
        $readmemh("dqe_w.mem", dqe_w);
    end

    wire        dq_req0, dq_req1, dq_vin0, dq_vin1, gl_vin0, gl_vin1;
    wire [11:0] dq_ra0, dq_ra1;
    wire signed [6:0] dq_frac0, dq_frac1;
    wire [P*32-1:0] dq_gemvy0, dq_gemvy1;
    wire [P*16-1:0] gl_x0, gl_x1;

    reg  dq_owner, dq_busy;
    always @(posedge clk) begin
        if (rst) begin dq_busy <= 1'b0; dq_owner <= 1'b0; end
        else if (!dq_busy) begin
            if (dq_req0)      begin dq_busy <= 1'b1; dq_owner <= 1'b0; end
            else if (dq_req1) begin dq_busy <= 1'b1; dq_owner <= 1'b1; end
        end else begin
            if ((dq_owner == 1'b0 && !dq_req0) || (dq_owner == 1'b1 && !dq_req1))
                dq_busy <= 1'b0;
        end
    end
    wire dq_gnt0 = dq_busy && (dq_owner == 1'b0);
    wire dq_gnt1 = dq_busy && (dq_owner == 1'b1);

    // free-running 2-stage ROM fetch on the granted cohort's row address —
    // stage 1 = the old per-cohort mwr/ewr register, stage 2 = the old
    // dq_mant/dq_exp input register. vin/gemvy are registered cohort-side at
    // the rv2 stage, so all vec_dequant inputs stay cycle-aligned.
    wire [11:0] sh_dq_ra = dq_gnt1 ? dq_ra1 : dq_ra0;
    reg [P*24-1:0] mwr_sh, mant_sh;
    reg [P*8-1:0]  ewr_sh, exp_sh;
    always @(posedge clk) begin
        mwr_sh  <= dqm_w[sh_dq_ra];
        ewr_sh  <= dqe_w[sh_dq_ra];
        mant_sh <= mwr_sh;
        exp_sh  <= ewr_sh;
    end

    wire           sh_dq_vin   = (dq_vin0 && dq_gnt0) || (dq_vin1 && dq_gnt1);
    wire signed [6:0] sh_dq_frac = dq_gnt1 ? dq_frac1  : dq_frac0;
    wire [P*32-1:0] sh_dq_gemvy = dq_gnt1 ? dq_gemvy1 : dq_gemvy0;
    wire           sh_dq_vout;
    wire [P*32-1:0] sh_dq_out;
    vec_dequant #(.P(P)) u_dq (
        .clk(clk), .rst(rst), .in_valid(sh_dq_vin), .frac(sh_dq_frac),
        .gemvy(sh_dq_gemvy), .mant(mant_sh), .exp(exp_sh),
        .out_valid(sh_dq_vout), .dq_out(sh_dq_out));

    wire           sh_gl_vin = (gl_vin0 && dq_gnt0) || (gl_vin1 && dq_gnt1);
    wire [P*16-1:0] sh_gl_x  = dq_gnt1 ? gl_x1 : gl_x0;
    wire           sh_gl_vout;
    wire [P*16-1:0] sh_gl_y;
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(sh_gl_vin), .x(sh_gl_x),
        .out_valid(sh_gl_vout), .y(sh_gl_y));

    // ================================================================
    //                       COHORT ENGINES (x2)
    // ================================================================
    wire [NC*9-1:0] tok_outs0, tok_outs1;
    wire        done_o0, done_o1;
    wire signed [63:0] rd_data0, rd_data1;

    cohort_engine #(.D(D), .D3(D3), .D_MLP(D_MLP), .P(P), .LANES(LANES), .N(NC),
                    .ND(ND), .VOCAB(VOCAB), .TMAX(TMAX), .GAMMA_N(GAMMA_N),
                    .DQ_N(DQ_N), .NSACT(NSACT), .WWORDS(WWORDS), .NLAYER(NLAYER),
                    .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .RESID_FRAC(RESID_FRAC),
                    .LN_OUT_FRAC(LN_OUT_FRAC), .VFRAC(VFRAC), .GELU_FRAC(GELU_FRAC),
                    .ISH(ISH)) coh0 (
        .clk(clk), .rst(rst), .go(go),
        .tok_ids(tok_ids[NC*9-1:0]), .pos(pos),
        .done_o(done_o0), .tok_outs(tok_outs0),
        .rd_stream(rd_stream[CSH-1:0]), .rd_sel(rd_sel), .rd_addr(rd_addr),
        .rd_data(rd_data0),
        .waddr(waddr0), .wword_rd(wword0),
        .emb_req(emb_req0), .emb_addr(emb_addr0), .emb_gnt(emb_gnt0), .emb_q(emb_q),
        .pw_we(pw_we), .pw_addr(pw_addr), .pw_data(pw_data),
        .ln_req(ln_req0), .ln_start_o(ln_start0), .ln_vin_o(ln_vin0),
        .ln_x_o(ln_x0), .ln_g_o(ln_g0),
        .ln_gnt(ln_gnt0), .ln_yv_i(sh_ln_yv), .ln_y_i(sh_ln_y), .ln_done_i(sh_ln_done),
        .at_req(at_req0), .at_start_o(at_start0), .at_tcount_o(at_tcount0),
        .at_ldv_o(at_ldv0), .at_ld_data_o(at_lddat0),
        .at_gnt(at_gnt0), .at_ldready_i(sh_at_ldready), .at_ctxv_i(sh_at_ctxv),
        .at_ctxidx_i(sh_at_ctxidx), .at_ctxdata_i(sh_at_ctxdata), .at_done_i(sh_at_done),
        .dq_req(dq_req0), .dq_gnt(dq_gnt0), .dq_raddr_o(dq_ra0),
        .dq_vin_o(dq_vin0), .dq_frac_o(dq_frac0), .dq_gemvy_o(dq_gemvy0),
        .dq_vout_i(sh_dq_vout), .dq_out_i(sh_dq_out),
        .gl_vin_o(gl_vin0), .gl_x_o(gl_x0),
        .gl_vout_i(sh_gl_vout), .gl_y_i(sh_gl_y));

    cohort_engine #(.D(D), .D3(D3), .D_MLP(D_MLP), .P(P), .LANES(LANES), .N(NC),
                    .ND(ND), .VOCAB(VOCAB), .TMAX(TMAX), .GAMMA_N(GAMMA_N),
                    .DQ_N(DQ_N), .NSACT(NSACT), .WWORDS(WWORDS), .NLAYER(NLAYER),
                    .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .RESID_FRAC(RESID_FRAC),
                    .LN_OUT_FRAC(LN_OUT_FRAC), .VFRAC(VFRAC), .GELU_FRAC(GELU_FRAC),
                    .ISH(ISH)) coh1 (
        .clk(clk), .rst(rst), .go(go),
        .tok_ids(tok_ids[N*9-1:NC*9]), .pos(pos),
        .done_o(done_o1), .tok_outs(tok_outs1),
        .rd_stream(rd_stream[CSH-1:0]), .rd_sel(rd_sel), .rd_addr(rd_addr),
        .rd_data(rd_data1),
        .waddr(waddr1), .wword_rd(wword1),
        .emb_req(emb_req1), .emb_addr(emb_addr1), .emb_gnt(emb_gnt1), .emb_q(emb_q),
        .pw_we(pw_we), .pw_addr(pw_addr), .pw_data(pw_data),
        .ln_req(ln_req1), .ln_start_o(ln_start1), .ln_vin_o(ln_vin1),
        .ln_x_o(ln_x1), .ln_g_o(ln_g1),
        .ln_gnt(ln_gnt1), .ln_yv_i(sh_ln_yv), .ln_y_i(sh_ln_y), .ln_done_i(sh_ln_done),
        .at_req(at_req1), .at_start_o(at_start1), .at_tcount_o(at_tcount1),
        .at_ldv_o(at_ldv1), .at_ld_data_o(at_lddat1),
        .at_gnt(at_gnt1), .at_ldready_i(sh_at_ldready), .at_ctxv_i(sh_at_ctxv),
        .at_ctxidx_i(sh_at_ctxidx), .at_ctxdata_i(sh_at_ctxdata), .at_done_i(sh_at_done),
        .dq_req(dq_req1), .dq_gnt(dq_gnt1), .dq_raddr_o(dq_ra1),
        .dq_vin_o(dq_vin1), .dq_frac_o(dq_frac1), .dq_gemvy_o(dq_gemvy1),
        .dq_vout_i(sh_dq_vout), .dq_out_i(sh_dq_out),
        .gl_vin_o(gl_vin1), .gl_x_o(gl_x1),
        .gl_vout_i(sh_gl_vout), .gl_y_i(sh_gl_y));

    assign tok_outs = {tok_outs1, tok_outs0};

    // done: pulse on rising edge of (both cohorts done)
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

    // host-debug readback mux: cohort selected by the high stream bit
    reg rd_coh_d;
    always @(posedge clk) rd_coh_d <= rd_stream[CSH];
    always @* rd_data = rd_coh_d ? rd_data1 : rd_data0;
endmodule
