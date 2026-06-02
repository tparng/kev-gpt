// -----------------------------------------------------------------------------
// sequencer — Tier-3 single-stream transformer control FSM (CPU out of the loop).
//
// Runs the MULTI-TOKEN KV-cached autoregressive decode dataflow as hardware, wiring
// the four GATED datapath blocks (gemv_banked, layernorm, softmax, gelu_lut) under one
// control FSM with ZERO host I/O between the `go` pulse and the N-token stream. The
// per-token forward (NLAYER blocks + final LN + head + argmax):
//
//   embed -> NLAYER x [ LN1 -> qkv GEMV -> attn(softmax+KV) -> proj GEMV -> +res
//                       -> LN2 -> mlp_fc GEMV -> GELU -> mlp_proj GEMV -> +res ]
//         -> LN_f -> head GEMV -> argmax -> tok_out
//
// is wrapped in an autoregressive loop. A prompt of PROMPT_LEN tokens (resident in
// prompt_rom) primes positions 0..PROMPT_LEN-1; the token predicted at position t is
// fed back as the input at position t+1, advancing tcur. The emitted greedy stream
// (the NGEN tokens predicted from position PROMPT_LEN-1 onward) is written to an
// on-chip tok_stream buffer the host drains after `done`.
//
// PER-BLOCK PERSISTENT KV: each of the NLAYER blocks has its own INT8/Q.VFRAC K and V
// cache, indexed [blk][position]. At position t, block b appends its k,v at slot t and
// attends over slots 0..t; the caches survive between tokens. This is the fix that
// makes the multi-block, multi-position attention correct (the old single shared KV
// buffer was correct only for one token at pos=0).
//
// Gates (fabric/stage3/run_sequencer.py), all vs fabric/stage3/seq_ref:
//   NLAYER=1 : one transformer block, residual x_out BIT-EXACT (mismatches=0/256)
//              vs seq_ref.IntSequencer.block0_signals.
//   NLAYER=4 single-token : full forward, emitted tok_out IDENTICAL to seq_ref.step.
//   NLAYER=4 multitoken   : the full NGEN-token greedy stream IDENTICAL to
//              seq_ref.IntSequencer.generate_greedy (the binding token-stream gate).
//
// Pinned formats (bit-true to seq_ref / the committed blocks):
//   residual x   signed Q6.25 (32-bit)     gamma  signed Q4.20   LN out  signed Q.22
//   GEMV act     INT8 = clip(round(real/s_act))   weight INT4   accum INT32
//   dequant      per-channel y_real = y_int * mant * 2^exp  (mant 24-bit, folded scale)
//   softmax      scores Q8.8, probs Q1.20  gelu Q4.12 in/out
//
// One shared gemv_banked instance runs all four GEMVs (weights streamed from a resident
// weight ROM each use) — correctness-first; a per-GEMV resident array is the throughput
// follow-up. All tables/weights are host-preloaded via $readmemh ROMs (the testbench
// writes the .mem files). Once `go` pulses, the FSM runs the block with zero host I/O.
//
// iverilog -g2012 safe: single-write memories, registered ROM reads, copy unpacked
// element to a plain vector before any variable index op, procedural always-FSM.
// All glue arithmetic (requant, dequant, score, ctx) mirrors seq_ref exactly.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer #(
    parameter integer D        = 256,
    parameter integer D3       = 768,
    parameter integer D_MLP    = 1024,
    parameter integer NHEAD    = 4,
    parameter integer HEAD_DIM = 64,
    parameter integer NLAYER   = 4,       // transformer blocks (full forward)
    parameter integer VOCAB    = 193,
    parameter integer LANES    = 16,
    parameter integer TMAX     = 256,
    parameter integer KVMAX      = 32,    // max cached positions (prompt+gen); KV mem depth
    parameter integer PROMPT_LEN = 8,     // resident prompt tokens (prime the cache)
    parameter integer NGEN       = 8,     // tokens to greedily generate after the prompt
    parameter integer RESID_FRAC  = 25,
    parameter integer LN_OUT_FRAC = 22,
    parameter integer GELU_FRAC   = 12,
    parameter integer SCORE_FRAC  = 8,
    parameter integer PROB_FRAC   = 20,
    parameter integer VFRAC       = 16,   // K/V stored Q.16
    parameter integer ISH         = 40    // inv_sact fixed-point shift
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    input  wire [7:0]  tok_id,           // legacy single-token input (pos=0 path); prompt[0]
    input  wire [8:0]  pos,              // legacy single-token start position
    output reg         busy,
    output reg         done,
    input  wire [8:0]  rd_addr,
    output reg signed [31:0] x_out,
    output reg  [8:0]  tok_out,          // argmax of logits (the most-recent emitted token id)
    // multi-token stream readout: tok_stream[0..ngen_out-1] is the greedy stream
    input  wire [8:0]  ts_addr,
    output reg  [8:0]  ts_out,
    output reg  [8:0]  ngen_out          // number of generated tokens written to tok_stream
);
    localparam integer WBITS   = LANES*4;
    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;
    localparam integer GW_MP   = ((D    + LANES-1)/LANES) * D_MLP;
    localparam integer GW_BLK  = GW_QKV + GW_PROJ + GW_FC + GW_MP;   // words / block
    localparam integer GW_HEAD = ((VOCAB + LANES-1)/LANES) * D;
    localparam integer WROM_N  = GW_BLK * NLAYER + GW_HEAD;
    localparam integer DQ_BLK  = D3 + D + D_MLP + D;                 // dequant / block
    localparam integer DQ_N    = DQ_BLK * NLAYER + VOCAB;            // + head channels
    localparam integer GAMMA_N = 2*NLAYER + 1;                       // ln1|ln2 /blk + ln_f
    localparam integer NSACT   = 4*NLAYER + 1;                       // 4 GEMV/blk + head

    // ------------------------------------------------------------------- ROMs --
    (* rom_style = "block" *) reg signed [31:0] tok_emb   [0:VOCAB*D-1];
    (* rom_style = "block" *) reg signed [31:0] pos_emb   [0:TMAX*D-1];
    (* rom_style = "block" *) reg signed [31:0] gamma_rom [0:GAMMA_N*D-1]; // (ln1|ln2)*L | ln_f
    (* rom_style = "block" *) reg [WBITS-1:0]   wrom      [0:WROM_N-1];
    (* rom_style = "block" *) reg signed [31:0] dq_mant   [0:DQ_N-1];
    (* rom_style = "block" *) reg signed [7:0]  dq_exp    [0:DQ_N-1];
    (* rom_style = "block" *) reg signed [63:0] inv_sact  [0:NSACT-1];
    (* rom_style = "block" *) reg [8:0]          prompt_rom[0:PROMPT_LEN-1];   // resident prompt
    initial begin
        $readmemh("tok_emb.mem", tok_emb);
        $readmemh("pos_emb.mem", pos_emb);
        $readmemh("gamma.mem",   gamma_rom);
        $readmemh("wrom.mem",    wrom);
        $readmemh("dq_mant.mem", dq_mant);
        $readmemh("dq_exp.mem",  dq_exp);
        $readmemh("inv_sact.mem",inv_sact);
        $readmemh("prompt.mem",  prompt_rom);
    end

    // --------------------------------------------------------------- scratch ---
    reg signed [31:0] xres  [0:D-1];        // residual Q6.25
    reg signed [31:0] lnbuf [0:D-1];        // LN input snapshot
    reg signed [63:0] lnout [0:D-1];        // LN out Q.22 (GEMV act source, K<=D)
    reg signed [63:0] mlpbuf[0:D_MLP-1];    // MLP hidden: Q4.12 (gelu in) then Q.22 (gelu out)
    reg signed [31:0] qvec  [0:D-1];        // q dequantized, stored Q.VFRAC
    // PER-BLOCK persistent K/V caches (Q.VFRAC), indexed [blk*KVMAX*D + tpos*D + d].
    reg signed [31:0] kcache[0:NLAYER*KVMAX*D-1];
    reg signed [31:0] vcache[0:NLAYER*KVMAX*D-1];
    reg signed [31:0] ctxv  [0:D-1];        // attention context, Q6.25
    reg [20:0]        probbuf[0:TMAX-1];    // current head probs Q1.20
    reg signed [31:0] gemvy [0:D_MLP-1];    // GEMV INT32 outputs
    reg [8:0]         tok_stream[0:KVMAX-1];// emitted greedy stream (host drains)

    // ----------------------------------------------------- block instances -----
    reg                gv_ldrst, gv_wwe, gv_xwe, gv_start;
    reg [WBITS-1:0]    gv_wdata;
    reg signed [7:0]   gv_xdata;
    reg [10:0]         gv_m, gv_k;
    wire               gv_done;
    reg [9:0]          gv_rdaddr;
    wire signed [31:0] gv_yout;
    gemv_banked #(.LANES(LANES), .MMAX(D_MLP), .KMAX(D_MLP), .RLAT(2)) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k),
        .ld_rst(gv_ldrst), .w_we(gv_wwe), .w_data(gv_wdata),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .rd_addr(gv_rdaddr), .y_out(gv_yout)
    );

    reg                ln_start, ln_vin;
    reg signed [31:0]  ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire signed [63:0] ln_y;
    layernorm u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done)
    );

    reg                sm_start, sm_invalid;
    reg [8:0]          sm_tcount;
    reg signed [15:0]  sm_score;
    wire               sm_inready, sm_outvalid, sm_done;
    wire [20:0]        sm_prob;
    softmax #(.TMAX(TMAX)) u_sm (
        .clk(clk), .rst(rst), .start(sm_start), .t_count(sm_tcount),
        .in_valid(sm_invalid), .score(sm_score), .in_ready(sm_inready),
        .out_valid(sm_outvalid), .prob(sm_prob), .done(sm_done)
    );

    reg signed [15:0]  gl_x;
    wire signed [15:0] gl_y;
    gelu_lut u_gelu (.clk(clk), .x(gl_x), .y(gl_y));

    // ===================================================================
    //  Master FSM
    // ===================================================================
    localparam [6:0]
      S_IDLE=0, S_EMBED=1,
      // generic GEMV sub-sequence is invoked via gemv_go/gemv_ret; states 10..19
      S_LN_SNAP=2, S_LN_FEED=3, S_LN_COLL=4,
      S_QKV_GO=5, S_QKV_DEQ=6,
      S_KVAP=7,
      S_HEAD=8, S_SCORE=9, S_SMFEED=10, S_SMCOLL=11, S_CTX=12, S_CTXST=13,
      S_PROJ_PRE=14, S_PROJ_GO=15, S_PROJ_DEQ=16, S_RES1=17,
      S_LN2_SNAP=18, S_LN2_FEED=19, S_LN2_COLL=20,
      S_FC_GO=21, S_FC_DEQ=22, S_GELU=23, S_GELU_W=24,
      S_MP_PRE=25, S_MP_GO=26, S_MP_DEQ=27, S_RES2=28,
      S_FINISH=29,
      // final stage: ln_f -> head GEMV -> argmax -> emit
      S_LNF_SNAP=30, S_LNF_FEED=31, S_LNF_COLL=32,
      S_HEAD_DEQ=33, S_ARGMAX=34, S_EMIT=35,
      // GEMV micro-sequence (callable): enter at G_LDRST
      G_LDRST=39, G_LDW=40, G_ACT=41, G_RUN=42, G_WAIT=43, G_RB0=44, G_RB1=45, G_RB2=46;
    reg [6:0] st, gret;          // st = master, gret = return state after GEMV

    integer ii;
    reg [11:0] ci, ki;
    reg [16:0] wcnt;             // up to ceil(1024/16)*1024 = 65536 words (15-17 bit)
    reg [19:0] wbase;
    reg [15:0] dqbase;           // up to DQ_BLK*NLAYER+VOCAB ~ 9409 (needs >12 bit)
    reg [11:0] gM, gK;           // GEMV dims for the current call
    reg [5:0]  actsel;           // act_scale index (into inv_sact, blk-absolute)
    reg        act_wide;         // 1: GEMV acts come from mlpbuf (K up to D_MLP); 0: lnout
    reg [11:0] gi;               // wide element counter (GELU / MLP, up to D_MLP)
    reg [2:0]  head_i;
    // GEMV readback address-shift pipeline (2-cycle gemv read latency)
    reg [11:0] rb_a0, rb_a1, rb_a2;
    reg        rb_v0, rb_v1, rb_v2;
    reg [11:0] rbcap;
    // full-forward block loop + argmax
    reg [3:0]  blk;                 // current transformer block 0..NLAYER-1
    reg [19:0] wblk;                // weight ROM base for current block = blk*GW_BLK
    reg [15:0] dqblk;               // dequant ROM base for current block = blk*DQ_BLK
    reg [5:0]  sablk;               // inv_sact base for current block = blk*4
    reg [19:0] kvblk;               // KV cache base for current block = blk*KVMAX*D
    reg signed [31:0] best_val;     // argmax running best logit (real, via dequant)
    reg [8:0]  best_idx;
    reg [8:0]  tpos, ji;
    // autoregressive multi-token loop state
    reg [8:0]  tcur;                // current decode position (0-based); pos_emb + KV slot
    reg [7:0]  cur_tok;            // input token id for this forward
    reg [8:0]  gcnt;               // number of generated tokens emitted so far

    // plain-vector temporaries
    reg signed [63:0] lntmp;
    reg signed [31:0] mant_v, kv_v, qv_v, vv_v;
    reg signed [7:0]  exp_v;
    reg [20:0]        prob_v;

    // requant (INT8) temporaries
    reg signed [95:0] aq_prod, aq_sh;
    reg signed [31:0] aq_int;
    // dequant temporaries
    reg signed [95:0] dq_prod;
    reg signed [8:0]  dq_shv;
    reg signed [95:0] dq_val;
    // score/ctx temporaries
    reg signed [63:0] score_acc;
    reg signed [63:0] score_q88;
    reg signed [95:0] ctx_acc;
`ifdef DBG_SEQ
    reg [31:0] dbg_cyc = 0;
`endif

    always @(posedge clk) begin
        gv_ldrst<=0; gv_wwe<=0; gv_xwe<=0; gv_start<=0;
        ln_start<=0; ln_vin<=0; sm_start<=0; sm_invalid<=0; done<=0;

`ifdef DBG_SEQ
        if (!rst) begin
            dbg_cyc <= dbg_cyc + 1;
            if (dbg_cyc % 50000 == 0)
                $display("[cyc %0d] st=%0d ci=%0d wcnt=%0d ji=%0d head=%0d tpos=%0d gvdone=%b lnyv=%b smov=%b smdone=%b",
                         dbg_cyc, st, ci, wcnt, ji, head_i, tpos, gv_done, ln_yv, sm_outvalid, sm_done);
        end
`endif
        if (rst) begin
            st <= S_IDLE; busy <= 0;
            ci<=0; gi<=0; wcnt<=0; ji<=0; head_i<=0; ki<=0;
            tcur<=0; gcnt<=0; ngen_out<=0;
        end else begin
        case (st)
        // ----------------------------------------------------------------
        S_IDLE: begin
            busy<=0;
            if (go) begin
                busy<=1; ci<=0; gi<=0;
                blk<=0; wblk<=0; dqblk<=0; sablk<=0; kvblk<=0; // start at block 0
                gcnt<=0; ngen_out<=0;
                // Multi-token (PROMPT_LEN>1): prime from prompt_rom starting at position 0,
                //   input token at position t = prompt_rom[t]; then autoregress.
                // Single-token (PROMPT_LEN<=1): legacy path — feed tok_id at position `pos`.
                if (PROMPT_LEN > 1) begin
                    tcur<=0; cur_tok <= prompt_rom[0][7:0];
                end else begin
                    tcur<=pos; cur_tok <= tok_id;
                end
                st<=S_EMBED;
            end
        end

        S_EMBED: begin
            // embed the current token at the current position: x = tok_emb[t] + pos_emb[tcur]
            xres[ci] <= tok_emb[cur_tok*D + ci] + pos_emb[tcur*D + ci];
            if (ci==D-1) begin ci<=0; st<=S_LN_SNAP; end else ci<=ci+1'b1;
        end

        // ---------------- LN1 ----------------
        S_LN_SNAP: begin
            lnbuf[ci] <= xres[ci];
            if (ci==D-1) begin ci<=0; ln_start<=1; st<=S_LN_FEED; end else ci<=ci+1'b1;
        end
        S_LN_FEED: begin
            ln_vin<=1; ln_x<=lnbuf[ci];
            ln_g<=gamma_rom[(2*blk)*D + ci];                     // ln1 gamma of block blk
            if (ci==D-1) begin ci<=0; st<=S_LN_COLL; end else ci<=ci+1'b1;
        end
        S_LN_COLL: begin
            if (ln_yv) begin
                lnout[ci]<=ln_y;
                if (ci==D-1) begin
                    // set up qkv GEMV call (block-relative ROM bases)
                    ci<=0; wcnt<=0; gM<=D3; gK<=D;
                    wbase<=wblk + 0; dqbase<=dqblk + 0;
                    actsel<=sablk + 6'd0; act_wide<=1'b0; gret<=S_QKV_DEQ; st<=G_LDRST;
                end else ci<=ci+1'b1;
            end
        end

        // =========================================================
        //  GEMV micro-sequence (callable): G_LDW -> ... -> G_DONE -> gret
        //  inputs: gM,gK,wbase,dqbase,actsel,act_from_ln ; src acts:
        //   act_from_ln=1 -> lnout[ci] (Q.22 via inv_sact[actsel])
        //   act_from_ln=0 -> ctxv/gelu pre-quantized into actbuf? Here we
        //     instead pre-fill gemvy-as-act? To keep one path we require the
        //     caller to have placed INT8 acts; but for ctx/gelu we quantize
        //     from a real-Q source: handled by S_PROJ_PRE / S_MP_PRE / S_GELU
        //     which write lnout[] with the to-be-quantized value in Q.22-equivalent.
        // =========================================================
        G_LDRST: begin gv_ldrst<=1; wcnt<=0; st<=G_LDW; end
        G_LDW: begin
            gv_wwe<=1; gv_wdata<=wrom[wbase + wcnt];
            if (wcnt==gM_words(gM,gK)-1) begin wcnt<=0; ci<=0; st<=G_ACT; end
            else wcnt<=wcnt+1'b1;
        end
        G_ACT: begin
            // INT8 act-quant source[ci] (carried as Q.LN_OUT_FRAC) by inv_sact[actsel]
            lntmp = act_wide ? mlpbuf[ci] : lnout[ci];
            aq_prod = $signed(lntmp) * $signed(inv_sact[actsel]);          // Q.(F+ISH)
            if (lntmp >= 0)
                aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
            else
                aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
            aq_int = aq_sh[31:0];
            if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
            gv_xwe<=1; gv_xdata<=aq_int[7:0];
            if (ci==gK-1) begin ci<=0; st<=G_RUN; end else ci<=ci+1'b1;
        end
        G_RUN: begin gv_m<=gM[10:0]; gv_k<=gK[10:0]; gv_start<=1; st<=G_WAIT; end
        G_WAIT: begin if (gv_done) begin ci<=0; gv_rdaddr<=0; rbcap<=0; st<=G_RB0; end end
        // Streaming readback: free-run gv_rdaddr = ci; capture gv_yout (2-cycle
        // gemv read latency) into gemvy[ci-2] via a 3-deep address-shift register.
        // rbcap counts captured outputs; finishes when rbcap==gM.
        G_RB0: begin
            // drive address ci (advances every cycle up to gM-1, then holds)
            if (ci < gM) begin gv_rdaddr <= ci[9:0]; rb_a0 <= ci; ci <= ci + 1'b1; end
            else gv_rdaddr <= gM[9:0]-1'b1;
            rb_a1 <= rb_a0; rb_a2 <= rb_a1;
            rb_v0 <= (ci < gM); rb_v1 <= rb_v0; rb_v2 <= rb_v1;
            // capture: gv_yout now corresponds to the address issued 2 cycles ago (rb_a2)
            if (rb_v2) begin
                gemvy[rb_a2] <= gv_yout;
                if (rb_a2 == gM-1) begin ci<=0; st<=gret; end
            end
        end

        // ---------------- qkv dequant -> split q,k,v (store Q.VFRAC) ----------
        S_QKV_DEQ: begin
            // dequant gemvy[ci] using dq_mant/exp at dqbase+ci -> Q.VFRAC
            mant_v = dq_mant[dqbase + ci]; exp_v = dq_exp[dqbase + ci];
            dq_prod = $signed(gemvy[ci]) * $signed(mant_v);
            dq_shv  = $signed(exp_v) + VFRAC;
            if (dq_shv >= 0) dq_val = dq_prod <<< dq_shv;
            else             dq_val = rsh_round(dq_prod, -dq_shv);
            // route: 0..D-1 -> q ; D..2D-1 -> k ; 2D..3D-1 -> v
            // K/V append at this block's slot tcur:  kvblk + tcur*D + offset
            if (ci < D)            qvec[ci]                          <= dq_val[31:0];
            else if (ci < 2*D)     kcache[kvblk + tcur*D + (ci-D)]   <= dq_val[31:0];
            else                   vcache[kvblk + tcur*D + (ci-2*D)] <= dq_val[31:0];
            if (ci==D3-1) begin head_i<=0; tpos<=tcur+1'b1; st<=S_HEAD; end
            else ci<=ci+1'b1;
        end

        // ---------------- attention per head ----------------
        // pulse softmax start with this head's key count (tpos), then feed scores.
        S_HEAD: begin
            ji<=0; ci<=0; sm_tcount<=tpos; sm_start<=1; st<=S_SMFEED;
        end
        // Feed one score per key while in_ready: score = (q.k_j)/sqrt(hd) -> Q8.8.
        S_SMFEED: begin
            if (sm_inready) begin
                score_acc = 64'sd0;
                for (ii=0; ii<HEAD_DIM; ii=ii+1) begin
                    qv_v = qvec[head_i*HEAD_DIM + ii];
                    kv_v = kcache[kvblk + ji*D + head_i*HEAD_DIM + ii];   // block-blk key j
                    score_acc = score_acc + $signed(qv_v) * $signed(kv_v);  // Q.(2*VFRAC)
                end
                // /sqrt(64)=>>3, to Q8.8: round(acc / 2^(2*VFRAC+3-8))
                score_q88 = rsh_round(score_acc, (2*VFRAC) + ISQRT - SCORE_FRAC);
                sm_invalid<=1; sm_score<=sat16(score_q88);
                if (ji==tpos-1) begin ji<=0; ci<=0; st<=S_SMCOLL; end
                else ji<=ji+1'b1;
            end
        end
        S_SMCOLL: begin
            if (sm_outvalid) begin probbuf[ci]<=sm_prob; ci<=ci+1'b1; end
            if (sm_done) begin ci<=0; st<=S_CTX; end
        end
        // ctx[d] = sum_j prob[j] * v_j[d]  (Q.(20+VFRAC)) -> Q6.25
        S_CTX: begin
            ctx_acc = 96'sd0;
            for (ii=0; ii<TMAX; ii=ii+1) begin
                if (ii < tpos) begin
                    prob_v = probbuf[ii];
                    vv_v   = vcache[kvblk + ii*D + head_i*HEAD_DIM + ci];   // block-blk value j, Q.VFRAC
                    ctx_acc = ctx_acc + $signed({75'd0, prob_v}) * $signed({{64{vv_v[31]}}, vv_v});
                end
            end
            ctxv[head_i*HEAD_DIM + ci] <= rsh_round(ctx_acc, PROB_FRAC + VFRAC - RESID_FRAC);
            if (ci==HEAD_DIM-1) st<=S_CTXST; else ci<=ci+1'b1;
        end
        S_CTXST: begin
            if (head_i==NHEAD-1) begin ci<=0; st<=S_PROJ_PRE; end
            else begin head_i<=head_i+1'b1; ci<=0; st<=S_HEAD; end
        end

        // ---------------- proj GEMV (acts = ctxv, real Q6.25) ----------------
        // Quantize ctxv (Q6.25) into lnout (reuse buffer) as Q.LN_OUT_FRAC equivalent
        // so the shared G_ACT path applies. ctxv is Q6.25 = real*2^25; to feed G_ACT
        // which treats lnout as Q.LN_OUT_FRAC(=22) and multiplies by inv_sact, we store
        // ctxv >> (RESID_FRAC-LN_OUT_FRAC)=>>3 into lnout.
        S_PROJ_PRE: begin
            lnout[ci] <= $signed({{32{ctxv[ci][31]}}, ctxv[ci]}) >>> (RESID_FRAC-LN_OUT_FRAC);
            if (ci==D-1) begin
                ci<=0; wcnt<=0; gM<=D; gK<=D;
                wbase<=wblk + GW_QKV; dqbase<=dqblk + D3;
                actsel<=sablk + 6'd1; act_wide<=1'b0; gret<=S_PROJ_DEQ; st<=G_LDRST;
            end else ci<=ci+1'b1;
        end
        S_PROJ_DEQ: begin
            mant_v = dq_mant[dqbase + ci]; exp_v = dq_exp[dqbase + ci];
            dq_prod = $signed(gemvy[ci]) * $signed(mant_v);
            dq_shv  = $signed(exp_v) + RESID_FRAC;          // -> Q6.25
            if (dq_shv>=0) dq_val = dq_prod <<< dq_shv;
            else           dq_val = rsh_round(dq_prod, -dq_shv);
            ctxv[ci] <= dq_val[31:0];                       // proj output (Q6.25)
            if (ci==D-1) begin ci<=0; st<=S_RES1; end else ci<=ci+1'b1;
        end
        S_RES1: begin
            xres[ci] <= xres[ci] + ctxv[ci];                // x = x + attn_out
            if (ci==D-1) begin ci<=0; st<=S_LN2_SNAP; end else ci<=ci+1'b1;
        end

        // ---------------- LN2 ----------------
        S_LN2_SNAP: begin
            lnbuf[ci]<=xres[ci];
            if (ci==D-1) begin ci<=0; ln_start<=1; st<=S_LN2_FEED; end else ci<=ci+1'b1;
        end
        S_LN2_FEED: begin
            ln_vin<=1; ln_x<=lnbuf[ci];
            ln_g<=gamma_rom[(2*blk+1)*D + ci];                   // ln2 gamma of block blk
            if (ci==D-1) begin ci<=0; st<=S_LN2_COLL; end else ci<=ci+1'b1;
        end
        S_LN2_COLL: begin
            if (ln_yv) begin
                lnout[ci]<=ln_y;
                if (ci==D-1) begin
                    ci<=0; gi<=0; wcnt<=0; gM<=D_MLP; gK<=D;
                    wbase<=wblk + GW_QKV+GW_PROJ; dqbase<=dqblk + D3+D; actsel<=sablk + 6'd2;
                    act_wide<=1'b0; gret<=S_FC_DEQ; st<=G_LDRST;
                end else ci<=ci+1'b1;
            end
        end
        // mlp_fc dequant -> store Q4.12 into mlpbuf (1024-wide) for GELU
        S_FC_DEQ: begin
            mant_v = dq_mant[dqbase + gi]; exp_v = dq_exp[dqbase + gi];
            dq_prod = $signed(gemvy[gi]) * $signed(mant_v);
            dq_shv  = $signed(exp_v) + GELU_FRAC;            // -> Q4.12
            if (dq_shv>=0) dq_val = dq_prod <<< dq_shv;
            else           dq_val = rsh_round(dq_prod, -dq_shv);
            mlpbuf[gi] <= {{32{dq_val[31]}}, sat16_32(dq_val)};  // Q4.12 sign-extended to 64b
            if (gi==D_MLP-1) begin gi<=0; st<=S_GELU; end else gi<=gi+1'b1;
        end
        // GELU pipeline (elementwise, 3-cycle latency): drive gl_x with mlpbuf[gi]
        // (Q4.12), capture gl_y 3 cycles later as Q.LN_OUT_FRAC back into mlpbuf.
        S_GELU: begin
            lntmp = mlpbuf[gi];          // plain-vector copy before the part-select (iverilog-safe)
            gl_x <= lntmp[15:0];
            st   <= S_GELU_W; ki<=0;
        end
        S_GELU_W: begin
            if (ki==3'd3) begin          // 4-cycle wait: 3-stage gelu pipeline + 1 margin
                // gl_y is Q4.12; shift to Q.LN_OUT_FRAC (<<10) for mlp_proj act-quant
                mlpbuf[gi] <= $signed({{48{gl_y[15]}}, gl_y}) <<< (LN_OUT_FRAC-GELU_FRAC);
                if (gi==D_MLP-1) begin
                    gi<=0; ci<=0; wcnt<=0; gM<=D; gK<=D_MLP;
                    wbase<=wblk + GW_QKV+GW_PROJ+GW_FC; dqbase<=dqblk + D3+D+D_MLP;
                    actsel<=sablk + 6'd3; act_wide<=1'b1; gret<=S_MP_DEQ; st<=G_LDRST;
                end else begin gi<=gi+1'b1; st<=S_GELU; end
            end else ki<=ki+1'b1;
        end
        S_MP_DEQ: begin
            mant_v = dq_mant[dqbase + ci]; exp_v = dq_exp[dqbase + ci];
            dq_prod = $signed(gemvy[ci]) * $signed(mant_v);
            dq_shv  = $signed(exp_v) + RESID_FRAC;
            if (dq_shv>=0) dq_val = dq_prod <<< dq_shv;
            else           dq_val = rsh_round(dq_prod, -dq_shv);
            ctxv[ci] <= dq_val[31:0];
            if (ci==D-1) begin ci<=0; st<=S_RES2; end else ci<=ci+1'b1;
        end
        S_RES2: begin
            xres[ci] <= xres[ci] + ctxv[ci];                // x = x + mlp_out
            if (ci==D-1) begin
                ci<=0;
                if (blk == NLAYER-1) st<=S_LNF_SNAP;        // last block -> final LN + head
                else begin
                    // advance to next block: bump all ROM bases + per-block KV base
                    blk<=blk+1'b1;
                    wblk<=wblk+GW_BLK; dqblk<=dqblk+DQ_BLK; sablk<=sablk+4'd4;
                    kvblk<=kvblk+KVMAX*D;
                    st<=S_LN_SNAP;
                end
            end else ci<=ci+1'b1;
        end

        // ---------------- final LN_f -> head GEMV -> argmax ----------------
        S_LNF_SNAP: begin
            lnbuf[ci]<=xres[ci];
            if (ci==D-1) begin ci<=0; ln_start<=1; st<=S_LNF_FEED; end else ci<=ci+1'b1;
        end
        S_LNF_FEED: begin
            ln_vin<=1; ln_x<=lnbuf[ci];
            ln_g<=gamma_rom[(2*NLAYER)*D + ci];             // ln_f gamma (last gamma block)
            if (ci==D-1) begin ci<=0; st<=S_LNF_COLL; end else ci<=ci+1'b1;
        end
        S_LNF_COLL: begin
            if (ln_yv) begin
                lnout[ci]<=ln_y;
                if (ci==D-1) begin
                    // head GEMV: M=VOCAB, K=D
                    ci<=0; wcnt<=0; gM<=VOCAB; gK<=D;
                    wbase<=GW_BLK*NLAYER; dqbase<=DQ_BLK*NLAYER; actsel<=4*NLAYER;
                    act_wide<=1'b0; gret<=S_HEAD_DEQ; st<=G_LDRST;
                end else ci<=ci+1'b1;
            end
        end
        // dequant head logits to real (Q6.25) and run a streaming argmax
        S_HEAD_DEQ: begin
            mant_v = dq_mant[dqbase + ci]; exp_v = dq_exp[dqbase + ci];
            dq_prod = $signed(gemvy[ci]) * $signed(mant_v);
            dq_shv  = $signed(exp_v) + RESID_FRAC;          // -> Q6.25 (monotone w/ logit)
            if (dq_shv>=0) dq_val = dq_prod <<< dq_shv;
            else           dq_val = rsh_round(dq_prod, -dq_shv);
            if (ci==0 || $signed(dq_val[31:0]) > best_val) begin
                best_val <= dq_val[31:0];
                best_idx <= ci[8:0];
            end
            if (ci==VOCAB-1) st<=S_EMIT; else ci<=ci+1'b1;
        end
        // ---------------- emit + autoregressive loop ----------------
        // best_idx = argmax at position tcur. Record it into the output stream iff this
        // position is in the generation window (tcur >= PROMPT_LEN-1). Then either feed
        // the next token (prompt token if still priming, else the just-emitted token) and
        // advance to tcur+1, or finish once NGEN tokens have been emitted.
        S_EMIT: begin
            tok_out <= best_idx;
            // is this position emitting a generated token? (predictions at PROMPT_LEN-1..)
            if (tcur >= PROMPT_LEN-1) begin
                tok_stream[gcnt] <= best_idx;
                gcnt    <= gcnt + 1'b1;
                ngen_out<= gcnt + 1'b1;
            end
`ifdef DBG_SEQ
            $display("EMIT pos=%0d tok_out=%0d gcnt=%0d best_val=%0d",
                     tcur, best_idx, gcnt, best_val);
`endif
            // stop after NGEN emitted tokens (or if the KV cache is full)
            if (((tcur >= PROMPT_LEN-1) && (gcnt + 1'b1 >= NGEN)) || (tcur+1'b1 >= KVMAX)) begin
                done<=1; busy<=0; st<=S_IDLE;
            end else begin
                // advance one position: choose next input token, reset per-forward state
                cur_tok <= (tcur+1'b1 < PROMPT_LEN) ? prompt_rom[tcur+1'b1][7:0] : best_idx[7:0];
                tcur <= tcur + 1'b1;
                ci<=0; gi<=0; blk<=0; wblk<=0; dqblk<=0; sablk<=0; kvblk<=0;
                st<=S_EMBED;
            end
        end
        default: st<=S_IDLE;
        endcase
        end
    end

    // readback of x_out (2-cycle, like gemv): registered
    reg signed [31:0] xrb;
    always @(posedge clk) begin
        xrb   <= xres[rd_addr[7:0]];
        x_out <= xrb;
    end

    // readback of the emitted token stream (2-cycle registered, like x_out)
    reg [8:0] tsrb;
    always @(posedge clk) begin
        tsrb   <= tok_stream[ts_addr[($clog2(KVMAX))-1:0]];
        ts_out <= tsrb;
    end

    // -------- functions (combinational, iverilog-safe) -----------------------
    localparam integer ISQRT = 3;   // /sqrt(64) = >>3

    function [16:0] gM_words(input [11:0] m, input [11:0] k);
        gM_words = (((m + LANES - 1)/LANES) * k);
    endfunction

    // arithmetic right shift with round-half-away-from-zero
    function signed [95:0] rsh_round(input signed [95:0] v, input integer s);
        reg signed [95:0] half;
        begin
            if (s <= 0) rsh_round = v <<< (-s);
            else begin
                half = (96'sd1 <<< (s-1));
                if (v >= 0) rsh_round = (v + half) >>> s;
                else        rsh_round = -(((-v) + half) >>> s);
            end
        end
    endfunction

    function signed [15:0] sat16(input signed [63:0] v);
        begin
            // NOTE: 16'sh8000 already equals -32768; the bound must be written as a
            // wide signed literal. (-16'sh8000 wrongly evaluates to +32768.)
            if (v >  64'sd32767)  sat16 = 16'sd32767;
            else if (v < -64'sd32768) sat16 = -16'sd32768;
            else sat16 = v[15:0];
        end
    endfunction
    function signed [31:0] sat16_32(input signed [95:0] v);
        begin
            if (v >  96'sd32767)  sat16_32 = 32'sd32767;
            else if (v < -96'sd32768) sat16_32 = -32'sd32768;
            else sat16_32 = v[31:0];
        end
    endfunction
endmodule
