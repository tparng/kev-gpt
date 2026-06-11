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
    localparam integer GRPSH = $clog2(LANES/P);   // P-rows per ymem group word (shift)
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

    // ---- embed image in the resident weight URAM's SPARE DEPTH (log §36 plan 2) -
    // The tok/pos embed ROMs (~92 BRAM tiles) are APPENDED to the wrom weight image
    // by write_mems_wideword and read through the GEMV bank's port B (phase-disjoint
    // with GEMV reads). EPW embed rows (P*32b each) pack into one LANES*4-bit word:
    // word = {row_{EPW-1}, .., row1, row0}; row r of a table lives at word
    // EMB_*_BASE + r/EPW. Bases are EVEN-ALIGNED so the DP=1 column-parity pair
    // read returns RPP = 2*EPW consecutive rows; slot select within the pair is
    // r % RPP. Requires EPW >= 1 (LANES >= 8*P, e.g. LANES=256 at P=8): smaller
    // LANES no longer carry embeds in the wrom image.
    localparam integer EPW   = (LANES*4)/(P*32);            // embed rows per wrom word
    localparam integer EPWS  = (EPW > 1) ? $clog2(EPW) : 0; // row -> word shift
    localparam integer RPP   = (EPW > 0) ? 2*EPW : 1;       // rows per pair read
    localparam [2:0]   RSELM = RPP - 1;                     // pair-slot mask (RPP<=8)
    localparam integer EMB_TOK0     = WB_HEAD + GW_HEAD;    // first word past the image
    localparam integer EMB_TOK_BASE = EMB_TOK0 + (EMB_TOK0 % 2);
    localparam integer EMB_TOKW     = (EPW > 0) ? (VOCAB*EROWS + EPW - 1)/EPW : 0;
    localparam integer EMB_POS0     = EMB_TOK_BASE + EMB_TOKW;
    localparam integer EMB_POS_BASE = EMB_POS0 + (EMB_POS0 % 2);

    // ---- FSM -------------------------------------------------------------------
    localparam [4:0]
      S_IDLE=0, S_EMB=1,
      L_GAM=2, L_FEED=3, L_COLL=4,                  // callable LN (L_GAM = start-only)
      G_AQ=5, G_RUN=6, G_WAIT=7, G_RB=8,            // callable GEMV (RB = fused rb+dq+gelu)
      S_QKVRET=10, S_AST=11, S_ALD=12, S_ACL=13,    // attention
      S_RES1=14, S_LN2=15, S_FCRET=16,              // proj/res1/LN2-call
      S_MPSET=17, S_RES2=20, S_FIN=21,              // mlp_proj setup (GELU folded into G_RB)
      S_HEADSET=22, S_ARGMAX=23,                    // final LN_f -> head -> argmax
      S_KVW_S=24, S_KVW_F=25, S_KVW_W=26,           // KV quant-write (doc-7 R1)
      S_CDR=27;                                     // ctx drain from the pair catchers
    reg [4:0] st;
    reg [3:0] blk;                           // transformer block 0..NLAYER-1
    reg [10:0] ci;
    reg [$clog2(ROWSM+1)-1:0] fr, orow, dor;
    // read-pipeline delayed addresses + valids (consume stage of each FSM loop)
    reg [10:0] cid;  reg civ;
    reg [$clog2(ROWSM+1)-1:0] frd;  reg frv;
    reg [$clog2(ARROWS+1)-1:0] ard;  reg arv;

    // ---- S_EMB embed fetch through the weight bank's embed port -----------------
    // Issue alternates tok (etp=0) / pos (etp=1) row fetches, one address/cycle;
    // the pair lands on emb_pair 1 cycle later (eb_* are the arrival-stage regs).
    // tok row = tok_id*EROWS+fr, pos row = pos*EROWS+fr; word = BASE + row/EPW;
    // pair slot = row % RPP (bases even-aligned). ~2*EROWS+2 cycles per token.
    reg        etp;                          // fetch phase: 0 = tok row, 1 = pos row
    reg        eb_v, eb_tp;                  // arrival valid + phase
    reg [2:0]  eb_sel;                       // arrival pair-slot (row % RPP)
    reg [$clog2(ROWSM+1)-1:0] eb_row;        // arrival xres destination row
    reg [P*32-1:0]    tacc;                  // tok row held for the tok+pos sum
    reg [LANES*8-1:0] epr;                   // plain-reg pair copy (safe part-select)
    reg [P*32-1:0]    erw;                   // selected embed row
    // TIMING (5ns cone weight-bank BRAM -> xres LUTRAM): the pair-slot select is
    // REGISTERED (erw_r + eb2_* tags); the tacc hold / tok+pos sum / xres commit
    // run one cycle behind the arrival. Same numbers, +1 cycle per token.
    reg        eb2_v, eb2_tp;                // select-stage valid + phase
    reg [$clog2(ROWSM+1)-1:0] eb2_row;       // select-stage xres destination row
    reg [P*32-1:0]    erw_r;                 // REGISTERED selected embed row
    wire [13:0] emb_row_w = (etp ? pos : tok_id) * EROWS
                            + {{(14-$clog2(ROWSM+1)){1'b0}}, fr};
    // 32-bit param + 14-bit row word offset, truncated to the address width
    // (both bases + the largest offset are < WWORDS by the spare-depth budget)
    wire [$clog2(WWORDS)-1:0] emb_addr_w =
        (etp ? EMB_POS_BASE : EMB_TOK_BASE) + (emb_row_w >> EPWS);
    wire emb_sel_w = (st == S_EMB);

    // ---- wide-word ROMs ($readmemh: one P-packed word per line) -----------------
    // All sync-read (registered) so they infer BLOCK RAM, not LUTs.
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w   [0:GAMMA_N*EROWS-1];// Q4.20
    reg signed [63:0] inv_sact [0:NSACT-1];                       // 17-deep: stays LUT
    (* rom_style = "block" *) reg [P*24-1:0] dqm_w [0:DQROWS-1];           // P mant / word
    (* rom_style = "block" *) reg [P*8-1:0]  dqe_w [0:DQROWS-1];           // P exp  / word
    initial begin
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
    (* ram_style = "block" *) reg [P*32-1:0] head_bank  [0:ARROWS-1];  // head logits Q6.25

    // ---- synchronous-read registers (BRAM output stage; declared before use) ---
    reg [P*32-1:0] xres_r, qkv_r, ctxv_r, attn_r, mlp_r, head_r;
    reg [P*64-1:0] lnout1_r, lnout2_r, mlpbuf_r;
    reg [P*32-1:0] gam_r;

    // ---- layernorm_vec ---------------------------------------------------------
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    // ---- GEMV (resident, P-wide boundary) ---------------------------------------
    // act feed: P INT8 lanes/cycle; readback: P INT32 outputs/cycle (rd_addr is a
    // P-group index). The MAC core is the proven gemv_banked_resident.
    reg                gv_ldrst, gv_xwe, gv_start;
    reg  [P*8-1:0]     gv_xdata;
    reg [10:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    wire [3:0]         gv_gdone;           // committed-group count (RB overlap)
    reg [10:0]         gv_rdaddr;
    wire [P*32-1:0]    gv_yout;
    wire [LANES*8-1:0] emb_pair;            // pair-read data (RPP embed rows)
    gemv_banked_resident_vec #(.LANES(LANES), .P(P), .MMAX(1024), .KMAX(1024), .RLAT(2),
                  .WWORDS(WWORDS), .K2(1)) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ldrst | wl_rst), .w_we(wl_we), .w_data(wl_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .gdone(gv_gdone),
        .rd_addr(gv_rdaddr[$clog2(1024/P)-1:0]), .y_out(gv_yout),
        .emb_sel(emb_sel_w), .emb_addr(emb_addr_w), .emb_pair(emb_pair));

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

    // ---- TWIN vec_attn_w engines (doc-7 R4e): heads ping-pong A/B by parity,
    // each on its OWN kv_bank read stream (the URAM/BRAM second port), so a head
    // PAIR runs fully concurrently — the attention slope halves again.
    reg               at_startA, at_startB, at_ldvA, at_ldvB;
    reg [8:0]         at_tcount;
    wire              at_kdoneA, at_ctxvA, at_doneA;
    wire              at_kdoneB, at_ctxvB, at_doneB;
    wire [6:0]        at_ctxidxA, at_ctxidxB;
    wire [P*32-1:0]   at_ctxdataA, at_ctxdataB;
    reg [1:0]  hh;                          // legacy name: head A of the current pair
    reg [1:0]  hB;                          // head B of the current pair
    reg        pair;                        // pair index (NHEAD/2 = 2 pairs)
    reg        adone_s, bdone_s;            // sticky engine-done flags per pair
    reg [8:0]  wi;                          // load-address counter (runs ahead)
    reg [8:0]  wic;                         // accepted-word counter (consume stage)
    reg        wiv;                         // (legacy)
    localparam integer HR = HEAD_DIM / P;
    reg        qsel;                        // whose q is streaming (0=A, 1=B)
    wire [10:0] aw_src = (qsel ? hB : hh)*HR + wi;
    reg  [P*32-1:0] q_data_q;                    // registered WITH at_ldv (aligned pair)
    reg             ldv0;                        // addr-stage valid (1 ahead of at_ldv)
    // BOTH engines' ctx strobes land in catch buffers, drained serially in S_CDR
    // (2*HR cycles). A direct ctxv_bank write from the catcher made the bank
    // 2W1R -> synthesis fell back to REGISTERS (take-5); the buffer keeps it 1W1R.
    reg [P*32-1:0] ctxbufA [0:HR-1];
    reg [P*32-1:0] ctxbufB [0:HR-1];
    reg [4:0]  cdr;                         // ctx drain counter (0..2*HR-1)
    // KV-write FEEDER (R4f, extended to ALL writes): every (head, K/V) of the
    // new position quantises into kv_bank through this mini-FSM, STARTED AT THE
    // QKV DISPATCH so the K writes hide under the qkv GEMV+readback. Ordering
    // (NHEAD=4, head pairs (0,1)/(2,3)): K h0,h1,h2,h3 then V h1,h0,h3,h2 —
    // K-first so attention pairs can start on kvp_done counts (pair0 >= 2,
    // pair1 >= 4); within each V pair, engine B's head FIRST so its commit
    // lands in stream A's K->V gap instead of stalling behind engine A's
    // T-beat V read (W_CWR commits only while read-stream A is idle).
    // Data-ready gate: row fr of the current vector may be consumed only once
    // the qkv readback has written it (kvw_src < qkv_wrow). The qkv_bank read
    // port is arbitrated: S_ALD's q streams win; the feeder pauses (addr
    // presented only when granted, the in-flight beat completes regardless).
    // V reads are interlocked: at_kdone latches a pending flag, kb_rstart
    // fires once kvp_done covers that engine's V write (vneedA/vneedB).
    localparam [1:0] KF_IDLE=2'd0, KF_S=2'd1, KF_F=2'd2, KF_W=2'd3;
    reg [1:0] kvf_st;
    reg       kvf_active;
    reg [3:0] kvp_done;                     // completed (head,K/V) writes 0..8
    reg [10:0] qkv_wrow;                    // qkv rows committed by G_RB so far
    reg       vpA, vpB;                     // V-read pending (kdone seen, write not)
    wire      kvf_grant = (st != S_ALD) && (st != S_KVW_F);
    wire [3:0] vneedA = {2'b0, hh} + 4'd6;  // V write index for engine A's head
    wire [3:0] vneedB = {2'b0, hB} + 4'd4;  // V write index for engine B's head

    // ---- kv_bank: the on-chip K8/V8 cache making decode faithful ---------------
    reg         kb_wstart, kb_wvalid, kb_rstart, kb_rkv, kb2_rstart, kb2_rkv;
    reg  [P*32-1:0] kb_wdata_q;                  // registered with kb_wvalid (L_FEED idiom)
    reg  [1:0]  kvw_h;                           // KV-write head loop
    reg         kvw_kv;                          // 0 = K, 1 = V
    wire        kb_wdone, kb_rvalid, kb_rdone, kb2_rvalid, kb2_rdone;
    wire [HEAD_DIM*32-1:0] kb_rdata, kb2_rdata;  // one dequantised position per beat
    reg  [1:0]  alds;                            // S_ALD phase: 0 qA, 1 qB
    wire [10:0] kvw_src = (kvw_kv ? 2*D/P : D/P) + kvw_h*HR
                          + {{(11-$clog2(ROWSM+1)){1'b0}}, fr};
    kv_bank #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER),
              .TMAX(TMAX), .KBITS(8)) u_kvb (
        .clk(clk), .rst(rst),
        .wq_start(kb_wstart), .wq_layer(blk), .wq_kv(kvw_kv), .wq_head(kvw_h),
        .wq_pos(pos), .wq_valid(kb_wvalid), .wq_data(kb_wdata_q), .wq_done(kb_wdone),
        .rd_start(kb_rstart), .rd_layer(blk), .rd_kv(kb_rkv), .rd_head(hh),
        .rd_tcount(pos + 9'd1),
        .rd_valid(kb_rvalid), .rd_data(kb_rdata), .rd_done(kb_rdone),
        .rd2_start(kb2_rstart), .rd2_layer(blk), .rd2_kv(kb2_rkv), .rd2_head(hB),
        .rd2_tcount(pos + 9'd1),
        .rd2_valid(kb2_rvalid), .rd2_data(kb2_rdata), .rd2_done(kb2_rdone));

    vec_attn_w #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(TMAX)) u_attnA (
        .clk(clk), .rst(rst), .start(at_startA), .tcount(at_tcount),
        .q_valid(at_ldvA), .q_data(q_data_q),
        .kv_valid(kb_rvalid), .kv_data(kb_rdata),
        .k_done(at_kdoneA),
        .ctx_valid(at_ctxvA), .ctx_idx(at_ctxidxA), .ctx_data(at_ctxdataA),
        .done(at_doneA));

    vec_attn_w #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(TMAX)) u_attnB (
        .clk(clk), .rst(rst), .start(at_startB), .tcount(at_tcount),
        .q_valid(at_ldvB), .q_data(q_data_q),
        .kv_valid(kb2_rvalid), .kv_data(kb2_rdata),
        .k_done(at_kdoneB),
        .ctx_valid(at_ctxvB), .ctx_idx(at_ctxidxB), .ctx_data(at_ctxdataB),
        .done(at_doneB));

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
    reg signed [95:0]  aq_prod_r [0:P-1];     // G_AQ stage-1 product registers
    reg                aq_neg_r  [0:P-1];
    reg signed [63:0]  lnt_r     [0:P-1];     // G_AQ stage-0 source registers
    // TIMING (5ns cone g_asel -> aq_prod_r): the 17-deep inv_sact LUT read was
    // muxed combinationally INSIDE the multiply cycle. g_asel is constant for a
    // whole GEMV call (set at dispatch, >=2 cycles before the first civ1 multiply),
    // so a free-running registered read is always settled in time.
    reg signed [63:0]  isact_r;
    reg [10:0]         cid1, cid2;  reg civ1, civ2;
    // TIMING (5ns cone aq_prod_r DSP-out -> gv_xdata): the 96-bit conditional
    // negate+round+shift and the clip+pack were one cycle. Stage 2a registers the
    // rounded value (only [31:0] is ever consumed downstream — identical
    // semantics), stage 2b clips+packs. +1 cycle per AQ phase.
    reg signed [31:0]  aq_sh_r [0:P-1];
    reg [10:0]         cid3;  reg civ3;
    reg signed [31:0]  cb, dqv, hv;
    reg [P*32-1:0]  ww, sw, dword, hw;
    reg [P*8-1:0]   aqw;                     // P-wide act-quant word
    reg [P*16-1:0]  mword;                   // P-wide sat16 word (GELU feed)
    reg [P*64-1:0]  gsw;
    reg [P*24-1:0]  mwr;
    reg [P*8-1:0]   ewr;
    reg signed [63:0] gl_sh;
    reg [10:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    reg [$clog2(ROWSM+1)-1:0] gor;
    integer pp;

    // free-running registered inv_sact read (see isact_r note above)
    always @(posedge clk) isact_r <= inv_sact[g_asel];

    reg signed [31:0] best_val; reg [8:0] best_idx, hidx;
    reg [$clog2(ARROWS+1)-1:0] ar;           // argmax row counter
    reg signed [31:0] wm_val;  reg [8:0] wm_idx;        // word-max tree temporaries
    reg signed [31:0] am_val;  reg [8:0] am_idx;        // stage-1 registers
    reg [$clog2(ARROWS+1)-1:0] amd;  reg amv;
    // argmax 3-stage pipeline (the head_bank -> compare chain was the OOC critical path):
    // stage A registers the row, stage B halves P -> P/2, stage C reduces + running best.
    reg [P*32-1:0] hw_r;
    reg signed [31:0] pv0, pv1, pv2, pv3;     // P/2 pair maxima
    reg [8:0]         pi0, pi1, pi2, pi3;
    reg [$clog2(ARROWS+1)-1:0] ad1;  reg av1;
    reg signed [31:0] va, vb;
    reg [8:0]         ia, ib;


    // ---- synchronous reads (one read register per memory, address muxed) -------
    wire [10:0] rbr = rd_addr >> LSH;            // board readback row (idle only)
    wire [10:0] xres_ra   = (st==L_FEED) ? {{(11-$clog2(ROWSM+1)){1'b0}}, fr} :
                            (st==S_RES1 || st==S_RES2) ? ci : rbr;
    // G_AQ counts P-rows directly (one wide word per cycle into the GEMV boundary)
    wire [10:0] lnout1_ra = (st==G_AQ) ? ci : rbr;
    wire [10:0] lnout2_ra = (st==G_AQ) ? ci : rbr;
    wire [10:0] ctxv_ra   = (st==G_AQ) ? ci : rbr;
    wire [10:0] mlpbuf_ra = (st==G_AQ) ? ci : rbr;
    wire [10:0] qkv_ra    = (st==S_ALD)   ? aw_src :
                            (st==S_KVW_F) ? kvw_src :
                            (kvf_active && kvf_st==KF_F) ? kvw_src : rbr;
    wire [10:0] attn_ra   = (st==S_RES1) ? ci : rbr;
    wire [10:0] mlp_ra    = (st==S_RES2) ? ci : rbr;
    wire [10:0] head_ra   = (st==S_ARGMAX) ? {{(11-$clog2(ARROWS+1)){1'b0}}, ar} : rbr;
    // LN FEED FUSION: every LN call is fed DURING the state that produces its
    // input rows (S_EMB commit / S_RES1 / S_RES2), so the gamma read tracks the
    // producer's row counter (eb_row in S_EMB — issued one cycle ahead of the
    // commit stage; ci elsewhere). L_GAM/L_FEED are retired.
    wire [10:0] gam_fr    = (st==S_EMB) ? {{(11-$clog2(ROWSM+1)){1'b0}}, eb_row} : ci;

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
        gam_r    <= gamma_w  [l_gbase*EROWS + gam_fr];
    end

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; gv_ldrst<=1'b0; gv_xwe<=1'b0; gv_start<=1'b0;
        dq_vin<=1'b0; gl_vin<=1'b0; done<=1'b0;
        at_startA<=1'b0; at_startB<=1'b0; at_ldvA<=1'b0; at_ldvB<=1'b0;
        kb_wstart<=1'b0; kb_wvalid<=1'b0; kb_rstart<=1'b0; kb2_rstart<=1'b0;
        if (rst) begin
            st<=S_IDLE; ci<=0; fr<=0; orow<=0; dor<=0; gor<=0;
            rv0<=0; rv1<=0; rv2<=0;
            civ<=0; frv<=0; arv<=0; wiv<=0; etp<=1'b0; eb_v<=1'b0; eb2_v<=1'b0;
            kvf_st<=KF_IDLE; kvf_active<=1'b0;
            kvp_done<=4'd0; qkv_wrow<=11'd0; vpA<=1'b0; vpB<=1'b0;
        end else begin
            case (st)
                S_IDLE: if (go) begin
                    fr<=0; frv<=0; etp<=1'b0; eb_v<=1'b0; eb2_v<=1'b0; blk<=4'd0;
                    // block-0 LN1 is FED during the S_EMB commits (LN fusion)
                    l_gbase<=4'd0; l_dst<=1'b0; l_ret<=S_QKVRET; ln_start<=1'b1;
                    st<=S_EMB;
                end
                // ---- embed -> xres via the weight bank's embed port (log §36 plan 2):
                // tok row then pos row per fr, serial (1 fetch/cycle, ~2*EROWS+2 cyc).
                S_EMB: begin
                    // issue stage: emb_addr_w is combinational from etp/fr this cycle
                    eb_v   <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    eb_tp  <= etp;
                    eb_sel <= emb_row_w[2:0] & RSELM;
                    eb_row <= fr;
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) begin
                        if (etp) begin etp <= 1'b0; fr <= fr + 1'b1; end
                        else etp <= 1'b1;
                    end
                    // select stage: register the pair-slot mux of last cycle's pair
                    if (eb_v) begin
                        epr = emb_pair;                         // plain-reg copy first
                        erw_r <= epr[eb_sel*(P*32) +: P*32];
                    end
                    eb2_v <= eb_v; eb2_tp <= eb_tp; eb2_row <= eb_row;
                    // commit stage: tacc hold / tok+pos sum, one cycle behind.
                    // LN FUSION: each committed row is ALSO fed to the LN (ln_x
                    // = the same sum, ln_g = the gamma row issued at the eb
                    // stage) — block-0 LN1 loads during the embed, L_FEED dies.
                    ln_vin <= eb2_v && eb2_tp;
                    ln_g   <= gam_r;
                    if (eb2_v) begin
                        if (!eb2_tp) tacc <= erw_r;             // hold the tok row
                        else begin                              // pos row: sum + commit
                            for (pp=0; pp<P; pp=pp+1)
                                ww[pp*32 +: 32] = $signed(tacc[pp*32 +: 32])
                                                + $signed(erw_r[pp*32 +: 32]);
                            xres_bank[eb2_row] <= ww;
                            ln_x <= ww;
                            if (eb2_row==ROWS-1) begin
                                fr<=0; frv<=0; eb_v<=1'b0; eb2_v<=1'b0; etp<=1'b0;
                                if (dbg_stop==2'd1) st<=S_FIN;       // DEBUG: stop after embed
                                else begin orow<=0; st<=L_COLL; end
                            end
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
                // The KV FEEDER is armed HERE: its K writes ride the qkv
                // readback (data-ready gated), so attention starts as soon as
                // the pair's K writes commit (R1 contract kept per-read via
                // the kvp_done gates + V interlocks, not a serial S_KVW block).
                S_QKVRET: begin
                    g_wbase<=blk*GW_BLK + WB_QKV; g_m<=D3[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=blk*4 + 6'd0; g_frac<=7'd16; g_dqrow<=blk*DQB_P + DR_QKV; g_dst<=3'd0;
                    g_ret<=S_AST; ci<=0; civ<=0;
                    hh<=2'd0; hB<=2'd1; pair<=1'b0;
                    kvw_h<=2'd0; kvw_kv<=1'b0;
                    kvf_active<=1'b1; kvf_st<=KF_S; kvp_done<=4'd0; qkv_wrow<=11'd0;
                    vpA<=1'b0; vpB<=1'b0;
                    gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ============ KV quant-write: k then v, per head ==============
                S_KVW_S: begin
                    kb_wstart<=1'b1; fr<=0; frv<=0; st<=S_KVW_F;
                end
                S_KVW_F: begin
                    frd <= fr; frv <= (fr != HR[$clog2(ROWSM+1)-1:0]);
                    if (fr != HR[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    kb_wvalid  <= frv;
                    kb_wdata_q <= qkv_r;
                    if (frv && frd==HR-1) begin fr<=0; frv<=0; st<=S_KVW_W; end
                end
                // main FSM writes heads 0..1 only; heads 2..3 go to the FEEDER,
                // which runs during pair-0's attention (R4f).
                S_KVW_W: if (kb_wdone) begin
                    if (!kvw_kv) begin kvw_kv<=1'b1; st<=S_KVW_S; end
                    else if (kvw_h == 2'd0) begin kvw_h<=2'd1; kvw_kv<=1'b0; st<=S_KVW_S; end
                    else begin
                        kvw_h<=2'd2; kvw_kv<=1'b0; kvf_active<=1'b1; kvf_st<=KF_S;
                        hh<=2'd0; hB<=2'd1; pair<=1'b0; st<=S_AST;
                    end
                end
                // ================= callable GEMV ==============================
                G_AQ: begin    // act-quant: P lanes/cycle, 3-STAGE (mux | mult | round+sat).
                    // Source mux + ctx-rounding registered BEFORE the 64x40 multiply —
                    // the BRAM-read -> mux -> DSP chain was the post-pipeline critical path.
                    cid <= ci; civ <= (ci != (g_k >> LSH));
                    cid1 <= cid; civ1 <= civ;
                    cid2 <= cid1; civ2 <= civ1;
                    cid3 <= cid2; civ3 <= civ2;
                    if (ci != (g_k >> LSH)) ci <= ci + 1'b1;
                    if (civ) begin                 // stage 0: source mux -> lnt_r
                        for (pp=0; pp<P; pp=pp+1) begin
                            case (g_asrc)
                                2'd0: lntmp = $signed(lnout1_r[pp*64 +: 64]);
                                2'd1: begin
                                    cb = ctxv_r[pp*32 +: 32];
                                    // ctx Q6.25 -> Q.22 rsh_round (matches _proj_after_attn)
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
                    if (civ1) begin                // stage 1: multiply (DSP-retimed)
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_prod_r[pp] <= $signed(lnt_r[pp]) * $signed(isact_r);
                            aq_neg_r[pp]  <= (lnt_r[pp] < 0);
                        end
                    end
                    if (civ2) begin                // stage 2a: round (negate/add/shift)
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_prod = aq_prod_r[pp];
                            if (!aq_neg_r[pp])
                                aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                            else
                                aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                            aq_sh_r[pp] <= aq_sh[31:0];
                        end
                    end
                    if (civ3) begin                // stage 2b: clip + pack -> INT8 lane
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_int = aq_sh_r[pp];
                            if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                            aqw[pp*8 +: 8] = aq_int[7:0];
                        end
                        gv_xwe<=1'b1; gv_xdata<=aqw;
                        // AQ/RUN OVERLAP (the §22/§24-proven SB lever, now here):
                        // start the GEMV WITH the first act-row write. xmem row r
                        // is written at edge firstxwe+1+r; the MAC (K2=1) first
                        // reads row r at edge firstxwe+2+4r — the feed stays >=1
                        // row ahead for all r, so the values the MAC consumes are
                        // identical and the AQ feed hides under the MAC compute.
                        if (cid3 == 11'd0) begin
                            gv_m<=g_m; gv_k<=g_k;
                            gv_wbase<=g_wbase[$clog2(WWORDS)-1:0];
                            gv_start<=1'b1;
                        end
                        if (cid3==(g_k >> LSH)-1) begin
                            ci<=0; civ<=0; civ1<=0; civ2<=0; civ3<=0; st<=G_WAIT;
                        end
                    end
                end
                // G_RUN retired: gv_start now fires inside G_AQ (AQ/RUN overlap)
                G_WAIT: if (gv_gdone != 4'd0) begin
                    // groupwise RB overlap: drain begins on the FIRST group commit,
                    // not gv_done — rows of committed groups are final (the MAC
                    // writes ymem[g] exactly once, at group g's commit).
                    ci<=0; gv_rdaddr<=0; rv0<=0; rv1<=0; rv2<=0; dor<=0; st<=G_RB;
                end
                G_RB: begin
                    // FUSED readback -> dequant -> dest. ymem readback is 2-cycle
                    // (rd_addr -> rd_word -> y_out): gv_yout for address rb0 lands at rv2.
                    // dqm/dqe BRAM reads at rb1 land in the same cycle -> stream the
                    // dequant unit at 1 word/cycle. For g_dst==2 (mlp hidden), dequant
                    // output feeds vec_gelu in flight; exit when GELU drains.
                    // Row issue is gated on the row's GROUP being committed
                    // (ci>>GRPSH < gv_gdone): the drain rides the MAC, stalling
                    // (rv0=0 bubbles) when it catches the in-flight group.
                    if (ci < ((g_m + P-1) >> LSH)) begin
                        if ({4'b0, ci} >> GRPSH < {11'b0, gv_gdone}) begin
                            gv_rdaddr<=ci; rb0<=ci; ci<=ci+1'b1;
                        end
                    end else gv_rdaddr <= ((g_m + P-1) >> LSH) - 1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(ci < ((g_m + P-1) >> LSH))
                         && ({4'b0, ci} >> GRPSH < {11'b0, gv_gdone});
                    rv1<=rv0; rv2<=rv1;
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
                        // feeder data-ready horizon: rows 0..dor are committed
                        if (g_dst == 3'd0)
                            qkv_wrow <= {{(11-$clog2(ROWSM+1)){1'b0}}, dor} + 11'd1;
                        case (g_dst)
                            3'd0: qkv_bank [dor] <= dword;
                            3'd1: attn_bank[dor] <= dword;
                            3'd2: begin gl_vin<=1'b1; gl_x<=mword; end // Q4.12 sat16 -> GELU
                            3'd3: mlp_bank [dor] <= dword;
                            default: head_bank[dor] <= dword;  // 4: head logits Q6.25
                        endcase
                        // ci must re-enter g_ret as 0 (S_RES1/S_RES2 stream ci=0..ROWS-1).
                        // On silicon the 6-bit address WRAPS (the §6 board bug); keep clean.
                        if (g_dst != 3'd2 && dor==((g_m + P-1) >> LSH)-1) begin
                            ci<=0; civ<=0; st<=g_ret;
                            // LN fusion: arm the LN for the residual-state feed
                            if (g_ret==S_RES1 || g_ret==S_RES2) ln_start<=1'b1;
                        end else dor<=dor+1'b1;
                    end
                    if (gl_vout) begin                     // collect GELU -> mlpbuf (Q.22)
                        for (pp=0; pp<P; pp=pp+1) begin
                            gl_sh = $signed(gl_y[pp*16 +: 16]) <<< (LN_OUT_FRAC-GELU_FRAC);
                            gsw[pp*64 +: 64] = gl_sh;
                        end
                        mlpbuf_bank[gor] <= gsw;
                        if (gor==ROWSM-1) begin ci<=0; civ<=0; st<=g_ret; end
                        else gor<=gor+1'b1;
                    end
                end
                // ============ attention (R4e: twin engines, head pairs) ========
                // a pair starts once ITS K writes have committed (pair0 needs
                // K h0,h1 = writes 1..2; pair1 needs K h2,h3 = writes 3..4);
                // the V writes are covered by the per-engine read interlocks.
                S_AST: if (kvp_done >= (pair ? 4'd4 : 4'd2)) begin
                    at_startA<=1'b1; at_tcount<=pos + 9'd1;
                    wi<=9'd0; wic<=9'd0; wiv<=1'b0; ldv0<=1'b0; qsel<=1'b0;
                    adone_s<=1'b0; bdone_s<=1'b0; st<=S_ALD;
                end
                S_ALD: begin
                    // stream q to A (qsel=0) then to B (qsel=1); each engine's K
                    // stream starts the moment its q is in (its OWN kv_bank port).
                    // V-starts and ctx collection are the catchers below the case.
                    ldv0     <= (wi != HR[8:0]);
                    q_data_q <= qkv_r;
                    if (!qsel) at_ldvA <= ldv0; else at_ldvB <= ldv0;
                    if (wi != HR[8:0]) wi <= wi + 1'b1;
                    if (qsel ? at_ldvB : at_ldvA) begin
                        if (wic == HR-1) begin
                            if (!qsel) begin
                                kb_rstart <= 1'b1; kb_rkv <= 1'b0;   // K for A (head hh)
                                at_startB <= 1'b1;
                                wi<=9'd0; wic<=9'd0; ldv0<=1'b0; qsel<=1'b1;
                            end else begin
                                kb2_rstart <= 1'b1; kb2_rkv <= 1'b0; // K for B (head hB)
                                st <= S_ACL;
                            end
                        end else wic <= wic + 1'b1;
                    end
                end
                S_ACL: if (adone_s && bdone_s) begin
                    cdr <= 5'd0; st <= S_CDR;
                end
                // drain both ctx catchers into ctxv_bank (the bank's ONLY writer)
                S_CDR: begin
                    if (cdr < HR[4:0])
                        ctxv_bank[hh*HR + {6'b0, cdr[2:0]}] <= ctxbufA[cdr[2:0]];
                    else
                        ctxv_bank[hB*HR + {6'b0, cdr[2:0]}] <= ctxbufB[cdr[2:0]];
                    if (cdr == 2*HR-1) begin
                        if (pair) begin                           // -> proj GEMV
                            g_wbase<=blk*GW_BLK + WB_PROJ; g_m<=D[10:0]; g_k<=D[10:0]; g_asrc<=2'd1;
                            g_asel<=blk*4 + 6'd1; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_PROJ; g_dst<=3'd1;
                            l_gbase<=blk*2 + 4'd1; l_dst<=1'b1; l_ret<=S_FCRET;  // LN2 (fed in S_RES1)
                            g_ret<=S_RES1; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else begin
                            pair<=1'b1; hh<=2'd2; hB<=2'd3; st<=S_AST;
                        end
                    end else cdr <= cdr + 5'd1;
                end
                // ---- res1: xres += attn_out (sync read, 1-cyc skew) ----
                // LN FUSION: LN2 is fed the residual sum AS IT IS WRITTEN
                // (ln_start fired at the proj G_RB exit; l_gbase/l_dst/l_ret
                // were set at the proj dispatch) -> straight to L_COLL.
                S_RES1: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    ln_vin <= civ;
                    ln_g   <= gam_r;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(attn_r[pp*32 +: 32]);
                        xres_bank[cid] <= sw;
                        ln_x <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0; orow<=0; st<=L_COLL;
                        end
                    end
                end
                // mlp_fc GEMV setup (after LN2) --------------------------------
                S_FCRET: if (dbg_stop==2'd2 && blk==4'd0) st<=S_FIN;  // DEBUG: stop after LN2
                  else begin
                    g_wbase<=blk*GW_BLK + WB_FC; g_m<=D_MLP[10:0]; g_k<=D[10:0]; g_asrc<=2'd2;
                    g_asel<=blk*4 + 6'd2; g_frac<=7'd12; g_dqrow<=blk*DQB_P + DR_FC; g_dst<=3'd2;
                    g_ret<=S_MPSET; ci<=0; civ<=0; gor<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // mlp_proj GEMV setup (GELU already streamed inside G_RB) -------
                S_MPSET: begin
                    g_wbase<=blk*GW_BLK + WB_MP; g_m<=D[10:0]; g_k<=D_MLP[10:0]; g_asrc<=2'd3;
                    g_asel<=blk*4 + 6'd3; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_MP; g_dst<=3'd3;
                    // next LN (fed in S_RES2): block blk+1's LN1, or LN_f after the last block
                    if (blk == NLAYER-1) begin
                        l_gbase<=NLAYER*2;     l_dst<=1'b0; l_ret<=S_HEADSET;
                    end else begin
                        l_gbase<=(blk+4'd1)*2; l_dst<=1'b0; l_ret<=S_QKVRET;
                    end
                    g_ret<=S_RES2; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ---- res2: xres += mlp_out ; next block LN1 or final LN_f ----
                // LN FUSION: the next LN (set up at S_MPSET) is fed the residual
                // sum as it is written -> straight to L_COLL.
                S_RES2: begin
                    cid <= ci; civ <= (ci != ROWS[10:0]);
                    if (ci != ROWS[10:0]) ci <= ci + 1'b1;
                    ln_vin <= civ;
                    ln_g   <= gam_r;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(mlp_r[pp*32 +: 32]);
                        xres_bank[cid] <= sw;
                        ln_x <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (dbg_stop==2'd3 && blk==4'd0) st<=S_FIN; // DEBUG: stop after block 0
                            else begin
                                if (blk != NLAYER-1) blk<=blk+1'b1;
                                orow<=0; st<=L_COLL;
                            end
                        end
                    end
                end
                // ---- head GEMV (act = LN_f out in lnout1) -> head_bank -------
                S_HEADSET: begin
                    g_wbase<=WB_HEAD; g_m<=VOCAB[10:0]; g_k<=D[10:0]; g_asrc<=2'd0;
                    g_asel<=4*NLAYER; g_frac<=7'd25; g_dqrow<=DR_HEAD[11:0]; g_dst<=3'd4;
                    g_ret<=S_ARGMAX; ci<=0; civ<=0; gv_ldrst<=1'b1;
                    best_val<=32'sh80000000; best_idx<=9'd0; ar<=0; arv<=0; av1<=0; amv<=0; st<=G_AQ;
                end
                // ---- P-wide argmax over the VOCAB logits (Q6.25, sync read) --
                // 3-stage pipeline (head_bank -> compare chain was the critical path):
                // A: register the row; B: P -> P/2 pairwise maxima; C: reduce + best.
                // First index wins ties (strict >), matching the scalar reference.
                S_ARGMAX: begin
                    ard <= ar; arv <= (ar != ARROWS[$clog2(ARROWS+1)-1:0]);
                    if (ar != ARROWS[$clog2(ARROWS+1)-1:0]) ar <= ar + 1'b1;
                    hw_r <= head_r; ad1 <= ard; av1 <= arv;          // stage A
                    if (av1) begin                                   // stage B: 4 pair maxima
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
                    if (amv) begin                                   // stage C: reduce + best
                        wm_val = pv0; wm_idx = pi0;
                        if (pv1 > wm_val) begin wm_val = pv1; wm_idx = pi1; end
                        if (pv2 > wm_val) begin wm_val = pv2; wm_idx = pi2; end
                        if (pv3 > wm_val) begin wm_val = pv3; wm_idx = pi3; end
                        if (wm_val > best_val) begin
                            best_val <= wm_val; best_idx <= wm_idx;
                        end
                        if (amd==ARROWS-1) begin
                            tok_out <= (wm_val > best_val) ? wm_idx : best_idx;
                            st <= S_FIN;
                        end
                    end
                end
                S_FIN: begin done<=1'b1; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
            // ---- R4e catchers: fire regardless of FSM state -------------------
            // V streams start when the engine's probs are ready AND that head's
            // V write has committed (kvp_done >= vneed); a not-yet-ready kdone
            // latches a pending flag and fires on the write's completion.
            if (at_kdoneA) vpA <= 1'b1;
            if (at_kdoneB) vpB <= 1'b1;
            if ((at_kdoneA || vpA) && (kvp_done >= vneedA)) begin
                kb_rstart  <= 1'b1; kb_rkv  <= 1'b1; vpA <= 1'b0;
            end
            if ((at_kdoneB || vpB) && (kvp_done >= vneedB)) begin
                kb2_rstart <= 1'b1; kb2_rkv <= 1'b1; vpB <= 1'b0;
            end
            if (at_ctxvA) ctxbufA[at_ctxidxA[2:0]] <= at_ctxdataA;
            if (at_ctxvB) ctxbufB[at_ctxidxB[2:0]] <= at_ctxdataB;
            if (at_doneA) adone_s <= 1'b1;
            if (at_doneB) bdone_s <= 1'b1;
            // ---- KV feeder: ALL (head,K/V) writes, armed at the qkv dispatch --
            // K h0..h3 ride the qkv readback (data-ready gated row by row),
            // then V h1,h0,h3,h2 hide under the attention pairs (B-head first,
            // see the declaration comment). Hardcodes NHEAD=4 head pairs.
            case (kvf_st)
                KF_S: if (kvf_active) begin
                    kb_wstart<=1'b1; fr<=0; frv<=0; kvf_st<=KF_F;
                end
                KF_F: begin
                    if (kvf_grant && ((fr == HR[$clog2(ROWSM+1)-1:0])
                                      || (kvw_src < qkv_wrow))) begin
                        frd <= fr; frv <= (fr != HR[$clog2(ROWSM+1)-1:0]);
                        if (fr != HR[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    end else frv <= 1'b0;
                    kb_wvalid  <= frv;
                    kb_wdata_q <= qkv_r;
                    if (frv && frd==HR-1) begin fr<=0; frv<=0; kvf_st<=KF_W; end
                end
                KF_W: if (kb_wdone) begin
                    kvp_done <= kvp_done + 4'd1;
                    if (!kvw_kv) begin                       // K phase: h0->h1->h2->h3
                        if (kvw_h != 2'd3) begin kvw_h<=kvw_h+2'd1; kvf_st<=KF_S; end
                        else begin kvw_kv<=1'b1; kvw_h<=2'd1; kvf_st<=KF_S; end
                    end else begin                           // V phase: h1->h0->h3->h2
                        case (kvw_h)
                            2'd1: begin kvw_h<=2'd0; kvf_st<=KF_S; end
                            2'd0: begin kvw_h<=2'd3; kvf_st<=KF_S; end
                            2'd3: begin kvw_h<=2'd2; kvf_st<=KF_S; end
                            default: begin kvf_active<=1'b0; kvf_st<=KF_IDLE; end
                        endcase
                    end
                end
                default: ;
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
