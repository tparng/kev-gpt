// mamba_seq.sv — the Mamba-2 token sequencer (doc 9 §7 rung 3): one full
// decode step, embed -> 7x(pre-norm -> in_proj -> conv+silu -> scan -> gated
// norm -> out_proj -> residual) -> final norm -> head -> argmax, entirely in
// fabric, driving the four gated sub-cores (gemv_i4i8, conv_silu,
// ssm_scan_row, rmsnorm_gated) through their load/start/done ports.
//
// Bring-up grade: sequential everything (~460k cyc/token); the wide GEMV and
// phase overlap are the speed campaign AFTER first tokens. Bit-honest: every
// datapath here reproduces model/mamba2_fixed.py exactly (integer phases) or
// at LUT resolution (norms); the per-phase harness compares each buffer.
//
// Host interface: a sel-addressed table-write port (weights/LUTs/consts, see
// WSEL_*), tok_in + start, done + tok_out, and a sel-addressed DEBUG read
// port over the internal buffers so run_mamba_seq.py can gate every phase.

`default_nettype none

module mamba_seq #(
    parameter int D     = 256,
    parameter int DIN   = 512,
    parameter int NST   = 64,
    parameter int H     = 8,
    parameter int L     = 7,
    parameter int V     = 1024,
    parameter int CONVD = DIN + 2*NST,       // 640
    parameter int INROWS = 2*DIN + 2*NST + H // 1160
) (
    input  wire        clk,
    input  wire        rst,
    output wire        ready,                 // scan clear-sweep done
    input  wire        start,
    input  wire [9:0]  tok_in,
    output reg         done,
    output reg  [9:0]  tok_out,

    // table writes: sel picks the target memory
    input  wire        wr_en,
    input  wire [3:0]  wr_sel,
    input  wire [18:0] wr_addr,
    input  wire [31:0] wr_data,

    // debug reads (combinational): sel picks the buffer
    input  wire [3:0]  dbg_sel,
    input  wire [10:0] dbg_addr,
    output reg  signed [31:0] dbg_data
);
    // write selectors
    localparam [3:0] WSEL_GW = 0, WSEL_EMB = 1, WSEL_ESC = 2, WSEL_CW = 3,
                     WSEL_CB = 4, WSEL_SLUT = 5, WSEL_ALUT = 6, WSEL_DLUT = 7,
                     WSEL_LNG = 8, WSEL_NRMG = 9, WSEL_NFG = 10,
                     WSEL_RSIN = 11, WSEL_RSOUT = 12, WSEL_RSHD = 13,
                     WSEL_CONST = 14;
    // debug selectors
    localparam [3:0] DBG_X = 0, DBG_XN = 1, DBG_ZX = 2, DBG_XBC = 3,
                     DBG_Y = 4, DBG_YN = 5, DBG_LOGIT = 6, DBG_X8 = 7;

    // ------------------------------------------------------------ tables ----
    (* ram_style = "ultra" *) reg [31:0] emb  [0:V*D/8-1];      // INT4 rows
    reg [15:0] esc   [0:V-1];                 // emb per-row fp16 scale
    reg [15:0] convw [0:L*CONVD*4-1];         // Q1.14
    reg [15:0] convb [0:L*CONVD-1];
    reg [15:0] slut  [0:255];                 // Q3.12 silu
    reg [15:0] alut  [0:L*H*256-1];           // Q0.16 decay
    reg [15:0] dlut  [0:L*H*256-1];           // Q1.14 dt
    reg [15:0] lng   [0:L*D-1];               // pre-norm gamma Q1.14
    reg [15:0] nrmg  [0:L*DIN-1];             // gated-norm gamma Q1.14
    reg [15:0] nfg   [0:D-1];                 // final gamma Q2.13
    reg [15:0] rsin  [0:L*INROWS-1];          // in_proj row scales fp16
    reg [15:0] rsout [0:L*D-1];
    reg [15:0] rshd  [0:V-1];
    reg [31:0] consts[0:127];                 // packed per-layer constants

    // consts layout (word index = l*16 + k):
    //  k=0 in-act recip  {e[7:0], m[15:0]}   (x/s = q*m >> (e+12))
    //  k=1 in-act scale  {e[7:0], m[15:0]}   (dequant second stage)
    //  k=2 out-act recip, k=3 out-act scale
    //  k=4 {bC[7:4], bB[3:0], bX[11:8]} packed as {20'b0, bX[3:0], bC[3:0], bB[3:0]}
    //  k=5..8 D_q per head, two Q2.13 per word {D[2h+1],D[2h]}
    //  112: head recip, 113: head scale (same packing)

    always @(posedge clk) if (wr_en) begin
        case (wr_sel)
            WSEL_GW:    ;                     // handled inside gemv (below)
            WSEL_EMB:   emb  [wr_addr[14:0]] <= wr_data;
            WSEL_ESC:   esc  [wr_addr[9:0]]  <= wr_data[15:0];
            WSEL_CW:    convw[wr_addr[14:0]] <= wr_data[15:0];
            WSEL_CB:    convb[wr_addr[12:0]] <= wr_data[15:0];
            WSEL_SLUT:  slut [wr_addr[7:0]]  <= wr_data[15:0];
            WSEL_ALUT:  alut [wr_addr[14:0]] <= wr_data[15:0];
            WSEL_DLUT:  dlut [wr_addr[14:0]] <= wr_data[15:0];
            WSEL_LNG:   lng  [wr_addr[10:0]] <= wr_data[15:0];
            WSEL_NRMG:  nrmg [wr_addr[11:0]] <= wr_data[15:0];
            WSEL_NFG:   nfg  [wr_addr[7:0]]  <= wr_data[15:0];
            WSEL_RSIN:  rsin [wr_addr[12:0]] <= wr_data[15:0];
            WSEL_RSOUT: rsout[wr_addr[10:0]] <= wr_data[15:0];
            WSEL_RSHD:  rshd [wr_addr[9:0]]  <= wr_data[15:0];
            WSEL_CONST: consts[wr_addr[6:0]] <= wr_data;
            default: ;
        endcase
    end

    // ------------------------------------------------------------ buffers ---
    reg signed [15:0] xbuf  [0:D-1];          // residual, Q3.12
    reg signed [15:0] xnbuf [0:DIN-1];        // normed (D used) / yn (DIN)
    reg signed [15:0] zxbuf [0:INROWS-1];     // dequantized in_proj out Q3.12
    reg signed [15:0] ybuf  [0:DIN-1];        // scan out + D-skip, Q3.12
    reg signed [7:0]  q8buf [0:DIN-1];        // quantized activations
    reg signed [15:0] logit [0:V-1];

    // ---------------------------------------------------------- sub-cores ---
    // gemv
    reg         g_start;  wire g_done;
    reg  [18:0] g_base;   reg [10:0] g_rows;  reg [6:0] g_wpr;
    reg         g_wrx;    reg [8:0] g_wrx_a;  reg signed [7:0] g_wrx_d;
    reg  [10:0] g_rda;    wire signed [31:0] g_acc;
    gemv_i4i8 #(.ROWS(INROWS), .D_IN(DIN), .WMEM(524288)) u_gemv (
        .clk(clk), .rst(rst), .start(g_start), .done(g_done),
        .base(g_base), .rows(g_rows), .wpr(g_wpr),
        .wr_w(wr_en && wr_sel == WSEL_GW), .wr_w_addr(wr_addr), .wr_w_data(wr_data),
        .wr_x(g_wrx), .wr_x_addr(g_wrx_a), .wr_x_data(g_wrx_d),
        .rd_acc_addr(g_rda), .rd_acc_data(g_acc));

    // conv+silu
    reg         c_start;  wire c_done;
    reg         c_wrw, c_wrb, c_wrx, c_wrl;
    reg  [11:0] c_wrw_a;  reg [9:0] c_wrb_a, c_wrx_a, c_rda;
    reg  [7:0]  c_wrl_a;
    reg signed [15:0] c_wrd;
    wire signed [15:0] c_y;
    conv_silu #(.CH(CONVD), .K(4)) u_conv (
        .clk(clk), .rst(rst), .start(c_start), .done(c_done),
        .wr_w(c_wrw), .wr_w_addr(c_wrw_a), .wr_w_data(c_wrd),
        .wr_b(c_wrb), .wr_b_addr(c_wrb_a), .wr_b_data(c_wrd),
        .wr_x(c_wrx), .wr_x_addr(c_wrx_a), .wr_x_data(c_wrd),
        .wr_lut(c_wrl), .wr_lut_addr(c_wrl_a), .wr_lut_data(c_wrd),
        .rd_y_addr(c_rda), .rd_y_data(c_y));

    // scan row (56 contexts)
    reg         s_start;  wire s_done, s_ready;
    reg  [5:0]  s_pbase;
    reg  [15:0] s_aq;     reg signed [5:0] s_shi, s_shy;
    reg         s_wrdtx, s_wrb, s_wrc;
    reg  [5:0]  s_wrdtx_a, s_wrb_a, s_wrc_a, s_rda;
    reg signed [15:0] s_dtx_d;
    reg signed [7:0]  s_bc_d;
    wire signed [15:0] s_y;
    ssm_scan_row #(.P(64), .N(NST), .CTX(L*H)) u_scan (
        .clk(clk), .rst(rst), .ready(s_ready), .start(s_start), .done(s_done),
        .pbase(s_pbase), .a_q(s_aq), .sh_i(s_shi), .sh_y(s_shy),
        .wr_dtx(s_wrdtx), .wr_dtx_addr(s_wrdtx_a), .wr_dtx_data(s_dtx_d),
        .wr_b(s_wrb), .wr_b_addr(s_wrb_a), .wr_b_data(s_bc_d),
        .wr_c(s_wrc), .wr_c_addr(s_wrc_a), .wr_c_data(s_bc_d),
        .rd_y_addr(s_rda), .rd_y_data(s_y));
    assign ready = s_ready;

    // gated rmsnorm
    reg         n_start;  wire n_done;
    reg         n_gated;
    reg         n_wry, n_wrz, n_wrg, n_wrl;
    reg  [8:0]  n_wry_a, n_wrz_a, n_wrg_a, n_rda;
    reg  [7:0]  n_wrl_a;
    reg signed [15:0] n_wrd;
    wire signed [15:0] n_out;
    rmsnorm_gated #(.D(DIN)) u_norm (
        .clk(clk), .rst(rst), .start(n_start), .done(n_done), .gated(n_gated),
        .wr_y(n_wry), .wr_y_addr(n_wry_a), .wr_y_data(n_wrd),
        .wr_z(n_wrz), .wr_z_addr(n_wrz_a), .wr_z_data(n_wrd),
        .wr_g(n_wrg), .wr_g_addr(n_wrg_a), .wr_g_data(n_wrd),
        .wr_lut(n_wrl), .wr_lut_addr(n_wrl_a), .wr_lut_data(n_wrd),
        .rd_o_addr(n_rda), .rd_o_data(n_out));

    // ------------------------------------------------------- scale helpers --
    // fp16 decode: value = {1,frac} * 2^(exp-25); dequant of INT32 acc to
    // Q3.12: t = acc * m11; shift = 25 - exp - 12 = 13 - exp + 24? — computed
    // below with explicit rounding; second stage multiplies the (m,e) act
    // scale. Saturation to INT16 at the end.
    function automatic signed [47:0] mul_m11(input signed [31:0] a,
                                             input [15:0] fp16);
        reg [10:0] m;
        begin
            m = {1'b1, fp16[9:0]};
            mul_m11 = a * $signed({1'b0, m});
        end
    endfunction

    function automatic signed [15:0] sat16f(input signed [47:0] v);
        sat16f = (v > 48'sd32767) ? 16'sd32767 :
                 (v < -48'sd32768) ? -16'sd32768 : v[15:0];
    endfunction

    // rounded arithmetic shift right (n >= 1)
    function automatic signed [47:0] rshr(input signed [47:0] v,
                                          input signed [7:0] n);
        if (n <= 0)      rshr = v <<< (-n);
        else             rshr = (v + (48'sd1 <<< (n - 1))) >>> n;
    endfunction

    // ---------------------------------------------------------- FSM ---------
    localparam [4:0]
        S_IDLE=0, S_EMB=1, S_NIN_LD=2, S_NIN=3, S_QIN=4, S_GIN_LD=5,
        S_GIN=6, S_GIN_RD=7, S_CONV_LDW=8, S_CONV_LDX=9, S_CONV=10,
        S_CONV_RD=11, S_SCAN_PREP=12, S_SCAN_H=13, S_SCAN_RD=14,
        S_NG_LD=15, S_NG=16, S_QOUT=17, S_GOUT_LD=18, S_GOUT=19,
        S_GOUT_RD=20, S_NF_LD=21, S_NF=22, S_QHD=23, S_GHD_LD=24,
        S_GHD=25, S_GHD_RD=26, S_DONE=27;

    reg [4:0]  st;
    reg [2:0]  li;                            // layer index
    reg [3:0]  hi;                            // head index
    reg [11:0] i;                             // generic element counter
    reg [1:0]  sub;                           // sub-step within a state
    reg signed [15:0] best;
    reg [9:0]  besti;

    // per-layer constants (latched at layer start)
    wire [31:0] c_inr  = consts[li*16 + 0];
    wire [31:0] c_ins  = consts[li*16 + 1];
    wire [31:0] c_outr = consts[li*16 + 2];
    wire [31:0] c_outs = consts[li*16 + 3];
    wire [3:0]  bB = consts[li*16 + 4][3:0];
    wire [3:0]  bC = consts[li*16 + 4][7:4];
    wire [3:0]  bX = consts[li*16 + 4][11:8];
    wire [31:0] c_hdr  = consts[112];
    wire [31:0] c_hds  = consts[113];

    // scan per-head registers
    reg [15:0] dtq_h;
    reg [4:0]  eh_h;

    // dt bit-length (priority encoder over 15 bits)
    function automatic [4:0] bitlen15(input [14:0] v);
        integer b;
        begin
            bitlen15 = 0;
            for (b = 14; b >= 0; b = b - 1)
                if (v[b] && bitlen15 == 0) bitlen15 = b + 1;
        end
    endfunction

    // temp regs used across cycles inside states
    reg signed [31:0] acc_t;
    reg signed [47:0] t1;
    reg [15:0] fp_t;
    integer kk;

    always @(posedge clk) begin
        done <= 1'b0;
        g_start <= 0; c_start <= 0; s_start <= 0; n_start <= 0;
        g_wrx <= 0; c_wrw <= 0; c_wrb <= 0; c_wrx <= 0; c_wrl <= 0;
        s_wrdtx <= 0; s_wrb <= 0; s_wrc <= 0;
        n_wry <= 0; n_wrz <= 0; n_wrg <= 0; n_wrl <= 0;

        if (rst) begin
            st <= S_IDLE;
        end else case (st)
          S_IDLE: if (start && s_ready) begin
              li <= 0; i <= 0; sub <= 0; st <= S_EMB;
          end

          // ---- embed: x[c] = nib(emb[tok])*escale, Q3.12 --------------------
          S_EMB: begin
              // i = element 0..D-1; word = tok*D/8 + i[?]; two-cycle per elem
              case (sub)
                0: begin acc_t <= 32'sd0; sub <= 1;
                     fp_t <= esc[tok_in];
                     t1 <= 0;
                end
                1: begin
                     // read word, extract nibble i%8, sign-extend
                     acc_t <= $signed({{28{emb[tok_in*(D/8) + i[11:3]][(i[2:0])*4+3]}},
                               emb[tok_in*(D/8) + i[11:3]][(i[2:0])*4 +: 4]});
                     sub <= 2;
                end
                2: begin
                     // x = nib * fp16 -> Q3.12: shift = 13 - exp (see notes)
                     xbuf[i[7:0]] <= sat16f(rshr(mul_m11(acc_t, fp_t),
                                    8'sd13 - $signed({3'b0, fp_t[14:10]})));
                     sub <= 0;
                     if (i == D-1) begin i <= 0; st <= S_NIN_LD; end
                     else i <= i + 1;
                end
              endcase
          end

          // ---- pre-norm: stream x + ln gamma into norm core (ungated) ------
          S_NIN_LD: begin
              n_gated <= 0;
              case (sub)
                0: begin n_wry <= 1; n_wry_a <= i[9:0]; n_wrd <= xbuf[i[7:0]];
                     sub <= 1; end
                1: begin n_wrg <= 1; n_wrg_a <= i[9:0];
                     n_wrd <= lng[li*D + i[7:0]];
                     sub <= 0;
                     if (i == D-1) begin i <= 0; st <= S_NIN; n_start <= 1; end
                     else i <= i + 1;
                end
              endcase
          end
          S_NIN: if (n_done) begin st <= S_QIN; i <= 0; end

          // ---- quantize xn -> INT8 via in-recip ----------------------------
          S_QIN: begin
              case (sub)
                0: begin n_rda <= i[9:0]; sub <= 1; end
                1: begin
                     t1 <= rshr($signed(n_out) * $signed({1'b0, c_inr[15:0]}),
                                $signed({1'b0, c_inr[23:16]}) + 8'sd12);
                     sub <= 2;
                end
                2: begin
                     q8buf[i[8:0]] <= (t1 > 48'sd127) ? 8'sd127 :
                                      (t1 < -48'sd128) ? -8'sd128 : t1[7:0];
                     sub <= 0;
                     if (i == D-1) begin i <= 0; st <= S_GIN_LD; end
                     else i <= i + 1;
                end
              endcase
          end

          // ---- in_proj GEMV -------------------------------------------------
          S_GIN_LD: begin
              g_wrx <= 1; g_wrx_a <= i[8:0]; g_wrx_d <= q8buf[i[8:0]];
              if (i == DIN-1) begin        // zero-pad 256..511
                  i <= 0; st <= S_GIN;
                  g_base <= li * (INROWS * (D/8));
                  g_rows <= INROWS; g_wpr <= D/8;
                  g_start <= 1;
              end else i <= i + 1;
          end
          S_GIN: if (g_done) begin st <= S_GIN_RD; i <= 0; sub <= 0; end
          S_GIN_RD: begin
              case (sub)
                0: begin g_rda <= i[10:0]; sub <= 1; end
                1: begin
                     fp_t <= rsin[li*INROWS + i[10:0]];
                     acc_t <= g_acc; sub <= 2;
                end
                2: begin  // stage 1: acc * row-scale mantissa, shift by exp
                     t1 <= rshr(mul_m11(acc_t, fp_t),
                                8'sd10 - $signed({3'b0, fp_t[14:10]}));
                     sub <= 3;
                end
                3: begin  // stage 2: * act scale (m,e) -> Q3.12
                     zxbuf[i[10:0]] <= sat16f(rshr(
                         t1 * $signed({1'b0, c_ins[15:0]}),
                         $signed({1'b0, c_ins[23:16]}) + 8'sd15 - 8'sd12));
                     sub <= 0;
                     if (i == INROWS-1) begin i <= 0; st <= S_CONV_LDW; end
                     else i <= i + 1;
                end
              endcase
          end

          // ---- conv: stream weights, bias, x --------------------------------
          S_CONV_LDW: begin
              c_wrw <= 1; c_wrw_a <= i[11:0];
              c_wrd <= convw[li*CONVD*4 + i[11:0]];
              if (i == CONVD*4-1) begin i <= 0; st <= S_CONV_LDX; sub <= 0; end
              else i <= i + 1;
          end
          S_CONV_LDX: begin
              case (sub)
                0: begin c_wrb <= 1; c_wrb_a <= i[9:0];
                     c_wrd <= convb[li*CONVD + i[9:0]]; sub <= 1; end
                1: begin c_wrx <= 1; c_wrx_a <= i[9:0];
                     c_wrd <= zxbuf[DIN + i[9:0]];      // xBC slice
                     sub <= 0;
                     if (i == CONVD-1) begin i <= 0; st <= S_CONV; c_start <= 1; end
                     else i <= i + 1;
                end
              endcase
          end
          S_CONV: if (c_done) begin st <= S_CONV_RD; i <= 0; sub <= 0; end
          S_CONV_RD: begin  // read conv y back into xnbuf (reuse as xBC')
              case (sub)
                0: begin c_rda <= i[9:0]; sub <= 1; end
                1: begin xnbuf[i[9:0]] <= c_y; sub <= 0;
                     if (i == CONVD-1) begin i <= 0; st <= S_SCAN_PREP; sub <= 0; end
                     else i <= i + 1;
                end
              endcase
          end

          // ---- scan prep: write B8/C8 once (shared across heads) ------------
          S_SCAN_PREP: begin
              case (sub)
                0: begin
                     t1 <= rshr($signed(xnbuf[DIN + i[6:0]]),
                                $signed(8'sd12) - $signed({4'b0, bB}));
                     sub <= 1;
                end
                1: begin
                     s_wrb <= 1; s_wrb_a <= i[5:0];
                     s_bc_d <= (t1 > 48'sd127) ? 8'sd127 :
                               (t1 < -48'sd128) ? -8'sd128 : t1[7:0];
                     sub <= 2;
                end
                2: begin
                     t1 <= rshr($signed(xnbuf[DIN + NST + i[6:0]]),
                                $signed(8'sd12) - $signed({4'b0, bC}));
                     sub <= 3;
                end
                3: begin
                     s_wrc <= 1; s_wrc_a <= i[5:0];
                     s_bc_d <= (t1 > 48'sd127) ? 8'sd127 :
                               (t1 < -48'sd128) ? -8'sd128 : t1[7:0];
                     sub <= 0;
                     if (i == NST-1) begin i <= 0; hi <= 0; st <= S_SCAN_H; sub <= 0; end
                     else i <= i + 1;
                end
              endcase
          end

          // ---- per-head: dt/a lookup, dtx write, scan run --------------------
          S_SCAN_H: begin
              case (sub)
                0: begin  // LUT index from dtraw (Q3.12): (v+32768)>>8
                     fp_t <= zxbuf[2*DIN + 2*NST + hi[2:0]] + 16'sd32768 >> 8;
                     sub <= 1;
                end
                1: begin
                     dtq_h <= dlut[li*H*256 + hi[2:0]*256 + fp_t[7:0]];
                     s_aq  <= alut[li*H*256 + hi[2:0]*256 + fp_t[7:0]];
                     sub <= 2;
                end
                2: begin
                     eh_h <= (21 - bitlen15(dtq_h[14:0]) > 20) ? 5'd20 :
                             (bitlen15(dtq_h[14:0]) > 21) ? 5'd0 :
                             5'd21 - bitlen15(dtq_h[14:0]);
                     sub <= 3;
                end
                3: begin  // per-channel dtx: x8 = rnd(xv>>(12-bX)) sat8,
                          // dtx = rnd((dt_q*x8)>>(14-eh))
                     t1 <= rshr($signed(xnbuf[hi[2:0]*64 + i[5:0]]),
                                $signed(8'sd12) - $signed({4'b0, bX}));
                     sub <= 1 + 3;  // -> 4
                end
                4: begin
                     acc_t <= (t1 > 48'sd127) ? 32'sd127 :
                              (t1 < -48'sd128) ? -32'sd128 : t1[31:0];
                     sub <= 5;
                end
                5: begin
                     s_wrdtx <= 1; s_wrdtx_a <= i[5:0];
                     s_dtx_d <= sat16f(rshr(acc_t * $signed({1'b0, dtq_h}),
                                            8'sd14 - $signed({3'b0, eh_h})));
                     if (i == 63) begin
                         i <= 0; sub <= 6;
                     end else begin i <= i + 1; sub <= 3; end
                end
                6: begin
                     s_pbase <= li*H + hi[2:0];
                     s_shi <= 6'sd16 + 6'sd13 - $signed({2'b0, bX})
                              - $signed({1'b0, eh_h}) - $signed({2'b0, bB});
                     s_shy <= $signed({2'b0, bC}) + 6'sd13 - 6'sd13;
                     s_start <= 1; sub <= 7;
                end
                7: if (s_done) begin st <= S_SCAN_RD; i <= 0; sub <= 0; end
              endcase
          end
          S_SCAN_RD: begin  // y = rnd(yq>>1) + rnd(D_q*xv>>13), Q3.12
              case (sub)
                0: begin s_rda <= i[5:0]; sub <= 1; end
                1: begin
                     acc_t <= s_y; sub <= 2;
                     t1 <= $signed(consts[li*16 + 5 + {2'b0, hi[2:1]}]
                            >> (hi[0] ? 16 : 0)) & 32'shFFFF;
                end
                2: begin
                     ybuf[hi[2:0]*64 + i[5:0]] <= sat16f(
                        rshr({{32{acc_t[15]}}, acc_t[15:0]}, 8'sd1)
                      + rshr($signed(t1[15:0]) * $signed(xnbuf[hi[2:0]*64 + i[5:0]]),
                             8'sd13));
                     sub <= 0;
                     if (i == 63) begin
                         i <= 0;
                         if (hi == H-1) begin hi <= 0; st <= S_NG_LD; sub <= 0; end
                         else begin hi <= hi + 1; st <= S_SCAN_H; sub <= 0; end
                     end else i <= i + 1;
                end
              endcase
          end

          // ---- gated norm ----------------------------------------------------
          S_NG_LD: begin
              n_gated <= 1;
              case (sub)
                0: begin n_wry <= 1; n_wry_a <= i[9:0]; n_wrd <= ybuf[i[8:0]];
                     sub <= 1; end
                1: begin n_wrz <= 1; n_wrz_a <= i[9:0]; n_wrd <= zxbuf[i[8:0]];
                     sub <= 2; end
                2: begin n_wrg <= 1; n_wrg_a <= i[9:0];
                     n_wrd <= nrmg[li*DIN + i[8:0]];
                     sub <= 0;
                     if (i == DIN-1) begin i <= 0; st <= S_NG; n_start <= 1; end
                     else i <= i + 1;
                end
              endcase
          end
          S_NG: if (n_done) begin st <= S_QOUT; i <= 0; sub <= 0; end

          // ---- quantize yn, out GEMV, residual add ---------------------------
          S_QOUT: begin
              case (sub)
                0: begin n_rda <= i[9:0]; sub <= 1; end
                1: begin
                     t1 <= rshr($signed(n_out) * $signed({1'b0, c_outr[15:0]}),
                                $signed({1'b0, c_outr[23:16]}) + 8'sd12);
                     sub <= 2;
                end
                2: begin
                     q8buf[i[8:0]] <= (t1 > 48'sd127) ? 8'sd127 :
                                      (t1 < -48'sd128) ? -8'sd128 : t1[7:0];
                     sub <= 0;
                     if (i == DIN-1) begin i <= 0; st <= S_GOUT_LD; end
                     else i <= i + 1;
                end
              endcase
          end
          S_GOUT_LD: begin
              g_wrx <= 1; g_wrx_a <= i[8:0]; g_wrx_d <= q8buf[i[8:0]];
              if (i == DIN-1) begin
                  i <= 0; st <= S_GOUT;
                  g_base <= consts[120][18:0] + li * (D * (DIN/8));
                  g_rows <= D; g_wpr <= DIN/8;
                  g_start <= 1;
              end else i <= i + 1;
          end
          S_GOUT: if (g_done) begin st <= S_GOUT_RD; i <= 0; sub <= 0; end
          S_GOUT_RD: begin
              case (sub)
                0: begin g_rda <= i[10:0]; sub <= 1; end
                1: begin fp_t <= rsout[li*D + i[7:0]]; acc_t <= g_acc; sub <= 2; end
                2: begin
                     t1 <= rshr(mul_m11(acc_t, fp_t),
                                8'sd10 - $signed({3'b0, fp_t[14:10]}));
                     sub <= 3;
                end
                3: begin
                     xbuf[i[7:0]] <= sat16f(
                         $signed({{32{xbuf[i[7:0]][15]}}, xbuf[i[7:0]]}) +
                         rshr(t1 * $signed({1'b0, c_outs[15:0]}),
                              $signed({1'b0, c_outs[23:16]}) + 8'sd15 - 8'sd12));
                     sub <= 0;
                     if (i == D-1) begin
                         i <= 0;
                         if (li == L-1) begin li <= 0; st <= S_NF_LD; end
                         else begin li <= li + 1; st <= S_NIN_LD; end
                     end else i <= i + 1;
                end
              endcase
          end

          // ---- final norm (ungated, Q2.13 gamma => shift 13 handled by
          //      exporting gamma pre-scaled; norm core treats gamma Q1.14, so
          //      nfg values are stored HALVED by the exporter contract) ------
          S_NF_LD: begin
              n_gated <= 0;
              case (sub)
                0: begin n_wry <= 1; n_wry_a <= i[9:0]; n_wrd <= xbuf[i[7:0]];
                     sub <= 1; end
                1: begin n_wrg <= 1; n_wrg_a <= i[9:0]; n_wrd <= nfg[i[7:0]];
                     sub <= 0;
                     if (i == D-1) begin i <= 0; st <= S_NF; n_start <= 1; end
                     else i <= i + 1;
                end
              endcase
          end
          S_NF: if (n_done) begin st <= S_QHD; i <= 0; sub <= 0; end
          S_QHD: begin
              case (sub)
                0: begin n_rda <= i[9:0]; sub <= 1; end
                1: begin
                     // Q2.13 gamma means norm output is HALF the true value:
                     // fold the x2 into the head recip shift (e-1)
                     t1 <= rshr($signed(n_out) * $signed({1'b0, c_hdr[15:0]}),
                                $signed({1'b0, c_hdr[23:16]}) + 8'sd11);
                     sub <= 2;
                end
                2: begin
                     q8buf[i[8:0]] <= (t1 > 48'sd127) ? 8'sd127 :
                                      (t1 < -48'sd128) ? -8'sd128 : t1[7:0];
                     sub <= 0;
                     if (i == D-1) begin i <= 0; st <= S_GHD_LD; end
                     else i <= i + 1;
                end
              endcase
          end
          S_GHD_LD: begin
              g_wrx <= 1; g_wrx_a <= i[8:0]; g_wrx_d <= q8buf[i[8:0]];
              if (i == DIN-1) begin
                  i <= 0; st <= S_GHD;
                  g_base <= consts[121][18:0];  // head image base
                  g_rows <= V; g_wpr <= D/8;
                  g_start <= 1;
              end else i <= i + 1;
          end
          S_GHD: if (g_done) begin st <= S_GHD_RD; i <= 0; sub <= 0;
                  best <= -16'sd32768; besti <= 0; end
          S_GHD_RD: begin
              case (sub)
                0: begin g_rda <= i[10:0]; sub <= 1; end
                1: begin fp_t <= rshd[i[9:0]]; acc_t <= g_acc; sub <= 2; end
                2: begin
                     t1 <= rshr(mul_m11(acc_t, fp_t),
                                8'sd10 - $signed({3'b0, fp_t[14:10]}));
                     sub <= 3;
                end
                3: begin
                     logit[i[9:0]] <= sat16f(rshr(
                         t1 * $signed({1'b0, c_hds[15:0]}),
                         $signed({1'b0, c_hds[23:16]}) + 8'sd15 - 8'sd12));
                     if (sat16f(rshr(t1 * $signed({1'b0, c_hds[15:0]}),
                         $signed({1'b0, c_hds[23:16]}) + 8'sd15 - 8'sd12)) > best) begin
                         best <= sat16f(rshr(t1 * $signed({1'b0, c_hds[15:0]}),
                             $signed({1'b0, c_hds[23:16]}) + 8'sd15 - 8'sd12));
                         besti <= i[9:0];
                     end
                     sub <= 0;
                     if (i == V-1) begin st <= S_DONE; end
                     else i <= i + 1;
                end
              endcase
          end
          S_DONE: begin tok_out <= besti; done <= 1'b1; st <= S_IDLE; end
          default: st <= S_IDLE;
        endcase

        // debug read mux (registered)
        case (dbg_sel)
            DBG_X:     dbg_data <= xbuf [dbg_addr[7:0]];
            DBG_XN:    dbg_data <= xnbuf[dbg_addr[9:0]];
            DBG_ZX:    dbg_data <= zxbuf[dbg_addr];
            DBG_XBC:   dbg_data <= xnbuf[dbg_addr[9:0]];
            DBG_Y:     dbg_data <= ybuf [dbg_addr[8:0]];
            DBG_YN:    dbg_data <= 32'sd0;
            DBG_LOGIT: dbg_data <= logit[dbg_addr[9:0]];
            DBG_X8:    dbg_data <= q8buf[dbg_addr[8:0]];
            default:   dbg_data <= 32'sd0;
        endcase
    end

endmodule

`default_nettype wire
