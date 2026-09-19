// -----------------------------------------------------------------------------
// conv1d_seq -- Stage 1 conv front-end's conv1d engine (gen2asr/
// ASR-ACCELERATOR-OP-SEQUENCE.md Stage 1 table: conv1/conv2/conv3, "No conv
// engine anywhere in kevgpt_seq -- every existing compute block
// (gemv_banked_resident_vec) is a GEMV (dense matmul), not a sliding-window
// op. This is the single biggest new hardware component."). This module
// reframes conv1d as a SEQUENCE of TOUT ordinary GEMV calls against the SAME
// resident weight image, each fed a gathered strided window -- no new MAC
// hardware, gemv_banked_resident_vec.sv (already proven, fabric/stage3/) is
// reused unmodified, same G_XRESET/G_XFEED/G_XSTART/G_WAIT/G_DRAIN dispatcher
// idiom as decoder_block_seq.sv/encoder_block_seq.sv.
//
// Window/weight layout (own choice, not PyTorch's): real conv weight is
// [COUT][CIN_real][KW] (PyTorch's own nn.Conv1d layout); this module wants a
// K-major/channel-minor window so it can be built out of PLAIN ROW COPIES of
// the t-major/channel-minor input storage every other block in this project
// already uses (groupnorm1_vec.sv's own xbank, encoder_block_seq.sv's own
// xres_bank) -- so the packer transposes the weight's last two axes at
// export time (W.transpose(0,2,1), trivial in numpy) to get
// W'[cout, k*CIN+ci] = W[cout, ci, k], and this module's own xtbank is
// [TIN][CIN/P] (row r = t*CGRP+cg). The payoff: for output position t_out,
// the whole KW*CIN-element window is exactly KMAX/P=KW*CGRP CONSECUTIVE
// xtbank rows starting at row (t_out*STRIDE)*CGRP -- a straight linear
// address, no gather/mux logic at all (t_out*STRIDE is always a multiple of
// CGRP's own row unit since STRIDE only ever appears as a row-COUNT
// multiplier here, not a bit position).
//
// CIN is the EFFECTIVE (possibly padded) channel count and MUST be a
// multiple of P: conv1's real input has 1 channel (raw audio), which does
// NOT divide P=8 -- rather than special-case that shape, this module always
// pads CIN up to P (CIN=8 for conv1, with the packer zeroing the weight's 7
// extra per-tap lanes so the extra MACs contribute exactly 0). This wastes
// 8x the K-dimension work for conv1 specifically (127 real taps -> 1016
// padded ones) -- a known, bounded, and cheap cost (conv1's real work is
// tiny next to conv2/conv3's own GEMVs or any of this project's LLM linear
// layers) traded for ONE shape-generic RTL module instead of two. Same
// "correct now, a real later resource-optimization option if needed, not a
// correctness requirement" idiom this project already uses for vec_tanh.sv/
// vec_silu.sv's own unpaired-lane tradeoff.
//
// Output format: ALWAYS wide Q6.25 (P*32b/lane, gdequant()'d via a single
// runtime dq_shift, optional per-output-channel bias added in the SAME
// domain) -- the SAME "real activation" format layernorm_vec_gendiv.sv/
// groupnorm1_vec.sv's own x_in expects, regardless of what real activation
// (tanh/gelu) comes next. Format conversion to whatever THAT LUT needs
// (Q4.12) and re-quantization back to INT8 for the NEXT conv's own x_we feed
// are a top-level-FSM/glue concern (conv_front_end_seq.sv), not this
// module's -- keeps this module bit-format-agnostic and reusable as-is for
// conv1/conv2/conv3 via 3 separate parameterizations.
//
// Protocol: preload weights (gv_ld_we/gv_ld_data, passthrough to the
// internal gemv core) and, if HAS_BIAS, bias (b_we/b_data, MROWS=COUT/P
// rows) ONCE; preload the whole input tensor (xt_we/xt_data, XTROWS=TIN*CGRP
// rows, t-major/channel-minor) ONCE; pulse `go` for the WHOLE conv (all TOUT
// output positions in one call); y_valid/y_data stream MROWS rows per
// output position, TOUT*MROWS rows total, same t-major/channel-minor order
// as the input (so this module's own output can feed directly into the
// next stage's input bank without reshuffling).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module conv1d_seq #(
    parameter integer P        = 8,
    parameter integer WBW      = 8,
    parameter integer CIN      = 8,        // effective (possibly padded) input channels, multiple of P
    parameter integer COUT     = 288,      // output channels, multiple of P
    parameter integer KW       = 127,      // kernel width (real taps; CIN padding does not affect this)
    parameter integer STRIDE   = 64,
    parameter integer TIN      = 3000,     // input timesteps
    parameter integer HAS_BIAS = 0,
    parameter integer LANES    = 128,
    parameter integer WWORDS   = 1 << 20
) (
    input  wire        clk,
    input  wire        rst,
    // one-time weight preload (passthrough to the internal gemv core)
    input  wire        gv_ld_rst,
    input  wire        gv_ld_we,
    input  wire [31:0] gv_ld_data,
    // one-time bias preload (only consumed if HAS_BIAS; MROWS rows, Q6.25, added post-dequant)
    input  wire            b_we,
    input  wire [P*32-1:0] b_data,
    // one-time input-tensor preload (XTROWS rows, INT8, t-major/channel-minor)
    input  wire           xt_we,
    input  wire [P*8-1:0] xt_data,
    // single-shift dequant, runtime (gdequant-style: raw_int32 >>> dq_shift, or <<< if negative)
    input  wire signed [7:0] dq_shift,
    // run: one pulse computes ALL TOUT output positions
    input  wire  go,
    output reg   done,
    output reg          y_valid,
    output reg [P*32-1:0] y_data
);
    localparam integer TOUT   = (TIN - KW) / STRIDE + 1;
    localparam integer CGRP   = CIN / P;          // input row-groups per timestep
    localparam integer MROWS  = COUT / P;         // output rows per timestep
    localparam integer KMAX   = KW * CIN;
    localparam integer MMAX   = COUT;
    localparam integer XTROWS = TIN * CGRP;
    localparam integer KROWS  = KMAX / P;         // feed cycles per output position

    // ---- input tensor storage: one wide row per (t,cg), t-major/channel-minor
    (* ram_style = "distributed" *) reg [P*8-1:0] xtbank [0:XTROWS-1];
    reg [$clog2(XTROWS+1)-1:0] xt_wptr;
    always @(posedge clk) begin
        if (rst) xt_wptr <= 0;
        else if (xt_we) begin xtbank[xt_wptr] <= xt_data; xt_wptr <= xt_wptr + 1'b1; end
    end

    // ---- bias storage (only meaningful if HAS_BIAS) ---------------------------
    (* ram_style = "distributed" *) reg [P*32-1:0] b_bank [0:(MROWS>0?MROWS-1:0)];
    reg [$clog2(MROWS+1)-1:0] b_wptr;
    always @(posedge clk) begin
        if (rst) b_wptr <= 0;
        else if (b_we) begin b_bank[b_wptr] <= b_data; b_wptr <= b_wptr + 1'b1; end
    end

    function automatic signed [31:0] gdequant;
        input signed [31:0] raw;
        input signed [7:0]  frac;
        begin
            gdequant = (frac >= 0) ? (raw >>> frac) : (raw <<< (-frac));
        end
    endfunction

    // ---- gemv_banked_resident_vec (real, WBW=8) --------------------------------
    reg                          gv_start;
    reg                          gv_xrst;
    wire                         gv_done;
    reg  [$clog2(MMAX+1)-1:0]    gv_m;
    reg  [$clog2(KMAX+1)-1:0]    gv_k;
    reg  [$clog2(WWORDS)-1:0]    gv_wbase;
    reg                          gv_xwe;
    reg  [P*8-1:0]               gv_xdata;
    wire [$clog2((MMAX+LANES-1)/LANES+1)-1:0] gv_gdone;
    wire [$clog2(MMAX/P + 2)-1:0] gv_rdaddr = gi;   // combinational -- see decoder_block_seq.sv's own gv_rdaddr comment
    wire [P*32-1:0]              gv_yout;
    gemv_banked_resident_vec #(.LANES(LANES), .WBW(WBW), .P(P), .MMAX(MMAX), .KMAX(KMAX),
                                .WWORDS(WWORDS), .RLAT(2), .K2(0), .MEM_PRIMITIVE("block")) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ld_rst || gv_xrst), .w_we(gv_ld_we), .w_data(gv_ld_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .gdone(gv_gdone),
        .rd_addr(gv_rdaddr[$clog2(MMAX/P)-1:0]), .y_out(gv_yout),
        .emb_sel(1'b0), .emb_addr({$clog2(WWORDS){1'b0}}), .emb_pair(),
        .wbdiag_addr({$clog2(WWORDS){1'b0}}), .wbdiag_pair()
    );

    // ---- FSM --------------------------------------------------------------------
    localparam [3:0] S_IDLE=0, S_TSTART=1, G_XRESET=2, G_XFEED=3, G_XSTART=4,
                      G_WAIT=5, G_DRAIN=6, S_TNEXT=7, S_DONE=8;
    reg [3:0] state;
    reg [$clog2(TOUT+1)-1:0]      t_out;
    reg [$clog2(XTROWS+1)-1:0]    t_base;
    reg [$clog2(KROWS+1)-1:0]     ri_g;
    reg [$clog2(MROWS+3)-1:0]     gi, gi2;
    integer biap;
    reg signed [31:0] bias_lane;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; y_valid <= 1'b0;
            t_out <= 0; ri_g <= 0; gi <= 0;
            gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0;
        end else begin
            done    <= 1'b0;
            y_valid <= 1'b0;
            gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0;
            case (state)
                S_IDLE: if (go) begin t_out <= 0; state <= S_TSTART; end
                S_TSTART: begin
                    t_base <= t_out * STRIDE * CGRP;
                    ri_g   <= 0;
                    gv_m   <= MMAX[$clog2(MMAX+1)-1:0];
                    gv_k   <= KMAX[$clog2(KMAX+1)-1:0];
                    gv_wbase <= {$clog2(WWORDS){1'b0}};
                    state  <= G_XRESET;
                end
                G_XRESET: begin gv_xrst <= 1'b1; state <= G_XFEED; end
                G_XFEED: begin
                    gv_xdata <= xtbank[t_base + ri_g];
                    gv_xwe   <= 1'b1;
                    if (ri_g != KROWS-1) ri_g <= ri_g + 1'b1;
                    else begin ri_g <= 0; state <= G_XSTART; end
                end
                G_XSTART: begin gv_start <= 1'b1; state <= G_WAIT; end
                G_WAIT: if (gv_done) begin gi <= 0; state <= G_DRAIN; end
                G_DRAIN: begin
                    if (gi >= 2) begin
                        gi2 = gi - 2;
                        for (biap = 0; biap < P; biap = biap + 1) begin
                            bias_lane = (HAS_BIAS != 0) ? $signed(b_bank[gi2[$clog2(MROWS+1)-1:0]][biap*32 +: 32])
                                                          : 32'sd0;
                            y_data[biap*32 +: 32] <= gdequant($signed(gv_yout[biap*32 +: 32]), dq_shift)
                                                     + bias_lane;
                        end
                        y_valid <= 1'b1;
                    end
                    if (gi == MROWS+1) state <= S_TNEXT;
                    else gi <= gi + 1'b1;
                end
                S_TNEXT: begin
                    if (t_out == TOUT-1) state <= S_DONE;
                    else begin t_out <= t_out + 1'b1; state <= S_TSTART; end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule // conv1d_seq
