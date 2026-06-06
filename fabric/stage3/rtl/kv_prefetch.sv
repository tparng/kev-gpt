// -----------------------------------------------------------------------------
// kv_prefetch — double-buffered (ping-pong) KV window prefetch (KV-DDR-100K §2-3).
//
// Rung 3 of the KV-DDR ladder. Sits between rtl/kv_dma (the burst-read + dequant
// engine, rung 2) and the attention unit (rtl/vec_attn's load phase). Its job is
// to HIDE the DDR round-trip behind attention's compute window: while the consumer
// works through the current KV WINDOW (all NHEAD heads of one layer), the engine
// pre-reads the NEXT window from DDR through kv_dma into the OTHER of two buffers,
// so the consumer never waits on DDR once the pipeline is full.
//
// WHY WINDOW-GRANULAR PING-PONG (and not per-head bursts). The pinned goformer_kvq
// DDR layout is POSITION-ROW major: one row per (layer,K|V,position) packs ALL
// NHEAD heads head-major (kv_dma reads one such row and reconstructs all NHEAD head
// words at once). So a single head's window is NOT one contiguous burst — it is one
// 1/NHEAD slice of each of T rows. The bandwidth-efficient access is therefore to
// read each position-row ONCE and demux all NHEAD heads into per-head slots; per
// window that is exactly 2*T row reads (T for K, T for V) regardless of NHEAD. After
// a window fill ALL heads are resident, so the consumer streams head 0..NHEAD-1 with
// ZERO per-head stall, and the ping-pong hides the NEXT window's 2*T row reads behind
// the current window's NHEAD-head compute. (This matches the §1 bandwidth budget,
// which counts the whole K+V window per layer, not per head.)
//
// BUFFER SHAPE (consumer-facing, vec_attn load-phase shaped). Each ping-pong half
// holds, for a whole window (all heads), the K window then the V window as
// HEAD_DIM-wide words, indexed [kv][head][position]:
//   slot(kv,head,pos) = kv*(NHEAD*TMAX) + head*TMAX + pos
//   buf[half][slot] = that (kv,head)'s HEAD_DIM Q.16 channels at position `pos`
// A word is HEAD_DIM channels of WORDW bits each (channel d in [d*WORDW +: WORDW]) —
// the SAME wide-word banking rule kv_dma's obuf uses (CLAUDE.md: variable index is a
// memory ADDRESS, not a per-lane mux). 2*NHEAD*TMAX words/half.
//
// CONSUMER INTERFACE (addressable buffer, valid/ready load-phase shaped):
//   * pulse `start` with `tcount`=T to arm the FIRST window (cold-fills it).
//   * `win_ready` pulses when a window (all heads) is fully resident in the read
//     buffer; the consumer drives (rd_kv, rd_head, rd_pos) and reads `rd_word`, the
//     HEAD_DIM-wide Q.16 word — exactly as vec_attn loads its kmem/vmem per head.
//   * to prefetch the next window, pulse `next_win` with the next window's tcount
//     (the host re-points the addr ROM to the next layer beforehand). The engine
//     fills the OTHER half while the consumer keeps reading the current one.
//   * assert `win_consume` for one cycle when done with all heads of the current
//     window; the engine swaps buffers and surfaces the prefetched next window.
//   * `all_done` pulses on win_consume when no further window was requested.
//
// LATENCY HIDING / STALL METRIC: `stall_cnt` counts the cycles the consumer would
// wait (win_consume asserted but the next window not yet resident). The whole point
// of the brick: stalls ~ 0 once the per-window compute window exceeds the 2*T-row
// fill. The cold (first) window's fill is the only un-hidden cost (reported as the
// head-0/window-0 figure).
//
// iverilog-2012 safe: flat plain-reg buffers, no $readmemh into a 2D row, no variable
// part-select on unpacked-array elements (kv_dma's obuf_word is a wire copied to a
// plain reg before indexing).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_prefetch #(
    parameter integer NHEAD    = 4,
    parameter integer HEAD_DIM = 64,
    parameter integer KBITS    = 4,
    parameter integer TMAX     = 32,
    parameter integer WORDW    = 24,      // reconstructed Q.16 word width (matches kv_dma)
    parameter integer P        = 8,       // channels/cycle kv_dma dequantizes (emit width)
    parameter integer ADDRW    = 24,
    parameter integer LENW     = 16
) (
    input  wire                      clk,
    input  wire                      rst,

    // ---- arm / advance the window ----
    input  wire                      start,        // pulse with tcount: cold-fill window 0
    input  wire                      next_win,      // pulse with tcount: prefetch the next window
    input  wire [8:0]                tcount,        // live positions T (1..TMAX)

    // ---- per-(kv,position) DDR row base addresses (host-driven ROM port) ----
    // The engine presents (addr_kv, addr_pos); the host returns the byte base addr of
    // that position-row (for the window currently being FILLED) on `addr_in`.
    output wire                      addr_kv,       // 0=K row, 1=V row wanted
    output wire [8:0]                addr_pos,      // position index wanted
    input  wire [ADDRW-1:0]          addr_in,       // base byte addr of that row

    // ---- consumer-facing addressable read buffer (vec_attn load-phase shaped) ----
    output reg                       win_ready,     // pulses: a window (all heads) resident
    input  wire                      rd_kv,         // 0=K window, 1=V window
    input  wire [$clog2(NHEAD)-1:0]  rd_head,       // 0..NHEAD-1
    input  wire [8:0]                rd_pos,        // 0..T-1
    output wire [HEAD_DIM*WORDW-1:0] rd_word,       // HEAD_DIM Q.16 channels of (kv,head,pos)
    input  wire                      win_consume,   // pulse: done with the current window
    output reg                       all_done,      // pulses: consumed with no next window
    output reg                       busy,

    // ---- diagnostics ----
    output reg  [31:0]               stall_cnt,     // total cycles consumer waited on prefetch

    // ---- burst-read port to the DDR (latency) model — wired straight to kv_dma ----
    output wire                      req,
    output wire [ADDRW-1:0]          req_addr,
    output wire [LENW-1:0]           req_len,
    output wire                      rd_ready,
    input  wire                      rd_valid,
    input  wire [7:0]                rd_data,
    input  wire                      rd_last
);
    // ---------------------------------------------------------------- geometry
    localparam integer HALFW = 2*NHEAD*TMAX;               // words per window per half
    localparam integer AB    = $clog2(HALFW);

    function [AB-1:0] slot;
        input integer kv;
        input integer head;
        input integer pos;
        slot = kv*(NHEAD*TMAX) + head*TMAX + pos;
    endfunction

    // ---------------------------------------------------------------- kv_dma inst
    reg                       dma_start;
    reg  [ADDRW-1:0]          dma_base;
    wire                      dma_busy;
    wire                      dma_row_valid;
    reg  [$clog2(NHEAD)-1:0]  dma_rd_head;
    wire [HEAD_DIM*WORDW-1:0] dma_obuf_word;

    kv_dma #(
        .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .KBITS(KBITS), .WORDW(WORDW),
        .P(P), .ADDRW(ADDRW), .LENW(LENW)
    ) u_dma (
        .clk(clk), .rst(rst),
        .start(dma_start), .base_addr(dma_base),
        .busy(dma_busy), .row_valid(dma_row_valid),
        .rd_head(dma_rd_head), .obuf_word(dma_obuf_word),
        .req(req), .req_addr(req_addr), .req_len(req_len),
        .rd_ready(rd_ready), .rd_valid(rd_valid), .rd_data(rd_data), .rd_last(rd_last)
    );

    // ---------------------------------------------------------------- ping-pong buffers
    reg [HEAD_DIM*WORDW-1:0] buf0 [0:HALFW-1];
    reg [HEAD_DIM*WORDW-1:0] buf1 [0:HALFW-1];

    reg                      rd_half;      // half the consumer reads from (0->buf0)
    reg                      wr_half;       // half the prefetch fills

    // consumer read mux (compute the slot in a wide unsigned then index)
    wire [31:0] rd_slot_w = (rd_kv ? (NHEAD*TMAX) : 0)
                          + {30'd0, rd_head} * TMAX
                          + {23'd0, rd_pos};
    wire [AB-1:0] rd_slot = rd_slot_w[AB-1:0];
    assign rd_word = rd_half ? buf1[rd_slot] : buf0[rd_slot];

    // address ROM request driven by the prefetch FSM
    reg          ainfo_kv;
    reg  [8:0]   ainfo_pos;
    assign addr_kv  = ainfo_kv;
    assign addr_pos = ainfo_pos;

    // ---------------------------------------------------------------- fill FSM
    // For the window being filled into wr_half: read each (kv,pos) row ONCE and demux
    // all NHEAD head words into the per-head slots. Per window = 2*T row reads.
    localparam [2:0]
        F_IDLE  = 3'd0,
        F_AREQ  = 3'd1,   // present (kv,pos) to addr ROM
        F_ISSUE = 3'd2,   // issue the kv_dma row read at the captured base
        F_WAIT  = 3'd3,   // wait dma_row_valid
        F_DEMUX = 3'd4,   // walk NHEAD heads: select head, capture word, store
        F_NEXT  = 3'd5;   // advance position / kv phase
    reg [2:0] fstate;

    reg  [8:0]  fill_pos;
    reg         fill_kv;          // 0=K phase, 1=V phase
    reg  [$clog2(NHEAD+1)-1:0] dh;   // demux head index 0..NHEAD-1
    reg  [8:0]  Tfill;            // T of the window being filled
    reg         fill_pending;     // a fill has been requested into wr_half
    reg         fill_done;        // wr_half holds a complete window
    reg [HEAD_DIM*WORDW-1:0] cap_word;

    // ---------------------------------------------------------------- consumer side
    reg  [8:0]  Trd;              // T of the window currently readable
    reg         have_read_win;    // a window is resident in rd_half
    reg         next_requested;   // consumer asked for a following window
    reg         waiting_consume;  // win_consume seen but next window not ready

    always @(posedge clk) begin
        win_ready <= 1'b0;
        all_done  <= 1'b0;
        dma_start <= 1'b0;

        if (rst) begin
            fstate          <= F_IDLE;
            busy            <= 1'b0;
            stall_cnt       <= 32'd0;
            rd_half         <= 1'b0;
            wr_half         <= 1'b0;
            fill_pos        <= 9'd0;
            fill_kv         <= 1'b0;
            dh              <= 0;
            Tfill           <= 9'd0;
            fill_pending    <= 1'b0;
            fill_done       <= 1'b0;
            Trd             <= 9'd0;
            have_read_win   <= 1'b0;
            next_requested  <= 1'b0;
            waiting_consume <= 1'b0;
            ainfo_kv        <= 1'b0;
            ainfo_pos       <= 9'd0;
            dma_rd_head     <= 0;
        end else begin
            // -------------------------------------------------- arm window 0 (cold)
            if (start) begin
                busy            <= 1'b1;
                rd_half         <= 1'b0;
                wr_half         <= 1'b0;     // cold-fill window 0 into the READ half
                Tfill           <= tcount;
                fill_pending    <= 1'b1;
                fill_done       <= 1'b0;
                have_read_win   <= 1'b0;
                next_requested  <= 1'b0;
                waiting_consume <= 1'b0;
                stall_cnt       <= 32'd0;
            end

            // -------------------------------------------------- request next window
            // Consumer points addr ROM at the next layer, then pulses next_win. We
            // fill the OTHER half. (Ignored if a fill is already pending.)
            if (next_win && !fill_pending && !fill_done) begin
                wr_half        <= ~rd_half;
                Tfill          <= tcount;
                fill_pending   <= 1'b1;
                next_requested <= 1'b1;
            end else if (next_win) begin
                next_requested <= 1'b1;
            end

            // -------------------------------------------------- fill FSM
            case (fstate)
                F_IDLE: begin
                    if (fill_pending && !fill_done) begin
                        fill_pos <= 9'd0;
                        fill_kv  <= 1'b0;
                        fstate   <= F_AREQ;
                    end
                end

                F_AREQ: begin
                    ainfo_kv  <= fill_kv;
                    ainfo_pos <= fill_pos;
                    fstate    <= F_ISSUE;
                end

                F_ISSUE: begin
                    if (!dma_busy) begin
                        dma_base    <= addr_in;
                        dma_start   <= 1'b1;
                        dma_rd_head <= 0;
                        dh          <= 0;
                        fstate      <= F_WAIT;
                    end
                end

                F_WAIT: begin
                    if (dma_row_valid) begin
                        // row resident in kv_dma.obuf; demux all heads. Present head 0
                        // address now; obuf_word is combinational on dma_rd_head.
                        dma_rd_head <= 0;
                        dh          <= 0;
                        fstate      <= F_DEMUX;
                    end
                end

                // walk heads: dma_rd_head selects head dh; capture & store its word,
                // advance dma_rd_head one ahead so its obuf_word is ready next cycle.
                F_DEMUX: begin
                    cap_word = dma_obuf_word;            // head dh's word (comb mux)
                    if (wr_half) buf1[slot(fill_kv, dh, fill_pos)] <= cap_word;
                    else         buf0[slot(fill_kv, dh, fill_pos)] <= cap_word;
                    if (dh == NHEAD-1) begin
                        fstate <= F_NEXT;
                    end else begin
                        dh          <= dh + 1'b1;
                        dma_rd_head <= dh[$clog2(NHEAD)-1:0] + 1'b1;
                    end
                end

                F_NEXT: begin
                    if (fill_pos == Tfill - 9'd1) begin
                        if (fill_kv == 1'b0) begin
                            fill_kv  <= 1'b1;     // K window done -> V window
                            fill_pos <= 9'd0;
                            fstate   <= F_AREQ;
                        end else begin
                            // whole window resident
                            fill_done    <= 1'b1;
                            fill_pending <= 1'b0;
                            fstate       <= F_IDLE;
                        end
                    end else begin
                        fill_pos <= fill_pos + 9'd1;
                        fstate   <= F_AREQ;
                    end
                end

                default: fstate <= F_IDLE;
            endcase

            // -------------------------------------------------- surface a filled window
            // fill_done just went (or is) high. If the consumer has no readable window
            // yet (cold case: wr_half==rd_half), surface it directly. Otherwise the
            // filled window waits in the write half for win_consume to swap it in.
            if (fstate == F_NEXT && fill_kv == 1'b1 && fill_pos == Tfill - 9'd1) begin
                if (!have_read_win) begin
                    // cold window: filled into the read half -> directly readable.
                    // Clear fill_done (set by this same completion below) so the NEXT
                    // window's prefetch can be requested into the write half.
                    Trd           <= Tfill;
                    have_read_win <= 1'b1;
                    win_ready     <= 1'b1;
                    fill_done     <= 1'b0;
                end
            end

            // -------------------------------------------------- consume / swap
            if (win_consume) begin
                if (fill_done && next_requested) begin
                    // next window already resident in the write half -> swap, no stall
                    rd_half        <= wr_half;
                    wr_half        <= rd_half;
                    Trd            <= Tfill;
                    have_read_win  <= 1'b1;
                    win_ready      <= 1'b1;
                    fill_done      <= 1'b0;
                    next_requested <= 1'b0;
                end else if (next_requested) begin
                    // next window requested but not ready -> stall until it lands
                    waiting_consume <= 1'b1;
                end else begin
                    // no further window -> done
                    all_done      <= 1'b1;
                    busy          <= 1'b0;
                    have_read_win <= 1'b0;
                end
            end

            // resolve a stalled consume once the next window lands
            if (waiting_consume) begin
                stall_cnt <= stall_cnt + 32'd1;
                if (fill_done) begin
                    rd_half         <= wr_half;
                    wr_half         <= rd_half;
                    Trd             <= Tfill;
                    have_read_win   <= 1'b1;
                    win_ready       <= 1'b1;
                    fill_done       <= 1'b0;
                    next_requested  <= 1'b0;
                    waiting_consume <= 1'b0;
                end
            end
        end
    end
endmodule
