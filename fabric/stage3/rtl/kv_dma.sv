// -----------------------------------------------------------------------------
// kv_dma — KV-cache burst read + dequant engine (Demo P1, KV-DDR-100K.md §2).
//
// Reads one per-position DDR row in the PINNED goformer_kvq layout through a
// simple byte-wide burst-read port (served in sim by ddr_latency_model.sv, in
// silicon by an S_AXI_HP master), parses the per-head header, unpacks the packed
// nibble codes LSB-first, and dequantizes each code to a signed Q.16 word:
//
//       x_hat = code * scale + lo            (exact integer, KV-DDR-100K §0)
//
// where `lo` is sign-extended from its stored low-16 bits and `scale` is an
// unsigned Q.16 magnitude (>=1). This is bit-true to model.goformer_kvq:
// dequant_head(codes, lo, scale), the reference k_deq_q16 / v_deq_q16 words.
//
// PINNED ROW LAYOUT (head-major within a position-row):
//   row = [ head0_hdr | head0_codes | head1_hdr | ... | head{NHEAD-1}_codes ]
//   per head:  hdr   = lo (int32, LSB-first) || scale (uint16, LSB-first)  = 6 B
//              codes = HEAD_DIM codes of KBITS each, packed LSB-first       = HEAD_DIM*KBITS/8 B
//   K4/V4: 6 + 64*4/8 = 38 B/head; NHEAD=4 -> 152 B/position.
//
// One position-row read = one INCR burst of ROW_BYTES beats. The engine streams
// the bytes in, and as each head completes it writes one WIDE WORD of HEAD_DIM
// reconstructed Q.16 channels into the output bank `obuf` (CLAUDE.md wide-word
// banking rule: the variable head index is a memory ADDRESS, not a per-lane mux —
// channel d of head h lives in bits [d*WORDW +: WORDW]).
//
// WIDE EMIT (P channels/cycle, KV-DDR-100K rung 4). The byte stream delivers
// CODES_PER_BYTE codes/byte; the row-emit throughput — not DDR latency/bandwidth —
// is the wall (rung 3 finding: 256 channel-emit cycles dominate ~420 cyc/row). So
// the engine STAGES BYTES_PER_EMIT bytes into a plain-reg code word holding P codes
// and dequantizes all P in ONE cycle through P parallel lanes, committing a P-wide
// slice of word_acc. The per-channel dequant (code*scale+lo, lo32 sign-extended,
// scale uint16) is BIT-IDENTICAL to the byte-serial version — only its parallelism
// changes. Per head emit drops from HEAD_DIM cycles to HEAD_DIM/P group cycles.
// Requires HEAD_DIM % P == 0 and P % CODES_PER_BYTE == 0 (P is a multiple of the
// codes carried by one byte, so each emit group consumes a whole number of bytes).
//
// The host drives one read per (K|V, position): assert `start` with the byte
// `base_addr` of the row; when `row_valid` pulses, the NHEAD wide words for that
// row are in `obuf[0..NHEAD-1]` and may be read out (one head-word per cycle via
// `rd_head` -> `obuf_word`). `busy` is high for the duration.
//
// iverilog-2012 safe: flat plain-reg part-selects only (variable base on a plain
// reg word_acc/emit_codes, never on an unpacked-array element); no $readmemh into a
// 2D row; the per-head reconstructed channels are assembled in a plain-reg word and
// committed as one wide bank word per head.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_dma #(
    parameter integer NHEAD    = 4,
    parameter integer HEAD_DIM = 64,
    parameter integer KBITS    = 4,        // bits per code (2/4/8; 8 % KBITS == 0)
    parameter integer WORDW    = 24,       // signed width of a reconstructed Q.16 word
                                           // (full code*scale+lo; ~+-350k needs >=20b)
    parameter integer P        = 8,        // channels dequantized/emitted per cycle
    parameter integer ADDRW    = 24,
    parameter integer LENW     = 16
) (
    input  wire                     clk,
    input  wire                     rst,
    // command
    input  wire                     start,       // pulse: read the row at base_addr
    input  wire [ADDRW-1:0]         base_addr,   // byte address of this position-row
    output reg                      busy,
    output reg                      row_valid,   // pulses 1 cycle when obuf is ready
    // read-out of the reconstructed row: present head index, get its wide word
    input  wire [$clog2(NHEAD)-1:0] rd_head,
    output wire [HEAD_DIM*WORDW-1:0] obuf_word,
    // burst-read port to the DDR (latency) model
    output reg                      req,
    output reg  [ADDRW-1:0]         req_addr,
    output reg  [LENW-1:0]          req_len,
    output reg                      rd_ready,
    input  wire                     rd_valid,
    input  wire [7:0]               rd_data,
    input  wire                     rd_last
);
    // ---- derived geometry --------------------------------------------------
    localparam integer HDR_BYTES      = 6;                         // lo int32 + scale uint16
    localparam integer CODES_PER_BYTE = 8 / KBITS;                 // 2 for K4
    localparam integer CODE_BYTES     = (HEAD_DIM * KBITS) / 8;    // 32 for K4
    localparam integer HEAD_BYTES     = HDR_BYTES + CODE_BYTES;    // 38 for K4
    localparam integer ROW_BYTES      = NHEAD * HEAD_BYTES;        // 152 for K4
    localparam [KBITS-1:0] CODE_MASK  = {KBITS{1'b1}};

    // wide-emit geometry: P codes/group, BYTES_PER_EMIT bytes feed one group.
    localparam integer BYTES_PER_EMIT = P / CODES_PER_BYTE;        // 4 for P=8,K4
    localparam integer NGROUP         = HEAD_DIM / P;              // 8 for P=8,hd=64

    // ---- output bank: one wide word per head (HEAD_DIM channels of WORDW) ---
    // Wide-word banking: channel d in bits [d*WORDW +: WORDW]; head = row address.
    reg [HEAD_DIM*WORDW-1:0] obuf [0:NHEAD-1];
    assign obuf_word = obuf[rd_head];

    // ---- per-head header / assembly registers ------------------------------
    reg [HEAD_DIM*WORDW-1:0] word_acc;   // plain-reg accumulator for current head
    reg signed [31:0]        lo_se;      // full signed lo (int32, Q.16)
    reg [15:0]               scale_u;    // unsigned scale (Q.16, >=1)
    reg [23:0]               lo_lsb;     // low 24b of lo as it is read in (bytes 0..2)
    reg [P*KBITS-1:0]        emit_codes; // staged P codes (LSB-first), BYTES_PER_EMIT bytes

    // ---- counters ----------------------------------------------------------
    reg [$clog2(NHEAD+1)-1:0]        h;        // current head 0..NHEAD-1
    reg [$clog2(NGROUP+1)-1:0]       g;        // current emit group 0..NGROUP-1
    reg [2:0]                        hdr_i;    // header byte index 0..5
    reg [$clog2(BYTES_PER_EMIT+1)-1:0] bsub;   // byte index within current group 0..BYTES_PER_EMIT-1

    // ---- FSM ---------------------------------------------------------------
    localparam [2:0] S_IDLE = 3'd0,
                     S_REQ  = 3'd1,   // assert req for the row burst
                     S_HDR  = 3'd2,   // consume 6 header bytes for head h
                     S_GETB = 3'd3,   // accept BYTES_PER_EMIT packed code bytes
                     S_EMIT = 3'd4,   // dequant P codes in one cycle, commit slice
                     S_HEND = 3'd5,   // commit head word, advance head
                     S_DONE = 3'd6;   // pulse row_valid
    reg [2:0] state;

    // combinational P-wide dequant of the staged codes in emit_codes.
    //   x_hat[l] = code[l]*scale + lo   (exact integer, signed Q.16), l in 0..P-1
    // code l lives in emit_codes[l*KBITS +: KBITS] (LSB-first, same nibble order as
    // the byte-serial version: within a byte, sub 0 is the low nibble; bytes earlier
    // in the stream are lower channels). Built per-lane so the arithmetic is the
    // SAME shape (unsigned mul, +signed int32 lo, truncate to WORDW) as before.
    wire [P*WORDW-1:0] emit_word;
    genvar gi;
    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : g_deq
            wire [KBITS-1:0]        lcode = (emit_codes >> (gi*KBITS)) & CODE_MASK;
            wire [KBITS+16-1:0]     lprod = lcode * scale_u;             // unsigned mul
            wire signed [33:0]      lfull = $signed({1'b0, lprod}) + lo_se; // + signed lo (int32)
            assign emit_word[gi*WORDW +: WORDW] = lfull[WORDW-1:0];      // exact code*scale+lo
        end
    endgenerate

    always @(posedge clk) begin
        // pulsed defaults
        row_valid <= 1'b0;

        if (rst) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            req       <= 1'b0;
            rd_ready  <= 1'b0;
            h         <= 0;
            g         <= 0;
            hdr_i     <= 3'd0;
            bsub      <= 0;
        end else begin
            case (state)
                // ----------------------------------------------------------
                S_IDLE: begin
                    busy     <= 1'b0;
                    req      <= 1'b0;
                    rd_ready <= 1'b0;
                    if (start) begin
                        req_addr <= base_addr;
                        req_len  <= ROW_BYTES[LENW-1:0];
                        req      <= 1'b1;
                        busy     <= 1'b1;
                        h        <= 0;
                        g        <= 0;
                        hdr_i    <= 3'd0;
                        bsub     <= 0;
                        state    <= S_REQ;
                    end
                end
                // ---------------- issue the burst, then accept header -------
                S_REQ: begin
                    req      <= 1'b0;          // 1-cycle req pulse
                    rd_ready <= 1'b1;          // ready to accept beats
                    hdr_i    <= 3'd0;
                    state    <= S_HDR;
                end
                // ---- 6 header bytes: lo int32 (b0..b3 LSB-first), scale16 (b4,b5) ----
                S_HDR: begin
                    rd_ready <= 1'b1;
                    if (rd_valid) begin
                        case (hdr_i)
                            3'd0: lo_lsb[7:0]   <= rd_data;
                            3'd1: lo_lsb[15:8]  <= rd_data;
                            3'd2: lo_lsb[23:16] <= rd_data;
                            3'd3: ;                       // lo high byte folded below
                            3'd4: scale_u[7:0]  <= rd_data;
                            3'd5: scale_u[15:8] <= rd_data;
                            default: ;
                        endcase
                        if (hdr_i == 3'd3) begin
                            // full signed lo = {b3, lo_lsb[23:0]} (int32, two's complement).
                            lo_se <= $signed({rd_data, lo_lsb[23:0]});
                            hdr_i <= 3'd4;
                        end else if (hdr_i == 3'd5) begin
                            g        <= 0;
                            bsub     <= 0;
                            rd_ready <= 1'b1;       // ready for the first code byte
                            state    <= S_GETB;
                        end else begin
                            hdr_i <= hdr_i + 3'd1;
                        end
                    end
                end
                // ---------------- accept BYTES_PER_EMIT packed code bytes ----
                // Stage CODES_PER_BYTE codes per accepted byte into emit_codes at
                // bsub*CODES_PER_BYTE (i.e. byte bsub fills bits [bsub*8 +: 8] of the
                // P*KBITS-wide code word). Once BYTES_PER_EMIT bytes are staged, the
                // P codes are ready for a single-cycle dequant.
                S_GETB: begin
                    rd_ready <= 1'b1;
                    if (rd_valid) begin
                        emit_codes[bsub*8 +: 8] <= rd_data;   // KBITS in {2,4,8} -> byte = whole codes
                        if (bsub == BYTES_PER_EMIT-1) begin
                            bsub     <= 0;
                            rd_ready <= 1'b0;           // stall the burst while emitting
                            state    <= S_EMIT;
                        end else begin
                            bsub     <= bsub + 1'b1;    // accept the next byte of this group
                        end
                    end
                end
                // ---------------- emit P codes from emit_codes in ONE cycle --
                // emit_word is combinational in (emit_codes, lo_se, scale_u), all
                // stable here -> commit a P-channel slice of word_acc per cycle.
                S_EMIT: begin
                    word_acc[g*P*WORDW +: P*WORDW] <= emit_word;
                    if (g == NGROUP-1) begin
                        // last group of this head -> commit the head word
                        rd_ready <= 1'b0;
                        state    <= S_HEND;
                    end else begin
                        // fetch the next BYTES_PER_EMIT bytes for the next group
                        g        <= g + 1'b1;
                        bsub     <= 0;
                        rd_ready <= 1'b1;
                        state    <= S_GETB;
                    end
                end
                // ---------------- commit head word, advance head -----------
                S_HEND: begin
                    obuf[h]  <= word_acc;
                    rd_ready <= 1'b0;
                    if (h == NHEAD-1) begin
                        state <= S_DONE;
                    end else begin
                        h     <= h + 1'b1;
                        hdr_i <= 3'd0;
                        rd_ready <= 1'b1;
                        state <= S_HDR;
                    end
                end
                // ----------------------------------------------------------
                S_DONE: begin
                    row_valid <= 1'b1;
                    busy      <= 1'b0;
                    rd_ready  <= 1'b0;
                    state     <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
