// -----------------------------------------------------------------------------
// gemv_resident — Stage 2 throughput step: the WHOLE model resident in URAM,
// loaded ONCE. Per layer the host sets w_base (byte offset of that layer's
// weights in the big URAM) + M/K, streams only the activation, and runs. No
// per-token weight re-streaming (that was ~83% of the time).
//
// URAM is efficient only wide, so weights live in a 64-bit-wide memory (8 bytes
// per word). A byte address -> (word = addr>>3, sel = addr[2:0]); the read has a
// configurable latency RLAT (cascaded URAM won't do 1-cycle at speed), and the
// MAC pipeline delays the byte/nibble/activation/valid controls by RLAT to match.
//
//   y[m] = sum_k W[m,k]*x[k]   W INT4 signed, x INT8 signed, y INT32 exact.
// Packed-byte layout = fabric/pack.py (low nibble = even k).  PE_LANES=1.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_resident #(
    parameter integer MMAX   = 1024,
    parameter integer KMAX   = 1024,
    parameter integer WWORDS = 262144,   // 64-bit words -> 2 MB URAM capacity
    parameter integer RLAT   = 2          // weight/activation read latency (cycles)
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS*8)-1:0]   w_base,    // byte offset of this layer
    // one-time load: byte stream assembled into the 64-bit URAM
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [7:0]                    w_data,
    // per-call activation
    input  wire                          x_we,
    input  wire [7:0]                    x_data,
    input  wire                          start,
    output reg                           done,
    input  wire [$clog2(MMAX)-1:0]       rd_addr,
    output reg  signed [31:0]            y_out
);
    localparam integer BAW  = $clog2(WWORDS*8);   // byte-address width
    localparam integer WAW  = $clog2(WWORDS);     // word-address width
    localparam integer XAW  = $clog2(KMAX);

    (* ram_style = "ultra" *) reg [63:0] wmem [0:WWORDS-1];
    reg signed [7:0]                     xmem [0:KMAX-1];
    reg signed [31:0]                    y    [0:MMAX-1];

    // ---- one-time load: assemble 8 incoming bytes into a 64-bit word ----------
    reg [WAW-1:0] wword;
    reg [2:0]     wsub;
    reg [63:0]    wbuf;
    reg [XAW-1:0] xptr;
    wire [63:0]   wnext = wbuf | ({56'h0, w_data} << (wsub*8));   // insert byte at wsub
    always @(posedge clk) begin
        if (ld_rst) begin wword <= 0; wsub <= 0; wbuf <= 64'h0; xptr <= 0; end
        else begin
            if (w_we) begin
                wmem[wword] <= wnext;                       // commit partial word every byte
                if (wsub == 3'd7) begin wword <= wword + 1'b1; wsub <= 0; wbuf <= 64'h0; end
                else begin wbuf <= wnext; wsub <= wsub + 1'b1; end
            end
            if (x_we) begin xmem[xptr] <= x_data; xptr <= xptr + 1'b1; end
        end
    end

    // ---- FSM + RLAT-deep read pipeline ----------------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]         state;
    reg [$clog2(MMAX):0] mrow;
    reg [XAW:0]       kc, kmac;
    reg [BAW-1:0]     row_base;          // running m*KB in bytes
    reg signed [31:0] acc;

    wire [XAW-1:0]    kb = k_count[XAW:1];
    wire [BAW-1:0]    byte_addr = w_base + row_base + kc[XAW:1];
    wire [WAW-1:0]    word_addr = byte_addr[BAW-1:3];
    wire [2:0]        bsel      = byte_addr[2:0];
    wire              issue     = (kc < k_count);

    // pipeline registers (RLAT stages)
    reg [63:0]        word_p [0:RLAT-1];
    reg signed [7:0]  x_p    [0:RLAT-1];
    reg [2:0]         bsel_p [0:RLAT-1];
    reg               kpar_p [0:RLAT-1];
    reg               v_p    [0:RLAT-1];
    integer i;

    wire [7:0]        wbyte = word_p[RLAT-1][bsel_p[RLAT-1]*8 +: 8];
    wire signed [3:0] w4 = kpar_p[RLAT-1] ? $signed(wbyte[7:4]) : $signed(wbyte[3:0]);
    wire signed [31:0] prod = w4 * x_p[RLAT-1];
    wire              mac_v = v_p[RLAT-1];

    always @(posedge clk) begin
        // read + control pipeline (shift)
        word_p[0] <= wmem[word_addr];
        x_p[0]    <= xmem[kc[XAW-1:0]];
        bsel_p[0] <= bsel;
        kpar_p[0] <= kc[0];
        v_p[0]    <= (state == RUN) && issue;
        for (i = 1; i < RLAT; i = i + 1) begin
            word_p[i] <= word_p[i-1]; x_p[i] <= x_p[i-1];
            bsel_p[i] <= bsel_p[i-1]; kpar_p[i] <= kpar_p[i-1]; v_p[i] <= v_p[i-1];
        end

        if (rst) begin
            state <= IDLE; done <= 1'b0;
            mrow <= 0; kc <= 0; kmac <= 0; acc <= 0; row_base <= 0;
            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        mrow <= 0; kc <= 0; kmac <= 0; acc <= 0; row_base <= 0;
                        for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + 1'b1;
                    if (mac_v) begin acc <= acc + prod; kmac <= kmac + 1'b1; end
                    if (kmac == k_count) begin
                        y[mrow[$clog2(MMAX)-1:0]] <= acc;
                        if (mrow == m_count - 1) state <= FIN;
                        else begin
                            mrow <= mrow + 1'b1; kc <= 0; kmac <= 0; acc <= 0;
                            row_base <= row_base + kb;
                            for (i = 0; i < RLAT; i = i + 1) v_p[i] <= 1'b0;
                        end
                    end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    always @(posedge clk) y_out <= y[rd_addr];
endmodule
