// Testbench for kv_dma: drive a schedule of per-position row reads against a
// $readmemh DDR image served through ddr_latency_model (configurable latency),
// dump every reconstructed Q.16 channel word to kv.out for the bit-exact compare
// in run_kv_dma.py, and emit per-row cycle counts (req -> row_valid) to cyc.out.
//
// The schedule is NROWS reads; addr.mem holds the byte base address of each row
// (one hex word per line). One row read = one INCR burst of ROW_BYTES beats; the
// row's NHEAD * HEAD_DIM channels are read out head-major, channel-major and
// written WORDW-bit hex, with a "ROW" marker between rows (Python splits on it).
//
// Latency knobs come from +define+ (FIRST_WORD_LAT / BEAT_GAP) so the harness
// sweeps them; this is the PRD's "sim DDR latency model in the gate harness".
`timescale 1ns / 1ps
`ifndef NROWS
 `define NROWS 1
`endif
`ifndef DDR_DEPTH
 `define DDR_DEPTH 65536
`endif
`ifndef FIRST_WORD_LAT
 `define FIRST_WORD_LAT 30
`endif
`ifndef BEAT_GAP
 `define BEAT_GAP 0
`endif
`ifndef KBITS
 `define KBITS 4
`endif
`ifndef P
 `define P 8
`endif

module tb;
    localparam integer NHEAD    = 4;
    localparam integer HEAD_DIM = 64;
    localparam integer KBITS    = `KBITS;
    localparam integer WORDW    = 24;
    localparam integer P        = `P;
    localparam integer ADDRW    = 24;
    localparam integer LENW     = 16;
    localparam integer NROWS    = `NROWS;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ---- kv_dma command/readout ----
    reg                      start;
    reg  [ADDRW-1:0]         base_addr;
    wire                     busy;
    wire                     row_valid;
    reg  [$clog2(NHEAD)-1:0] rd_head;
    wire [HEAD_DIM*WORDW-1:0] obuf_word;

    // ---- burst-read wires between kv_dma (master) and ddr model (slave) ----
    wire                     req;
    wire [ADDRW-1:0]         req_addr;
    wire [LENW-1:0]          req_len;
    wire                     rd_ready;
    wire                     rd_valid;
    wire [7:0]               rd_data;
    wire                     rd_last;

    kv_dma #(
        .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .KBITS(KBITS), .WORDW(WORDW),
        .P(P), .ADDRW(ADDRW), .LENW(LENW)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .base_addr(base_addr),
        .busy(busy), .row_valid(row_valid),
        .rd_head(rd_head), .obuf_word(obuf_word),
        .req(req), .req_addr(req_addr), .req_len(req_len),
        .rd_ready(rd_ready), .rd_valid(rd_valid), .rd_data(rd_data), .rd_last(rd_last)
    );

    ddr_latency_model #(
        .DEPTH(`DDR_DEPTH), .ADDRW(ADDRW), .LENW(LENW),
        .FIRST_WORD_LAT(`FIRST_WORD_LAT), .BEAT_GAP(`BEAT_GAP),
        .MEMFILE("ddr.mem")
    ) ddr (
        .clk(clk), .rst(rst),
        .req(req), .addr(req_addr), .len(req_len),
        .ready(rd_ready), .valid(rd_valid), .data(rd_data), .last(rd_last)
    );

    // schedule of row base addresses
    reg [ADDRW-1:0] rowaddr [0:NROWS-1];

    integer r, hh, dd, f, fc, cyc;
    reg signed [WORDW-1:0] chan;

    initial begin
        $readmemh("addr.mem", rowaddr);
        f  = $fopen("kv.out",  "w");
        fc = $fopen("cyc.out", "w");

        rst = 1'b1; start = 1'b0; base_addr = 0; rd_head = 0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        for (r = 0; r < NROWS; r = r + 1) begin
            base_addr = rowaddr[r];
            start     = 1'b1;
            @(posedge clk); #1;
            start     = 1'b0;
            cyc       = 1;                 // count cycles from the start pulse
            // wait for row_valid, counting cycles
            while (!row_valid) begin
                @(posedge clk); #1;
                cyc = cyc + 1;
            end
            $fwrite(fc, "%0d\n", cyc);

            // read out every head's channels (head-major, channel-major)
            for (hh = 0; hh < NHEAD; hh = hh + 1) begin
                rd_head = hh[$clog2(NHEAD)-1:0];
                #1;                         // settle the combinational obuf read
                for (dd = 0; dd < HEAD_DIM; dd = dd + 1) begin
                    chan = obuf_word[dd*WORDW +: WORDW];
                    // emit WORDW-bit two's-complement hex (6 nibbles for 24b)
                    $fwrite(f, "%06x\n", chan & 24'hFFFFFF);
                end
            end
            $fwrite(f, "ROW\n");
            @(posedge clk); #1;             // settle before next row
        end

        $fclose(f);
        $fclose(fc);
        $display("TB_DONE NROWS=%0d FIRST_WORD_LAT=%0d BEAT_GAP=%0d",
                 NROWS, `FIRST_WORD_LAT, `BEAT_GAP);
        $finish;
    end

    // safety timeout
    initial begin
        #5000000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
