// Testbench for kv_prefetch: cold-fill window 0, then for each window (= one layer's
// full KV: all NHEAD heads, K then V) consume every head's K window then V window
// (mimicking vec_attn's per-head load phase) while the engine prefetches the NEXT
// window into the other ping-pong half. We:
//   * dump every presented HEAD_DIM-wide Q.16 channel word to kv.out (per window, per
//     head: K window then V window, channel-major) for the bit-exact compare in
//     run_kv_prefetch.py;
//   * record the stall_cnt the prefetch reports per WINDOW -> stall.out;
//   * model a per-window COMPUTE WINDOW (NHEAD*PER_HEAD_CYC cycles) during which the
//     consumer is busy and the engine overlaps the next window's DDR reads. The whole
//     point of the brick: stalls ~ 0 once the compute window > the 2*T-row fill.
//
// DDR image is the pinned goformer_kvq layout served by ddr_latency_model
// (configurable first-word latency). The addr ROM maps (layer,kv,pos) -> row base;
// the prefetch drives (addr_kv,addr_pos) for the window being FILLED, so before
// pulsing next_win the tb points `cur_fill_layer` at the next layer.
`timescale 1ns / 1ps
`ifndef TPOS
 `define TPOS 8
`endif
`ifndef NLAYERS
 `define NLAYERS 2
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
`ifndef PER_HEAD_CYC
 `define PER_HEAD_CYC 1300
`endif
`ifndef P
 `define P 8
`endif

module tb;
    localparam integer NHEAD    = 4;
    localparam integer HEAD_DIM = 64;
    localparam integer KBITS    = `KBITS;
    localparam integer TMAX     = 32;
    localparam integer WORDW    = 24;
    localparam integer P        = `P;
    localparam integer ADDRW    = 24;
    localparam integer LENW     = 16;
    localparam integer TPOS     = `TPOS;
    localparam integer NLAYERS  = `NLAYERS;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    reg                       start;
    reg                       next_win;
    reg  [8:0]                tcount;
    wire                      win_ready;
    reg                       rd_kv;
    reg  [$clog2(NHEAD)-1:0]  rd_head;
    reg  [8:0]                rd_pos;
    wire [HEAD_DIM*WORDW-1:0] rd_word;
    reg                       win_consume;
    wire                      all_done;
    wire                      busy;
    wire [31:0]               stall_cnt;

    wire                      addr_kv;
    wire [8:0]                addr_pos;
    reg  [ADDRW-1:0]          addr_in;

    wire                      req;
    wire [ADDRW-1:0]          req_addr;
    wire [LENW-1:0]           req_len;
    wire                      rd_ready;
    wire                      rd_valid;
    wire [7:0]                rd_data;
    wire                      rd_last;

    kv_prefetch #(
        .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .KBITS(KBITS), .TMAX(TMAX),
        .WORDW(WORDW), .P(P), .ADDRW(ADDRW), .LENW(LENW)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .next_win(next_win), .tcount(tcount),
        .addr_kv(addr_kv), .addr_pos(addr_pos), .addr_in(addr_in),
        .win_ready(win_ready),
        .rd_kv(rd_kv), .rd_head(rd_head), .rd_pos(rd_pos), .rd_word(rd_word),
        .win_consume(win_consume), .all_done(all_done), .busy(busy),
        .stall_cnt(stall_cnt),
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

    // addr ROM: rom[(layer*2 + kv)*TPOS + pos] = base addr. The prefetch's row reads
    // target the window being FILLED, so we serve cur_fill_layer's rows.
    reg [ADDRW-1:0] rom [0:NLAYERS*2*TPOS-1];
    integer cur_fill_layer;
    wire [31:0] rom_idx = (cur_fill_layer*2 + addr_kv)*TPOS + addr_pos;
    always @(*) addr_in = rom[rom_idx[$clog2(NLAYERS*2*TPOS)-1:0]];

    // latch win_ready so a 1-cycle pulse is never missed
    reg ready_latch, ready_ack;
    always @(posedge clk) begin
        if (rst)             ready_latch <= 1'b0;
        else if (win_ready)  ready_latch <= 1'b1;   // priority
        else if (ready_ack)  ready_latch <= 1'b0;
    end

    // measure the FILL DURATION: cycles from a next_win pulse to the fill landing
    // (dut.fill_done rising). This is the cost the per-token compute must hide.
    reg        fill_run;
    integer    fill_cyc, last_fill_cyc;
    reg        fd_prev;
    always @(posedge clk) begin
        if (rst) begin fill_run<=0; fill_cyc<=0; last_fill_cyc<=0; fd_prev<=0; end
        else begin
            fd_prev <= dut.fill_done;
            if (next_win)               begin fill_run<=1; fill_cyc<=0; end
            else if (fill_run)          fill_cyc <= fill_cyc + 1;
            if (fill_run && dut.fill_done && !fd_prev) begin
                fill_run <= 0; last_fill_cyc <= fill_cyc;
            end
        end
    end

    integer L, hh, dd, f, fc, prev_stall, this_stall, cw;
    reg signed [WORDW-1:0] chan;

    initial begin
        $readmemh("addr.mem", rom);
        f  = $fopen("kv.out",  "w");
        fc = $fopen("stall.out", "w");

        rst = 1'b1; start = 1'b0; next_win = 1'b0; tcount = 0;
        rd_kv = 0; rd_head = 0; rd_pos = 0; win_consume = 1'b0;
        cur_fill_layer = 0; ready_ack = 1'b0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        // cold-fill window 0 (layer 0)
        cur_fill_layer = 0;
        tcount = TPOS;
        start  = 1'b1;
        @(posedge clk); #1;
        start  = 1'b0;
        prev_stall = stall_cnt;

        for (L = 0; L < NLAYERS; L = L + 1) begin
            // wait for this window to be resident
            while (!ready_latch) begin @(posedge clk); #1; end
            ready_ack = 1'b1; @(posedge clk); #1; ready_ack = 1'b0;

            // as soon as the window is readable, request the NEXT window's prefetch
            if (L + 1 < NLAYERS) begin
                cur_fill_layer = L + 1;        // point addr ROM at the next layer
                tcount   = TPOS;
                next_win = 1'b1;
                @(posedge clk); #1;
                next_win = 1'b0;
            end

            // ---- consume every head: K window then V window, channel-major ----
            for (hh = 0; hh < NHEAD; hh = hh + 1) begin
                rd_head = hh[$clog2(NHEAD)-1:0];
                rd_kv = 1'b0;
                for (rd_pos = 0; rd_pos < TPOS; rd_pos = rd_pos + 1) begin
                    #1;
                    for (dd = 0; dd < HEAD_DIM; dd = dd + 1) begin
                        chan = rd_word[dd*WORDW +: WORDW];
                        $fwrite(f, "%06x\n", chan & 24'hFFFFFF);
                    end
                end
                rd_kv = 1'b1;
                for (rd_pos = 0; rd_pos < TPOS; rd_pos = rd_pos + 1) begin
                    #1;
                    for (dd = 0; dd < HEAD_DIM; dd = dd + 1) begin
                        chan = rd_word[dd*WORDW +: WORDW];
                        $fwrite(f, "%06x\n", chan & 24'hFFFFFF);
                    end
                end
                $fwrite(f, "HEAD\n");
                // model this head's compute window (the engine overlaps next-window DDR)
                for (cw = 0; cw < `PER_HEAD_CYC; cw = cw + 1) begin @(posedge clk); #1; end
            end
            $fwrite(f, "LAYER\n");

            // done with this whole window -> consume (swap to prefetched next window)
            win_consume = 1'b1;
            @(posedge clk); #1;
            win_consume = 1'b0;

            this_stall = stall_cnt - prev_stall;
            prev_stall = stall_cnt;
            $fwrite(fc, "%0d\n", this_stall);

            if (L == NLAYERS-1) begin
                while (!all_done) begin @(posedge clk); #1; end
            end
        end

        $fclose(f);
        $fclose(fc);
        $display("TB_FILLCYC %0d", last_fill_cyc);
        $display("TB_DONE NLAYERS=%0d TPOS=%0d FIRST_WORD_LAT=%0d PER_HEAD_CYC=%0d",
                 NLAYERS, TPOS, `FIRST_WORD_LAT, `PER_HEAD_CYC);
        $finish;
    end

    initial begin
        #50000000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
