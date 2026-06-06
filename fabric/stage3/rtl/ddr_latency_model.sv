// -----------------------------------------------------------------------------
// ddr_latency_model — sim-only DDR read model serving a simple burst-read port.
//
// PURPOSE (5-demo-prd.md P1 / KV-DDR-100K.md §5 "Bit-honesty"): the KV-DDR path
// introduces a DDR round-trip that iverilog MUST model, "or the async/sync class
// of bug (the log's §6/§7) returns." This module is that model — a $readmemh DDR
// image served back with CONFIGURABLE latency so the kv_dma FSM's handshake is
// exercised against realistic first-word latency + per-beat cadence before any
// real AXI exists.
//
// Port (byte-wide simple burst-read; kv_dma is the master):
//   req      : master asserts to start a burst at `addr`, `len` beats (bytes).
//   addr     : byte address of the first beat (into the flat DDR image).
//   len      : number of beats (bytes) in this burst, 1..MAXLEN.
//   ready    : master can accept a beat this cycle (back-pressure).
//   valid    : a data beat is presented on `data` this cycle (and accepted iff ready).
//   data     : the byte at the running address.
//   last     : asserted with the final beat of the burst.
//
// Timing model (both knobs are parameters, swept by the gate harness):
//   FIRST_WORD_LAT : cycles from accepting `req` to the FIRST `valid` beat
//                    (models DDR open-row / read latency, e.g. ~1 / 8 / 30).
//   BEAT_GAP       : extra idle cycles BETWEEN beats within a burst (0 => 1 beat/
//                    cycle, the ideal INCR cadence). Sweepable to model contention.
//
// The model honours `ready` back-pressure: it will hold `valid`+`data` stable
// until the beat is accepted, so a master that stalls cannot drop a byte. This is
// exactly the surface that flushes ready/valid handshake bugs.
//
// Plain procedural SV, single registered read of a flat byte ROM (iverilog-2012
// safe: $readmemh into a flat 1-D reg array, plain-reg part-selects only).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module ddr_latency_model #(
    parameter integer DEPTH          = 65536,   // bytes in the DDR image
    parameter integer ADDRW          = 24,      // byte-address width
    parameter integer LENW           = 16,      // burst-length width (beats)
    parameter integer FIRST_WORD_LAT = 30,      // cycles to first beat
    parameter integer BEAT_GAP       = 0,       // idle cycles between beats
    parameter         MEMFILE        = "ddr.mem"
) (
    input  wire                 clk,
    input  wire                 rst,
    // burst-read request
    input  wire                 req,
    input  wire [ADDRW-1:0]     addr,
    input  wire [LENW-1:0]      len,
    // data channel (valid/ready)
    input  wire                 ready,
    output reg                  valid,
    output reg  [7:0]           data,
    output reg                  last
);
    // flat byte ROM (one byte per entry) — $readmemh into a 1-D array (iverilog-safe)
    reg [7:0] mem [0:DEPTH-1];
    initial $readmemh(MEMFILE, mem);

    localparam [1:0] D_IDLE = 2'd0,   // waiting for a request
                     D_WAIT = 2'd1,   // counting first-word latency
                     D_BEAT = 2'd2,   // presenting a beat (held until accepted)
                     D_GAP  = 2'd3;   // inter-beat idle (BEAT_GAP cycles)
    reg [1:0]            dstate;
    reg [ADDRW-1:0]      cur_addr;    // running byte address
    reg [LENW-1:0]       remaining;   // beats left to deliver (incl. current)
    reg [31:0]           wait_cnt;    // first-word / gap countdown

    always @(posedge clk) begin
        if (rst) begin
            dstate    <= D_IDLE;
            valid     <= 1'b0;
            last      <= 1'b0;
            data      <= 8'd0;
            cur_addr  <= {ADDRW{1'b0}};
            remaining <= {LENW{1'b0}};
            wait_cnt  <= 32'd0;
        end else begin
            case (dstate)
                // ---------------------------------------------------------
                D_IDLE: begin
                    valid <= 1'b0;
                    last  <= 1'b0;
                    if (req) begin
                        cur_addr  <= addr;
                        remaining <= len;
                        // FIRST_WORD_LAT>=1: that many idle cycles before beat 0.
                        if (FIRST_WORD_LAT <= 1) begin
                            // present beat 0 next cycle
                            data      <= mem[addr];
                            valid     <= 1'b1;
                            last      <= (len == 1);
                            cur_addr  <= addr + 1'b1;
                            remaining <= len - 1'b1;
                            dstate    <= D_BEAT;
                        end else begin
                            wait_cnt <= FIRST_WORD_LAT - 1;
                            dstate   <= D_WAIT;
                        end
                    end
                end
                // ---------------- first-word latency countdown ------------
                D_WAIT: begin
                    valid <= 1'b0;
                    if (wait_cnt <= 1) begin
                        // present beat 0
                        data      <= mem[cur_addr];
                        valid     <= 1'b1;
                        last      <= (remaining == 1);
                        cur_addr  <= cur_addr + 1'b1;
                        remaining <= remaining - 1'b1;
                        dstate    <= D_BEAT;
                    end else begin
                        wait_cnt <= wait_cnt - 1;
                    end
                end
                // ---------------- present a beat, hold until accepted -----
                D_BEAT: begin
                    // valid stays high (data/last held) until `ready` accepts it.
                    if (valid && ready) begin
                        if (last) begin
                            valid  <= 1'b0;
                            last   <= 1'b0;
                            dstate <= D_IDLE;
                        end else if (BEAT_GAP != 0) begin
                            valid    <= 1'b0;
                            wait_cnt <= BEAT_GAP;
                            dstate   <= D_GAP;
                        end else begin
                            // next beat immediately (1 beat/cycle)
                            data      <= mem[cur_addr];
                            valid     <= 1'b1;
                            last      <= (remaining == 1);
                            cur_addr  <= cur_addr + 1'b1;
                            remaining <= remaining - 1'b1;
                        end
                    end
                    // else: hold valid/data/last (back-pressure)
                end
                // ---------------- inter-beat gap --------------------------
                D_GAP: begin
                    valid <= 1'b0;
                    if (wait_cnt <= 1) begin
                        data      <= mem[cur_addr];
                        valid     <= 1'b1;
                        last      <= (remaining == 1);
                        cur_addr  <= cur_addr + 1'b1;
                        remaining <= remaining - 1'b1;
                        dstate    <= D_BEAT;
                    end else begin
                        wait_cnt <= wait_cnt - 1;
                    end
                end
                default: dstate <= D_IDLE;
            endcase
        end
    end
endmodule
