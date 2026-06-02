// Testbench for the Tier-3 sequencer. The driver writes the small-table .mem files
// (tok_emb/pos_emb/gamma/dq_*/inv_sact/prompt) which the RTL $readmemh-loads, AND
// wrom.mem (the transposed INT4 weight image). This tb STREAMS wrom.mem into the
// sequencer's runtime-loadable URAM through the wl_* LOAD PORT (NOT $readmemh init) —
// proving the synth-safe weight path. It also writes the prompt over the pl_* port.
// Then it pulses `go`, waits for `done`, and dumps:
//   xout.out      — the 256-element Q6.25 residual x_out (block-0 bit-exact gate)
//   tokout.out    — the final emitted token id (single-token full-forward gate)
//   tokstream.out — the NGEN-token greedy stream (multi-token gate)
// for the Python compare (run_sequencer.py) against seq_ref.IntSequencer.
`timescale 1ns / 1ps
`ifndef TOKID
 `define TOKID 0
`endif
`ifndef POS
 `define POS 0
`endif
`ifndef LANES
 `define LANES 16
`endif
`ifndef PNLAYER
 `define PNLAYER 1
`endif
`ifndef PPROMPT
 `define PPROMPT 1
`endif
`ifndef PNGEN
 `define PNGEN 1
`endif
`ifndef PKVMAX
 `define PKVMAX 32
`endif
// When defined, stream wrom.mem through the wl_* load port (the synth-safe path) and
// write the prompt over pl_*. (Always defined by run_sequencer.py.)
`ifndef PWLOAD
 `define PWLOAD 1
`endif

module tb;
    localparam integer D     = 256;
    localparam integer D3    = 768;
    localparam integer D_MLP = 1024;
    localparam integer VOCAB = 193;
    localparam integer LANES = `LANES;
    localparam integer NLAYER= `PNLAYER;
    localparam integer WBITS = LANES*4;
    localparam integer SUBW  = (WBITS > 32) ? (WBITS/32) : 1;     // 32-bit chunks / wide word

    // weight word count (mirrors sequencer's WROM_N), used to size the URAM + load loop
    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;
    localparam integer GW_MP   = ((D    + LANES-1)/LANES) * D_MLP;
    localparam integer GW_BLK  = GW_QKV + GW_PROJ + GW_FC + GW_MP;
    localparam integer GW_HEAD = ((VOCAB + LANES-1)/LANES) * D;
    localparam integer WROM_N  = GW_BLK*NLAYER + GW_HEAD;
    // round WROM_N up to a power of two for the URAM depth (addr slice is clean)
    localparam integer WWORDS  = 1 << ($clog2(WROM_N));

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg               rst, go;
    reg  [7:0]        tok_id;
    reg  [8:0]        pos;
    wire              busy, done;
    reg  [8:0]        rd_addr;
    wire signed [31:0] x_out;
    wire [8:0]        tok_out;
    reg  [8:0]        ts_addr;
    wire [8:0]        ts_out;
    wire [8:0]        ngen_out;
    // weight + prompt runtime load ports
    reg               wl_rst, wl_we;
    reg  [31:0]       wl_data;
    reg               pl_we;
    reg  [8:0]        pl_addr, pl_data;

    sequencer #(.LANES(LANES), .NLAYER(NLAYER),
                .PROMPT_LEN(`PPROMPT), .NGEN(`PNGEN), .KVMAX(`PKVMAX),
                .WWORDS(WWORDS)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok_id), .pos(pos),
        .busy(busy), .done(done), .rd_addr(rd_addr), .x_out(x_out), .tok_out(tok_out),
        .ts_addr(ts_addr), .ts_out(ts_out), .ngen_out(ngen_out),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data),
        .pl_we(pl_we), .pl_addr(pl_addr), .pl_data(pl_data)
    );

    integer i, f, ng, s;
    // tb-local image of the transposed weights (loaded from wrom.mem, NOT into the DUT)
    reg [WBITS-1:0] wimg [0:WROM_N-1];
    // the resident prompt ids (from prompt.mem) to push over the pl_* port
    reg [8:0] pimg [0:`PPROMPT-1];
    reg [WBITS-1:0] wword;

    initial begin
        rst = 1'b1; go = 1'b0; tok_id = `TOKID; pos = `POS; rd_addr = 0; ts_addr = 0;
        wl_rst = 1'b0; wl_we = 1'b0; wl_data = 32'b0;
        pl_we = 1'b0; pl_addr = 0; pl_data = 0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

`ifdef PWLOAD
        // ---- stream the transposed INT4 image into the URAM through the LOAD PORT ----
        $readmemh("wrom.mem", wimg);
        $readmemh("prompt.mem", pimg);
        wl_rst = 1'b1; @(posedge clk); #1; wl_rst = 1'b0;   // reset the load assembler
        for (i = 0; i < WROM_N; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1) begin          // SUBW 32-bit chunks, low first
                wl_we   = 1'b1;
                wl_data = wword[s*32 +: 32];
                @(posedge clk); #1;
            end
        end
        wl_we = 1'b0;
        // ---- write the resident prompt over the prompt load port ----
        for (i = 0; i < `PPROMPT; i = i + 1) begin
            pl_we = 1'b1; pl_addr = i[8:0]; pl_data = pimg[i];
            @(posedge clk); #1;
        end
        pl_we = 1'b0;
        @(posedge clk); #1;
`endif

        // start the (multi-token autoregressive) decode
        go = 1'b1; @(posedge clk); #1; go = 1'b0;

        // wait for done (with a generous timeout guard below)
        wait (done == 1'b1);
        @(posedge clk); #1;

        // dump x_out (Q6.25) — 2-cycle readback latency (single-token block gate)
        f = $fopen("xout.out", "w");
        for (i = 0; i < D; i = i + 1) begin
            rd_addr = i[8:0];
            @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%08x\n", x_out);
        end
        $fclose(f);

        // dump the most-recently emitted token id (single-token full-forward gate)
        f = $fopen("tokout.out", "w");
        $fwrite(f, "%0d\n", tok_out);
        $fclose(f);

        // dump the full generated token stream (multi-token gate)
        ng = ngen_out;
        f = $fopen("tokstream.out", "w");
        for (i = 0; i < ng; i = i + 1) begin
            ts_addr = i[8:0];
            @(posedge clk); @(posedge clk); #1;
            $fwrite(f, "%0d\n", ts_out);
        end
        $fclose(f);

        $display("TB_DONE tok=%0d pos=%0d tok_out=%0d ngen=%0d", `TOKID, `POS, tok_out, ng);
        $finish;
    end

    initial begin
        #3000000000;          // generous: ~49k weight words stream serially per token * NGEN
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
