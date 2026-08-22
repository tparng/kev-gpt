// -----------------------------------------------------------------------------
// xheep_kevgpt_peripheral — X-HEEP ext_peripheral register-interface shell
// around the UNMODIFIED sequencer_vec core, replacing gemv_axi_seq_vec.v's
// AXI4-Lite shell for the Genesys2 (X-HEEP + cv32e40px) port.
//
// Same register map, register semantics, and IDCODE ("SQRV" 0x53515256) as
// gemv_axi_seq_vec.v (fabric/stage3/rtl/gemv_axi_seq_vec.v) -- only the outer
// bus protocol changes. X-HEEP's reg_req_t/reg_rsp_t (reg_pkg.sv) is a flat,
// single-cycle valid/ready register interface with no separate address/data
// phases (unlike AXI4-Lite's AW/W/B/AR/R channels), so one request per cycle
// carries address + write-data + write-enable together, and this peripheral
// always completes it the same cycle (reg_rsp_o.ready == reg_req_i.valid) --
// there is no internal wait-state condition here to gate on, unlike a
// multi-cycle CSR block might need.
//
// Register map (addr[7:2], the same 6-bit word index the AXI shell used):
//  0x00 CTRL (b0 go, b1 wl_rst, b2 soft_reset, b4:3 dbg_stop)
//  0x04 STATUS (b0 done, b1 busy)
//  0x08 TOK_ID   0x0C POS    0x10 W_DATA (wl_we) 0x14 RD_SEL  0x18 RD_ADDR
//  0x1C RD_DATA_LO   0x20 RD_DATA_HI   0x24 TOK_OUT   0x28 CYCLES   0x2C IDCODE
//  0x30 SEED (write-only): nonzero => load xorshift state + enable on-chip
//       Gumbel-max sampling (persists across GOs); 0/never-written => greedy
//       argmax (bit-exact) -- identical semantics to the AXI shell.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module xheep_kevgpt_peripheral #(
    // P=16 (this instance's old, never-actually-verified default) produces
    // X-state tok_out at the full-sequencer level -- found while gating
    // Phase 5's real weight image (fabric.stage3.run_vec_kv --p 16 --lanes 128,
    // VEC_KV_VERDICT match=False, hw gen=[None]*6). Every previously-green
    // run_vec_kv result in this project (including the Option A and KV260
    // regression checks in PORT-NOTES.md) used P=8; P=16 is not exercised by
    // any gate in the repo (run_banked.sv sweeps `lanes`, not `P`) and is
    // apparently untested/broken at the sequencer_vec level, not just unlucky
    // parameterization. Not investigated further -- out of scope for this
    // phase, flagged rather than silently worked around. Defaulting to the
    // one P value this project has ever gated green.
    parameter integer P      = 8,
    // LANES=128 (inherited unmodified from the KV260 config) drove the
    // systolic GEMV/attention datapath to 252,194 LUTs (123.75% of the
    // xc7k325t's 203,800) and 804/840 DSP48E1 (95.7%) at synth alone --
    // found via a synth-only Vivado checkpoint, doesn't fit this part at
    // all. LANES=64 (gated bit-exact against Option A, avg_cyc 2498->4038
    // per pass -- a real, accepted throughput cost) is the deployed value.
    parameter integer LANES  = 64,
    // Defaults are sizing.py's Option A shape (fabric/genesys2/PORT-NOTES.md's
    // "sizing decision"): d=128, n_layer=2, n_head=2, ctx=128, vocab=57 (this
    // port's trained Kevin-speak char vocab). Was silently left at
    // sequencer_vec's own KV260-deployed defaults (D=256/NHEAD=4/VOCAB=193)
    // here -- this instance never actually overrode them even though NLAYER/
    // WWORDS/TMAX below look like they were port-tuned; caught when wiring up
    // Phase 5's real weight image. HEAD_DIM stays fixed at sequencer_vec's own
    // default (64) per that same sizing decision -- only D/NHEAD/VOCAB vary.
    parameter integer D      = 128,
    parameter integer D3     = 384,
    parameter integer D_MLP  = 512,
    parameter integer NHEAD  = 2,
    parameter integer VOCAB  = 57,
    parameter integer NLAYER = 2,
    // sequencer_vec's own default (262144 wide words = 16MB) is a generic
    // upper bound, never tightened per-model -- Option A's real footprint is
    // ~4680 wide words (74,880 32-bit W_DATA writes / SUBW=16), found to be
    // eating 256/445 RAMB36 (57.5%) at the untightened default during a
    // synth-only Vivado checkpoint. 8192 (power-of-2 headroom over the real
    // need) is the deployed value here.
    // WWORDS counts LANES*4-bit wide words, so halving LANES (above) halves
    // the bytes/word too -- doubled from 8192 to keep the same real byte
    // capacity margin over Option A's actual need.
    parameter integer WWORDS = 16384,
    parameter integer TMAX   = 128,
    // Genesys2 has no URAM at all -- see weight_bank_tdp.sv/kv_bank.sv.
    parameter               MEM_PRIMITIVE = "block",
    // DDR3-backed KV cache (fabric/genesys2/PORT-NOTES.md "Phase 2
    // architecture") -- 0 (default) keeps this peripheral's existing,
    // fully-resident build untouched. 1 threads kv_wr_*/kv_rd_* straight
    // through to sequencer_vec's own KV_DDR_BACKED=1 path; see that
    // parameter's comment in sequencer_vec.sv for the full picture.
    parameter               KV_DDR_BACKED = 0,
    parameter type reg_req_t = reg_pkg::reg_req_t,
    parameter type reg_rsp_t = reg_pkg::reg_rsp_t
) (
    input  logic     clk_i,
    input  logic     rst_ni,
    input  reg_req_t reg_req_i,
    output reg_rsp_t reg_rsp_o,

    // ---- DDR-backed KV cache DMA ports (KV_DDR_BACKED=1 only; idle-tied by
    // sequencer_vec itself when 0, see that module) -- wire straight to a
    // fabric/genesys2/rtl/kevgpt_ddr_bundle.sv instance's kv_wr_*/kv_rd_*
    // ports at the top level. ---------------------------------------------
    output logic                 kv_wr_pkt_valid,
    input  logic                 kv_wr_pkt_ready,
    output logic [28:0]          kv_wr_pkt_addr,
    output logic [255:0]         kv_wr_pkt_data,
    output logic [31:0]          kv_wr_pkt_mask,
    input  logic                 kv_wr_ack_valid,
    output logic                 kv_wr_ack_ready,
    output logic                 kv_rd_req_valid,
    input  logic                 kv_rd_req_ready,
    output logic [28:0]          kv_rd_req_addr,
    input  logic                 kv_rd_ret_valid,
    output logic                 kv_rd_ret_ready,
    input  logic [255:0]         kv_rd_ret_data
);
    wire clk = clk_i;
    wire [5:0] windex = reg_req_i.addr[7:2];

    reg          go_pulse, wl_rst, soft_reset, wl_we;
    reg [1:0]    dbg_stop;
    reg [31:0]   wl_data;
    reg [8:0]    tok_id, pos;
    reg [3:0]    rd_sel;
    reg [10:0]   rd_addr;
    reg [31:0]   seed;
    reg          seed_we;

    // ---- write side: one register-file update per accepted write request ----
    // (matches the AXI shell's pulse semantics for go/wl_we/seed_we -- each is
    // high for exactly the one cycle after the write request lands, which is
    // what sequencer_vec's start-pulse inputs expect).
    always @(posedge clk) begin
        if (!rst_ni) begin
            go_pulse<=0; wl_rst<=0; soft_reset<=0; wl_we<=0; wl_data<=0;
            tok_id<=0; pos<=0; rd_sel<=0; rd_addr<=0; dbg_stop<=0;
            seed<=0; seed_we<=0;
        end else if (reg_req_i.valid && reg_req_i.write) begin
            go_pulse<=0; wl_rst<=0; wl_we<=0; seed_we<=0;   // 1-cycle pulses default low
            case (windex)
                6'h0: begin go_pulse<=reg_req_i.wdata[0]; wl_rst<=reg_req_i.wdata[1];
                            soft_reset<=reg_req_i.wdata[2]; dbg_stop<=reg_req_i.wdata[4:3]; end
                6'h2: tok_id  <= reg_req_i.wdata[8:0];
                6'h3: pos     <= reg_req_i.wdata[8:0];
                6'h4: begin wl_we<=1; wl_data<=reg_req_i.wdata; end
                6'h5: rd_sel  <= reg_req_i.wdata[3:0];
                6'h6: rd_addr <= reg_req_i.wdata[10:0];
                6'hC: begin seed <= reg_req_i.wdata; seed_we <= 1'b1; end   // 0x30 SEED
                default: ;
            endcase
        end else begin
            go_pulse<=0; wl_rst<=0; wl_we<=0; seed_we<=0;
        end
    end

    // ---- read side: combinational mux, same word map as the AXI shell ----
    reg [31:0] rdata_mux;
    always @(*) begin
        case (windex)
            6'h1: rdata_mux = {30'b0, core_busy, done_latched};
            6'h7: rdata_mux = core_rd_data[31:0];
            6'h8: rdata_mux = core_rd_data[63:32];
            6'h9: rdata_mux = {23'b0, core_tok_out};
            6'hA: rdata_mux = cycles_latched;
            6'hB: rdata_mux = 32'h5351_5256;              // "SQRV"
            default: rdata_mux = 32'b0;
        endcase
    end

    // X-HEEP's register_interface is one-cycle decoupled; `ready` means the
    // transaction (read or write) completes this cycle -- no wait states.
    assign reg_rsp_o.ready = reg_req_i.valid;
    assign reg_rsp_o.error = 1'b0;
    assign reg_rsp_o.rdata = rdata_mux;

    // ---- the unmodified sequencer core + its run/cycle-count bookkeeping,
    //      identical to gemv_axi_seq_vec.v ----
    wire               core_done_w;
    wire [8:0]         core_tok_out;
    wire signed [63:0] core_rd_data;
    reg                core_busy, done_latched;
    reg  [31:0]        cycles_run, cycles_latched;
    wire               core_rst = ~rst_ni | soft_reset;

    always @(posedge clk) begin
        if (core_rst) begin core_busy<=0; cycles_run<=0; cycles_latched<=0; done_latched<=0; end
        else if (go_pulse) begin core_busy<=1; cycles_run<=0; done_latched<=0; end
        else if (core_busy) begin
            cycles_run<=cycles_run+1'b1;
            if (core_done_w) begin core_busy<=0; cycles_latched<=cycles_run; done_latched<=1; end
        end
    end

    sequencer_vec #(.P(P), .LANES(LANES), .D(D), .D3(D3), .D_MLP(D_MLP), .NHEAD(NHEAD),
                     .VOCAB(VOCAB), .NLAYER(NLAYER), .WWORDS(WWORDS), .TMAX(TMAX),
                     .MEM_PRIMITIVE(MEM_PRIMITIVE), .KV_DDR_BACKED(KV_DDR_BACKED)) u_seq (
        .clk(clk), .rst(core_rst), .go(go_pulse),
        .tok_id(tok_id), .pos(pos), .done(core_done_w), .tok_out(core_tok_out),
        .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(core_rd_data),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data), .dbg_stop(dbg_stop),
        .seed(seed), .seed_we(seed_we),
        .kv_wr_pkt_valid(kv_wr_pkt_valid), .kv_wr_pkt_ready(kv_wr_pkt_ready),
        .kv_wr_pkt_addr(kv_wr_pkt_addr), .kv_wr_pkt_data(kv_wr_pkt_data), .kv_wr_pkt_mask(kv_wr_pkt_mask),
        .kv_wr_ack_valid(kv_wr_ack_valid), .kv_wr_ack_ready(kv_wr_ack_ready),
        .kv_rd_req_valid(kv_rd_req_valid), .kv_rd_req_ready(kv_rd_req_ready), .kv_rd_req_addr(kv_rd_req_addr),
        .kv_rd_ret_valid(kv_rd_ret_valid), .kv_rd_ret_ready(kv_rd_ret_ready), .kv_rd_ret_data(kv_rd_ret_data)
    );
endmodule
