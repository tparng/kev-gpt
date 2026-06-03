// -----------------------------------------------------------------------------
// sequencer_vec — the P-WIDE vector-datapath sequencer (the road to ~10k tok/s).
// Built up PHASE BY PHASE, each gated bit-exact vs seq_ref.block0_phase_signals.
//
// MILESTONE 1 (this file, so far): embed -> LN1, producing lnout (Q.22) for one token.
//   Proves the bank-interleaved scratch scheme (element e -> bank e%P, row e/P; one row
//   feeds P banks in a cycle) AND the layernorm_vec (P-wide-I/O LN) integration end to end.
//   Gate: lnout == seq_ref.block0_phase_signals(tok)["ln1_out_q22"] (tok at pos 0).
// Later milestones add qkv-GEMV+dequant (vec_dequant), attention (vec_attn), GELU (vec_gelu),
// residuals, and scale to NLAYER blocks + head — each against its phase key.
//
// iverilog-2012 safe: copy unpacked-array elements to plain vectors before any indexed
// use; packed-vector +: part-selects only.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_vec #(
    parameter integer D     = 256,
    parameter integer P     = 8,            // vector lanes (D % P == 0)
    parameter integer VOCAB = 193,
    parameter integer TMAX  = 256,
    parameter integer GAMMA_N = 9,          // (ln1|ln2)*NLAYER + ln_f rows in gamma_rom
    parameter integer GBASE = 0             // gamma row-base for the LN under test (block0 ln1 = 0)
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    input  wire [8:0]  tok_id,
    input  wire [8:0]  pos,
    output reg         done,
    // lnout readback (Q.22, 2-cycle registered)
    input  wire [8:0]  rd_addr,
    output reg signed [63:0] ln_rd
);
    localparam integer ROWS = D / P;
    localparam integer LSH  = $clog2(P);

    // ---- small ROMs ($readmemh, registered async-read like sequencer_fast) ------
    (* rom_style = "block" *) reg signed [31:0] tok_emb   [0:VOCAB*D-1];
    (* rom_style = "block" *) reg signed [31:0] pos_emb   [0:TMAX*D-1];
    (* rom_style = "block" *) reg signed [31:0] gamma_rom [0:GAMMA_N*D-1];
    initial begin
        $readmemh("tok_emb.mem", tok_emb);
        $readmemh("pos_emb.mem", pos_emb);
        $readmemh("gamma.mem",   gamma_rom);
    end

    // ---- P-banked scratch: element e -> bank e%P, row e/P ----------------------
    reg signed [31:0] xres_bank  [0:P-1][0:ROWS-1];   // residual, Q6.25
    reg signed [31:0] gamma_bank [0:P-1][0:ROWS-1];   // gamma,    Q4.20
    reg signed [63:0] lnout_bank [0:P-1][0:ROWS-1];   // LN out,   Q.22

    // ---- layernorm_vec instance -------------------------------------------------
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    // ---- FSM --------------------------------------------------------------------
    localparam [2:0] S_IDLE=0, S_EMB=1, S_FEED=2, S_COLL=3, S_FIN=4;
    reg [2:0] st;
    reg [8:0] ci;                 // embed element counter 0..D-1
    reg [$clog2(ROWS+1)-1:0] fr, orow;
    integer pp;
    reg signed [31:0] e_tok, e_pos;
    reg signed [31:0] xb, gb;     // plain-vector bank copies for feed

    always @(posedge clk) begin
        ln_start <= 1'b0; ln_vin <= 1'b0; done <= 1'b0;
        if (rst) begin
            st <= S_IDLE; ci <= 0; fr <= 0; orow <= 0;
        end else begin
            case (st)
                S_IDLE: if (go) begin ci <= 0; st <= S_EMB; end
                // ---- embed (1/cyc into banks): x = tok_emb[tok] + pos_emb[pos] -------
                S_EMB: begin
                    e_tok = tok_emb[tok_id*D + ci];
                    e_pos = pos_emb[pos*D + ci];
                    xres_bank [ci[LSH-1:0]][ci >> LSH] <= e_tok + e_pos;
                    gamma_bank[ci[LSH-1:0]][ci >> LSH] <= gamma_rom[GBASE*D + ci];
                    if (ci == D-1) begin ci <= 0; fr <= 0; ln_start <= 1'b1; st <= S_FEED; end
                    else ci <= ci + 1'b1;
                end
                // ---- feed LN P-wide: one banked row per cycle, ROWS cycles ----------
                S_FEED: begin
                    ln_vin <= 1'b1;
                    for (pp = 0; pp < P; pp = pp + 1) begin
                        xb = xres_bank [pp][fr];
                        gb = gamma_bank[pp][fr];
                        ln_x[pp*32 +: 32] <= xb;
                        ln_g[pp*32 +: 32] <= gb;
                    end
                    if (fr == ROWS-1) begin orow <= 0; st <= S_COLL; end
                    else fr <= fr + 1'b1;
                end
                // ---- collect LN output P-wide (gated on y_valid), ROWS rows ---------
                S_COLL: begin
                    if (ln_yv) begin
                        for (pp = 0; pp < P; pp = pp + 1)
                            lnout_bank[pp][orow] <= ln_y[pp*64 +: 64];
                        if (orow == ROWS-1) st <= S_FIN;
                        else orow <= orow + 1'b1;
                    end
                end
                S_FIN: begin done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

    // ---- lnout readback: element rd_addr -> bank rd_addr%P, row rd_addr/P --------
    reg signed [63:0] ln_rd0;
    always @(posedge clk) begin
        ln_rd0 <= lnout_bank[rd_addr[LSH-1:0]][rd_addr >> LSH];
        ln_rd  <= ln_rd0;
    end
endmodule
