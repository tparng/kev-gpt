# Stage 3 CAPSTONE — PS+PL block design for the TIER-3 SEQUENCER wrapped in AXI4-Lite
# (gemv_axi_seq + sequencer + gemv_banked + layernorm + softmax + gelu_lut). The whole
# single-stream autoregressive transformer runs in fabric with the CPU out of the loop:
# the host streams the INT4 weight image into a RUNTIME-LOADABLE URAM through W_DATA,
# writes the prompt over PL_*, pulses GO, polls DONE, drains the token stream over TS_*.
# Build-only; impl_seq.tcl does synth/impl/bitstream.
#
#   vivado -mode batch -source fabric/stage3/tcl/build_bd_seq.tcl
#
# PREREQUISITE — the small-table BRAM ROMs ($readmemh) need their .mem init files on the
# synth search path. Generate them once into $bdir/mems before this build:
#   python -m fabric.stage3.run_sequencer --multitoken --nlayer 4 \
#          --prompt-len 4 --ngen 6 --seed 3 --dir C:/kevbuild/stage3_seqbit/mems
# (that run writes tok_emb/pos_emb/gamma/dq_mant/dq_exp/inv_sact/prompt/seed/exp_lut/
#  gelu_lut.mem). build_bd_seq.tcl add_files them so Vivado copies them into the run dir.
# NOTE: wrom.mem is NOT used by synth — weights load at runtime into URAM, never $readmemh.
#
# URAM note: WWORDS=262144 wide words at LANES=16 (64-bit words) = 64b x 262144. URAM is
# 72b x 4096 = 4096 deep/block; 262144/4096 = 64 blocks deep x 1 wide = 64 URAM (the full
# xck26 budget). The 4-layer model uses 199936 words (49 URAM if depth-packed). If
# placement overflows URAM, lower WWORDS to 204800 (>= 199936) — still 50 blocks deep.
set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set bdir  "C:/kevbuild/stage3_seqbit"   ;# outside OneDrive (cldflt locks files)
set mems  "$bdir/mems"

file mkdir $bdir
create_project gemv_seq_pl "$bdir/gemv_seq_pl" -part $part -force
set_property board_part $board [current_project]

# core + submodules are .sv; module-ref top (the BD cell) MUST be the .v wrapper
add_files -norecurse [list \
    "$root/fabric/stage3/rtl/gemv_axi_seq.v" \
    "$root/fabric/stage3/rtl/sequencer.sv" \
    "$root/fabric/stage3/rtl/gemv_banked.sv" \
    "$root/fabric/stage3/rtl/layernorm.sv" \
    "$root/fabric/stage3/rtl/softmax.sv" \
    "$root/fabric/stage3/rtl/gelu_lut.sv"]

# BRAM-ROM init files ($readmemh of the small tables); add them so synth finds them.
set memfiles [list tok_emb.mem pos_emb.mem gamma.mem dq_mant.mem dq_exp.mem \
                   inv_sact.mem prompt.mem seed.mem exp_lut.mem gelu_lut.mem]
foreach mf $memfiles {
    if {[file exists "$mems/$mf"]} {
        add_files -norecurse "$mems/$mf"
        set_property file_type {Memory Initialization Files} [get_files "$mems/$mf"]
    } else {
        puts "WARNING: missing $mems/$mf — run the PREREQUISITE python mem-writer first"
    }
}
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] [get_bd_cells ps]

set g [create_bd_cell -type module -reference gemv_axi_seq seq]
set_property -dict [list CONFIG.NLAYER {4} CONFIG.LANES {16} CONFIG.KVMAX {32} \
    CONFIG.PROMPT_LEN {8} CONFIG.NGEN {8} CONFIG.WWORDS {262144} \
    CONFIG.C_S_AXI_ADDR_WIDTH {8}] [get_bd_cells seq]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/seq/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /seq/S_AXI]

# pin the slave at 0xA000_0000 (64K window covers the 256-byte reg map)
assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */seq/*] }
regenerate_bd_layout
save_bd_design
validate_bd_design

make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
update_compile_order -fileset sources_1

puts "BD_BUILD_OK top=[get_property top [current_fileset]]"
