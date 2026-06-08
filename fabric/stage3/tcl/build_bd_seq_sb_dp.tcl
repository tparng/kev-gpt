# Stage 1b DOUBLE-PUMP — PS+PL block design for the split-brain sequencer with the
# clk2x MMCM. One Clocking Wizard fans pl_clk0 into clk_out1=clk (the AXI+fabric
# clock) and clk_out2=clk2x (2x, 0-deg aligned — what mac_bank_dp needs). Both come
# from the SAME MMCM so they share insertion delay (true 0-deg). The board driver
# sweeps fclk0 (the MMCM input); clk = 2*fclk0, clk2x = 4*fclk0. fclk0=100 -> 200/400.
#   vivado -mode batch -source build_bd_seq_sb_dp.tcl -tclargs <P> <LANES> <WWORDS> <FIN_MHZ> <TMAX> <ND> <BDIR> <NC>
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set wwords [lindex $argv 2]
set fin    [lindex $argv 3]
set tmax   [lindex $argv 4]
set nd     [lindex $argv 5]
set bdir   [lindex $argv 6]
set nc     [lindex $argv 7]
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$wwords eq ""} { set wwords 25600 }
if {$fin    eq ""} { set fin 200.0 }      ;# pl_clk0 = clk (fabric/AXI); clk2x = 2x
if {$tmax   eq ""} { set tmax 16 }
if {$nd     eq ""} { set nd 0 }
if {$bdir   eq ""} { set bdir "C:/kevbuild/stage3_seqsb_dp_bit" }
if {$nc     eq ""} { set nc 8 }
set nn [expr {2 * $nc}]

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set mems  "C:/kevbuild/stage3_seq_sb"

file mkdir $bdir
create_project gemv_seqsb_dp_pl "$bdir/gemv_seqsb_dp_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage3/rtl/gemv_axi_seq_sb.v" \
    "$root/fabric/stage3/rtl/sequencer_sb.sv" \
    "$root/fabric/stage3/rtl/cohort_engine.sv" \
    "$root/fabric/stage3/rtl/nl_engine.sv" \
    "$root/fabric/stage3/rtl/layernorm_vec.sv" \
    "$root/fabric/stage3/rtl/vec_dequant.sv" \
    "$root/fabric/stage3/rtl/vec_attn.sv" \
    "$root/fabric/stage3/rtl/vec_gelu.sv" \
    "$root/fabric/stage3/rtl/gelu_lut.sv" \
    "$root/fabric/stage3/rtl/gelu_lut2.sv" \
    "$root/fabric/stage3/rtl/softmax.sv" \
    "$root/fabric/stage3/rtl/weight_bank_tdp.sv" \
    "$root/fabric/stage3/rtl/embed_bank_tdp.sv" \
    "$root/fabric/stage3/rtl/gemm_cohort_vec.sv" \
    "$root/fabric/stage3/rtl/gemm_banked_resident_vec.sv" \
    "$root/fabric/stage3/rtl/mac_bank_dp.sv" \
    "$root/fabric/stage3/rtl/mac_bank_dsp_dp.sv"]

set memfiles [list gamma_w.mem dqm_w.mem dqe_w.mem \
                   inv_sact.mem seed.mem exp_lut.mem gelu_lut_e.mem gelu_lut_o.mem]
foreach mf $memfiles {
    if {[file exists "$mems/$mf"]} {
        add_files -norecurse "$mems/$mf"
        set_property file_type {Memory Initialization Files} [get_files "$mems/$mf"]
    } else { puts "WARNING: missing $mems/$mf" }
}
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} [get_bd_cells ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $fin] [get_bd_cells ps]

# ---- Clocking Wizard (MMCM): pl_clk0 (=clk) -> clk2x (2x, 0-deg aligned) --------
# The fabric/AXI clock stays pl_clk0 (exactly like the record build). The MMCM only
# makes clk2x = 2x pl_clk0, phase-aligned to its input via the feedback path -> clk2x
# is aligned to clk (both reference pl_clk0). Read pl_clk0's actual FREQ_HZ (the PS
# rounds $fin, e.g. 199.998 for 200) so the wizard input freq matches (BD 41-238).
set cw [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0]    [get_bd_pins clk_wiz/clk_in1]
# Read pl_clk0's ACTUAL freq (the PS rounds 200 -> 199.998001 MHz); the wizard input
# must match it exactly or validate_bd_design fails BD 41-238.
set in_hz  [get_property CONFIG.FREQ_HZ [get_bd_pins ps/pl_clk0]]
set in_mhz [expr {$in_hz / 1.0e6}]
set out2x  [expr {2.0 * $in_mhz}]
# No reset (free-running MMCM, locks on power-up) and no LOCKED port -> only clk_in1
# + clk_out1 pins, so the net connects can't miss a pin.
set_property -dict [list \
    CONFIG.PRIMITIVE {MMCM} \
    CONFIG.PRIM_IN_FREQ $in_mhz \
    CONFIG.CLKOUT1_USED {true} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $out2x CONFIG.CLKOUT1_REQUESTED_PHASE {0.000} \
    CONFIG.USE_LOCKED {false} CONFIG.USE_RESET {false} \
    CONFIG.NUM_OUT_CLKS {1}] [get_bd_cells clk_wiz]
puts "CLKWIZ in=${in_mhz}MHz -> clk2x=${out2x}"

set g [create_bd_cell -type module -reference gemv_axi_seq_sb seq]
set_property -dict [list CONFIG.P $pp CONFIG.LANES $lanes CONFIG.N $nn CONFIG.NC $nc \
    CONFIG.ND $nd CONFIG.NLAYER {4} CONFIG.WWORDS $wwords CONFIG.TMAX $tmax \
    CONFIG.C_S_AXI_ADDR_WIDTH {8} CONFIG.DBG {0} CONFIG.ATT2 {0} CONFIG.DP {1}] [get_bd_cells seq]

# AXI clocked by pl_clk0 (Auto) — same as the single-clock build.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/seq/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /seq/S_AXI]
connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins seq/clk2x]

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */seq/*] }
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
update_compile_order -fileset sources_1
puts "BD_BUILD_SB_DP_OK p=$pp lanes=$lanes wwords=$wwords clk=${in_mhz} clk2x=${out2x} tmax=$tmax nd=$nd n=$nn nc=$nc"
