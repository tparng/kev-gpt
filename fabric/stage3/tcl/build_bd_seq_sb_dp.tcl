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
if {$fin    eq ""} { set fin 100.0 }      ;# MMCM input (pl_clk0); clk=2x, clk2x=4x
if {$tmax   eq ""} { set tmax 16 }
if {$nd     eq ""} { set nd 0 }
if {$bdir   eq ""} { set bdir "C:/kevbuild/stage3_seqsb_dp_bit" }
if {$nc     eq ""} { set nc 8 }
set nn [expr {2 * $nc}]
set clkmhz  [expr {2.0 * $fin}]           ;# clk_out1 (fabric/AXI)
set clk2mhz [expr {4.0 * $fin}]           ;# clk_out2 (clk2x)

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

# ---- Clocking Wizard (MMCM): pl_clk0 -> clk (2x) + clk2x (4x), both 0-deg ----
set cw [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz]
set_property -dict [list \
    CONFIG.PRIMITIVE {MMCM} \
    CONFIG.PRIM_IN_FREQ $fin \
    CONFIG.CLKOUT1_USED {true}  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $clkmhz  CONFIG.CLKOUT1_REQUESTED_PHASE {0.000} \
    CONFIG.CLKOUT2_USED {true}  CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $clk2mhz CONFIG.CLKOUT2_REQUESTED_PHASE {0.000} \
    CONFIG.USE_LOCKED {true} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_LOW} \
    CONFIG.NUM_OUT_CLKS {2}] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0]      [get_bd_pins clk_wiz/clk_in1]
connect_bd_net [get_bd_pins ps/pl_resetn0]   [get_bd_pins clk_wiz/resetn]

set g [create_bd_cell -type module -reference gemv_axi_seq_sb seq]
set_property -dict [list CONFIG.P $pp CONFIG.LANES $lanes CONFIG.N $nn CONFIG.NC $nc \
    CONFIG.ND $nd CONFIG.NLAYER {4} CONFIG.WWORDS $wwords CONFIG.TMAX $tmax \
    CONFIG.C_S_AXI_ADDR_WIDTH {8} CONFIG.DBG {0} CONFIG.ATT2 {0} CONFIG.DP {1}] [get_bd_cells seq]

# AXI on clk_out1 (the fabric clock); clk2x from clk_out2.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {/clk_wiz/clk_out1 (200 MHz)} Clk_slave {/clk_wiz/clk_out1 (200 MHz)} \
    Clk_xbar {/clk_wiz/clk_out1 (200 MHz)} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/seq/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /seq/S_AXI]
connect_bd_net [get_bd_pins clk_wiz/clk_out2] [get_bd_pins seq/clk2x]

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */seq/*] }
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
update_compile_order -fileset sources_1
puts "BD_BUILD_SB_DP_OK p=$pp lanes=$lanes wwords=$wwords fin=$fin clk=$clkmhz clk2x=$clk2mhz tmax=$tmax nd=$nd n=$nn nc=$nc"
