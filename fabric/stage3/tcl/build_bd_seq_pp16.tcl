# Stage 3 10k — PS+PL block design for the P-WIDE datapath sequencer (gemv_axi_seq_pp16 +
# sequencer_pp + layernorm_vec + vec_dequant + vec_attn + vec_gelu + gelu_lut + softmax +
# gemv_banked_resident). Build-only; impl_seq_vec.tcl does synth/impl/bitstream.
#   vivado -mode batch -source build_bd_seq_vec.tcl -tclargs <P> <LANES> <WWORDS> <FREQMHZ>
# PREREQUISITE: the small-table + P-banked-dq .mem files in $mems (a run_vec_seq sim dir).
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set wwords [lindex $argv 2]
set freq   [lindex $argv 3]
set tmax   [lindex $argv 4]
set nn     [lindex $argv 5]
set bdir   [lindex $argv 6]                       ;# 7th arg: build dir (was hardcoded
                                                  ;# and silently clobbered old builds)
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$wwords eq ""} { set wwords 25600 }
if {$freq   eq ""} { set freq 125 }
# TMAX=32 keeps the embed ROMs + N=4 stream scratch inside the 144-BRAM budget.
if {$tmax   eq ""} { set tmax 32 }
if {$nn     eq ""} { set nn 16 }
if {$bdir   eq ""} { set bdir "C:/kevbuild/stage3_seqpp16_bit" }
set nd 8                                          ;# DSP-packed streams (wrapper default)

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set mems  "C:/kevbuild/stage3_seq_vec_p$pp"       ;# wide-word ROMs — P-DEPENDENT (packed P/word)

file mkdir $bdir
create_project gemv_seqpp16_pl "$bdir/gemv_seqpp16_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage3/rtl/gemv_axi_seq_pp16.v" \
    "$root/fabric/stage3/rtl/sequencer_pp.sv" \
    "$root/fabric/stage3/rtl/layernorm_vec.sv" \
    "$root/fabric/stage3/rtl/vec_dequant.sv" \
    "$root/fabric/stage3/rtl/vec_attn.sv" \
    "$root/fabric/stage3/rtl/vec_gelu.sv" \
    "$root/fabric/stage3/rtl/gelu_lut.sv" \
    "$root/fabric/stage3/rtl/gelu_lut2.sv" \
    "$root/fabric/stage3/rtl/softmax.sv" \
    "$root/fabric/stage3/rtl/gemm_banked_resident_vec.sv"]

# wide-word BRAM-ROM init files (one P-packed word per line) + submodule LUTs
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
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq] [get_bd_cells ps]

set g [create_bd_cell -type module -reference gemv_axi_seq_pp16 seq]
set_property -dict [list CONFIG.P $pp CONFIG.LANES $lanes CONFIG.N $nn CONFIG.ND $nd \
    CONFIG.NLAYER {4} CONFIG.WWORDS $wwords CONFIG.TMAX $tmax \
    CONFIG.C_S_AXI_ADDR_WIDTH {8}] [get_bd_cells seq]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/seq/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /seq/S_AXI]

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */seq/*] }
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
update_compile_order -fileset sources_1
puts "BD_BUILD_VEC_OK p=$pp lanes=$lanes wwords=$wwords freq=$freq tmax=$tmax"
