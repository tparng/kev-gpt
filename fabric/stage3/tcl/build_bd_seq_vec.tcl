# Stage 3 10k — PS+PL block design for the P-WIDE datapath sequencer (gemv_axi_seq_vec +
# sequencer_vec + layernorm_vec + vec_dequant + vec_attn + vec_gelu + gelu_lut + softmax +
# gemv_banked_resident). Build-only; impl_seq_vec.tcl does synth/impl/bitstream.
#   vivado -mode batch -source build_bd_seq_vec.tcl -tclargs <P> <LANES> <WWORDS> <FREQMHZ>
# PREREQUISITE: the small-table + P-banked-dq .mem files in $mems (a run_vec_seq sim dir).
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set wwords [lindex $argv 2]
set freq   [lindex $argv 3]
set tmax   [lindex $argv 4]
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$wwords eq ""} { set wwords 25600 }
if {$freq   eq ""} { set freq 40 }
# TMAX=64 keeps both embed ROMs inside the 144-BRAM budget (256 needs 174 tiles).
if {$tmax   eq ""} { set tmax 64 }

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set bdir  "C:/kevbuild/stage3_seqvec_bit"
set mems  "C:/kevbuild/stage3_seq_vec_p$pp"       ;# wide-word ROMs — P-DEPENDENT (packed P/word)

file mkdir $bdir
create_project gemv_seqvec_pl "$bdir/gemv_seqvec_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage3/rtl/gemv_axi_seq_vec.v" \
    "$root/fabric/stage3/rtl/sequencer_vec.sv" \
    "$root/fabric/stage3/rtl/layernorm_vec.sv" \
    "$root/fabric/stage3/rtl/vec_dequant.sv" \
    "$root/fabric/stage3/rtl/vec_attn.sv" \
    "$root/fabric/stage3/rtl/vec_gelu.sv" \
    "$root/fabric/stage3/rtl/gelu_lut.sv" \
    "$root/fabric/stage3/rtl/softmax.sv" \
    "$root/fabric/stage3/rtl/gemv_banked_resident.sv"]

# wide-word BRAM-ROM init files (one P-packed word per line) + submodule LUTs
set memfiles [list tok_emb_w.mem pos_emb_w.mem gamma_w.mem dqm_w.mem dqe_w.mem \
                   inv_sact.mem seed.mem exp_lut.mem gelu_lut.mem]
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

set g [create_bd_cell -type module -reference gemv_axi_seq_vec seq]
set_property -dict [list CONFIG.P $pp CONFIG.LANES $lanes CONFIG.NLAYER {4} \
    CONFIG.WWORDS $wwords CONFIG.TMAX $tmax CONFIG.C_S_AXI_ADDR_WIDTH {8}] [get_bd_cells seq]

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
