# Stage 3 LEAP — PS+PL block design for the RESIDENT-READ sequencer (gemv_axi_seq_fast +
# sequencer_fast + gemv_banked_resident + layernorm + softmax + gelu_lut). Same flow as
# build_bd_seq.tcl but the GEMV core is resident-read (no per-matmul reload) and LANES is
# a tclarg (PE width). Build-only; impl_seq_fast.tcl does synth/impl/bitstream.
#
#   vivado -mode batch -source build_bd_seq_fast.tcl -tclargs <LANES> <WWORDS> <FREQMHZ>
#
# PREREQUISITE: the small-table .mem files must be on the synth search path in $mems.
# The PE=256 sim already wrote them to C:/kevbuild/stage3_seq_fast256 (small tables are
# lane-independent). wrom.mem is NOT used (weights load at runtime into the resident URAM).
set lanes   [lindex $argv 0]
set wwords  [lindex $argv 1]
set freq    [lindex $argv 2]
if {$lanes  eq ""} { set lanes 256 }
if {$wwords eq ""} { set wwords 16384 }
if {$freq   eq ""} { set freq 40 }

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set bdir  "C:/kevbuild/stage3_seqfast_bit"
set mems  "C:/kevbuild/stage3_seq_fast256"        ;# has the small-table .mem (lane-indep)

file mkdir $bdir
create_project gemv_seqfast_pl "$bdir/gemv_seqfast_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage3/rtl/gemv_axi_seq_fast.v" \
    "$root/fabric/stage3/rtl/sequencer_fast.sv" \
    "$root/fabric/stage3/rtl/gemv_banked_resident.sv" \
    "$root/fabric/stage3/rtl/layernorm.sv" \
    "$root/fabric/stage3/rtl/softmax.sv" \
    "$root/fabric/stage3/rtl/gelu_lut.sv"]

set memfiles [list tok_emb.mem pos_emb.mem gamma.mem dq_mant.mem dq_exp.mem \
                   inv_sact.mem prompt.mem seed.mem exp_lut.mem gelu_lut.mem]
foreach mf $memfiles {
    if {[file exists "$mems/$mf"]} {
        add_files -norecurse "$mems/$mf"
        set_property file_type {Memory Initialization Files} [get_files "$mems/$mf"]
    } else {
        puts "WARNING: missing $mems/$mf — run the PE=256 sim first to emit the small tables"
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
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq] [get_bd_cells ps]
# NOTE: the PS-side FREQMHZ is advisory for a FLAT fpgautil load (it is NOT applied) — the
# host driver forces fclk0 to this value at runtime (set_and_verify_fclk). Keep them equal.

set g [create_bd_cell -type module -reference gemv_axi_seq_fast seq]
set_property -dict [list CONFIG.NLAYER {4} CONFIG.LANES $lanes CONFIG.KVMAX {32} \
    CONFIG.PROMPT_LEN {8} CONFIG.NGEN {8} CONFIG.WWORDS $wwords \
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

puts "BD_BUILD_FAST_OK lanes=$lanes wwords=$wwords freq=$freq top=[get_property top [current_fileset]]"
