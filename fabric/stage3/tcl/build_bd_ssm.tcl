# Mamba-2 bring-up — PS+PL block design for ssm_scan_axi (the scan core's first
# silicon). Build-only; impl_ssm.tcl does synth/impl/bitstream.
#   vivado -mode batch -source build_bd_ssm.tcl -tclargs <FREQMHZ> <BDIR>
# No .mem prerequisites — the core clears its own state.
set freq [lindex $argv 0]
set bdir [lindex $argv 1]
if {$freq eq ""} { set freq 100 }
if {$bdir eq ""} { set bdir "/tmp/kevbuild/ssm_bit" }

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]

file mkdir $bdir
create_project ssm_pl "$bdir/ssm_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage3/rtl/ssm_scan_axi.v" \
    "$root/fabric/stage3/rtl/ssm_scan.sv"]
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} [get_bd_cells ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq] [get_bd_cells ps]

set g [create_bd_cell -type module -reference ssm_scan_axi ssm]
set_property -dict [list CONFIG.C_S_AXI_ADDR_WIDTH {8}] [get_bd_cells ssm]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/ssm/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /ssm/S_AXI]

assign_bd_address
catch { set_property offset 0xA0000000 [get_bd_addr_segs */ssm/*] }
regenerate_bd_layout
save_bd_design
validate_bd_design
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
puts "BD_BUILD_SSM_OK freq=$freq"
