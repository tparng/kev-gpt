# Stage 2 throughput — PS+PL block design for the RESIDENT GEMV engine
# (gemv_axi_resident + gemv_resident): whole model loaded once into URAM, per
# layer set W_BASE + stream activation. Build-only; impl.tcl does synth/impl/bits.
#   vivado -mode batch -source fabric/stage2/tcl/build_bd_resident.tcl
set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set bdir  "C:/kevbuild/stage2"      ;# outside OneDrive (cldflt locks files)

file mkdir $bdir
create_project gemv_pl "$bdir/gemv_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage2/rtl/gemv_resident.sv" \
    "$root/fabric/stage2/rtl/gemv_axi_resident.v"]
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] [get_bd_cells ps]

set g [create_bd_cell -type module -reference gemv_axi_resident gemv]
set_property -dict [list CONFIG.MMAX {1024} CONFIG.KMAX {1024} \
    CONFIG.WWORDS {200704} CONFIG.RLAT {2}] [get_bd_cells gemv]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/ps/M_AXI_HPM0_FPD} Slave {/gemv/S_AXI} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins /gemv/S_AXI]

assign_bd_address
regenerate_bd_layout
save_bd_design
validate_bd_design

make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all [get_files design_1.bd]
update_compile_order -fileset sources_1

puts "BD_BUILD_OK top=[get_property top [current_fileset]]"
