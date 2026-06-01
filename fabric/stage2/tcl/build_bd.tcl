# Stage 2 — assemble the PS+PL block design for the "first PL silicon" GEMV demo
# on the KV260, and validate it. Build-only (no synth) for fast feedback; the
# long synth/impl/bitstream is run by impl.tcl after this passes.
#
#   vivado -mode batch -source fabric/stage2/tcl/build_bd.tcl
#
# PS (zynq_ultra_ps_e, KV260 board preset) drives our gemv_axi (AXI-Lite shell
# over gemv_int4) over M_AXI_HPM0_FPD at pl_clk0=100 MHz. Weights + one golden
# activation are baked into the core ROMs (blocks_0_proj layer).

set part  "xck26-sfvc784-2LV-c"
set board "xilinx.com:kv260_som:part0:1.4"
set root  [file normalize [file dirname [info script]]/../../..]
set bdir  "C:/kevbuild/stage2"
# build OUTSIDE OneDrive: the cloud-files filter locks/virtualises files and
# breaks Vivado's project cleanup. Sources stay in-repo; only build output moves.
set wfile [file normalize "$root/fabric/export/blocks_0_proj.w.mem"]
set xfile [file normalize "$root/fabric/export/tb/blocks_0_proj.x.mem"]
# PE_LANES: 1 = single-port memories (synthesis-clean, bit-exact on silicon).
# >1 needs the per-lane banked ROM redesign or it mis-synthesizes (zeros).
set PE 16
if {$argc >= 1} { set PE [lindex $argv 0] }

file mkdir $bdir
create_project gemv_pl "$bdir/gemv_pl" -part $part -force
set_property board_part $board [current_project]

add_files -norecurse [list \
    "$root/fabric/stage2/rtl/gemv_core.sv" \
    "$root/fabric/stage2/rtl/gemv_axi.v"]
# ROM-init images as design sources too, so $readmemh resolves them at synth.
add_files -norecurse [list $wfile $xfile]
update_compile_order -fileset sources_1

# ---- block design --------------------------------------------------------
create_bd_design design_1

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps]
# one AXI master (HPM0 FPD), one PL clock at 100 MHz, fabric reset
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
] [get_bd_cells ps]

# our core as a module-reference cell, weights+activation baked in
set g [create_bd_cell -type module -reference gemv_axi gemv]
set_property -dict [list \
    CONFIG.M {256} CONFIG.K {256} CONFIG.PE_LANES $PE \
    CONFIG.WFILE $wfile CONFIG.XFILE $xfile] [get_bd_cells gemv]

# auto-wire AXI (inserts SmartConnect + proc_sys_reset, assigns address)
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
puts "ADDR_SEGS:"
foreach s [get_bd_addr_segs -quiet] { puts "  $s" }
