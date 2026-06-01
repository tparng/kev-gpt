# Fast OOC synth of gemv_core to confirm the weight init SURVIVES synthesis
# (the old gemv_int4 dropped it: "Net wmem does not have driver" -> pruned).
# Checks: BRAM used (weights resident), no prune, real utilisation. ~5 min.
#   vivado -mode batch -source fabric/stage2/tcl/synth_core.tcl
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set out  "C:/kevbuild/stage2/core_synth"
file mkdir $out
set wfile [file normalize "$root/fabric/export/blocks_0_proj.w.mem"]
set xfile [file normalize "$root/fabric/export/tb/blocks_0_proj.x.mem"]

read_verilog -sv "$root/fabric/stage2/rtl/gemv_core.sv"
synth_design -top gemv_core -part $part -mode out_of_context \
    -generic M=256 -generic K=256 -generic WFILE=$wfile -generic XFILE=$xfile

report_utilization -file "$out/util.rpt"
set rams [get_cells -hier -quiet -filter {REF_NAME =~ RAMB*}]
set nlut [llength [get_cells -hier -quiet -filter {REF_NAME =~ LUT*}]]
puts "SYNTH_CORE_DONE ramb_cells=[llength $rams] lut_cells=$nlut"
