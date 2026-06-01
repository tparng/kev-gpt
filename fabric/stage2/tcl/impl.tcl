# Stage 2 — synth + implement + write bitstream for the PS+PL GEMV design.
# Run after build_bd.tcl succeeds. Long (~30-45 min on a laptop).
#
#   vivado -mode batch -source fabric/stage2/tcl/impl.tcl
#
# Weights/activation come from the BD gemv cell's WFILE/XFILE config (set in
# build_bd.tcl), so no -generic here. Emits design_1_wrapper.bit + reports.

set root [file normalize [file dirname [info script]]/../../..]
set bdir "C:/kevbuild/stage2"
open_project "$bdir/gemv_pl/gemv_pl.xpr"

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "SYNTH_FAILED"
    error "synthesis did not complete"
}
puts "SYNTH_OK"

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "IMPL_FAILED"
    error "implementation did not complete"
}

open_run impl_1
report_utilization    -file "$bdir/util_impl.rpt"
report_timing_summary -file "$bdir/timing_impl.rpt"
set wns [get_property SLACK [get_timing_paths -delay_type min_max -nworst 1]]

set bit "$bdir/gemv_pl/gemv_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/gemv.bit"
    # also emit the .bin for the Kria fpga_manager flow
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/gemv" -quiet
    puts "BITSTREAM -> $bdir/gemv.bit"
}
puts "IMPL_DONE WNS=$wns"
