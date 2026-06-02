# Stage 3 — synth + implement + write bitstream for the RESIDENT BANKED GEMV design.
# Run AFTER build_bd_banked.tcl succeeds. Long (~30-60 min on a laptop; the wide
# 1024-bit URAM datapath is heavier than the PE=1 resident core).
#
#   vivado -mode batch -source fabric/stage3/tcl/impl_banked.tcl
#
# Emits design_1_wrapper.bit (+ .bin for the Kria fpga_manager flow) + util/timing.
set root [file normalize [file dirname [info script]]/../../..]
set bdir "C:/kevbuild/stage3_bit"
open_project "$bdir/gemv_banked_pl/gemv_banked_pl.xpr"

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

set bit "$bdir/gemv_banked_pl/gemv_banked_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/gemv_banked.bit"
    # also emit the .bin for the Kria fpga_manager flow
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/gemv_banked" -quiet
    puts "BITSTREAM -> $bdir/gemv_banked.bit"
}
puts "IMPL_DONE WNS=$wns"
