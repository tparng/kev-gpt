# Stage 3 LEAP — synth + implement + write bitstream for the RESIDENT-READ sequencer.
# Run AFTER build_bd_seq_fast.tcl. Emits gemv_seqfast.bit/.bin + util/timing.
#   vivado -mode batch -source fabric/stage3/tcl/impl_seq_fast.tcl
set bdir "C:/kevbuild/stage3_seqfast_bit"
open_project "$bdir/gemv_seqfast_pl/gemv_seqfast_pl.xpr"

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH_FAILED"; error "synth" }
puts "SYNTH_OK"

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "IMPL_FAILED"; error "impl" }

open_run impl_1
report_utilization    -file "$bdir/util_impl.rpt"
report_timing_summary -file "$bdir/timing_impl.rpt"
set wns [get_property SLACK [get_timing_paths -delay_type min_max -nworst 1]]

set bit "$bdir/gemv_seqfast_pl/gemv_seqfast_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/gemv_seqfast.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/gemv_seqfast" -quiet
    puts "BITSTREAM -> $bdir/gemv_seqfast.bit"
}
puts "IMPL_FAST_DONE WNS=$wns"
