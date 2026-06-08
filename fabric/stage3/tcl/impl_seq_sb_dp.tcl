# Stage 1b DOUBLE-PUMP — synth + impl + bitstream for the double-pump split-brain
# sequencer. Run AFTER build_bd_seq_sb_dp.tcl.
#   vivado -mode batch -source impl_seq_sb_dp.tcl [-tclargs <bdir>]
set bdir [lindex $argv 0]
if {$bdir eq ""} { set bdir "C:/kevbuild/stage3_seqsb_dp_bit" }
open_project "$bdir/gemv_seqsb_dp_pl/gemv_seqsb_dp_pl.xpr"

launch_runs synth_1 -jobs 12
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH_FAILED"; error "synth" }
puts "SYNTH_OK"

# Dense design (97% FF / 100% URAM): spread logic + congestion-aware routing, and
# add a post-place phys-opt so the near-full placement still routes at 200.
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting [get_runs impl_1]

launch_runs impl_1 -to_step write_bitstream -jobs 12
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "IMPL_FAILED"; error "impl" }

open_run impl_1
report_utilization    -file "$bdir/util_impl.rpt"
report_timing_summary -file "$bdir/timing_impl.rpt"
set wns [get_property SLACK [get_timing_paths -delay_type min_max -nworst 1]]

set bit "$bdir/gemv_seqsb_dp_pl/gemv_seqsb_dp_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/gemv_seqsb_dp.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/gemv_seqsb_dp" -quiet
    puts "BITSTREAM -> $bdir/gemv_seqsb_dp.bit"
}
puts "IMPL_SB_DP_DONE WNS=$wns"
