# Stage 3 10k — synth + implement + bitstream for the P-wide datapath sequencer.
# Run AFTER build_bd_seq_vec.tcl. Emits gemv_seqvec.bit/.bin + util/timing.
set bdir "C:/kevbuild/stage3_seqvec_bit"
open_project "$bdir/gemv_seqvec_pl/gemv_seqvec_pl.xpr"

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH_FAILED"; error "synth" }
puts "SYNTH_OK"

# the P-wide datapath is LUT/SLICEM-dense (distributed-RAM buffers); spread logic across the
# whole device + a congestion-aware router so placement doesn't choke near the URAM/DSP columns.
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting [get_runs impl_1]

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "IMPL_FAILED"; error "impl" }

open_run impl_1
report_utilization    -file "$bdir/util_impl.rpt"
report_timing_summary -file "$bdir/timing_impl.rpt"
set wns [get_property SLACK [get_timing_paths -delay_type min_max -nworst 1]]

set bit "$bdir/gemv_seqvec_pl/gemv_seqvec_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/gemv_seqvec.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/gemv_seqvec" -quiet
    puts "BITSTREAM -> $bdir/gemv_seqvec.bit"
}
puts "IMPL_VEC_DONE WNS=$wns"
