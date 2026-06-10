# Doc-7 "Kevin remembers" — synth + implement + bitstream for the KV-faithful sequencer.
# Run AFTER build_bd_seq_kv.tcl. Emits seq_kv.bit/.bit.bin + util/timing.
set bdir "C:/kevbuild/stage3_seqkv_bit"
open_project "$bdir/gemv_seqkv_pl/gemv_seqkv_pl.xpr"

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH_FAILED"; error "synth" }
# GATE: any $readmem the synthesiser could not open means a ROM is all-zero in the
# bitstream (the log §38 first-silicon bug). Scan every synth runme.log, fail loudly.
foreach slog [glob -nocomplain "$bdir/gemv_seqkv_pl/gemv_seqkv_pl.runs/*synth*/runme.log"] {
    set fh [open $slog r]; set txt [read $fh]; close $fh
    if {[regexp {Synth 8-4445[^\n]*} $txt line]} {
        puts "SYNTH_READMEM_FAIL $slog: $line"
        error "readmem"
    }
}
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

set bit "$bdir/gemv_seqkv_pl/gemv_seqkv_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/seq_kv.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/seq_kv" -quiet
    puts "BITSTREAM -> $bdir/seq_kv.bit"
}
puts "IMPL_KV_DONE WNS=$wns"
