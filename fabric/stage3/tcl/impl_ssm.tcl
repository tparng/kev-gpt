# Mamba-2 bring-up — synth + implement + bitstream for ssm_scan_axi.
# Run AFTER build_bd_ssm.tcl.
#   vivado -mode batch -source impl_ssm.tcl -tclargs <BDIR>
set bdir [lindex $argv 0]
if {$bdir eq ""} { set bdir "/tmp/kevbuild/ssm_bit" }
open_project "$bdir/ssm_pl/ssm_pl.xpr"

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

set bit "$bdir/ssm_pl/ssm_pl.runs/impl_1/design_1_wrapper.bit"
if {[file exists $bit]} {
    file copy -force $bit "$bdir/ssm_scan.bit"
    write_cfgmem -force -format BIN -interface SMAPx32 -disablebitswap \
        -loadbit "up 0x0 $bit" "$bdir/ssm_scan" -quiet
    puts "BITSTREAM -> $bdir/ssm_scan.bit (wns=$wns)"
} else { puts "NO_BITSTREAM" }
puts "IMPL_SSM_DONE wns=$wns"
