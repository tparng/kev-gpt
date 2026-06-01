# OOC synth of the transposed banked GEMV throughput core -> real Fmax/util on
# the KV260 -2LV part. This is the gating experiment for the ~100k tok/s path:
# does a wide banked array (LANES lanes fed by wide URAM words) close timing, and
# what does each lane cost in LUT/DSP/URAM?
#
#   vivado -mode batch -source synth_banked.tcl -tclargs <RTL> <LANES> <MMAX> <KMAX> <PERIOD_ns> <OUTDIR>
#
# Footprint is held ~constant (MMAX*KMAX*4 bits ~= the 1.5 MB model) so Fmax-vs-LANES
# isolates the array-width cost (x-broadcast fanout, wide read) at fixed URAM.

set RTL    [lindex $argv 0]
set LANES  [lindex $argv 1]
set MMAX   [lindex $argv 2]
set KMAX   [lindex $argv 3]
set PERIOD [lindex $argv 4]
set OUT    [lindex $argv 5]
set PART   xck26-sfvc784-2LV-c

file mkdir $OUT
read_verilog -sv $RTL
synth_design -top gemv_banked -part $PART -mode out_of_context \
    -generic LANES=$LANES -generic MMAX=$MMAX -generic KMAX=$KMAX

create_clock -name clk -period $PERIOD [get_ports clk]

report_timing_summary -max_paths 3 -file $OUT/timing_L$LANES.rpt
report_utilization                  -file $OUT/util_L$LANES.rpt

set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
set achieved [expr {$PERIOD - $wns}]
set fmax [expr {1000.0 / $achieved}]
puts [format "SYNTH_RESULT LANES=%s PERIOD=%s WNS=%.3f ACHIEVED_ns=%.3f FMAX_MHz=%.1f" \
      $LANES $PERIOD $wns $achieved $fmax]

# one-line machine-readable summary, appended across the sweep
set fh [open $OUT/summary.txt a]
puts $fh [format "LANES=%s target_period=%s WNS=%.3f Fmax_MHz=%.1f" $LANES $PERIOD $wns $fmax]
close $fh
