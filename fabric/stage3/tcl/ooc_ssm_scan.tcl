# OOC fit/Fmax check for ssm_scan (the Mamba-2 recurrent scan core, doc 9 §4).
# Answers: does the sequential gate core fit and at what Fmax? Expected to be
# tiny (1 MAC datapath + one 4Kx16 state RAM = 2 BRAM36); this run is about
# the DSP inference and the h read-modify-write path timing, not area.
#   vivado -mode batch -source ooc_ssm_scan.tcl -tclargs <PERIOD_NS>
# Run from any scratch dir (no .mem files needed — state resets in logic).
set period [lindex $argv 0]
if {$period eq ""} { set period 4.0 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

set top [lindex $argv 1]
if {$top eq ""} { set top ssm_scan }
read_verilog -sv "$rtl/$top.sv"

synth_design -top $top -part $part -mode out_of_context
create_clock -name clk -period $period [get_ports clk]
opt_design
place_design
route_design

report_utilization -file ${top}_util.rpt
report_timing_summary -file ${top}_timing.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
if {$wns >= 0} { set verdict "MET" } else { set verdict "FAILED" }
puts "SSM_SCAN_OOC top=$top period=$period wns=$wns $verdict"
