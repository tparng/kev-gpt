# OOC fit/Fmax for the WEIGHT-STATIONARY cohort GEMV (Rung B). The weight image
# (URAM) is SHARED and O(1) in N; only the N MAC banks + N per-stream activation
# memories scale. This confirms the amortization's area cost: cycles stay flat
# (~one weight pass, measured GC_CYC=2325 for N=8) while DSP/LUT grow ~N x.
#   vivado -mode batch -source ooc_gemv_cohort.tcl -tclargs <N> <PE> <ROWS> <D_IN> <WMEM> <PERIOD_NS>
set n      [lindex $argv 0]
set pe     [lindex $argv 1]
set rows   [lindex $argv 2]
set din    [lindex $argv 3]
set wmem   [lindex $argv 4]
set period [lindex $argv 5]
if {$n      eq ""} { set n 8 }
if {$pe     eq ""} { set pe 16 }
if {$rows   eq ""} { set rows 1160 }
if {$din    eq ""} { set din 512 }
if {$wmem   eq ""} { set wmem 409600 }
if {$period eq ""} { set period 4.0 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv [list "$rtl/gemv_i4i8_cohort.sv"]

synth_design -top gemv_i4i8_cohort -part $part -mode out_of_context \
    -generic N=$n -generic PE=$pe -generic ROWS=$rows -generic D_IN=$din \
    -generic WMEM=$wmem

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC gemv_i4i8_cohort N=$n PE=$pe ROWS=$rows D_IN=$din WMEM=$wmem @ ${period}ns ====="
report_utilization
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
puts "OOC_GEMV_COHORT_DONE n=$n pe=$pe period=$period"
