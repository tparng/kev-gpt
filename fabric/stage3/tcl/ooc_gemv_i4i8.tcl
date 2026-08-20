# OOC fit/Fmax check for the WIDE gemv_i4i8 (Rung A). Reports URAM/BRAM/DSP/LUT
# and Fmax for the standalone core at a chosen PE.
#   vivado -mode batch -source ooc_gemv_i4i8.tcl -tclargs <PE> <ROWS> <D_IN> <WMEM> <PERIOD_NS>
set pe     [lindex $argv 0]
set rows   [lindex $argv 1]
set din    [lindex $argv 2]
set wmem   [lindex $argv 3]
set period [lindex $argv 4]
if {$pe     eq ""} { set pe 16 }
if {$rows   eq ""} { set rows 1160 }
if {$din    eq ""} { set din 512 }
if {$wmem   eq ""} { set wmem 409600 }
if {$period eq ""} { set period 4.0 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv [list "$rtl/gemv_i4i8.sv"]

synth_design -top gemv_i4i8 -part $part -mode out_of_context \
    -generic PE=$pe -generic ROWS=$rows -generic D_IN=$din -generic WMEM=$wmem

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC gemv_i4i8 PE=$pe ROWS=$rows D_IN=$din WMEM=$wmem @ ${period}ns ====="
report_utilization
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
puts "OOC_GEMV_I4I8_DONE pe=$pe period=$period"
