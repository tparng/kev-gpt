set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
read_verilog -sv "$root/fabric/stage2/rtl/gemv_resident.sv"
synth_design -top gemv_resident -part $part -mode out_of_context \
    -generic WWORDS=200704 -generic RLAT=2
create_clock -name clk -period 10.000 [get_ports clk]
set u [report_utilization -return_string]
foreach line [split $u "\n"] { if {[regexp {URAM|Block RAM Tile|CLB LUTs |DSPs} $line]} { puts "UTIL $line" } }
set wns [get_property SLACK [get_timing_paths -delay_type min_max -nworst 1]]
puts "RESIDENT_SYNTH WNS=$wns (period 10ns / 100MHz, RLAT=2)"
