# Out-of-context synthesis of the Stage 1 GEMV for the KV260 SOM, to get the
# real utilisation (DSP/LUT/FF/URAM) and timing (Fmax) numbers doc 2's stage 1
# is about. No board required — this is synth-only.
#
#   vivado -mode batch -source fabric/stage1/tcl/synth.tcl
#   vivado -mode batch -source fabric/stage1/tcl/synth.tcl -tclargs 256 256 16
#
# Args (optional): M K PE_LANES. Defaults 256 256 16 (the attn/proj shape).
# Reports land in fabric/stage1/synth/.

set part "xck26-sfvc784-2LV-c"
set M  256
set K  256
set PE 16
if {$argc >= 1} { set M  [lindex $argv 0] }
if {$argc >= 2} { set K  [lindex $argv 1] }
if {$argc >= 3} { set PE [lindex $argv 2] }

set root   [file normalize [file dirname [info script]]/../../..]
set outdir "$root/fabric/stage1/synth"
file mkdir $outdir

puts "synth gemv_int4  part=$part  M=$M K=$K PE_LANES=$PE"
read_verilog -sv "$root/fabric/stage1/rtl/gemv_int4.sv"

# OOC synth: weights/activation ROMs left uninitialised (WFILE/XFILE empty) so
# this measures the datapath + memory resources, not a specific weight image.
synth_design -top gemv_int4 -part $part -mode out_of_context \
    -generic M=$M -generic K=$K -generic PE_LANES=$PE

# 300 MHz target (3.333 ns). Tighten to find the real Fmax once it closes.
create_clock -name clk -period 3.333 [get_ports clk]

report_utilization      -file "$outdir/util_${M}x${K}_pe${PE}.rpt"
report_timing_summary   -file "$outdir/timing_${M}x${K}_pe${PE}.rpt"

# headline numbers to the console / log
set wns [get_property SLACK [get_timing_paths -delay_type min_max -nworst 1]]
puts "SYNTH_DONE M=$M K=$K PE=$PE WNS=$wns"
puts "reports -> $outdir"
