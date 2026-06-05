# OOC fit check for sequencer_pp with the GEMM core linked from a pre-synthesized
# DCP (bottom-up hierarchical flow). The core's .sv is NOT read: the module stays
# a black box through elaboration and links from the checkpoint, so the parent's
# global optimization pass cannot break its URAM mapping (the N=8-in-context
# killer, WIDE-WORD-DATAPATH-LOG §16). Run from a dir holding the .mem files.
#   vivado -mode batch -source ooc_seq_pp_linkdcp.tcl -tclargs <P> <LANES> <PERIOD> <WWORDS> <TMAX> <N> <core.dcp>
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set period [lindex $argv 2]
set wwords [lindex $argv 3]
set tmax   [lindex $argv 4]
set nn     [lindex $argv 5]
set dcp    [lindex $argv 6]
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$period eq ""} { set period 8.0 }
if {$wwords eq ""} { set wwords 25600 }
if {$tmax   eq ""} { set tmax 32 }
if {$nn     eq ""} { set nn 8 }
if {$dcp    eq ""} { set dcp "C:/kevbuild/dcpflow/gemm_core_n${nn}.dcp" }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_checkpoint $dcp
read_verilog -sv [list \
    "$rtl/sequencer_pp.sv" \
    "$rtl/layernorm_vec.sv" \
    "$rtl/vec_dequant.sv" \
    "$rtl/vec_attn.sv" \
    "$rtl/vec_gelu.sv" \
    "$rtl/gelu_lut.sv" \
    "$rtl/gelu_lut2.sv" \
    "$rtl/softmax.sv"]

synth_design -top sequencer_pp -part $part -mode out_of_context \
    -generic P=$pp -generic LANES=$lanes -generic N=$nn -generic NLAYER=4 \
    -generic WWORDS=$wwords -generic TMAX=$tmax

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC sequencer_pp (core via DCP) P=$pp LANES=$lanes N=$nn @ ${period}ns ====="
report_utilization
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
report_timing -max_paths 30 -nworst 1 -unique_pins -file timing_paths.rpt
puts "OOC_SEQ_PP_LINKDCP_DONE p=$pp lanes=$lanes n=$nn period=$period dcp=$dcp"
