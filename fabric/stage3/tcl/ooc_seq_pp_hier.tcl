# Synth-only OOC of sequencer_pp + HIERARCHICAL utilization (no timing reports)
# — for chasing WHERE the LUTs are. Args as ooc_seq_pp.tcl.
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set period [lindex $argv 2]
set wwords [lindex $argv 3]
set tmax   [lindex $argv 4]
set nn     [lindex $argv 5]
set nd     [lindex $argv 6]
set gwait  [lindex $argv 7]
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$wwords eq ""} { set wwords 25600 }
if {$tmax   eq ""} { set tmax 32 }
if {$nn     eq ""} { set nn 16 }
if {$nd     eq ""} { set nd 8 }
if {$gwait  eq ""} { set gwait 120000 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv [list \
    "$rtl/sequencer_pp.sv" \
    "$rtl/nl_engine.sv" \
    "$rtl/layernorm_vec.sv" \
    "$rtl/vec_dequant.sv" \
    "$rtl/vec_attn.sv" \
    "$rtl/vec_gelu.sv" \
    "$rtl/gelu_lut.sv" \
    "$rtl/gelu_lut2.sv" \
    "$rtl/softmax.sv" \
    "$rtl/gemm_banked_resident_vec.sv"]

synth_design -top sequencer_pp -part $part -mode out_of_context \
    -generic P=$pp -generic LANES=$lanes -generic N=$nn -generic G=[expr {$nn/2}] \
    -generic ND=$nd -generic GWAIT=$gwait -generic NLAYER=4 \
    -generic WWORDS=$wwords -generic TMAX=$tmax

report_utilization -hierarchical -hierarchical_depth 4 -file util_hier.rpt
report_utilization
puts "OOC_HIER_DONE n=$nn nd=$nd"
