# OOC fit/Fmax check for sequencer_sb (split-brain: two N=8 cohorts on the TDP
# weight image). Answers: does 2x {GE FSM + dequant + gelu} FIT next to the shared
# LN/attn/embed (vs sequencer_pp16's 107.4k LUT), and at what Fmax? Run from a dir
# holding the small-table + banked-dq .mem files (e.g. the run_sb_seq sim dir).
#   vivado -mode batch -source ooc_seq_sb.tcl -tclargs <P> <LANES> <PERIOD_NS> <WWORDS> <TMAX> <ND> <NC>
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set period [lindex $argv 2]
set wwords [lindex $argv 3]
set tmax   [lindex $argv 4]
set nd     [lindex $argv 5]
set nc     [lindex $argv 6]
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$period eq ""} { set period 6.0 }
if {$wwords eq ""} { set wwords 25600 }
if {$tmax   eq ""} { set tmax 32 }
# ND = DSP-packed GEMM streams PER COHORT (6 -> 12 of 16 total, the SS18 config)
if {$nd     eq ""} { set nd 6 }
# NC = streams per cohort (8 -> N=16; 7 -> N=14, the -2-LUT-bank fit variant)
if {$nc     eq ""} { set nc 8 }
set nn [expr {2 * $nc}]
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv [list \
    "$rtl/sequencer_sb.sv" \
    "$rtl/cohort_engine.sv" \
    "$rtl/nl_engine.sv" \
    "$rtl/layernorm_vec.sv" \
    "$rtl/vec_dequant.sv" \
    "$rtl/vec_attn.sv" \
    "$rtl/vec_gelu.sv" \
    "$rtl/gelu_lut.sv" \
    "$rtl/gelu_lut2.sv" \
    "$rtl/softmax.sv" \
    "$rtl/weight_bank_tdp.sv" \
    "$rtl/gemm_cohort_vec.sv" \
    "$rtl/gemm_banked_resident_vec.sv"]

synth_design -top sequencer_sb -part $part -mode out_of_context \
    -generic P=$pp -generic LANES=$lanes -generic N=$nn -generic NC=$nc \
    -generic ND=$nd -generic NLAYER=4 \
    -generic WWORDS=$wwords -generic TMAX=$tmax

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC sequencer_sb P=$pp LANES=$lanes N=$nn NC=$nc ND=$nd @ ${period}ns ====="
report_utilization
report_utilization -hierarchical -hierarchical_depth 3 -file util_hier.rpt
write_checkpoint -force post_synth_sb.dcp
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
report_timing -max_paths 30 -nworst 1 -unique_pins -file timing_paths.rpt
puts "OOC_SEQ_SB_DONE p=$pp lanes=$lanes nd=$nd period=$period"
