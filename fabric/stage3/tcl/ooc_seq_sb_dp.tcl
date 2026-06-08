# OOC fit/timing check for the DOUBLE-PUMPED sequencer_sb (DP=1): clk2x MAC +
# 2-wide AQ + 2-wide RB. Answers: does the double-pump design FIT on the KV260,
# and what is the clk (200) / clk2x (400) timing? Run from a dir holding the
# .mem ROM-init files (a run_sb_seq sim dir, e.g. C:/kevbuild/stage3_seq_rb_full).
#   vivado -mode batch -source ooc_seq_sb_dp.tcl -tclargs <P> <LANES> <CLK_NS> <CLK2X_NS> <WWORDS> <TMAX> <ND> <NC> <ATT2>
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set clkns  [lindex $argv 2]
set clk2xns [lindex $argv 3]
set wwords [lindex $argv 4]
set tmax   [lindex $argv 5]
set nd     [lindex $argv 6]
set nc     [lindex $argv 7]
set att2   [lindex $argv 8]
if {$pp     eq ""} { set pp 8 }
if {$lanes  eq ""} { set lanes 128 }
if {$clkns  eq ""} { set clkns 5.0 }
if {$clk2xns eq ""} { set clk2xns 2.5 }
if {$wwords eq ""} { set wwords 25600 }
if {$tmax   eq ""} { set tmax 16 }
if {$nd     eq ""} { set nd 0 }
if {$nc     eq ""} { set nc 8 }
if {$att2   eq ""} { set att2 0 }
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
    "$rtl/embed_bank_tdp.sv" \
    "$rtl/gemm_cohort_vec.sv" \
    "$rtl/gemm_banked_resident_vec.sv" \
    "$rtl/mac_bank_dp.sv" \
    "$rtl/mac_bank_dsp_dp.sv"]

synth_design -top sequencer_sb -part $part -mode out_of_context \
    -generic P=$pp -generic LANES=$lanes -generic N=$nn -generic NC=$nc \
    -generic ND=$nd -generic NLAYER=4 -generic DP=1 \
    -generic WWORDS=$wwords -generic TMAX=$tmax -generic ATT2=$att2 \
    -verilog_define SYNTHESIS

create_clock -name clk   -period $clkns   [get_ports clk]
create_clock -name clk2x -period $clk2xns [get_ports clk2x]
# clk and clk2x are MMCM-related (2x); group them so cross-paths use the right edges
set_clock_groups -asynchronous -group [get_clocks clk] -group [get_clocks clk2x]

puts "===== OOC sequencer_sb DP=1 P=$pp LANES=$lanes N=$nn NC=$nc ND=$nd ATT2=$att2 clk=${clkns} clk2x=${clk2xns} ====="
report_utilization
report_utilization -hierarchical -hierarchical_depth 3 -file util_hier_dp.rpt
write_checkpoint -force post_synth_sb_dp.dcp
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
puts "OOC_SEQ_SB_DP_DONE p=$pp lanes=$lanes nd=$nd clk=$clkns clk2x=$clk2xns"
