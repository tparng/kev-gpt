# OOC fit/Fmax check for sequencer_fast at a chosen LANES (PE width). Answers the two
# questions a full BD build can't cheaply: does PE=256 FIT (URAM/BRAM/LUT/DSP <= device)
# and does it CLOSE at 40 MHz (the deployable clock)? Out-of-context synth only.
#   vivado -mode batch -source ooc_seq_fast.tcl -tclargs <LANES> <PERIOD_NS>
# Run from a dir that has the small-table .mem files ($readmemh of tok_emb/pos_emb/gamma/
# dq_mant/dq_exp/inv_sact/prompt) so BRAM inference is faithful; e.g. the sim out dir
# C:/kevbuild/stage3_seq_fast256.
set lanes  [lindex $argv 0]
set period [lindex $argv 1]
if {$lanes  eq ""} { set lanes 256 }
if {$period eq ""} { set period 25.0 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv [list \
    "$rtl/sequencer_fast.sv" \
    "$rtl/gemv_banked_resident.sv" \
    "$rtl/layernorm.sv" \
    "$rtl/softmax.sv" \
    "$rtl/gelu_lut.sv"]

synth_design -top sequencer_fast -part $part -mode out_of_context \
    -generic LANES=$lanes -generic NLAYER=4 -generic KVMAX=32 \
    -generic PROMPT_LEN=8 -generic NGEN=8

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC sequencer_fast LANES=$lanes @ ${period}ns ====="
report_utilization -hierarchical -hierarchical_depth 1
puts "----- primitive util -----"
report_utilization
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max -report_unconstrained
puts "OOC_SEQ_FAST_DONE lanes=$lanes period=$period"
