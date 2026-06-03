# OOC fit/Fmax check for sequencer_vec (the P-wide datapath sequencer) at a chosen P/LANES.
# Answers: does it FIT (URAM/BRAM/LUT/DSP) and at what Fmax? Run from a dir holding the
# small-table + banked-dq .mem files (e.g. a run_vec_seq sim dir).
#   vivado -mode batch -source ooc_seq_vec.tcl -tclargs <P> <LANES> <PERIOD_NS>
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set period [lindex $argv 2]
if {$pp     eq ""} { set pp 16 }
if {$lanes  eq ""} { set lanes 128 }
if {$period eq ""} { set period 8.0 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv [list \
    "$rtl/sequencer_vec.sv" \
    "$rtl/layernorm_vec.sv" \
    "$rtl/vec_dequant.sv" \
    "$rtl/vec_attn.sv" \
    "$rtl/vec_gelu.sv" \
    "$rtl/gelu_lut.sv" \
    "$rtl/softmax.sv" \
    "$rtl/gemv_banked_resident.sv"]

synth_design -top sequencer_vec -part $part -mode out_of_context \
    -generic P=$pp -generic LANES=$lanes -generic NLAYER=4

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC sequencer_vec P=$pp LANES=$lanes @ ${period}ns ====="
report_utilization
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
puts "OOC_SEQ_VEC_DONE p=$pp lanes=$lanes period=$period"
