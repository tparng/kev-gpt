# OOC fit/Fmax check for sequencer_vec (the P-wide datapath sequencer) at a chosen P/LANES.
# Answers: does it FIT (URAM/BRAM/LUT/DSP) and at what Fmax? Run from a dir holding the
# small-table + banked-dq .mem files (e.g. a run_vec_seq sim dir).
#   vivado -mode batch -source ooc_seq_vec.tcl -tclargs <P> <LANES> <PERIOD_NS>
set pp     [lindex $argv 0]
set lanes  [lindex $argv 1]
set period [lindex $argv 2]
set wwords [lindex $argv 3]
set tmax   [lindex $argv 4]
if {$pp     eq ""} { set pp 16 }
if {$lanes  eq ""} { set lanes 128 }
if {$period eq ""} { set period 8.0 }
# TMAX=64 keeps both embed ROMs inside the 144-BRAM budget (256 needs 174 tiles).
if {$tmax   eq ""} { set tmax 64 }
# WWORDS = resident weight URAM depth (wide words). The image is ~12.6 Mbit fixed, so this
# must scale INVERSELY with LANES (word = LANES*4 bits): L=16->262144, L=128->32768, L=256->16384.
if {$wwords eq ""} { set wwords [expr {3276800 / $lanes}] }
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
    "$rtl/gemv_banked_resident_vec.sv"]

synth_design -top sequencer_vec -part $part -mode out_of_context \
    -generic P=$pp -generic LANES=$lanes -generic NLAYER=4 -generic WWORDS=$wwords \
    -generic TMAX=$tmax

create_clock -name clk -period $period [get_ports clk]

puts "===== OOC sequencer_vec P=$pp LANES=$lanes @ ${period}ns ====="
report_utilization
puts "----- timing -----"
report_timing_summary -max_paths 3 -delay_type max
# the layers underneath the worst path: worst path per unique endpoint group
report_timing -max_paths 30 -nworst 1 -unique_pins -file timing_paths.rpt
puts "OOC_SEQ_VEC_DONE p=$pp lanes=$lanes period=$period"
