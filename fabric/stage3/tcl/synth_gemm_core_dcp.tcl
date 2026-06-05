# Bottom-up OOC synth of the GEMM core -> DCP. The core is URAM-clean standalone
# (56 URAM) but Vivado's global pass kills its weight-bank URAM whenever the
# N=8 core sits inside ANY parent (see WIDE-WORD-DATAPATH-LOG §16). Hierarchical
# DCP linking freezes the proven netlist so the parent synth cannot touch it.
#   vivado -mode batch -source synth_gemm_core_dcp.tcl -tclargs <LANES> <N> <WWORDS> <P> [core.sv]
set lanes  [lindex $argv 0]
set nn     [lindex $argv 1]
set wwords [lindex $argv 2]
set pp     [lindex $argv 3]
set rtl    [lindex $argv 4]
if {$lanes  eq ""} { set lanes 128 }
if {$nn     eq ""} { set nn 8 }
if {$wwords eq ""} { set wwords 25600 }
if {$pp     eq ""} { set pp 8 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
if {$rtl eq ""} { set rtl "$root/fabric/stage3/rtl/gemm_banked_resident_vec.sv" }

read_verilog -sv $rtl
# generics MUST match the sequencer_pp instantiation exactly: a DCP-resolved
# black box ignores parameter overrides at the instance.
synth_design -top gemm_banked_resident_vec -part $part -mode out_of_context \
    -generic LANES=$lanes -generic N=$nn -generic P=$pp -generic MMAX=1024 \
    -generic KMAX=1024 -generic WWORDS=$wwords -generic RLAT=2
report_utilization
write_checkpoint -force gemm_core_n${nn}.dcp
puts "CORE_DCP_DONE n=$nn lanes=$lanes wwords=$wwords -> gemm_core_n${nn}.dcp"
