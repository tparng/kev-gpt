# OOC fit/Fmax check for mamba_pipe — the pipelined NC-stream Mamba-2 engine.
# Answers: does the NC-stream wave engine fit the KV260 (the tight URAM/BRAM
# budget with per-stream scan h banked x NC) and at what Fmax? Reports the
# post-synth primitive estimate (URAM/BRAM/DSP/LUT), then attempts place+route
# for the routed WNS. Run FROM a dir that has seed.mem (rmsnorm rsqrt seed ROM).
#   vivado -mode batch -source ooc_mamba_pipe.tcl -tclargs <PERIOD_NS> <NC> <NST> <DBG>
set period [lindex $argv 0]
if {$period eq ""} { set period 8.0 }
set nc [lindex $argv 1]
if {$nc eq ""} { set nc 2 }
set nst [lindex $argv 2]
if {$nst eq ""} { set nst 64 }
set dbg [lindex $argv 3]
if {$dbg eq ""} { set dbg 0 }
set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

read_verilog -sv "$rtl/mamba_pipe.sv"
read_verilog -sv "$rtl/gemv_i4i8.sv"
read_verilog -sv "$rtl/conv_silu.sv"
read_verilog -sv "$rtl/ssm_scan_row.sv"
read_verilog -sv "$rtl/rmsnorm_gated.sv"

synth_design -top mamba_pipe -part $part -mode out_of_context \
    -generic NC=$nc -generic NST=$nst -generic DBG=$dbg \
    -generic TMAX=2 -generic T_TOKENS=2
create_clock -name clk -period $period [get_ports clk]

# ---- post-synth primitive estimate (survives even if place overflows) ----
report_utilization -file mamba_pipe_synth_util.rpt
set s_uram   [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.URAM.*}]]
set s_ramb36 [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]
set s_ramb18 [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]
set s_dsp    [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ ARITHMETIC.*}]]
set s_lutl   [llength [get_cells -hier -filter {REF_NAME =~ LUT* && SOFT_HLUTNM == ""}]]
puts "MAMBA_PIPE_SYNTH nc=$nc nst=$nst dbg=$dbg uram=$s_uram ramb36=$s_ramb36 ramb18=$s_ramb18 dsp=$s_dsp"
puts "KV260_AVAIL uram=64 ramb36=144 dsp=1248 lut=117120"

# ---- attempt place + route for routed WNS (guarded) ----
set placed 0
if {[catch {
    opt_design
    place_design
    route_design
    set placed 1
} emsg]} {
    puts "MAMBA_PIPE_PLACE_FAILED: $emsg"
}

if {$placed} {
    report_utilization -file mamba_pipe_impl_util.rpt
    report_timing_summary -file mamba_pipe_timing.rpt
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
    if {$wns >= 0} { set verdict "MET" } else { set verdict "FAILED" }
    set fmax [expr {1000.0/($period - $wns)}]
    puts "MAMBA_PIPE_OOC nc=$nc period=$period wns=$wns fmax_mhz=$fmax $verdict"
} else {
    puts "MAMBA_PIPE_OOC nc=$nc period=$period ROUTE_SKIPPED (see SYNTH util / PLACE_FAILED above)"
}
