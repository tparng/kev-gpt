# =============================================================================
# ooc_macdsp_dp.tcl — OOC Fmax probe for the DOUBLE-PUMPED DSP-packed MAC bank.
#
# The companion to ooc_macdp.tcl (which probed the LUT leaf at 392.9 MHz). This
# probes mac_bank_dsp_dp — the WP487 packed-pair leaf whose accumulator IS the
# DSP48E2 P-register clocked at clk2x. The whole double-pump thesis is that the
# DSP cascade (PCIN/PCOUT, dedicated routing) closes 400-500+ MHz where the
# general fabric is stuck at 200; THIS is the test of that claim for the DSP path.
#
#   vivado -mode batch -source ooc_macdsp_dp.tcl -tclargs <LANES> <ABITS> [periods...]
#
# Defaults: LANES=128 ABITS=24, sweep 2.5ns(400) / 2.0ns(500) / 3.0ns(333).
# One Vivado at a time — do NOT launch while an impl build is running.
# =============================================================================
set lanes [lindex $argv 0]
set abits [lindex $argv 1]
if {$lanes eq ""} { set lanes 128 }
if {$abits eq ""} { set abits 24 }

set periods {}
for {set i 2} {$i < [llength $argv]} {incr i} { lappend periods [lindex $argv $i] }
if {[llength $periods] == 0} { set periods {2.5 2.0 3.0} }

set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

puts "===== OOC mac_bank_dsp_dp Fmax sweep  LANES=$lanes ABITS=$abits ====="

foreach period $periods {
    puts "----- clk2x period = ${period} ns ([format %.1f [expr {1000.0/$period}]] MHz target) -----"

    read_verilog -sv [list "$rtl/mac_bank_dsp_dp.sv"]

    # use_dsp="yes" on the accumulator -> the P-register is the clk2x domain; let
    # Vivado map the packed-pair multiply to DSP48E2 as it does in the real core.
    synth_design -top mac_bank_dsp_dp -part $part -mode out_of_context \
        -generic LANES=$lanes -generic ABITS=$abits \
        -verilog_define SYNTHESIS

    # clk2x is the real timing clock (the DSP P-register accumulator domain).
    create_clock -name clk2x -period $period               [get_ports clk2x]
    create_clock -name clk   -period [expr {2.0 * $period}] [get_ports clk]

    report_utilization
    puts "----- timing (clk2x ${period} ns) -----"
    report_timing_summary -max_paths 5 -delay_type max

    set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -delay_type max]]
    set achieved [expr {1000.0 / ($period - $wns)}]
    puts [format "OOC_MACDSP_DP period=%s WNS=%.3f achieved_fmax=%.1f MHz lanes=%d" \
          $period $wns $achieved $lanes]

    close_design
}

puts "OOC_MACDSP_DP_DONE lanes=$lanes (sweep: $periods)"
puts "READ: the period where WNS first crosses >=0 is the OOC Fmax; >=400 MHz (2.5ns MET) = GO for the DSP double-pump."
