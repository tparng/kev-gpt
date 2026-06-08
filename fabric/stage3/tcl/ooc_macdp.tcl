# =============================================================================
# ooc_macdp.tcl — OOC Fmax probe for the double-pumped MAC bank (Stage 0 Phase B).
#
# The GO/NO-GO timing signal: synth `mac_bank_dp` out-of-context at the KV260
# part with a clk2x constraint, sweep the clk2x period, and report WNS at each.
# This is the FIRST signal on whether the double-pumped MAC is in the 400 MHz
# ballpark. No Pblock yet (synth-Fmax precedes the floorplanned impl); a clean
# OOC close at 2.5 ns is the green light to spend an impl on the Pblocked island.
#
#   vivado -mode batch -source ooc_macdp.tcl -tclargs <LANES> <ABITS> [periods...]
#
# Defaults: LANES=128 ABITS=24, sweep 2.5ns(400MHz) / 2.0ns(500MHz) / 3.0ns(333MHz).
# Reports, per period, the WNS and the achieved Fmax (1/(period-WNS)). The
# period where WNS first crosses >=0 is the OOC Fmax ceiling.
#
# IMPORTANT (project rule): one Vivado at a time. Do NOT launch this while an
# impl build is running.
# =============================================================================
set lanes [lindex $argv 0]
set abits [lindex $argv 1]
if {$lanes eq ""} { set lanes 128 }
if {$abits eq ""} { set abits 24 }

# clk2x periods to sweep (ns). Override by passing extra tclargs after abits.
set periods {}
for {set i 2} {$i < [llength $argv]} {incr i} { lappend periods [lindex $argv $i] }
if {[llength $periods] == 0} { set periods {2.5 2.0 3.0} }

set part "xck26-sfvc784-2LV-c"
set root [file normalize [file dirname [info script]]/../../..]
set rtl  "$root/fabric/stage3/rtl"

puts "===== OOC mac_bank_dp Fmax sweep  LANES=$lanes ABITS=$abits ====="

foreach period $periods {
    puts "----- clk2x period = ${period} ns ([format %.1f [expr {1000.0/$period}]] MHz target) -----"

    read_verilog -sv [list "$rtl/mac_bank_dp.sv"]

    # SYNTHESIS define keeps parity with the impl flow (mac_bank_dp has no
    # SYNTHESIS-gated code today, but stay consistent with the other OOC tcls).
    synth_design -top mac_bank_dp -part $part -mode out_of_context \
        -generic LANES=$lanes -generic ABITS=$abits \
        -verilog_define SYNTHESIS

    # clk2x is the real timing clock (the accumulator domain). clk only feeds the
    # phase mux (a level into the multiply path), constrained at 2x the period so
    # it never dominates. Both come from the same MMCM on silicon.
    create_clock -name clk2x -period $period               [get_ports clk2x]
    create_clock -name clk   -period [expr {2.0 * $period}] [get_ports clk]

    report_utilization
    puts "----- timing (clk2x ${period} ns) -----"
    report_timing_summary -max_paths 5 -delay_type max

    # pull WNS for clk2x and compute the achieved Fmax
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -delay_type max]]
    set achieved [expr {1000.0 / ($period - $wns)}]
    puts [format "OOC_MACDP period=%s WNS=%.3f achieved_fmax=%.1f MHz lanes=%d" \
          $period $wns $achieved $lanes]

    # fresh project state for the next period in the sweep
    close_design
}

puts "OOC_MACDP_DONE lanes=$lanes (sweep: $periods)"
puts "READ: the period where WNS first crosses >=0 is the OOC Fmax; >=400 MHz (2.5ns MET) = GO."
