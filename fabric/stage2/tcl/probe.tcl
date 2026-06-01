# Probe the Vivado environment for the Stage 2 PS+PL build: version, the KV260
# part, whether KV260 board files are installed (decides PS-config approach), and
# the MPSoC PS IP. No project created; just reports and exits.
puts "PROBE_VIVADO_VERSION [version -short]"
set part xck26-sfvc784-2LV-c
puts "PROBE_PART_FOUND [llength [get_parts -quiet $part]]"
set bps [get_board_parts -quiet *kv260*]
puts "PROBE_KV260_BOARDPARTS $bps"
puts "PROBE_ZYNQ_PS_IP [llength [get_ipdefs -quiet *zynq_ultra_ps_e*]]"
puts "PROBE_AXI_BRAM_IP [llength [get_ipdefs -quiet *axi_bram_ctrl*]]"
puts "PROBE_DONE"
