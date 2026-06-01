#!/usr/bin/env bash
# Run all 17 model layers through the loadable PL GEMV engine from ONE bitstream,
# streaming each layer's weights over AXI and checking bit-exact vs golden.
# Run as root (needs /dev/mem):  sudo bash run_all_layers.sh
BASE=0xA0000000
DIR=/home/ubuntu/kevpl/export
DRV=/home/ubuntu/kevpl/gemv_load_drv
LAYERS="blocks_0_qkv blocks_0_proj blocks_0_mlp_fc blocks_0_mlp_proj \
        blocks_1_qkv blocks_1_proj blocks_1_mlp_fc blocks_1_mlp_proj \
        blocks_2_qkv blocks_2_proj blocks_2_mlp_fc blocks_2_mlp_proj \
        blocks_3_qkv blocks_3_proj blocks_3_mlp_fc blocks_3_mlp_proj head"
pass=0; tot=0
for L in $LAYERS; do
    read -r M K < <(python3 -c "import json;d=json.load(open('$DIR/tb/$L.dims.json'));print(d['M'],d['K'])")
    "$DRV" "$BASE" "$M" "$K" "$DIR/$L.w.mem" "$DIR/tb/$L.x.mem" "$DIR/tb/$L.y.mem" "$L"
    [ $? -eq 0 ] && pass=$((pass+1)); tot=$((tot+1))
done
echo "STAGE2_ALL_LAYERS pass=$pass/$tot"
