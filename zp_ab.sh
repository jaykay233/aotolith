#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1
echo "=== ZP_OFF bin=mega_minicpm_pf.new"
./mega_minicpm_pf.new -m $WORK/mc-q4o.gguf -f /home/bianbu/opt10k/ctx10k.txt -n 32 -c 12288 -t 8
echo "EXIT_ZP_OFF=$?"
echo "=== ZP_ON bin=mega_minicpm_pf.zp"
./mega_minicpm_pf.zp -m $WORK/mc-q4o.gguf -f /home/bianbu/opt10k/ctx10k.txt -n 32 -c 12288 -t 8
echo "EXIT_ZP_ON=$?"
echo "=== ZP_SPEC_K4 bin=mega_minicpm_spec.zp"
MEGA_DRAFT=4 ./mega_minicpm_spec.zp -m $WORK/mc-q4o.gguf -f /home/bianbu/opt10k/ctx10k.txt -n 64 -c 12288 -t 8
echo "EXIT_ZP_SPEC_K4=$?"
echo "=== ZP_DONE"
