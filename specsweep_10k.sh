#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1
for K in 2 4 5 6; do
  echo "=== SWEEP_K$K env=MEGA_DRAFT=$K"
  MEGA_DRAFT=$K ./mega_minicpm_spec.new -m $WORK/mc-q4o.gguf -f /home/bianbu/opt10k/ctx10k.txt -n 64 -c 12288 -t 8
  echo "EXIT_SWEEP_K$K=$?"
done
echo "=== SWEEP_DONE"
