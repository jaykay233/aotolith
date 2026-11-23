#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# 10K-context benchmark using the megakernel binary only.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/mega10k.log
: > "$LOG"
for T in 8 4; do
  echo "=== MEGA t$T 10K" | tee -a "$LOG"
  timeout 1800 ./mega_minicpm.new -m "$M" -f "$P" -n 128 -c 12288 -t "$T" >> "$LOG" 2>&1
  echo "EXIT_T$T=$?" | tee -a "$LOG"
done
echo "=== DONE" | tee -a "$LOG"
