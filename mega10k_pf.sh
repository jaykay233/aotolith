#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# 10K-depth benchmark, megakernel binary only (plus one llama.cpp ref pass for token check).
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/mega10k_pf.log
: > "$LOG"

echo "=== MEGA t=8 n=128" >> "$LOG"
timeout 2400 ./mega_minicpm_pf -m "$M" -f "$P" -n 128 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_T8=$?" >> "$LOG"

echo "=== MEGA t=4 n=128" >> "$LOG"
timeout 2400 ./mega_minicpm_pf -m "$M" -f "$P" -n 128 -c 12288 -t 4 >> "$LOG" 2>&1
echo "EXIT_T4=$?" >> "$LOG"

echo "=== REF t=8 n=4 (token check)" >> "$LOG"
timeout 2400 ./mega_minicpm_pf --ref -m "$M" -f "$P" -n 4 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_REF=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
