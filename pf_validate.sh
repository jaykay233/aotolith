#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Validate batched-prefill megakernel against llama.cpp on the real 10K prompt.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/pf_validate.log
: > "$LOG"
echo "=== ref 10K n=16" >> "$LOG"
timeout 1200 ./mega_minicpm_pf -m "$M" -f "$P" -n 16 -c 12288 -t 8 --ref >> "$LOG" 2>&1
echo "EXIT_REF=$?" >> "$LOG"
echo "=== mega 10K n=16" >> "$LOG"
timeout 1200 ./mega_minicpm_pf -m "$M" -f "$P" -n 16 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_MEGA=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
