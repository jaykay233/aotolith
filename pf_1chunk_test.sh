#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=$WORK/mc-q4o.gguf
LOG=$WORK/pf_1chunk_test.log
: > "$LOG"
echo "=== one chunk 458, kpack=1" >> "$LOG"
GGML_TILED_KPACK=1 MEGA_PF_BATCH=458 timeout 600 ./mega_minicpm_pf -m "$M" -f /dev/shm/p1200.txt -n 8 -c 4096 -t 8 >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
echo "=== one chunk 458, kpack=0" >> "$LOG"
GGML_TILED_KPACK=0 MEGA_PF_BATCH=458 timeout 600 ./mega_minicpm_pf -m "$M" -f /dev/shm/p1200.txt -n 8 -c 4096 -t 8 >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
echo "=== 2 chunks 256, pf-decode, kpack=1" >> "$LOG"
GGML_TILED_KPACK=1 MEGA_PF_BATCH=256 MEGA_PF_DECODE=1 timeout 600 ./mega_minicpm_pf -m "$M" -f /dev/shm/p1200.txt -n 8 -c 4096 -t 8 >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
