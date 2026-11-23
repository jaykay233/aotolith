#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Is the >256-key failure thread- or TCM-sensitive?
cd "$WORK" || exit 1
M=$WORK/mc-q4o.gguf
P=/dev/shm/p1200.txt
LOG=$WORK/pf_thread_test.log
: > "$LOG"

echo "=== T1" >> "$LOG"
env SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1 MEGA_PF_BATCH=458 \
    timeout 600 ./mega_minicpm_pf -m "$M" -f "$P" -n 4 -c 4096 -t 1 >> "$LOG" 2>&1
echo "EXIT_T1=$?" >> "$LOG"

env SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1 MEGA_PF_BATCH=458 \
    timeout 600 ./mega_minicpm_pf -m "$M" -f "$P" -n 4 -c 4096 -t 4 >> "$LOG" 2>&1
echo "EXIT_T4=$?" >> "$LOG"

# no TCM hold
env -u SPACEMIT_TCM_HOLD SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1 MEGA_PF_BATCH=458 \
    timeout 600 ./mega_minicpm_pf -m "$M" -f "$P" -n 4 -c 4096 -t 8 >> "$LOG" 2>&1
echo "EXIT_NOTCM=$?" >> "$LOG"

# no AI-core pinning
env SPACEMIT_TCM_HOLD=1 GGML_TILED_KPACK=1 MEGA_PF_BATCH=458 \
    timeout 600 ./mega_minicpm_pf -m "$M" -f "$P" -n 4 -c 4096 -t 8 >> "$LOG" 2>&1
echo "EXIT_NOCORES=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
