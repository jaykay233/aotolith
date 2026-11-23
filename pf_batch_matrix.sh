#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
LOG=$WORK/pf_batch_matrix.log
: > "$LOG"
run() {
  echo "=== $*" >> "$LOG"
  env "$@" timeout 600 ./mega_minicpm_pf -m "$M" -f "$P" -n 8 -c 4096 -t 8 >> "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
}
P=/dev/shm/p600.txt  run MEGA_PF_BATCH=244
P=/dev/shm/p1200.txt run MEGA_PF_BATCH=458
P=/dev/shm/p1200.txt run MEGA_PF_BATCH=128
P=/dev/shm/p1200.txt run MEGA_PF_BATCH=200
P=/dev/shm/p1200.txt run MEGA_PF_BATCH=457
echo "=== DONE" >> "$LOG"
