#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# 10K speculative decode with split-KV covering verify batches.
# Verify row counts: K=3 -> 4 rows x 16 heads = 64 (split up to 512), K=7 -> 128.
cd "$WORK" || exit 1
set -o pipefail
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1 FA_TRACE=1 FA_PROF=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/specsplit_10k.log
: > "$LOG"

run() { # label extra_env...
  local label=$1; shift
  echo "=== $label env=$*" >> "$LOG"
  timeout 1200 env "$@" ./mega_minicpm_spec.new -m "$M" -f "$P" -n 64 -c 12288 -t 8 >> "$LOG" 2>&1
  echo "EXIT_$label=$?" >> "$LOG"
}

run PLAIN     MEGA_DRAFT=0
run SPEC_K3   MEGA_DRAFT=3
run SPEC_K7   MEGA_DRAFT=7
echo "=== DONE" >> "$LOG"
