#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Which FA kernel path breaks at >256 keys? Same prompt, single 458-row chunk.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=$WORK/mc-q4o.gguf
P=/dev/shm/p1200.txt
LOG=$WORK/pf_kernel_ab.log
: > "$LOG"

run() {
  local tag="$1"; shift
  echo "=== $tag" >> "$LOG"
  env "$@" MEGA_PF_BATCH=458 timeout 300 ./mega_minicpm_pf \
      -m "$M" -f "$P" -n 4 -c 4096 -t 8 >> "$LOG" 2>&1
  echo "EXIT_$tag=$?" >> "$LOG"
}

# baseline: tiled kernel, kpack on (the failing config)
run BASE      GGML_TILED_KPACK=1
# force the one_chunk kernel for prefill too (neq1 >= FA_TILED_MIN never holds)
run ONECHUNK  GGML_TILED_KPACK=1 FA_TILED_MIN=100000000
# tiled kernel, kpack explicitly disabled
run NOKPACK   GGML_KPACK_OFF=1
# tiled kernel with 64-wide KV tiles
run KVT64     GGML_TILED_KPACK=1 GGML_FA_KVT64=1
# tiled kernel without q-head grouping inside a tile
run NOKVGRP   GGML_TILED_KPACK=1 GGML_FA_KVGROUP_OFF=1
echo "=== DONE" >> "$LOG"
