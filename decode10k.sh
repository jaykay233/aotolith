#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# 10K decode-path A/B in the megakernel: which attention path serves single-token decode best.
# Each run pays the 146 s 10K prefill; decode n=32 keeps the measurement ~4 s but stable.
#   D_T8           one_chunk decode (current default, 8 chunks over 16 q-head rows)
#   D_T8_TILED     route decode through the tiled kernel (FA_TILED_MIN=1): packs each KV tile
#                  once and shares it across rows, but still 8 row-chunks -> same KV traffic
#   D_T8_TILED_I8  same + int8 QK (GGML_FA_ATTN_I8=1) -> tests whether the int8 path also
#                  speeds single-token decode
#   D_NC2_T8       one_chunk with FA_NCHUNK=2 (one thread per kv-head group, half the KV passes,
#                  half the threads)
cd "$WORK" || exit 1
set -o pipefail
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/decode10k.log
: > "$LOG"

run() { # label t n extra_env...
  local label=$1 t=$2 n=$3; shift 3
  echo "=== $label t=$t n=$n env=$*" >> "$LOG"
  timeout 2400 env "$@" ./mega_minicpm_pf -m "$M" -f "$P" -n "$n" -c 12288 -t "$t" 2>&1 \
    | grep -E '^\[mega\]|^TOKENS|^FA_PROF|^IME' >> "$LOG"
  echo "EXIT_$label=$?" >> "$LOG"
}

run D_T8         8 32
run D_T8_TILED   8 32 FA_TILED_MIN=1
run D_T8_TILED_I8 8 32 FA_TILED_MIN=1 GGML_FA_ATTN_I8=1
run D_NC2_T8     8 32 FA_NCHUNK=2
echo "=== DONE" >> "$LOG"
