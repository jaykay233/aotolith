#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Where does a 4-token verify spend its time? FA_PROF instruments the tiled kernel section by
# section (qk / softmax / pv, rdtime @ 24 MHz), and prints per 6720 tiles.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/specprof10k.log
: > "$LOG"

echo "=== PROF_SPEC_K3 t=8 n=16" >> "$LOG"
timeout 2400 env MEGA_DRAFT=3 FA_PROF=1 IME_PROF=1 ./mega_minicpm_spec -m "$M" -f "$P" -n 16 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_PROF_SPEC_K3=$?" >> "$LOG"

echo "=== PROF_PLAIN_TILED t=8 n=16" >> "$LOG"
timeout 2400 env MEGA_DRAFT=0 FA_TILED_MIN=1 FA_PROF=1 IME_PROF=1 ./mega_minicpm_spec -m "$M" -f "$P" -n 16 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_PROF_PLAIN_TILED=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
