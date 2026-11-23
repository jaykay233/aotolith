#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Correctness + speed of prompt-lookup speculative decode in the megakernel.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
LOG=$WORK/spec_test.log
: > "$LOG"

head -c 1800 /home/bianbu/opt10k/ctx10k.txt > /dev/shm/short1.txt

# --- short context: greedy tokens must match between plain decode, speculative decode, and --ref
echo "=== SHORT plain (MEGA_DRAFT=0) n=24" >> "$LOG"
timeout 600 env MEGA_DRAFT=0 ./mega_minicpm_spec -m "$M" -f /dev/shm/short1.txt -n 24 -c 2048 -t 8 >> "$LOG" 2>&1
echo "EXIT_S_PLAIN=$?" >> "$LOG"
echo "=== SHORT spec k=3 n=24" >> "$LOG"
timeout 600 env MEGA_DRAFT=3 ./mega_minicpm_spec -m "$M" -f /dev/shm/short1.txt -n 24 -c 2048 -t 8 >> "$LOG" 2>&1
echo "EXIT_S_K3=$?" >> "$LOG"
echo "=== SHORT spec k=7 n=24" >> "$LOG"
timeout 600 env MEGA_DRAFT=7 ./mega_minicpm_spec -m "$M" -f /dev/shm/short1.txt -n 24 -c 2048 -t 8 >> "$LOG" 2>&1
echo "EXIT_S_K7=$?" >> "$LOG"
echo "=== SHORT ref n=24" >> "$LOG"
timeout 900 ./mega_minicpm_spec --ref -m "$M" -f /dev/shm/short1.txt -n 24 -c 2048 -t 8 >> "$LOG" 2>&1
echo "EXIT_S_REF=$?" >> "$LOG"
echo "=== DONE_SHORT" >> "$LOG"
