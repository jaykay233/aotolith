#!/bin/sh
# Greedy decoding with the exact fp32 numpy reference as the judge.
#
#   rf_greedy.sh ids.txt nsteps
#
# Each step re-runs ref_f32.py with the token appended, so this is true greedy
# (no sampling, no KV reuse) and costs one full pass per token. Slow but it is
# the ground truth the megakernel's chain is checked against.
P=${1:?ids file}
N=${2:-4}
G=/dev/shm/mc-q4o.gguf
cp "$P" /tmp/rfg.txt
echo "== fp32 greedy from $P"
for s in $(seq 1 "$N"); do
    top=$(RF_LOG=/tmp/rfg_log.bin python3 ref_f32.py $G /tmp/rfg.txt 2>/dev/null |
          awk '/f32 reference top-8/{getline; print $1}')
    echo "  step $((s - 1)) -> $top"
    [ -z "$top" ] && break
    printf " %s" "$top" >> /tmp/rfg.txt
done
