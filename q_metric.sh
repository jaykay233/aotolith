#!/bin/sh
# How far is each weight format from exact fp32? The judge is ref_f32.py's logit
# row, not llama.cpp's Q4_0: Q4_0 itself sits 0.66 cosine from fp32, so using it
# as the yardstick would reward the wrong thing.
#
#   q_metric.sh [ids]
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/px64.txt}
B=${B:-./m3c.new}

for cfg in "-X" "-K8 -q3 -X" "-K8 -q2 -X"; do
    tag=$(echo "$cfg" | tr -d ' -')
    MEGA_LOGD=/tmp/qm_$tag.bin $B -m "$M" -F "$P" -n 1 $cfg 2>&1 | grep -E "^logits|step 0 ->" || true
done
for f in /tmp/qm_*.bin; do
    echo "-- $(basename $f)"
    python3 cmpvec.py "$f" /tmp/rf_log64.bin | grep -E "cosine|rms diff|top8 ref"
done
