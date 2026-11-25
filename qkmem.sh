#!/bin/sh
# Is decode QK's dot loop limited by memory, or by the vector unit?
#
# The earlier "qing tile 0 changes nothing" test grepped the prefill qk line, not
# the decode one, so it never tested decode. This pins every position block to
# tile 0 of each layer (42 layers x 2 kv x 16 KB = 1.3 MB, L2-resident) and reads
# the instrumented dot-loop time, which is measured inside the loop that runs.
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 MEGA_QKSPLIT=1 MEGA_STEPS=1
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/qs.res

for arm in stream pinned; do
    if [ "$arm" = pinned ]; then E="MEGA_QKSAME=1"; else E=""; fi
    echo "== $arm" >> /tmp/qs.res
    env $E ./m3c.sp -m mc-q4o.gguf -F /tmp/p10k.txt -n 16 -X \
        -force refs/golden_10k_tokens.txt > /tmp/qs_$arm.raw 2>&1
    grep -E "qk split" /tmp/qs_$arm.raw >> /tmp/qs.res
    grep -E "attn split" /tmp/qs_$arm.raw >> /tmp/qs.res
    grep -E "^decode: [0-9]+ tokens in" /tmp/qs_$arm.raw >> /tmp/qs.res
done
echo QSDONE >> /tmp/qs.res
