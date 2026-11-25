#!/bin/sh
# Pipelined K loads (MEGA_QKPIPE=1) against the plain loop, on the default f16
# KV / 32-wide QK path. The pipeline only reorders loads against FMAs, so the two
# arms must write identical logits.
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 MEGA_QKSPLIT=1 MEGA_STEPS=1
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/pp.res

for arm in pipe nopipe; do
    if [ "$arm" = pipe ]; then E="MEGA_QKPIPE=1"; else E="MEGA_QKPIPE=0"; fi
    echo "== $arm" >> /tmp/pp.res
    env $E MEGA_STEPLOG=/tmp/pp_$arm.logits ./m3c.pp -m mc-q4o.gguf -F /tmp/p10k.txt \
        -n 32 -X -force refs/golden_10k_tokens.txt > /tmp/pp_$arm.raw 2>&1
    echo "exit=$?" >> /tmp/pp.res
    md5sum /tmp/pp_$arm.logits >> /tmp/pp.res
    grep -E "qk split|attn split" /tmp/pp_$arm.raw >> /tmp/pp.res
    grep -E "^decode: [0-9]+ tokens in" /tmp/pp_$arm.raw >> /tmp/pp.res
done
echo PPDONE >> /tmp/pp.res
