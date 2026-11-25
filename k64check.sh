#!/bin/sh
# Did the 64-wide (m2, vl=64) QK branch actually run? The section probe lives only
# in the 32-wide branch, so a zeroed dot time means the 64-wide branch was taken.
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 MEGA_QKSPLIT=1 MEGA_STEPS=1
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/k64chk.res

for arm in k32 k64; do
    if [ "$arm" = k64 ]; then E="MEGA_K64=1"; else E="MEGA_K64=0"; fi
    echo "== $arm" >> /tmp/k64chk.res
    env $E ./m3c.sp -m mc-q4o.gguf -F /tmp/p10k.txt -n 8 -X > /tmp/k64chk_$arm.raw 2>&1
    grep -E "qk split" /tmp/k64chk_$arm.raw >> /tmp/k64chk.res
    grep -E "attn split" /tmp/k64chk_$arm.raw >> /tmp/k64chk.res
    grep -E "^decode: [0-9]+ tokens in" /tmp/k64chk_$arm.raw >> /tmp/k64chk.res
done
echo K64CHKDONE >> /tmp/k64chk.res
