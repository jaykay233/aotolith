#!/bin/sh
# Where does decode QK's 12.9 ms actually go, on the path that runs by default
# (32-wide tiles, f16 KV)? Times the dot loop, the score stores and the per-head
# max sweep separately, summed over all threads.
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 MEGA_QKSPLIT=1 MEGA_STEPS=1
cd /home/bianbu/workspace/mega || exit 1
: > /tmp/sp.res

./m3c.sp -m mc-q4o.gguf -F /tmp/p10k.txt -n 16 -X \
    -force refs/golden_10k_tokens.txt > /tmp/sp.raw 2>&1
echo "exit=$?" >> /tmp/sp.res
grep -E "qk split|attn split|attn work \(ms" /tmp/sp.raw >> /tmp/sp.res
grep -E "^decode: [0-9]+ tokens in" /tmp/sp.raw >> /tmp/sp.res
echo SPDONE >> /tmp/sp.res
