#!/bin/sh
# Where does PV's 23.9 ms go: the V stream, or the eight scattered score loads?
# MEGA_PVPROBE drops the score loads (results are garbage, timing is the point).
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p4k.txt}
N=${N:-8}
B=${B:-./m3c.new}
for pre in "" "MEGA_PVPROBE=1"; do
    echo "== p4k -K8 [$pre]"
    env $pre $B -m "$M" -F "$P" -n "$N" -K8 -X 2>&1 |
        grep -E "decode path:|attn split|decode: .*tok/s" || true
done
