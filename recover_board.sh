#!/bin/sh
# Bring the board back to a known-good state after a recovery.
#
# The board has been coming back on its own (no power cycle), which is why this is
# written to diagnose first and change nothing. Two things used to break every
# recovery: /dev/shm is tmpfs, so the model, the weight caches and the vendor
# objects vanished, and there was no record of how ime2.o / rvv.o were built.
# Everything now lives on UFS under $WORK and survives.
#
#   ./recover_board.sh          # diagnose, then rebuild and verify the baseline
#   ./recover_board.sh -n       # diagnose only, change nothing
#   ./recover_board.sh -m       # migrate an existing /dev/shm install to UFS, then verify
set -u
HERE=$(dirname "$0")
. "$HERE/board_env.sh"
DRY=0
MIGRATE=0
for a in "$@"; do
    [ "$a" = "-n" ] && DRY=1
    [ "$a" = "-m" ] && MIGRATE=1
done

say() { printf '\n=== %s\n' "$1"; }

say "reachability"
if ! $SSH 'true' 2>/dev/null; then
    echo "UNREACHABLE. The gateway handles TLS; a timeout at banner exchange means"
    echo "the board itself is not answering. Nothing here can fix that -- wait for it"
    echo "to come back on its own, or power-cycle it."
    exit 1
fi
echo "ssh ok"

say "identity"
$SSH 'hostname; uptime; uname -r'

say "did it reboot, and was it an OOM"
# The failure mode being watched for: the kernel's own OOM killer taking something
# out. A short uptime plus OOM lines is confirmation; absence of them means the
# hang was elsewhere and the memory accounting below is the thing to read.
$SSH '
    echo "uptime_s=$(cut -d. -f1 /proc/uptime)"
    n=$(dmesg 2>/dev/null | grep -ciE "out of memory|oom-killer|killed process" || true)
    echo "oom_lines=$n"
    dmesg 2>/dev/null | grep -iE "out of memory|killed process" | tail -5
'

say "memory and filesystems"
$SSH '
    grep -E "^(MemTotal|MemAvailable|AnonHugePages|CmaTotal|CmaFree|Shmem):" /proc/meminfo | sed "s/^/  /"
    echo "  --- mounts ---"
    df -h / /home /dev/shm 2>/dev/null | sed "s/^/  /"
'

say "stray kernels"
$SSH 'pgrep -af "mega3|mega_minicpm|mega_decode|probe_" || echo "  none"'

say "work directory"
$SSH "
    if [ -d '$WORK' ]; then
        ls -l '$WORK' | sed 's/^/  /'
    else
        echo '  $WORK missing'
    fi
"

if [ "$MIGRATE" = "1" ]; then
    say "migrating /dev/shm -> UFS"
    # Copy rather than move: if the copy is interrupted, /dev/shm still has the
    # originals for this session. Verify sizes before deleting anything.
    $SSH "
        set -e
        mkdir -p '$WORK'
        cd /dev/shm || exit 1
        for f in mega3.cpp mega3 mega_decode.cpp mega_decode ime2.o rvv.o \
                 mc-q4o.gguf mega-w.bin mega-w3.bin mega-w8b.bin; do
            [ -f \"\$f\" ] || continue
            if [ -f '$WORK'/\$f ] && [ \"\$(stat -c %s \"\$f\")\" = \"\$(stat -c %s '$WORK'/\$f)\" ]; then
                echo \"  keep \$f (already there, same size)\"
            else
                echo \"  copy \$f (\$(stat -c %s \"\$f\") bytes)\"
                cp \"\$f\" '$WORK'/\"\$f\".part && mv '$WORK'/\"\$f\".part '$WORK'/\$f
            fi
        done
        echo '--- result ---'
        ls -l '$WORK' | sed 's/^/  /'
        echo '--- reclaimable from tmpfs ---'
        du -sh mc-q4o.gguf mega-w.bin mega-w3.bin mega-w8b.bin 2>/dev/null | sed 's/^/  /'
    "
fi

say "vendor objects"
# ime2.o and rvv.o come from the llama.cpp SpacemiT backend (ime2_kernels.cpp,
# rvv_kernels.cpp). If they are missing they can be recovered from that checkout's
# build tree instead of guessing a compile line, since its own CMake rules already
# built them.
$SSH "
    for d in '$WORK' /dev/shm; do
        [ -f \"\$d/ime2.o\" ] && [ -f \"\$d/rvv.o\" ] && { echo \"  found in \$d\"; exit 0; }
    done
    echo '  not found; searching the llama.cpp build tree'
    find /home/bianbu/workspace -path '*spacemit*' -name 'ime2_kernels*.o' 2>/dev/null | head -3 | sed 's/^/    /'
    find /home/bianbu/workspace -path '*spacemit*' -name 'rvv_kernels*.o' 2>/dev/null | head -3 | sed 's/^/    /'
"

say "clusters"
$SSH '
    echo "  cpus=$(grep -c ^processor /proc/cpuinfo)"
    for c in 0 8; do
        f=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || echo "?")
        echo "  cpu$c freq=$f"
    done
    echo "  (bind is /proc/set_ai_thread, not an affinity mask: without it a worker"
    echo "   lands on X100 and every timing changes while the numbers stay wrong)"
'

if [ "$DRY" = "1" ]; then
    say "dry run, stopping here"
    exit 0
fi

say "rebuild mega3 and confirm the archived baseline"
# The control run: if this does not reproduce ~31.4 s attention and ~27.4 s GEMM,
# something in the environment changed and no new measurement is comparable.
$SSH "
    cd '$WORK' || exit 1
    if [ ! -f mega3.cpp ] || [ ! -f ime2.o ] || [ ! -f rvv.o ]; then
        echo '  skip: need mega3.cpp and both objects'; exit 1
    fi
    g++ -O1 -std=c++20 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii \
        -o mega3 mega3.cpp ime2.o rvv.o -lpthread -lm > bld3.log 2>&1
    if [ \$? -ne 0 ]; then echo 'BUILD FAILED'; head -20 bld3.log; exit 1; fi
    grep -iE 'error|illegal' bld3.log | head -5
    echo 'build ok'
    ls -l mega3
"
