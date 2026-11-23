#!/bin/sh
# Run a heavy kernel build/bench on the board without being able to leave
# orphans behind.
#
# Why this exists: mega3 at pp10240 maps ~1.6 GB of huge pages plus the int8 KV
# planes. Launching three of them detached with "( ... &)" and then losing the
# ssh session left all three running; together they exhausted the board's memory
# and took sshd with it, which looked exactly like "the tunnel broke". Every run
# now goes through here: it kills strays first, runs in the foreground with a
# hard timeout, and kills the whole process group on any exit path.
#
# The working files live on UFS ($WORK), not /dev/shm: tmpfs is RAM, and 3.6 GB of
# model plus caches sitting in RAM is what made the board fragile in the first
# place. See board_env.sh.
#
#   ./run_board.sh "make -j8"                  # 900 s default
#   ./run_board.sh -t 300 "./mega3p -A 10240 -J -K"
#   ./run_board.sh -k                          # just reap strays, run nothing
set -u
HERE=$(dirname "$0")
. "$HERE/board_env.sh"

TIMEOUT=900
REAP_ONLY=0
while getopts "t:k" o; do
    case $o in
        t) TIMEOUT=$OPTARG ;;
        k) REAP_ONLY=1 ;;
        *) echo "usage: $0 [-t secs] [-k] <command>" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

PAT='mega3|mega_decode|probe_'

# Kill leftovers from earlier sessions and report memory before we add to it.
# meminfo matters: the earlier failure was invisible until the board stopped
# answering.
reap="
pkill -9 -f '$PAT' 2>/dev/null
sleep 0.3
echo strays_left=\$(pgrep -fc '$PAT' 2>/dev/null || echo 0)
grep -E '^(MemTotal|MemAvailable|MemFree|Shmem):' /proc/meminfo
"

if [ "$REAP_ONLY" = "1" ]; then
    $SSH "$reap"
    exit $?
fi

CMD=$*

# setsid + timeout: the command gets its own process group on the board, so a
# dropped ssh cannot orphan it, and timeout bounds it. The trailing reap runs
# even if the payload is killed or the connection drops mid-flight.
$SSH "
    $reap
    mkdir -p '$WORK' && cd '$WORK' || exit 1
    setsid sh -c 'env MEGA_WORK=$WORK timeout -s KILL $TIMEOUT sh -c \"$CMD\"; echo payload_exit=\$?' </dev/null
    rc=\$?
    pkill -9 -f '$PAT' 2>/dev/null
    grep -E '^(MemAvailable|AnonHugePages|Shmem):' /proc/meminfo
    exit \$rc
"
