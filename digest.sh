#!/bin/bash
# Publishes the digest on a fixed daily schedule.
#
# Fixed times rather than an interval, because a digest is something a reader
# expects at a time of day, not every N hours from whenever the container last
# restarted. The window each digest covers comes from the last published one, so
# a missed slot is not a gap: the next digest simply covers more ground.
set -uo pipefail

# Kyiv time, comma-separated hours. The default follows the pattern the reader
# already knows: morning, midday, evening, night.
DIGEST_HOURS=${DIGEST_HOURS:-8,14,18,23}
export TZ=${NYAN_TIMEZONE:-Europe/Kyiv}

seconds_until_next() {
    local now next best hour
    now=$(date +%s)
    best=""
    for hour in ${DIGEST_HOURS//,/ }; do
        next=$(date -d "today ${hour}:00" +%s 2>/dev/null) || return 1
        # Already past today: that slot comes round tomorrow.
        [ "$next" -le "$now" ] && next=$(date -d "tomorrow ${hour}:00" +%s)
        if [ -z "$best" ] || [ "$next" -lt "$best" ]; then
            best=$next
        fi
    done
    echo $(( best - now ))
}

while :; do
    WAIT=$(seconds_until_next)
    if [ -z "$WAIT" ] || [ "$WAIT" -le 0 ]; then
        echo "Cannot parse DIGEST_HOURS='$DIGEST_HOURS', waiting an hour" >&2
        WAIT=3600
    else
        echo "Next digest at $(date -d "@$(( $(date +%s) + WAIT ))" '+%H:%M %Z')" >&2
    fi
    sleep "$WAIT"

    python3 -m nyan.digest \
        --mongo-config-path configs/mongo_config.json \
        --client-config-path configs/client_config.json \
        --duration-hours "${DIGEST_DURATION_HOURS:-8}" \
        --min-news-count "${DIGEST_MIN_NEWS:-5}" \
        --auto
    CODE=$?
    [ "$CODE" -ne 0 ] && echo "Digest exited with code $CODE" >&2

    # A slot can be reached a second early by rounding, which would run the same
    # digest twice; the next wait is then computed from a time safely past it.
    sleep 60
done
