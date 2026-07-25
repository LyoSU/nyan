#!/bin/bash
# Crawls the channel list continuously. Per-channel pacing is handled by the
# spider's recrawl_time, so a finished crawl restarts immediately; a crawl that
# failed within seconds gets an increasing delay so a broken config does not
# turn into a busy loop.
set -uo pipefail

# State, not code: it belongs next to the other data. It used to live in
# crawler/, which forced deployments to mount the source directory to keep it —
# and that mount shadowed the code in the image, so rebuilds changed nothing.
FETCH_TIMES=${FETCH_TIMES:-data/fetch_times.json}

mkdir -p "$(dirname "$FETCH_TIMES")"

if [ ! -f "$FETCH_TIMES" ] && [ -f crawler/fetch_times.json ]; then
    echo "Carrying fetch times over from crawler/ to $FETCH_TIMES" >&2
    cp crawler/fetch_times.json "$FETCH_TIMES"
fi

# Pause between passes. Which channels are actually read is decided by the
# spider's recrawl_time, so this only stops the loop from spinning once every
# channel has been read recently and a pass returns immediately.
CRAWL_INTERVAL=${CRAWL_INTERVAL:-60}
RECRAWL_TIME=${RECRAWL_TIME:-300}

BACKOFF=1
MAX_BACKOFF=60

while :; do
    START=$(date +%s)

    scrapy crawl telegram \
        -a channels_file=channels.json \
        -a fetch_times="$FETCH_TIMES" \
        -a recrawl_time="$RECRAWL_TIME" \
        -a hours=24
    CODE=$?

    RUNTIME=$(( $(date +%s) - START ))
    if [ "$CODE" -eq 0 ] || [ "$RUNTIME" -ge 60 ]; then
        BACKOFF=1
        sleep "$CRAWL_INTERVAL"
    else
        BACKOFF=$(( BACKOFF * 2 ))
        [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
        echo "Crawl exited with code $CODE after ${RUNTIME}s, retrying in ${BACKOFF}s" >&2
        sleep "$BACKOFF"
    fi
done
