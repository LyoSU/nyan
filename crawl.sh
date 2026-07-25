#!/bin/bash
# Crawls the channel list continuously. Per-channel pacing is handled by the
# spider's recrawl_time, so a finished crawl restarts immediately; a crawl that
# failed within seconds gets an increasing delay so a broken config does not
# turn into a busy loop.
set -uo pipefail

BACKOFF=1
MAX_BACKOFF=60

while :; do
    START=$(date +%s)

    scrapy crawl telegram \
        -a channels_file=channels.json \
        -a fetch_times=crawler/fetch_times.json \
        -a hours=24
    CODE=$?

    RUNTIME=$(( $(date +%s) - START ))
    if [ "$CODE" -eq 0 ] || [ "$RUNTIME" -ge 60 ]; then
        BACKOFF=1
    else
        BACKOFF=$(( BACKOFF * 2 ))
        [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
        echo "Crawl exited with code $CODE after ${RUNTIME}s, retrying in ${BACKOFF}s" >&2
        sleep "$BACKOFF"
    fi
done
