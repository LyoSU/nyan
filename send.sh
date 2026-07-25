#!/bin/bash
# Runs the sender in a loop. The daemon is meant to run forever, so an exit
# means it crashed: back off before restarting instead of spinning on a
# misconfiguration and filling the log with the same traceback.
set -uo pipefail

mkdir -p data

BACKOFF=1
MAX_BACKOFF=60

while :; do
    START=$(date +%s)

    python3 -m nyan.send \
        --channels-info-path channels.json \
        --client-config-path configs/client_config.json \
        --mongo-config-path configs/mongo_config.json \
        --annotator-config-path configs/annotator_config.json \
        --clusterer-config-path configs/clusterer_config.json \
        --ranker-config-path configs/ranker_config.json \
        --renderer-config-path configs/renderer_config.json \
        --daemon-config-path configs/daemon_config.json
    CODE=$?

    RUNTIME=$(( $(date +%s) - START ))

    # A run that lasted a while hit something transient; one that died
    # immediately will keep dying, so wait longer each time.
    if [ "$RUNTIME" -ge 60 ]; then
        BACKOFF=1
    else
        BACKOFF=$(( BACKOFF * 2 ))
        [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
    fi

    echo "Sender exited with code $CODE after ${RUNTIME}s, restarting in ${BACKOFF}s" >&2
    sleep "$BACKOFF"
done
