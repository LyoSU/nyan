#!/bin/bash
# Rebuilds the channel graph once a night.
#
# The graph is what lets the site say how many of a story's sources are not each
# other. Without it the site does not fail — `getChannelGraph` returns an empty
# list on any error and independence quietly falls back to the three rungs a
# single story can prove — which is exactly why this needs a schedule rather than
# a person remembering: nobody will notice the number drifting upwards as new
# clone networks appear unmeasured.
#
# Fixed hour rather than an interval, for the same reason as digest.sh: it is a
# pass over months of clusters and belongs at the quietest part of the night, not
# at whatever time the container happened to last restart.
set -uo pipefail

# Kyiv time. 4am: after the night's crawling has settled and long before the
# morning digest reads anything.
#
# Deliberately not listed in docker-compose.yml's environment, unlike every other
# knob in this project. Coolify turns each variable there into an `ARG` plus a
# `--mount=type=secret` on every `RUN` of a Dockerfile it inlines into a shell
# command once per service — so a variable costs bytes in a command line that has
# already overflowed ARG_MAX once. These two have working defaults and nobody has
# ever needed to change them, which is not worth spending that budget on. To
# override, add them to the nyan-app service.
GRAPH_HOUR=${GRAPH_HOUR:-4}
GRAPH_WINDOW_DAYS=${GRAPH_WINDOW_DAYS:-60}
export TZ=${NYAN_TIMEZONE:-Europe/Kyiv}

build() {
    python3 -m scripts.build_channel_graph \
        --mongo-config-path configs/mongo_config.json \
        --window-days "$GRAPH_WINDOW_DAYS"
    CODE=$?
    [ "$CODE" -ne 0 ] && echo "Channel graph exited with code $CODE" >&2
    return 0
}

# Once at startup, before the first sleep. A fresh database — or a deploy onto
# one where this has never run — otherwise serves up to a full day of pages with
# the fourth rung missing and no sign that anything is absent. The loop below
# never exits on its own, so this runs on a real restart and not in a crash loop.
echo "Building the channel graph at startup" >&2
build

while :; do
    NOW=$(date +%s)
    NEXT=$(date -d "today ${GRAPH_HOUR}:00" +%s 2>/dev/null)
    if [ -z "${NEXT:-}" ]; then
        echo "Cannot parse GRAPH_HOUR='$GRAPH_HOUR', waiting an hour" >&2
        sleep 3600
        continue
    fi
    [ "$NEXT" -le "$NOW" ] && NEXT=$(date -d "tomorrow ${GRAPH_HOUR}:00" +%s)

    echo "Next channel graph rebuild at $(date -d "@$NEXT" '+%d.%m %H:%M %Z')" >&2
    sleep $(( NEXT - NOW ))

    build

    # A slot can be reached a second early by rounding, which would rebuild
    # twice; the next wait is then computed from a time safely past the hour.
    sleep 60
done
