"""Telling the site that the channel has a new post.

The site behind ``PUBLISH_URL`` answers a POST naming the stories by their
cluster ids: it drops its caches for those pages, pings the WebSub hub and
submits the URLs to IndexNow itself. So the ping is worth sending only after
the story is on record where the site reads it, and one ping per post is
enough.
"""

import logging
import os

import httpx

# Both come from the deployment, neither has a default: where the site lives is
# not the daemon's to know, and an empty token means off, so that dev runs and
# tests never reach the production site. `or` rather than a getenv default:
# docker-compose passes unset variables through as empty strings.
PUBLISH_URL = os.getenv("PUBLISH_URL") or ""
PUBLISH_TOKEN = os.getenv("PUBLISH_TOKEN") or ""

# The ping runs in the daemon loop right after a send; a slow site must not
# hold the next post back for long.
PUBLISH_TIMEOUT_SECONDS = 5.0


def notify_published(clids: list[int], *, transport: httpx.BaseTransport | None = None) -> bool:
    """Announce the stories with these cluster ids. True when the site took it.

    Never raises: a post that is already in the channel must not be reported
    as failed because the site was down. The failure is logged instead.
    """
    if not PUBLISH_URL or not PUBLISH_TOKEN:
        return False
    headers = {"Authorization": f"Bearer {PUBLISH_TOKEN}"}
    try:
        with httpx.Client(timeout=PUBLISH_TIMEOUT_SECONDS, transport=transport) as client:
            response = client.post(PUBLISH_URL, json={"clids": clids}, headers=headers)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logging.warning("Publish ping to %s failed: %s", PUBLISH_URL, e)
        return False
    return True
