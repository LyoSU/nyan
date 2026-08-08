"""Fetching a media file the bot has to hand to Telegram itself.

Almost every attachment goes out as a URL and is never downloaded here. The
exception is a video Telegram refuses to fetch on its own, which has to be
re-sent as bytes — see `nyan.client` for when that happens and why.
"""

import logging
from urllib.parse import urlparse

import requests

DOWNLOAD_TIMEOUT_SECONDS = 30
_CHUNK_BYTES = 1 << 16

# Where media is allowed to come from. Every one of the 14 179 video URLs on
# record is https on a cdnN.telesco.pe host, and cdn-telegram.org is what
# `nyan.rich.fix_media_url` rewrites those into — so an allowlist costs nothing
# and is exact, where a blocklist of private ranges would still permit any
# public address and would not survive a DNS rebind.
ALLOWED_MEDIA_HOSTS = ("telesco.pe", "cdn-telegram.org")


def is_allowed_media_url(url: str) -> bool:
    """Whether `url` may be fetched: https, on a Telegram CDN host.

    These URLs are read off crawled channel pages, which makes this the one
    place in the project where outside input becomes an outbound request — and
    whatever comes back is handed to Telegram to publish.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    # Compared against the host's own labels, so a "telesco.pe.evil.com" that
    # merely contains an allowed name is not mistaken for one.
    return any(
        host == allowed or host.endswith("." + allowed)
        for allowed in ALLOWED_MEDIA_HOSTS
    )


def fetch_media(url: str, size_limit: int) -> bytes | None:
    """The file at `url`, or None if it cannot be had within `size_limit`.

    Never raises: every caller has a working fallback — sending the URL and
    letting Telegram try — and a download failure must not cost the post.
    """
    if not is_allowed_media_url(url):
        logging.warning("Refusing to download media from outside the CDN: %s", url)
        return None

    try:
        with requests.get(
            url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            # Otherwise an allowed host could redirect the request to any
            # address at all, and the host check would have decided nothing.
            allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                logging.warning(
                    "Media not downloadable (HTTP %d): %s", response.status_code, url
                )
                return None

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > size_limit:
                logging.warning("Media too large to upload (%s bytes): %s", declared, url)
                return None

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(_CHUNK_BYTES):
                total += len(chunk)
                # Checked while reading, not only against the header: a missing
                # or lying content-length would otherwise buy an unbounded read
                # straight into memory.
                if total > size_limit:
                    logging.warning("Media exceeded the upload limit mid-read: %s", url)
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception:
        logging.warning("Could not download media: %s", url)
        return None
