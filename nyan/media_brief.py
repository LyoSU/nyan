"""What the author is told about a story's pictures, and the pictures themselves.

The author decides where the media goes — in the body, under "Докладніше",
or nowhere — and which picture leads. Blind, it can only guess from the
captions; so each attachment is described in words ("фото", "відео, 0:40") and,
when the vision flag is on, its first few stills are attached at low quality.

A video is never downloaded or cut: only the poster Telegram rendered for it is
shown, next to the words "відео, 0:40", so the model does not take the poster
for a photograph. A video without a poster is the words alone.

Fetching must never hold a post up: a few seconds per picture, and a picture
that does not come is simply not shown.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image

from nyan.media import MEDIA_VIDEO, MediaItem

# Enough to tell a strike site from a channel's logo, and no more: OpenAI's
# "low" detail reads a picture at 512 px in any case.
MAX_SIDE = 512
JPEG_QUALITY = 60
MAX_PICTURES = 3
FETCH_TIMEOUT = 3


@dataclass(frozen=True)
class Attachment:
    """One attachment as the author is told about it."""

    label: str
    #: The still to show: the photo itself, or a video's poster. Empty when
    #: there is nothing to look at.
    still_url: str = ""


def describe(media: list[MediaItem], docs: list[Any]) -> list[Attachment]:
    """Every attachment the post may show, in its order, with a label.

    A video's poster and length live on the document that carried it, index
    aligned with its `videos`; MediaItem keeps neither.
    """
    posters: dict[str, tuple[str, int]] = {}
    for doc in docs:
        thumbs = list(getattr(doc, "video_thumbs", ()) or ())
        durations = list(getattr(doc, "video_durations", ()) or ())
        for index, url in enumerate(getattr(doc, "videos", ()) or ()):
            thumb = thumbs[index] if index < len(thumbs) else ""
            duration = durations[index] if index < len(durations) else 0
            posters.setdefault(url, (thumb, duration))
    described = []
    for item in media:
        if item.type == MEDIA_VIDEO:
            thumb, duration = posters.get(item.url, ("", 0))
            length = f", {duration // 60}:{duration % 60:02d}" if duration else ""
            label = f"відео{length}" + ("" if thumb else ", без прев'ю")
            described.append(Attachment(label=label, still_url=thumb))
        else:
            described.append(Attachment(label="фото", still_url=item.url))
    return described


def media_note(attachments: list[Attachment], shown: int) -> str:
    """The lines the user message ends with, naming each attachment."""
    if not attachments:
        return ""
    lines = [f"{index}. {item.label}" for index, item in enumerate(attachments, start=1)]
    tail = f" Перші {shown} з них прикладено зображеннями в тому самому порядку." if shown else ""
    return "\n\nМЕДІА ІСТОРІЇ, по порядку:" + tail + "\n" + "\n".join(lines)


def fetch_still(url: str) -> str | None:
    """A small JPEG of the picture as a data URL, or None if it does not come."""
    try:
        response = requests.get(url, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=JPEG_QUALITY)
    except Exception as error:
        logging.info("No still for the author from %s: %s", url, error)
        return None
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def stills(attachments: list[Attachment]) -> list[str]:
    """Data URLs of the first few attachments that have something to look at.

    In attachment order and stopping at the first one missing, so the n-th
    image is always the n-th line of `media_note`.
    """
    found: list[str] = []
    for item in attachments[:MAX_PICTURES]:
        if not item.still_url:
            break
        data = fetch_still(item.still_url)
        if data is None:
            break
        found.append(data)
    return found


def with_stills(message: dict[str, Any], data_urls: list[str]) -> dict[str, Any]:
    """The user message with the pictures after its text.

    At the very end: the system message and the text before the pictures stay
    the cacheable prefix they were.
    """
    if not data_urls:
        return message
    parts: list[dict[str, Any]] = [{"type": "text", "text": message["content"]}]
    parts.extend(
        {"type": "image_url", "image_url": {"url": url, "detail": "low"}} for url in data_urls
    )
    return {**message, "content": parts}


def parse_media(raw: Any, count: int) -> dict[str, Any] | None:
    """The author's decision on the media, or None when it gave none usable.

    `place`: "show" in the body, "fold" under "Докладніше", "none" not at all.
    `first`: which attachment leads (1-based). `notes`: what each picture
    shows, kept for the log and for a caption later.
    """
    if not isinstance(raw, dict) or count == 0:
        return None
    place = raw.get("place")
    if place not in ("show", "fold", "none"):
        return None
    first = raw.get("first")
    decision: dict[str, Any] = {"place": place}
    if isinstance(first, int) and not isinstance(first, bool) and 1 <= first <= count:
        decision["first"] = first
    notes = raw.get("notes")
    if isinstance(notes, list):
        decision["notes"] = [
            {
                "what": str(note.get("what") or "")[:200],
                "kind": str(note.get("kind") or ""),
                "sensitive": note.get("sensitive") is True,
            }
            for note in notes[:MAX_PICTURES]
            if isinstance(note, dict)
        ]
    return decision
