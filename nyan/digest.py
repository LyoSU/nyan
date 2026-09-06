"""The periodic digest: one post collecting everything published since the last.

Four things make this more than a loop over clusters.

It reads what НЯН already wrote rather than what the channels wrote. Every
cluster carries a headline and, if several sources covered it, a summary that
has already been paid for, cleaned of boilerplate and checked for invented
facts. Feeding the raw channel posts back to the model would pay twice and get
a worse answer.

It publishes to its own channel. The digest is a different product from the
feed — someone who wants both subscribes to both — so it goes to its own issue
and nothing is mirrored.

It never loses a shift. The window runs from the last published digest to now,
so a quiet shift that did not meet the minimum is not dropped: its posts simply
appear in the next digest, because the watermark only advances when something
is actually published.

It knows what the last digest said. A long story reaches the reader as several
posts across several shifts — the strike, then the confirmed toll — so the
headlines of the previous digest go into the prompt. Without them every later
stage is written as if the event had just happened.

It goes out as ordinary HTML, not as a rich message. A digest is a page of
one-line headlines under a few bold section names, and rich blocks give every
heading, list item and divider its own padding: twenty headlines became three
screens, which is the opposite of what a digest is for. HTML has everything
the digest uses — bold, links, a quotation, an expandable quote for the tail
of a busy shift — at the density of a plain message. The feed posts stay rich,
because they use what rich buys: a slideshow, collapsed sources, a dim footer.
"""

import argparse
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, cast

from nyan.client import TelegramClient
from nyan.clusters import Clusters, render_prompt_files
from nyan.logs import setup_logging
from nyan.mongo import get_topics_collection
from nyan.openai import openai_completion, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from nyan.markup import link_emphasis, parse_markup, strip_markup, to_html
from nyan import summary as nyan_summary
from nyan.summary import DIGEST_LIMITS, SUBHEADING, Summary, SummaryBlock, parse_summary
from nyan.util import (
    PUBLISH_CHANNEL_URL,
    format_date_uk,
    format_dt_uk,
    format_period_uk,
    get_current_ts,
    ts_to_dt,
)


BASE_DIR = Path(__file__).parent

# Issue the digest is published to. A separate channel: the digest and the feed
# are different products, and a reader who wants both subscribes to both.
DIGEST_ISSUE = "digest"

# Where that channel is configured. An entry in the client config wins — that is
# where a separate bot token would go — and this is the shortcut for the common
# case of one bot posting to two channels. A public channel can be named by its
# username: DIGEST_CHANNEL_ID=@ShortUA.
DIGEST_CHANNEL_ID = os.getenv("DIGEST_CHANNEL_ID") or ""

# The marker in front of every headline. Not an HTML list, because Telegram
# has none: a marker is what separates one wrapped headline from the next on
# a phone, where most of them take two lines.
BULLET = "•"

# Telegram's limit on the text of one message, entities excluded. A digest of
# forty posts stays well under it, but the check is what turns a rare overflow
# into a shorter post rather than a 400 and a lost shift.
MAX_MESSAGE_LENGTH = 4096

# How far back the window may stretch when digests keep failing to publish.
# Without a bound, a week of quiet shifts would eventually build a prompt too
# large to send and a digest nobody would read.
MAX_CATCHUP_HOURS = 48

# Cluster text sent to the model, in characters. The summary is short already;
# this only bounds the fallback path, where a raw channel post can be long.
MAX_CLUSTER_TEXT = 600


def read_last_digest(mongo_config_path: str) -> dict[str, Any] | None:
    """The digest published most recently, or None before the first one."""
    collection = get_topics_collection(mongo_config_path)
    record = collection.find_one(
        {"published_until": {"$exists": True}}, sort=[("published_until", -1)]
    )
    return dict(record) if record else None


def read_watermark(mongo_config_path: str) -> int | None:
    """When the last published digest stopped counting."""
    last = read_last_digest(mongo_config_path)
    if not last:
        return None
    watermark = last.get("published_until")
    return int(watermark) if watermark else None


def previous_form(record: dict[str, Any] | None) -> dict[str, Any]:
    """What the last digest already told the reader — headlines and nothing else.

    A long story runs across several digests: the strike lands in one shift and
    the confirmed toll arrives in the next, as a separate post with a link of its
    own. The watermark keeps that post out of two digests, but it cannot tell the
    model that the reader already knows what happened, so without this every
    later stage is written as if the event were new.

    Deliberately no bodies. A fact from the previous digest belongs to no link in
    this one, and a digest may only state what the posts it lists actually say —
    so the model is given enough to recognize a continuation and not enough to
    describe one.
    """
    if not record:
        return {}
    summary = Summary.fromdict(record.get("summary") or {})
    return {
        "headline": strip_markup(summary.headline),
        "topics": [
            strip_markup(block.text)
            for block in summary.blocks
            if block.type == SUBHEADING
        ],
        # Markup stripped because in a digest headline the ** span picks the link
        # anchor: left in, it would teach the model to mark up its own headlines
        # by copying, in places where the asterisks mean something else.
        "headlines": [
            strip_markup(link["text"])
            for block in summary.blocks
            if block.type != nyan_summary.TEXT
            for link in block.links
        ]
        # A lede covers its posts in one or two sentences rather than a line
        # each, so the sentences are what the reader saw.
        + [
            strip_markup(block.text)
            for block in summary.blocks
            if block.type == nyan_summary.TEXT and block.links
        ],
    }


def window(mongo_config_path: str, duration_hours: float, now: int) -> tuple[int, int]:
    """The half-open range of creation times this digest covers.

    Starts where the last published digest stopped, so nothing falls between
    two digests and nothing is counted twice. Falls back to `duration_hours`
    on the first ever run, and is clamped so a long silence cannot produce an
    unbounded window.
    """
    default_start = now - int(duration_hours * 3600)
    watermark = read_watermark(mongo_config_path)
    if watermark is None:
        return default_start, now

    earliest = now - int(MAX_CATCHUP_HOURS * 3600)
    if watermark < earliest:
        logging.warning(
            "Last digest covered up to %s, further back than %d hours: starting there instead",
            format_dt_uk(ts_to_dt(watermark)),
            MAX_CATCHUP_HOURS,
        )
        return earliest, now
    if watermark >= now:
        return default_start, now
    return watermark, now


def collect_clusters(
    mongo_config_path: str, start_ts: int, end_ts: int, issue_name: str
) -> list[dict[str, Any]]:
    """The published posts in the window, as НЯН wrote them."""
    clusters_obj = Clusters.load_from_mongo(
        mongo_config_path, end_ts, end_ts - start_ts, until_ts=end_ts
    )
    clusters = list(clusters_obj.clid2cluster.values())
    clusters.sort(key=lambda cl: cl.create_time if cl.create_time else 0)

    collected = []
    for cluster in clusters:
        messages = [m for m in cluster.messages if m.issue == issue_name]
        if not messages:
            continue
        # What the post says, not what the channel said: already summarized,
        # already stripped of boilerplate, already paid for.
        summary = cluster.stored_summary
        text = (
            summary.as_text()
            if summary
            else (cluster.annotation_doc.patched_text or "")
        )
        collected.append(
            {
                "url": f"{PUBLISH_CHANNEL_URL}/{messages[0].message_id}",
                "headline": cluster.stored_headline or "",
                "text": " ".join(text.split())[:MAX_CLUSTER_TEXT],
                "views": cluster.views,
                "sources_count": len(cluster.channels),
            }
        )
    return collected


def write_digest(
    clusters: list[dict[str, Any]],
    prompt_path: str,
    model_name: str,
    period: str,
    today: str = "",
    previous: dict[str, Any] | None = None,
) -> Summary:
    # The date the digest is written on. Without it the model cannot tell that
    # "до кінця року" and "до кінця 2026 року" name the same deadline, and
    # writes both — the sort of line a reader spots immediately.
    today = today or format_date_uk(ts_to_dt(get_current_ts()))
    # Rules and material as two messages, the same split the per-story prompts
    # use. The texts below are our own summaries, but they were written from
    # posts other people wrote, so they arrive as the user half and fenced —
    # and the rules, which are the whole of the system half, stay one prefix.
    system_path = Path(prompt_path)
    messages = render_prompt_files(
        system_path,
        system_path.with_name(f"{system_path.stem}_input.txt"),
        clusters=clusters,
        period=period,
        today=today,
        # Always a mapping: the template asks for `previous.headlines`, and
        # an undefined name would raise instead of skipping the section.
        previous=previous or {},
    )

    try:
        content = openai_completion(
            messages=cast(list[dict[str, Any]], messages),
            model_name=model_name or DEFAULT_MODEL,
            response_format={"type": "json_object"},
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            prompt_cache_key="digest",
        )
        content = content[content.find("{") : content.rfind("}") + 1]
        raw = json.loads(content)
    except Exception:
        logging.exception("Digest generation failed")
        return Summary()

    # Only links the model was given may survive: an invented one sends the
    # reader to a post that does not exist.
    return parse_summary(
        raw,
        context="digest",
        allowed_urls={cluster["url"] for cluster in clusters},
        limits=DIGEST_LIMITS,
    )


def format_span(start_ts: int, end_ts: int) -> str:
    """Which shift the digest covers: "1 вересня, 14:00–18:00".

    The span, not just a date: two digests a day share one. The date is named
    once when both ends fall on it, since "1 вересня, 14:00 — 1 вересня, 18:00"
    makes a reader check whether the two dates differ.
    """
    start, end = ts_to_dt(start_ts), ts_to_dt(end_ts)
    if start.date() == end.date():
        return f"{format_dt_uk(start)}–{end.strftime('%H:%M')}"
    return f"{format_dt_uk(start)} — {format_dt_uk(end)}"


def _headline_lines(links: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{BULLET} {to_html(link_emphasis(link['text'], link['url']))}"
        for link in links
    )


def render_block(block: SummaryBlock) -> str:
    """One of the model's blocks as HTML, or "" for one with nothing to show."""
    kind = block.type
    if kind == nyan_summary.TEXT:
        return to_html(parse_markup(block.text))
    if kind == nyan_summary.SUBHEADING:
        return f"<b>{html.escape(block.text, quote=False)}</b>"
    if kind == nyan_summary.LINKS:
        return _headline_lines(block.links)
    if kind == nyan_summary.LIST:
        return "\n".join(
            f"{BULLET} {html.escape(item, quote=False)}" for item in block.items
        )
    if kind == nyan_summary.QUOTE:
        text = html.escape(block.text, quote=False)
        author = html.escape(block.author, quote=False)
        return f"<blockquote>{text}\n— {author}</blockquote>"
    if kind == nyan_summary.HIDDEN:
        # Expandable: the summary line shows, the rest opens on a tap. The
        # closest HTML has to a rich `details`, and enough for a digest whose
        # tail is a list of headlines the reader may or may not want.
        inside = [to_html(parse_markup(block.text))] if block.text else []
        if block.links:
            inside.append(_headline_lines(block.links))
        summary = html.escape(block.summary, quote=False)
        return "<blockquote expandable><b>{}</b>\n{}</blockquote>".format(
            summary, "\n".join(inside)
        )
    # The attributed kinds never survive parsing here — a digest names no
    # channels — and a disputed line without them is a plain sentence.
    return to_html(parse_markup(block.text)) if block.text else ""


def render_digest(summary: Summary, start_ts: int, end_ts: int, period: str) -> str:
    """The whole digest as one HTML message.

    Blocks are separated by a blank line, except that a section name sits
    directly on top of its headlines: a gap there would make the name look
    like a stray line rather than a label for what follows.
    """
    headline = summary.headline or f"Головне за {period}"
    body = ""
    tight = False
    for block in summary.blocks:
        rendered = render_block(block)
        if not rendered:
            continue
        if body:
            body += "\n" if tight else "\n\n"
        body += rendered
        tight = block.type == nyan_summary.SUBHEADING
    parts = [f"<b>{html.escape(headline, quote=False)}</b>", body]
    # The span in the smallest voice HTML has. Under the text, where a reader
    # who wants it looks and a reader who does not reads past it.
    parts.append(f"<i>{format_span(start_ts, end_ts)}</i>")
    return "\n\n".join(part for part in parts if part)


_TAG = re.compile(r"<[^>]+>")


def visible_length(text: str) -> int:
    """How much of the HTML Telegram counts against its message limit."""
    return len(html.unescape(_TAG.sub("", text)))


def fit_to_limit(
    summary: Summary, start_ts: int, end_ts: int, period: str
) -> tuple[str, int]:
    """The rendered digest, shortened from the end until Telegram will take it.

    Whole blocks go, never half a line, so what remains is still a digest.
    Returns the text and how many blocks were dropped; the caller logs the
    latter, since a digest that quietly lost its last section would hide the
    exact problem this is meant to surface.
    """
    kept = Summary(headline=summary.headline, blocks=list(summary.blocks))
    dropped = 0
    text = render_digest(kept, start_ts, end_ts, period)
    while visible_length(text) > MAX_MESSAGE_LENGTH and kept.blocks:
        kept.blocks.pop()
        dropped += 1
        text = render_digest(kept, start_ts, end_ts, period)
    return text, dropped


def count_missing(summary: Summary, clusters: list[dict[str, Any]]) -> list[str]:
    """Posts the model left out. Not fatal, but worth seeing in the log."""
    listed = {link["url"] for block in summary.blocks for link in block.links}
    return [cluster["url"] for cluster in clusters if cluster["url"] not in listed]


def main(
    mongo_config_path: str,
    client_config_path: str,
    duration_hours: float,
    max_news_count: int,
    min_news_count: int,
    issue_name: str,
    digest_issue_name: str,
    prompt_path: str,
    model_name: str,
    auto: bool,
) -> None:
    setup_logging()

    now = get_current_ts()
    start_ts, end_ts = window(mongo_config_path, duration_hours, now)
    logging.info(
        "Digest window: %s — %s",
        format_dt_uk(ts_to_dt(start_ts)),
        format_dt_uk(ts_to_dt(end_ts)),
    )

    clusters = collect_clusters(mongo_config_path, start_ts, end_ts, issue_name)
    if len(clusters) < min_news_count:
        # The watermark stays where it is, so these posts are not lost: they
        # will be part of the next digest instead.
        logging.info(
            "Only %d posts in the window, need %d: leaving them for the next digest",
            len(clusters),
            min_news_count,
        )
        return
    if len(clusters) > max_news_count:
        logging.info(
            "Digesting the %d most recent of %d posts", max_news_count, len(clusters)
        )
        clusters = clusters[-max_news_count:]

    # The period as a headline would name it, from the window actually covered
    # rather than from the requested duration: a digest that caught up after a
    # quiet shift covers more than eight hours and should say so.
    period = format_period_uk((end_ts - start_ts) / 3600)
    previous = previous_form(read_last_digest(mongo_config_path))
    if previous.get("headlines"):
        logging.info(
            "Previous digest listed %d posts, passing their headlines for context",
            len(previous["headlines"]),
        )
    summary = write_digest(
        clusters,
        prompt_path=prompt_path,
        model_name=model_name,
        period=period,
        today=format_date_uk(ts_to_dt(end_ts)),
        previous=previous,
    )
    if not summary:
        logging.warning("Nothing usable came back, leaving the window for next time")
        return

    missing = count_missing(summary, clusters)
    if missing:
        logging.warning("Digest left out %d of %d posts", len(missing), len(clusters))

    text, dropped = fit_to_limit(summary, start_ts, end_ts, period)
    if dropped:
        logging.warning(
            "Digest exceeded %d characters, dropped its last %d blocks",
            MAX_MESSAGE_LENGTH,
            dropped,
        )

    should_publish = auto
    if not auto:
        print(json.dumps(summary.asdict(), ensure_ascii=False, indent=2))
        should_publish = input("Publish? y/n ").strip() == "y"
    if not should_publish:
        return

    with TelegramClient(client_config_path) as client:
        if digest_issue_name not in client.issues:
            if not DIGEST_CHANNEL_ID:
                logging.error(
                    "No '%s' issue in the client config and no DIGEST_CHANNEL_ID set, "
                    "so there is nowhere to publish",
                    digest_issue_name,
                )
                return
            client.clone_issue(digest_issue_name, DIGEST_CHANNEL_ID, like=issue_name)
        message = client.send_message(text, issue_name=digest_issue_name)
    if message is None:
        # Nothing was published, so nothing has been digested: keeping the
        # watermark means the next run tries the same posts again.
        logging.error("Digest was not published, watermark left in place")
        return

    collection = get_topics_collection(mongo_config_path)
    collection.insert_one(
        {
            "clusters": clusters,
            "summary": summary.asdict(),
            "published_from": start_ts,
            "published_until": end_ts,
            "message_id": message.message_id,
        }
    )
    logging.info("Digest published as message %d", message.message_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-config-path", type=str, required=True)
    parser.add_argument("--client-config-path", type=str, required=True)
    parser.add_argument("--duration-hours", type=float, default=8)
    parser.add_argument("--max-news-count", type=int, default=40)
    parser.add_argument("--min-news-count", type=int, default=5)
    parser.add_argument("--issue-name", type=str, default="main")
    parser.add_argument("--digest-issue-name", type=str, default=DIGEST_ISSUE)
    parser.add_argument(
        "--prompt-path", type=str, default=str(BASE_DIR / "prompts/digest.txt")
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--auto", default=False, action="store_true")
    args = parser.parse_args()
    main(**vars(args))
