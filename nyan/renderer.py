import copy
import logging
import os
import re
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader

from nyan import rich
from nyan import summary as nyan_summary
from nyan.channels import GROUP_ORDER, Channels
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.markup import find_links, link_emphasis, parse_markup
from nyan.media import MEDIA_ANIMATION, MEDIA_PHOTO, MEDIA_VIDEO, MediaItem
from nyan.restatement import (
    LEDE_RESTATED,
    QUOTED_RESTATED,
    find_sentence_end,
    restatement,
)
from nyan.rich import Block, RenderedPost, RichText
from nyan.summary import Summary, SummaryBlock
from nyan.util import (
    DEFAULT_TIMEZONE,
    UK_MONTHS_GENITIVE,
    get_current_ts,
    normalize_channel_id,
    ts_to_dt,
)


# Used to derive a headline from the text itself when the LLM did not supply
# one — because the call failed, or the cluster predates headlines. Where the
# first sentence ends is `nyan.restatement.find_sentence_end`'s call.
MAX_DERIVED_HEADLINE_LENGTH = 120

# Titles for the two attributed blocks. Both are stated as a relation between
# the sources rather than as a label on the text, because that relation is the
# finding: one says the sources contradict each other, the other says a claim
# stands on fewer sources than the post does. A reader skimming has to see which
# of the two it is before reading the sentence under it.
DISPUTED_TITLE = "Джерела різняться"
ATTRIBUTED_TITLE = "Пишуть окремі джерела"

# Heading size of the post's headline, 1-6, where 1 is the largest. 3 is the
# smallest size that still reads as a headline rather than as emphasized body
# text — checked against the whole ladder in a real client. The section heading
# derives from it, so tuning this one keeps the hierarchy intact.
#
# Every headline gets this size, a breaking story included. A bigger heading on
# important clusters used to say "breaking", but the flag behind it fires on
# whichever story happened to gather four channels inside a quarter of an hour,
# which is most of them — so the size varied for a reason no reader could infer
# from the post, and read as a rendering glitch rather than as emphasis.
DEFAULT_HEADLINE_SIZE = 3

# How a newsroom labels an attribution under a picture. The label matters as
# much as the names: a bare list of channels under a carousel reads as a
# caption about the story, while "Фото: УНІАН" says what the names are — whose
# file this is, not who the story is about.
PHOTO_CREDIT_LABEL = "Фото"
VIDEO_CREDIT_LABEL = "Відео"
MIXED_CREDIT_LABEL = "Фото і відео"

# An em dash, the way print attributes a passage to its author. Not a hyphen:
# "- Укрінформ" reads as a bullet point, which is the wrong signal entirely.
CREDIT_DASH = "—"

# The tier whose share of a story's coverage is worth stating outright.
ANONYMOUS_GROUP = "grey"

# Below this many sources the share is noise: "з 2 джерел 1 — анонімний" tells a
# reader nothing they cannot see by reading the two names above it.
MIN_SOURCES_FOR_COMPOSITION = 4

MIN_HEADING_SIZE = 1
MAX_HEADING_SIZE = 6


def clamp_heading_size(size: int) -> int:
    return max(MIN_HEADING_SIZE, min(MAX_HEADING_SIZE, size))


LIST_BLOCK = "list"
PARAGRAPH_BLOCK = "paragraph"


def media_position(story: Sequence[Block], after_lede: int) -> int:
    """Where the attachments go, given where the lede ends.

    Past the list that follows the lede, because the two are one telling —
    first in prose, then itemised — and a carousel wedged between them leaves
    the bullets pressed under the picture, which reads as a cramped block
    rather than as a list.

    It stops before a paragraph that introduces a list of its own, though: that
    paragraph is a bold label ("Джерела різняться") and the list under it is
    what it labels. Media dropped in between separates the two.
    """
    index = after_lede
    while index < len(story) and story[index].get("type") == LIST_BLOCK:
        index += 1
    return index


def media_credit(media: Sequence[MediaItem]) -> RichText | None:
    """Whose files these are, named once for the whole attachment block.

    Every channel in the carousel, in the order its frame appears, and each one
    only once — a channel with two frames is still one source. The credit names
    who published the copy on screen, not whoever photographed the scene: where
    the rendition came from is what is actually known here.
    """
    seen: set[str] = set()
    parts: list[RichText] = []
    for item in media:
        if not item.channel_title or not item.source_url:
            continue
        if item.channel_title in seen:
            continue
        seen.add(item.channel_title)
        parts.append(rich.link(item.channel_title, item.source_url))
    if not parts:
        return None
    kinds = {item.type for item in media}
    moving = bool(kinds & {MEDIA_VIDEO, MEDIA_ANIMATION})
    still = MEDIA_PHOTO in kinds
    if moving and still:
        label = MIXED_CREDIT_LABEL
    elif moving:
        label = VIDEO_CREDIT_LABEL
    else:
        label = PHOTO_CREDIT_LABEL
    return [f"{label}: ", *rich.join(parts, ", ")]


def claim_credit(
    channel_ids: Sequence[str], sources: Mapping[str, RichText]
) -> RichText:
    """Who said it, in the same form the source list names them.

    Unresolvable ids are skipped rather than printed raw: a bare `some_channel`
    in the middle of a sentence is noise to a reader who cannot click it. The
    caller drops the claim if nothing survives here.
    """
    parts = [sources[cid] for cid in channel_ids if cid in sources]
    return rich.join(parts, ", ")


def claim_items(
    claims: Sequence[Mapping[str, Any]], sources: Mapping[str, RichText]
) -> list[Sequence[Block]]:
    """Claims as list items, each ending in the channels behind it.

    The credit sits at the end of the line, the way a byline follows a passage —
    put in front, the channel names would be read as the subject of the sentence.
    """
    items: list[Sequence[Block]] = []
    for claim in claims:
        credit = claim_credit(claim.get("channels", ()), sources)
        if not credit:
            continue
        text = str(claim.get("text", "")).rstrip(".")
        items.append([rich.paragraph(rich.join([text, credit], f" {CREDIT_DASH} "))])
    return items


def lede_index(blocks: Sequence[SummaryBlock]) -> int | None:
    """Which block is the lede: the paragraph the reader meets under the headline.

    The first block when it is prose, or the second when a quotation leads — a
    quote lead is legitimate, and the paragraph after it is still the first
    thing said in the post's own voice, right under the bold line. Anything
    further down has other blocks between it and the headline, so repeating the
    headline there is a different defect and not this function's business.
    """
    for index in range(min(2, len(blocks))):
        if blocks[index].type == nyan_summary.TEXT:
            return index
        if blocks[index].type != nyan_summary.QUOTE:
            return None
    return None


def unrestated(summary: Summary) -> Summary:
    """`summary` without a lede sentence that says the headline again.

    The headline sits in bold directly above the lede, and a lede that opens by
    restating it — the same subject, the same verb, the same object with a
    clause of padding — reads as the post stuttering. The prompt forbids this
    and the model still does it, so the renderer cuts the sentence: what is left
    of the paragraph is what the reader did not already get from the headline,
    and if nothing is left the paragraph goes and the headline carries the post.

    Only the first sentence is measured. A lede that restates and then goes on
    keeps its continuation; a second sentence is never checked, because by then
    the reader is past the headline.
    """
    if not summary.headline:
        return summary
    blocks = list(summary.blocks)
    index = lede_index(blocks)
    if index is None:
        return summary
    lede = blocks[index]
    end = find_sentence_end(lede.text)
    sentence, rest = (lede.text, "") if end is None else (lede.text[:end], lede.text[end:])
    if restatement(summary.headline, sentence) < LEDE_RESTATED:
        return summary
    logging.info(
        "Cutting a lede sentence that restates the headline %r: %r",
        summary.headline,
        sentence,
    )
    rest = rest.strip()
    if rest:
        blocks[index] = SummaryBlock(type=nyan_summary.TEXT, text=rest, links=find_links(rest))
    else:
        del blocks[index]
    return Summary(headline=summary.headline, blocks=blocks)


def summary_blocks(
    summary: Summary,
    section_size: int,
    sources: Mapping[str, RichText] | None = None,
    layout: bool = False,
) -> list[Block]:
    """The model's blocks as Telegram blocks.

    With `layout` the author's own layout choices are kept: a table, a folded
    quotation, folded claims. Without it each is set the way the old template
    sets what is closest: rows as a list, a quotation open, claims open.

    The model chooses the shape — neither a news story nor a digest comes in one
    shape — and this is where its vocabulary maps onto the API's. Unknown kinds
    cannot arrive here: `nyan.summary` has already dropped them. A story and a
    digest share this function because they share the vocabulary; only the
    frame around it differs.
    """
    by_channel = sources or {}
    blocks: list[Block] = []
    for block in summary.blocks:
        if block.type == nyan_summary.TEXT:
            blocks.append(rich.paragraph(parse_markup(block.text)))
        elif block.type == nyan_summary.LIST:
            blocks.append(
                rich.bullet_list(*[[rich.paragraph(item)] for item in block.items])
            )
        elif block.type == nyan_summary.LINKS:
            # Only the phrase the model marked up is the link. A digest where
            # every headline is blue from end to end emphasizes nothing, which
            # is what linking whole headlines produced.
            blocks.append(
                rich.bullet_list(
                    *[
                        [rich.paragraph(link_emphasis(link["text"], link["url"]))]
                        for link in block.links
                    ]
                )
            )
        elif block.type == nyan_summary.QUOTE:
            if layout and block.fold:
                blocks.append(rich.expandable_blockquote(block.text, credit=block.author))
            else:
                blocks.append(
                    rich.blockquote(rich.paragraph(block.text), credit=block.author)
                )
        elif block.type == nyan_summary.TABLE:
            if layout:
                blocks.append(rich.table(block.rows, header=block.header or None))
            else:
                blocks.append(
                    rich.bullet_list(*[[rich.paragraph(" — ".join(row))] for row in block.rows])
                )
        elif block.type == nyan_summary.HIDDEN:
            inside: list[Block] = []
            if block.text:
                inside.append(rich.paragraph(parse_markup(block.text)))
            if block.links:
                inside.append(
                    rich.bullet_list(
                        *[
                            [rich.paragraph(link_emphasis(link["text"], link["url"]))]
                            for link in block.links
                        ]
                    )
                )
            blocks.append(rich.details(block.summary, *inside))
        elif block.type == nyan_summary.SUBHEADING:
            blocks.append(rich.heading(block.text, size=section_size))
        elif block.type in (nyan_summary.DISPUTED, nyan_summary.ATTRIBUTED):
            title = (
                DISPUTED_TITLE
                if block.type == nyan_summary.DISPUTED
                else ATTRIBUTED_TITLE
            )
            items = claim_items(block.claims, by_channel)
            if items:
                # Title on its own line above the versions, not a prefix to the
                # first one: with two or more sides there is no single sentence
                # for it to introduce, and a bullet that starts with bold text
                # reads as the heading of the list rather than as a member of it.
                if layout and block.fold:
                    blocks.append(rich.details(title, rich.bullet_list(*items)))
                else:
                    blocks.append(rich.paragraph(rich.bold(title)))
                    blocks.append(rich.bullet_list(*items))
            elif block.text:
                # The old unattributed shape, still stored on earlier posts.
                blocks.append(
                    rich.paragraph(rich.join([rich.bold(title), block.text], ": "))
                )
    return blocks


def pluralize_sources(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "джерело"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "джерела"
    return "джерел"


# A quoted channel's text longer than this is collapsed in a minor post.
MINOR_QUOTE_FROM = 200


QUOTE_MARKS = re.compile(r"^(?:[ \t]*>[ \t]?)+", re.MULTILINE)


def without_quote_marks(text: RichText) -> RichText:
    """The channel's own "> " markers taken off the start of every line.

    A channel quoting someone writes "> " in front of each line; inside a
    quotation of ours they are a second, literal quotation mark. Only the
    starts of lines are touched, and a line can start inside a later piece of
    a mixed text, so the pieces are walked in order with that in mind.
    """
    at_line_start = True

    def strip(value: RichText) -> RichText:
        nonlocal at_line_start
        if isinstance(value, str):
            if not value:
                return value
            if at_line_start:
                value = QUOTE_MARKS.sub("", value)
            else:
                head, newline, tail = value.partition("\n")
                value = head + newline + (QUOTE_MARKS.sub("", tail) if newline else "")
            if value:
                at_line_start = value.endswith("\n")
            return value
        if isinstance(value, list):
            return [strip(item) for item in value]
        if isinstance(value, dict) and "text" in value:
            return {**value, "text": strip(value["text"])}
        return value

    return strip(text)


def normalized(text: str) -> str:
    """Lower case, letters and digits only, single spaces: for comparing wording."""
    text = text.lower().replace("’", "").replace("'", "")
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def said_in(change: str, headline: str) -> bool:
    """Whether the change line would only repeat the headline."""
    short = normalized(change)
    return bool(short) and short in normalized(headline)


def original_headline(cluster: Cluster) -> str | None:
    """The headline the post first went out with, if it is known.

    The first revision a rewrite records is the published version, timed at
    the post's creation. Older ones fall off past MAX_REVISIONS; then the
    start is no longer known and is not made up.
    """
    analysis = cluster.saved_analysis or {}
    revisions = analysis.get("revisions") or []
    if revisions:
        first = revisions[0]
        created = cluster.create_time or 0
        if abs(int(first.get("time") or 0) - created) <= 60 and first.get("headline"):
            return str(first["headline"])
        return None
    return cluster.stored_headline


def is_attributed(block: Block) -> bool:
    """The "Пишуть окремі джерела" title, in the list shape or the old inline one."""
    if block.get("type") != "paragraph":
        return False
    title = rich.bold(ATTRIBUTED_TITLE)
    text = block.get("text")
    return text == title or (isinstance(text, list) and bool(text) and text[0] == title)


# Said on its own line above a breaking post's headline, unless the config says otherwise.
BREAKING_LABEL = "🔴 Важливо"


def is_byline(block: Block) -> bool:
    """The "— Channel" line `render_credit` puts under a quoted text."""
    text = block.get("text")
    if block.get("type") != "paragraph" or not isinstance(text, dict):
        return False
    inner = text.get("text")
    return (
        text.get("type") == "italic"
        and isinstance(inner, list)
        and bool(inner)
        and inner[0] == CREDIT_DASH
    )


def is_sources_block(block: Block) -> bool:
    """The collapsed "12 джерел" list, which closes every rich post."""
    summary = block.get("summary")
    return (
        block.get("type") == "details"
        and isinstance(summary, str)
        and summary.split(" ")[0].isdigit()
    )


class Renderer:
    def __init__(self, config_path: str, channels: Channels) -> None:
        assert os.path.exists(config_path)
        with open(config_path) as r:
            config = json.load(r)

        self.channels = channels

        file_loader = FileSystemLoader(".")
        env = Environment(loader=file_loader)
        self.cluster_template = env.get_template(config["cluster_template"])
        # A named zone rather than a fixed offset, so daylight saving time is
        # handled instead of being an hour wrong for half the year.
        self.tz_name = config.get("tz_name", DEFAULT_TIMEZONE)

        # Where the story is on the web, if it is anywhere. Absent from the
        # config means no site link in the footer at all, which is what a
        # deployment without a site should get.
        self.site_url = (config.get("site_url") or "").rstrip("/")

        # "rich" builds a block tree for sendRichMessage; "legacy" keeps the
        # old HTML caption path, so a bad release can be rolled back by
        # editing the config instead of redeploying.
        self.post_format = config.get("post_format", "rich")
        self.sources_open = config.get("sources_open", False)

        # One knob for the whole post: the section heading is one step below it,
        # so the two cannot cross over when it is retuned.
        self.headline_size = clamp_heading_size(
            int(config.get("headline_size", DEFAULT_HEADLINE_SIZE))
        )
        self.section_size = clamp_heading_size(self.headline_size + 1)

        # How the editor's decisions show in the post. Every part is off unless
        # the config turns it on, and with all of them off a post renders
        # exactly as it did before the editor.
        presentation = config.get("presentation") or {}
        self.show_levels = bool(presentation.get("levels"))
        self.show_updates = bool(presentation.get("update_line"))
        self.show_storyline = bool(presentation.get("more_on_this"))
        self.quote_parent = bool(presentation.get("quote_parent"))
        # The legend of source marks on short lines, not one that runs off a
        # phone screen.
        self.short_legend = bool(presentation.get("short_legend"))
        # The line above a post the editor called important. "Важливо", not
        # "Терміново": the post is read hours later, when "urgent" is no
        # longer true and "important" still is.
        self.breaking_label = str(presentation.get("breaking_label") or BREAKING_LABEL)
        # The author lays out the body (tables, folded quotes, where the media
        # goes) and, with `vision`, sees the pictures it decides on. A post
        # laid out this way carries the old-template version of itself for
        # Telegram to fall back on.
        self.layout = bool(presentation.get("layout"))
        self.vision = bool(presentation.get("vision")) and self.layout
        # Where each issue's channel lives, to link its earlier posts. An issue
        # with no address here gets no "Раніше на цю тему": a link has to lead to the
        # channel the reader is in, and on a host with one channel per rubric
        # the main channel's address would lead elsewhere.
        self.channel_urls: dict[str, str] = {
            issue: url.rstrip("/")
            for issue, url in (presentation.get("channel_urls") or {}).items()
        }

    def render_cluster(
        self, cluster: Cluster, issue_name: str, post_format: str | None = None
    ) -> RenderedPost | None:
        """Render a post, in `post_format` if given, otherwise the configured one.

        Updating an existing message has to use the format that message was sent
        in: a caption cannot be replaced by a block tree, and vice versa.
        """
        groups = self.group_docs(cluster, issue_name)
        if not groups:
            logging.warning(
                "No documents left for issue '%s', skipping cluster", issue_name
            )
            return None

        if (post_format or self.post_format) == "legacy":
            return self.render_legacy_cluster(cluster, groups)
        post = self.present(self.render_rich_cluster(cluster, groups), cluster, issue_name)
        if self.layout:
            # The same post in the old template, from the same answer: sent if
            # Telegram rejects the layout, so no story is lost to it.
            self.layout = False
            try:
                plain = self.present(self.render_rich_cluster(cluster, groups), cluster, issue_name)
            finally:
                self.layout = True
            if plain.blocks != post.blocks:
                post.fallback = plain.blocks
        return post

    def group_docs(
        self, cluster: Cluster, issue_name: str
    ) -> list[tuple[str, list[Document]]]:
        """Bucket the cluster's documents by trust group, one doc per channel."""
        groups: dict[str, list[Document]] = defaultdict(list)
        for doc in cluster.docs:
            if doc.channel_id not in self.channels:
                # A channel removed from channels.json still has documents in
                # Mongo for another day. Without this the whole iteration dies
                # on a KeyError the first time such a cluster is rendered.
                logging.warning(
                    "Channel %s is not in the channel list, skipping %s",
                    doc.channel_id,
                    doc.url,
                )
                continue
            channel = self.channels[doc.channel_id]
            # A channel we watch but do not republish. It stays in the cluster so
            # the site can report that it carried the story; naming it here would
            # be handing it our readers.
            if channel.monitor_only:
                continue
            # A channel with no group for this issue is not republished in this
            # feed — the ordinary case, not a fault: 126 of the 163 channels
            # carry no group for 'war'. Logged at debug because a war cluster
            # otherwise warns once per document for working as configured, and
            # drowns the warning above, which means a channel has disappeared.
            if issue_name not in channel.groups:
                logging.debug(
                    "Channel %s has no group for issue '%s', skipping %s",
                    doc.channel_id,
                    issue_name,
                    doc.url,
                )
                continue
            groups[channel.groups[issue_name]].append(doc)

        used_channels = set()
        for group_name, group_docs in groups.items():
            group_docs.sort(key=lambda x: x.pub_time)
            filtered_group = list()
            for doc in group_docs:
                if doc.channel_id in used_channels:
                    continue
                used_channels.add(doc.channel_id)
                filtered_group.append(doc)
            groups[group_name] = filtered_group

        # Explicit tier order, not alphabetical. Sorting by the group key put
        # "grey" between "blue" and "red", so the reader met the anonymous
        # channels in the middle of the list instead of at the end of a scale.
        return sorted(
            ((name, docs) for name, docs in groups.items() if docs),
            key=lambda x: (
                GROUP_ORDER.index(x[0]) if x[0] in GROUP_ORDER else len(GROUP_ORDER),
                x[0],
            ),
        )

    # ------------------------------------------------------------------ rich

    def render_rich_cluster(
        self, cluster: Cluster, groups: list[tuple[str, list[Document]]]
    ) -> RenderedPost:
        """A post, written from every source if possible, quoted if not.

        The two bodies differ in who wrote them, and the post says so: a
        summary written from the whole cluster carries no byline, because
        crediting one channel for text it did not write would be a lie, while a
        quoted post names the channel whose words these are.
        """
        summary = cluster.summary
        blocks: list[Block] = []

        headline = summary.headline or self.split_headline(cluster)[0]
        if headline:
            # Bold on top of the heading: a client renders a heading in a
            # semibold weight, which at this size reads as body text with a
            # larger font rather than as a headline. Compared side by side, the
            # bold one is the one that looks like a news headline.
            blocks.append(rich.heading(rich.bold(headline), size=self.headline_size))
        media = self.render_media(cluster)
        media_place = "show"
        folded_media: list[Block] = []
        if self.layout and summary:
            decided = (cluster.saved_analysis or {}).get("media") or {}
            media_place = decided.get("place") or "show"
            first = decided.get("first")
            if isinstance(first, int) and first > 1:
                media = self.render_media(cluster, lead=first - 1)
            if media_place == "none":
                media = []
            elif media_place == "fold" and media:
                folded_media, media = media, []

        if summary:
            # What the reader has not already read in the bold line above: a
            # lede that opens by saying the headline again loses that sentence.
            summary = unrestated(summary)
            # Headline, lede, photo: the reader gets what happened before the
            # picture of it, instead of scrolling a slideshow to reach the
            # first sentence. Where the lede ends is `lede_length`'s decision,
            # and `media_position` moves past what must not be split from it.
            story = self.render_summary(summary, groups)
            at = media_position(story, self.lede_length(summary))
            blocks.extend(story[:at])
            blocks.extend(media)
            blocks.extend(story[at:])
            if folded_media:
                blocks.append(rich.details("Фото й відео", *folded_media))
        else:
            blocks.extend(media)
            body = self.split_headline(cluster)[1]
            if body:
                blocks.append(rich.paragraph(body))
            blocks.append(self.render_credit(cluster))

        blocks.append(self.render_sources(cluster, groups))
        blocks.append(rich.divider())
        blocks.append(self.render_footer(cluster))

        return RenderedPost(blocks=blocks)

    # ------------------------------------------------------------ presentation

    def present(self, post: RenderedPost, cluster: Cluster, issue_name: str) -> RenderedPost:
        """The editor's decisions on top of the rendered post, as configured.

        Applied to the finished block list rather than threaded through the
        rendering, so the post with every option off is the post as it was.
        """
        assert post.blocks is not None
        blocks = list(post.blocks)
        if self.show_levels:
            # The author saw the pictures: one it put in the body stays there
            # even in a short post — the coin is the news of a coin.
            shown = self.layout and (
                ((cluster.saved_analysis or {}).get("media") or {}).get("place") == "show"
            )
            blocks = self.render_level(blocks, cluster.level, keep_media=shown)
            post.notify = cluster.level == "breaking"

        # The headline is a heading, or the first paragraph of a minor post.
        headline_at = next(
            (i for i, block in enumerate(blocks) if block.get("type") == "heading"),
            None,
        )
        if headline_at is None and self.show_levels and cluster.level == "minor":
            headline_at = 0
        # The moment the post went out, against which its dates are said.
        published = cluster.create_time or get_current_ts()
        if self.show_updates and cluster.updates:
            at = 0 if headline_at is None else headline_at + 1
            blocks.insert(
                at,
                self.render_update_line(
                    cluster.updates[-1], published, cluster.stored_headline or ""
                ),
            )

        before_sources: list[Block] = []
        if self.show_updates and len(cluster.updates) > 1:
            before_sources.append(self.render_update_history(cluster, published))
        if self.show_storyline and cluster.storyline_posts:
            before_sources.extend(
                self.render_more_on_this(cluster.storyline_posts, issue_name, published)
            )
        sources_at = next(
            (i for i, block in enumerate(blocks) if is_sources_block(block)), len(blocks)
        )
        blocks[sources_at:sources_at] = before_sources

        if self.quote_parent and cluster.reply_to_clid and cluster.reply_to_headline:
            # Quoted exactly as the parent shows it — bold inside its heading —
            # or Telegram will not find the quote. The client sends the reply
            # without it if the parent's headline has since been rewritten.
            post.reply_quote = [rich.bold(cluster.reply_to_headline)]

        post.blocks = blocks
        return post

    def render_level(
        self, blocks: list[Block], level: str, keep_media: bool = False
    ) -> list[Block]:
        """Breaking gets a label and a sound; minor is set short.

        The headline keeps its size at every level: size used to signal
        "breaking" and read as a glitch, because the flag behind it fired on
        most stories. The label says it in words, and only the editor sets it.
        A minor story keeps its first paragraph and folds the rest away, and
        a quoted channel's text goes into a collapsed quotation under its
        name, so the post takes two or three lines of the feed.
        """
        if level == "breaking":
            # 🔴, not ⚡: the headline prompts put ⚡ on power outages, so a
            # reader already takes it for energy, and an outage post would show
            # two. No `marked`: highlighted, the line reads as a small badge.
            return [rich.paragraph(rich.bold(self.breaking_label)), *blocks]
        if level != "minor":
            return blocks
        result: list[Block] = []
        # Photos go under "Докладніше", first: a square photo in the body made
        # the two-line story the tallest post of the feed.
        media: list[Block] = []
        rest: list[Block] = []
        seen_paragraph = False
        skip_list = False
        for block in blocks:
            kind = block.get("type")
            if skip_list:
                skip_list = False
                if kind == "list":
                    continue
            if kind == "heading":
                result.append(rich.paragraph(block["text"]))
            elif kind == "slideshow" or kind in rich.MEDIA_BLOCK_TYPES:
                (result if keep_media else media).append(block)
            elif is_attributed(block):
                # What single channels add is a third level of detail; for a
                # minor story the sources are one tap away anyway.
                skip_list = True
            elif kind == "paragraph" and (not seen_paragraph or is_byline(block)):
                seen_paragraph = True
                result.append(block)
            elif kind in ("details", "divider", "footer"):
                if media or rest:
                    result.append(rich.details("Докладніше", *media, *rest))
                    media, rest = [], []
                result.append(block)
            else:
                rest.append(block)
        if media or rest:
            result.append(rich.details("Докладніше", *media, *rest))
        return self.quote_minor_body(result)

    @staticmethod
    def quote_minor_body(blocks: list[Block]) -> list[Block]:
        """A one-channel post's text, then its byline, as a collapsed quotation."""
        for i in range(len(blocks) - 1):
            body, credit = blocks[i], blocks[i + 1]
            if (
                body.get("type") == "paragraph"
                and is_byline(credit)
                and len(str(body.get("text"))) > MINOR_QUOTE_FROM
            ):
                byline = credit["text"]["text"]
                if isinstance(byline, list) and byline and byline[0] == CREDIT_DASH:
                    byline = byline[-1]
                return [
                    *blocks[:i],
                    rich.expandable_blockquote(without_quote_marks(body["text"]), credit=byline),
                    *blocks[i + 2 :],
                ]
        return blocks

    # Dates in the presentation are plain text in Kyiv time, not `date_time`
    # entities: a client formats those in its own language ("nov 10 15:41"),
    # and the feed is Ukrainian. Absolute dates, never "сьогодні"/"вчора": a
    # post is read for days after it went out, and "вчора" would go stale.

    def say_date(self, moment: int, published: int) -> str:
        """ "10 листопада", with the year only when it is not the post's year."""
        when = ts_to_dt(moment, self.tz_name)
        date = f"{when.day} {UK_MONTHS_GENITIVE[when.month - 1]}"
        if when.year != ts_to_dt(published, self.tz_name).year:
            date += f" {when.year}"
        return date

    def say_moment(self, moment: int, published: int) -> str:
        """ "10 листопада, 15:41"."""
        return f"{self.say_date(moment, published)}, {ts_to_dt(moment, self.tz_name):%H:%M}"

    def say_update_time(self, moment: int, published: int) -> str:
        """ "о 14:38" on the day the post went out, "12 листопада о 09:10" after."""
        when = ts_to_dt(moment, self.tz_name)
        if when.date() == ts_to_dt(published, self.tz_name).date():
            return f"о {when:%H:%M}"
        return f"{self.say_date(moment, published)} о {when:%H:%M}"

    def render_update_line(
        self, update: Mapping[str, Any], published: int, headline: str = ""
    ) -> Block:
        """"оновлено о 14:20: загиблих троє" — the edit, said out loud.

        Only the time when the change is what the rewritten headline already
        says: "Загинула 44-річна жінка" stood in the headline, the line and the
        table, three times over.
        """
        when = self.say_update_time(int(update["time"]), published)
        change = str(update.get("text") or "")
        if not change or said_in(change, headline):
            return rich.paragraph(rich.italic(f"оновлено {when}"))
        return rich.paragraph(rich.italic(f"оновлено {when}: {change}"))

    def render_update_history(self, cluster: Cluster, published: int) -> Block:
        """Every update in a folded table, starting from the post as it first went out.

        The first row is the original headline: by now the headline has been
        rewritten, and without it the reader cannot see where the story began.
        """
        rows = []
        original = original_headline(cluster)
        if original is not None:
            rows.append([self.say_table_time(published, published), f"опубліковано: {original}"])
        for update in cluster.updates:
            rows.append(
                [self.say_table_time(int(update["time"]), published), str(update["text"])]
            )
        # No width can be set for a column (a table cell has text, spans and
        # alignment only), so the time is kept short instead.
        return rich.details(
            f"Оновлення: {len(cluster.updates)}",
            rich.table(rows, header=["Коли", "Що змінилося"]),
        )

    def say_table_time(self, moment: int, published: int) -> str:
        when = ts_to_dt(moment, self.tz_name)
        if when.date() == ts_to_dt(published, self.tz_name).date():
            return f"{when:%H:%M}"
        return self.say_moment(moment, published)

    def render_more_on_this(
        self, posts: Sequence[Mapping[str, Any]], issue_name: str, published: int
    ) -> list[Block]:
        """Links to the channel's earlier posts of the same storyline.

        The date stays outside the link: only the headline is the thing to tap.
        """
        base = self.channel_urls.get(issue_name)
        if not base:
            return []
        items = [
            [
                rich.paragraph(
                    [
                        f"{self.say_moment(int(post['time'] or 0), published)} — ",
                        rich.link(str(post["headline"]), f"{base}/{post['message_id']}"),
                    ]
                )
            ]
            for post in posts
        ]
        if not items:
            return []
        return [rich.paragraph(rich.bold("Раніше на цю тему")), rich.bullet_list(*items)]

    def render_summary(
        self,
        summary: Summary,
        groups: list[tuple[str, list[Document]]] | None = None,
    ) -> list[Block]:
        return summary_blocks(
            summary,
            section_size=self.section_size,
            sources=self.index_sources(groups) if groups else None,
            layout=self.layout,
        )

    def index_sources(
        self, groups: list[tuple[str, list[Document]]]
    ) -> dict[str, RichText]:
        """How to name each channel when a claim is credited to it.

        Built from the grouped documents rather than from `cluster.docs`, which
        gets the tier glyph for free — and the tier is the point. A detail only
        one channel carries reads differently depending on who that channel is:
        an anonymous aggregator alone on a claim is the shape a planted item
        takes, and a state body alone on one is simply the body announcing its
        own business. The reader can only tell those apart if the mark travels
        with the name.

        `group_docs` has already reduced this to one post per channel, so the
        claim links to the channel's first post in the cluster.
        """
        sources: dict[str, RichText] = {}
        for group, docs in groups:
            emoji = self.channels.group_emoji(group)
            for doc in docs:
                name = self.render_source(doc)
                sources[doc.channel_id] = (
                    rich.join([emoji, name], " ") if emoji else name
                )
        return sources

    @staticmethod
    def lede_length(summary: Summary) -> int:
        """How many rendered blocks the post's lede occupies, for media to follow.

        One if the post opens with a paragraph, zero otherwise — which puts the
        media straight under the headline, as it was before the lede existed.
        Only a paragraph counts: every other opening shape introduces the blocks
        under it, and media dropped in between separates a label from the thing
        it labels. Read from the model's blocks rather than the rendered ones
        because two rendered paragraphs are not the same thing: a `disputed`
        block also renders as a paragraph, holding the bold title of the list
        beneath it.

        The one block returned by `summary_blocks` for a `text` block is what
        makes this a length rather than a search.
        """
        opening = summary.blocks[0].type if summary.blocks else ""
        return 1 if opening == nyan_summary.TEXT else 0

    def split_headline(self, cluster: Cluster) -> tuple[str | None, str | None]:
        """Return (headline, body) for the cluster's text.

        With an LLM headline the full text becomes the body, the way a
        headline and a lede work in print — unless the text is one short
        sentence and the headline is that sentence in nine words, which is the
        shape of most one-channel posts. Then the body is left out: the words
        are the channel's and are quoted whole or not at all, so nothing is
        cut from them, and the headline alone says what the post has to say.
        Without an LLM headline, the first sentence stands in, and the rest
        becomes the body so nothing is said twice.
        """
        text = (cluster.annotation_doc.patched_text or "").strip()
        headline = cluster.headline
        if headline:
            if text and restatement(headline, text) >= QUOTED_RESTATED:
                logging.info(
                    "Leaving out a quoted body that is the headline %r again", headline
                )
                return headline, None
            return headline, text or None
        if not text:
            return None, None

        split_at = find_sentence_end(text)
        if split_at is None or split_at > MAX_DERIVED_HEADLINE_LENGTH:
            return None, text
        return text[:split_at].strip(), text[split_at:].strip() or None


    def render_media(self, cluster: Cluster, lead: int = 0) -> list[Block]:
        """The cluster's attachments as blocks, in the order it chose them.

        Photos and videos are not ranked here. They were chosen together, as one
        list, precisely so that this function and the legacy client cannot
        disagree about which of them a reader sees — which is what happened when
        the video block was returned *instead of* the photos.
        """
        media = list(cluster.media)[: rich.MAX_MEDIA]
        if 0 < lead < len(media):
            # The author saw the pictures and named the one that is the news.
            media.insert(0, media.pop(lead))
        if not media:
            return []
        if len(media) == 1:
            return [self.render_media_item(media[0], credit=media_credit(media))]
        # No credit on the frames: Telegram accepts a caption on a block inside
        # a slideshow and renders it nowhere, so a per-frame byline is silently
        # lost. Only the slideshow's own caption shows, which makes one credit
        # naming every channel in the carousel the available honest answer.
        return [
            rich.slideshow(
                *[self.render_media_item(item) for item in media],
                credit=media_credit(media),
            )
        ]

    @staticmethod
    def render_media_item(item: MediaItem, credit: RichText | None = None) -> Block:
        """One attachment, credited only where a credit is actually rendered.

        Which is not inside a slideshow: Telegram accepts a caption on a nested
        block and shows it nowhere. So the caller passes a credit for a lone
        attachment and none for a frame of a carousel, where the byline sits on
        the carousel instead.
        """
        if item.type == MEDIA_VIDEO:
            return rich.video(item.url, credit=credit)
        if item.type == MEDIA_ANIMATION:
            return rich.animation(item.url, credit=credit)
        return rich.photo(item.url, credit=credit)

    def render_credit(self, cluster: Cluster) -> Block:
        """Whose text this is, right under the text — a byline, not a footnote.

        The words in the post belong to one channel out of the cluster, and the
        reader has to know which one before deciding what to make of them. That
        makes it part of the story rather than metadata, so it sits next to the
        paragraph instead of below the source list, where it used to live.
        """
        doc = cluster.annotation_doc
        return rich.paragraph(
            rich.italic(
                rich.join(
                    [
                        CREDIT_DASH,
                        rich.link(doc.channel_title or doc.channel_id, doc.url),
                    ],
                    " ",
                )
            )
        )

    def render_sources(
        self, cluster: Cluster, groups: list[tuple[str, list[Document]]]
    ) -> Block:
        """Collapsed source list under a plain count.

        The summary is just "12 джерел": a per-group breakdown of emoji and
        digits read as a technical badge rather than as information. The groups
        themselves stay inside, where there is room to name them.
        """
        items: list[Sequence[Block]] = []
        total = 0
        for group, docs in groups:
            emoji = self.channels.group_emoji(group)
            total += len(docs)
            title = f"{emoji} {self.channels.group_title(group)}".strip()
            channels = rich.join([self.render_source(doc) for doc in docs])
            items.append([rich.paragraph(rich.bold(title)), rich.paragraph(channels)])

        blocks: list[Block] = [rich.bullet_list(*items)]

        composition = self.render_composition(groups, total)
        if composition:
            blocks.append(composition)

        blocks.extend(self.render_marks_legend(groups))

        provenance = self.render_provenance(cluster)
        if provenance:
            blocks.append(rich.divider())
            blocks.extend(provenance)

        summary = f"{total} {pluralize_sources(total)}"
        return rich.details(summary, *blocks, is_open=self.sources_open)

    def render_source(self, doc: Document) -> RichText:
        """A channel's name, then the glyphs that say what kind of channel it is.

        The link keeps the name alone: a marker inside the link text reads as
        part of the outlet's title, which is exactly what it is not.
        """
        link = rich.link(doc.channel_title or doc.channel_id, doc.url)
        marks = self.channels.marks(doc.channel_id)
        return rich.join([link, marks], " ") if marks else link

    def render_composition(
        self, groups: list[tuple[str, list[Document]]], total: int
    ) -> Block | None:
        """How much of this story's coverage came from anonymous channels.

        The one number in the post that no source can report about itself, and
        the reason the anonymous tier exists at all: a reader who sees that nine
        of twelve sources will not say who owns them knows something about the
        story that no amount of reading the story would tell them.
        """
        anonymous = sum(len(docs) for group, docs in groups if group == ANONYMOUS_GROUP)
        if not anonymous or total < MIN_SOURCES_FOR_COMPOSITION:
            return None
        return rich.paragraph(
            f"З {total} {pluralize_sources(total)} {anonymous} — "
            f"{'анонімний канал' if anonymous == 1 else 'анонімні канали'}."
        )

    def render_marks_legend(
        self, groups: list[tuple[str, list[Document]]]
    ) -> list[Block]:
        """Names only the glyphs this post actually used.

        A fixed legend of every possible marker is longer than the source list it
        explains, and after two posts a reader stops reading it. Listing what is
        on screen keeps it to a line or two.
        """
        kinds: list[str] = []
        badges: list[str] = []
        for _, docs in groups:
            for doc in docs:
                channel = self.channels.channels.get(
                    normalize_channel_id(doc.channel_id)
                )
                if channel is None:
                    continue
                if channel.kind and self.channels.kind_emoji(channel.kind):
                    kinds.append(channel.kind)
                badges.extend(channel.badges)

        # Kinds before badges, and not in the order the sources happened to fall:
        # the two answer different questions, and interleaving them made the line
        # read as one undifferentiated row of symbols.
        parts = [
            f"{self.channels.kind_emoji(kind)} {self.channels.kind_title(kind)}"
            for kind in dict.fromkeys(kinds)
        ] + [
            f"{self.channels.badge_emoji(badge)} {self.channels.badge_title(badge)}"
            for badge in dict.fromkeys(badges)
        ]
        if not parts:
            return []
        if self.short_legend and len(parts) > 2:
            return self.render_short_legend(parts, len(dict.fromkeys(kinds)))
        return [rich.paragraph(rich.italic(" · ".join(parts)))]

    @staticmethod
    def render_short_legend(parts: list[str], kind_count: int) -> list[Block]:
        """Three marks or more on two lines: kinds on one, badges on the other,
        or halves when there is only one sort. One line of four ran off the
        right edge of a phone. Two paragraphs, since a line break inside a rich
        text is not something the API documents."""
        if 0 < kind_count < len(parts):
            lines = [parts[:kind_count], parts[kind_count:]]
        else:
            middle = (len(parts) + 1) // 2
            lines = [parts[:middle], parts[middle:]]
        return [rich.paragraph(rich.italic(" · ".join(line))) for line in lines]

    def render_provenance(self, cluster: Cluster) -> list[Block]:
        """Who published first and where the story probably originated.

        Neither answers "what happened", so both live next to the sources
        rather than in the always-visible part of the post.
        """
        first_doc = cluster.first_doc
        clock = ts_to_dt(first_doc.pub_time, self.tz_name).strftime("%H:%M")
        blocks: list[Block] = [
            rich.paragraph(
                rich.join(
                    [
                        "Першим",
                        rich.join(
                            [
                                rich.link(
                                    first_doc.channel_title or first_doc.channel_id,
                                    first_doc.url,
                                ),
                                rich.date_time(clock, first_doc.pub_time),
                            ],
                            ", ",
                        ),
                    ],
                    " — ",
                )
            )
        ]

        external_link = self.find_external_link(cluster)
        if external_link:
            blocks.append(
                rich.paragraph(
                    rich.join(
                        [
                            "Ймовірне першоджерело",
                            rich.link(external_link["host"], external_link["url"]),
                        ],
                        " — ",
                    )
                )
            )
        return blocks

    def render_footer(self, cluster: Cluster) -> Block:
        """Reach, in two facts, the second of them clickable.

        The quoted channel used to be named here as well, but it is now the
        credit under the text, and one post naming the same channel twice reads
        as a rendering bug.

        The site link is phrased as the second half of a sentence about reach
        rather than as a call to action, and that is the whole design of it. The
        post already carries the story, the sources and who was first, so
        "read more" would promise text the reader has just read and burn itself
        out within a fortnight. What the site has and the post does not is when
        each tier of channels picked the story up, how the wave compares with
        the week, and who stayed quiet — so the label names that, and the reader
        who does not care about it sees a dim clause in the smallest type on the
        post and reads past it.
        """
        reach: list[RichText] = [f"👁 {self.views_to_str(cluster.views)}"]
        story_url = self.story_url(cluster)
        if story_url:
            reach.append(rich.link("Як поширювалося", story_url))
        return rich.footer(rich.join(reach))

    def story_url(self, cluster: Cluster) -> str | None:
        """The story on the web, addressed by id rather than by headline.

        The site's own canonical URL carries a slug transliterated from the
        headline, and headlines are rewritten while a cluster is still taking
        sources — so that URL is not stable enough to put in a post that stays
        in the archive forever. The numeric form never changes and the site
        redirects it to whatever the canonical is when the link is followed.

        Duplicating the site's transliteration here would buy nothing: the
        headline nyan posts and the headline the site renders are derived
        separately, so the slugs could differ, and a wrong slug lands on the
        same redirect as no slug at all.
        """
        if not self.site_url or cluster.clid is None:
            return None
        return f"{self.site_url}/n/{cluster.clid}"

    # ---------------------------------------------------------------- legacy

    def render_legacy_cluster(
        self, cluster: Cluster, groups: list[tuple[str, list[Document]]]
    ) -> RenderedPost:
        """The pre-rich HTML caption, kept as a rollback path.

        Deliberately not given the summary: this format exists so that a bad
        release can be undone by editing a config, which means it has to stay
        the simple thing it was — one channel's text, quoted.
        """
        emojis = {group: self.channels.group_emoji(group) for group, _ in groups}
        marks = {
            doc.channel_id: self.channels.marks(doc.channel_id)
            for _, docs in groups
            for doc in docs
        }
        first_doc = copy.deepcopy(cluster.first_doc)
        first_doc.pub_time_dt = ts_to_dt(first_doc.pub_time, self.tz_name)

        text = self.cluster_template.render(
            annotation_doc=cluster.annotation_doc,
            first_doc=first_doc,
            groups=groups,
            emojis=emojis,
            marks=marks,
            views=self.views_to_str(cluster.views),
            is_important=cluster.is_important,
            external_link=self.find_external_link(cluster),
            tz_name=self.tz_name,
        )
        return RenderedPost(text=text, media=cluster.media)

    # ----------------------------------------------------------------- misc

    def find_external_link(self, cluster: Cluster) -> dict[str, str] | None:
        """The most linked external URL, if at least two channels cite it."""
        if not cluster.external_links:
            return None
        most_common = cluster.external_links.most_common()
        if not most_common:
            return None
        url, count = most_common[0]
        if count < 2:
            return None
        return {"url": url, "host": urlsplit(url).netloc}

    def render_discussion_message(self, doc: Document) -> str:
        return f'<a href="{doc.url}">{doc.channel_title}</a>'

    @staticmethod
    def views_to_str(views: int) -> str:
        if views >= 1000000:
            return f"{views / 1000000:.1f}M".replace(".", ",")
        elif views >= 1000:
            return f"{views / 1000:.1f}K".replace(".", ",")
        return str(views)
