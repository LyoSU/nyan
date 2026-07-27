import copy
import logging
import os
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
from nyan.markup import link_emphasis, parse_markup
from nyan.rich import Block, RenderedPost, RichText
from nyan.summary import Summary
from nyan.util import DEFAULT_TIMEZONE, normalize_channel_id, ts_to_dt


# Used to derive a headline from the text itself when the LLM did not supply
# one — because the call failed, or the cluster predates headlines.
#
# A newline counts as a sentence break: Telegram posts separate their lead from
# the body with one, and "…— Reuters.\n«Я зателефонував…" contains no period
# followed by a space at all. Looking only for ". " left such posts without any
# heading, which is most of them.
_SENTENCE_END_CHARS = ".!?"
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
# text — checked against the whole ladder in a real client. The other two sizes
# in a post derive from it, so tuning this one keeps the hierarchy: the headline
# stays above the section heading, an important story above an ordinary one.
DEFAULT_HEADLINE_SIZE = 3

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


def summary_blocks(
    summary: Summary,
    section_size: int,
    sources: Mapping[str, RichText] | None = None,
) -> list[Block]:
    """The model's blocks as Telegram blocks.

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
            blocks.append(
                rich.blockquote(rich.paragraph(block.text), credit=block.author)
            )
        elif block.type == nyan_summary.HIDDEN:
            blocks.append(
                rich.details(block.summary, rich.paragraph(parse_markup(block.text)))
            )
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

        # "rich" builds a block tree for sendRichMessage; "legacy" keeps the
        # old HTML caption path, so a bad release can be rolled back by
        # editing the config instead of redeploying.
        self.post_format = config.get("post_format", "rich")
        self.sources_open = config.get("sources_open", False)

        # One knob for the whole post: the two other heading sizes are one step
        # either side of it, so they cannot cross over when it is retuned.
        self.headline_size = clamp_heading_size(
            int(config.get("headline_size", DEFAULT_HEADLINE_SIZE))
        )
        self.important_headline_size = clamp_heading_size(self.headline_size - 1)
        self.section_size = clamp_heading_size(self.headline_size + 1)

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
        return self.render_rich_cluster(cluster, groups)

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
            # Important clusters get a bigger heading rather than a badge:
            # a marker that appears often stops reading as a marker.
            size = (
                self.important_headline_size
                if cluster.is_important
                else self.headline_size
            )
            # Bold on top of the heading: a client renders a heading in a
            # semibold weight, which at this size reads as body text with a
            # larger font rather than as a headline. Compared side by side, the
            # bold one is the one that looks like a news headline.
            blocks.append(rich.heading(rich.bold(headline), size=size))
        blocks.extend(self.render_media(cluster))

        if summary:
            blocks.extend(self.render_summary(summary, groups))
        else:
            body = self.split_headline(cluster)[1]
            if body:
                blocks.append(rich.paragraph(body))
            blocks.append(self.render_credit(cluster))

        blocks.append(self.render_sources(cluster, groups))
        blocks.append(rich.divider())
        blocks.append(self.render_footer(cluster))

        return RenderedPost(blocks=blocks)

    def render_summary(
        self,
        summary: Summary,
        groups: list[tuple[str, list[Document]]] | None = None,
    ) -> list[Block]:
        return summary_blocks(
            summary,
            section_size=self.section_size,
            sources=self.index_sources(groups) if groups else None,
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

    def split_headline(self, cluster: Cluster) -> tuple[str | None, str | None]:
        """Return (headline, body) for the cluster's text.

        With an LLM headline the full text becomes the body, the way a
        headline and a lede work in print. Without one, the first sentence
        stands in, and the rest becomes the body so nothing is said twice.
        """
        text = (cluster.annotation_doc.patched_text or "").strip()
        headline = cluster.headline
        if headline:
            return headline, text or None
        if not text:
            return None, None

        split_at = self.find_sentence_end(text)
        if split_at is None or split_at > MAX_DERIVED_HEADLINE_LENGTH:
            return None, text
        return text[:split_at].strip(), text[split_at:].strip() or None

    @staticmethod
    def find_sentence_end(text: str) -> int | None:
        """Index just past the first sentence, or None if there is only one.

        A sentence ends at .!? followed by whitespace, or at a line break —
        Telegram posts put their lead on its own line, often with no trailing
        punctuation at all.
        """
        for index, char in enumerate(text):
            if char == "\n":
                return index
            if char not in _SENTENCE_END_CHARS:
                continue
            following = text[index + 1 : index + 2]
            # End of text is not a split: there is no second sentence.
            if following and following.isspace():
                return index + 1
        return None

    def render_media(self, cluster: Cluster) -> list[Block]:
        if cluster.videos:
            return [rich.video(cluster.videos[0])]
        images = list(cluster.images)[: rich.MAX_MEDIA]
        if len(images) == 1:
            return [rich.photo(images[0])]
        if len(images) > 1:
            return [rich.slideshow(*[rich.photo(url) for url in images])]
        return []

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

        legend = self.render_marks_legend(groups)
        if legend:
            blocks.append(legend)

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
    ) -> Block | None:
        """Names only the glyphs this post actually used.

        A fixed legend of every possible marker is longer than the source list it
        explains, and after two posts a reader stops reading it. Listing what is
        on screen keeps it to a line or two.
        """
        kinds: list[str] = []
        badges: list[str] = []
        for _, docs in groups:
            for doc in docs:
                channel = self.channels.channels.get(normalize_channel_id(doc.channel_id))
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
            return None
        return rich.paragraph(rich.italic(" · ".join(parts)))

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
        """Reach only.

        The quoted channel used to be named here as well, but it is now the
        credit under the text, and one post naming the same channel twice reads
        as a rendering bug.
        """
        return rich.footer(f"👁 {self.views_to_str(cluster.views)}")

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
        return RenderedPost(
            text=text, photos=cluster.images, videos=cluster.videos
        )

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
