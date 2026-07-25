import copy
import logging
import os
import json
from collections import defaultdict
from collections.abc import Sequence
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader

from nyan import rich
from nyan import summary as nyan_summary
from nyan.channels import Channels
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.markup import link_emphasis, parse_markup
from nyan.rich import Block, RenderedPost
from nyan.summary import Summary
from nyan.util import DEFAULT_TIMEZONE, ts_to_dt


# Used to derive a headline from the text itself when the LLM did not supply
# one — because the call failed, or the cluster predates headlines.
#
# A newline counts as a sentence break: Telegram posts separate their lead from
# the body with one, and "…— Reuters.\n«Я зателефонував…" contains no period
# followed by a space at all. Looking only for ". " left such posts without any
# heading, which is most of them.
_SENTENCE_END_CHARS = ".!?"
MAX_DERIVED_HEADLINE_LENGTH = 120

# Prefix for the line that says the sources contradict each other. A reader
# skimming has to be able to see the disagreement without reading the sentence.
DISPUTED_TITLE = "Джерела різняться"

# Heading size of the post's headline, 1-6, where 1 is the largest. Every post
# has a headline, so it competes with nothing and does not need to shout: 4 is
# a shade above body text. The other two sizes in a post derive from it, so
# tuning this one keeps the hierarchy — the headline stays above the section
# heading, an important story stays above an ordinary one.
DEFAULT_HEADLINE_SIZE = 4

# An em dash, the way print attributes a passage to its author. Not a hyphen:
# "- Укрінформ" reads as a bullet point, which is the wrong signal entirely.
CREDIT_DASH = "—"

MIN_HEADING_SIZE = 1
MAX_HEADING_SIZE = 6


def clamp_heading_size(size: int) -> int:
    return max(MIN_HEADING_SIZE, min(MAX_HEADING_SIZE, size))


def summary_blocks(summary: Summary, section_size: int) -> list[Block]:
    """The model's blocks as Telegram blocks.

    The model chooses the shape — neither a news story nor a digest comes in one
    shape — and this is where its vocabulary maps onto the API's. Unknown kinds
    cannot arrive here: `nyan.summary` has already dropped them. A story and a
    digest share this function because they share the vocabulary; only the
    frame around it differs.
    """
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
        elif block.type == nyan_summary.DISPUTED:
            # Marked rather than merely stated: a reader skimming has to see
            # that the sources do not agree.
            blocks.append(
                rich.paragraph(rich.join([rich.bold(DISPUTED_TITLE), block.text], ": "))
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
            # Skip documents from channels that don't have this issue configured
            if issue_name not in channel.groups:
                logging.warning(
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

        return sorted(
            ((name, docs) for name, docs in groups.items() if docs),
            key=lambda x: x[0],
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
            blocks.append(rich.heading(headline, size=size))
        blocks.extend(self.render_media(cluster))

        if summary:
            blocks.extend(self.render_summary(summary))
        else:
            body = self.split_headline(cluster)[1]
            if body:
                blocks.append(rich.paragraph(body))
            blocks.append(self.render_credit(cluster))

        blocks.append(self.render_sources(cluster, groups))
        blocks.append(rich.divider())
        blocks.append(self.render_footer(cluster))

        return RenderedPost(blocks=blocks)

    def render_summary(self, summary: Summary) -> list[Block]:
        return summary_blocks(summary, section_size=self.section_size)

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
            channels = rich.join(
                [rich.link(doc.channel_title or doc.channel_id, doc.url) for doc in docs]
            )
            items.append([rich.paragraph(rich.bold(title)), rich.paragraph(channels)])

        blocks: list[Block] = [rich.bullet_list(*items)]

        provenance = self.render_provenance(cluster)
        if provenance:
            blocks.append(rich.divider())
            blocks.extend(provenance)

        summary = f"{total} {pluralize_sources(total)}"
        return rich.details(summary, *blocks, is_open=self.sources_open)

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
        first_doc = copy.deepcopy(cluster.first_doc)
        first_doc.pub_time_dt = ts_to_dt(first_doc.pub_time, self.tz_name)

        text = self.cluster_template.render(
            annotation_doc=cluster.annotation_doc,
            first_doc=first_doc,
            groups=groups,
            emojis=emojis,
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
