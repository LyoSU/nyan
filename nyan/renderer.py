import copy
import logging
import os
import json
from collections import defaultdict
from collections.abc import Sequence
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader

from nyan import rich
from nyan.channels import Channels
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.rich import Block, RenderedPost, RichText
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

# At most this many channels are credited under a single difference; beyond
# that the credit line stops being readable.
MAX_DIFF_CREDITS = 3


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
        blocks: list[Block] = []

        headline, body = self.split_headline(cluster)
        if headline:
            # Important clusters get a bigger heading rather than a badge:
            # a marker that appears often stops reading as a marker.
            blocks.append(rich.heading(headline, size=2 if cluster.is_important else 3))
        blocks.extend(self.render_media(cluster))
        if body:
            blocks.append(rich.paragraph(body))
        blocks.extend(self.render_differences(cluster))
        blocks.append(self.render_sources(cluster, groups))
        blocks.append(rich.divider())
        blocks.append(self.render_footer(cluster))

        return RenderedPost(blocks=blocks)

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

    def render_differences(self, cluster: Cluster) -> list[Block]:
        """One quotation per difference, credited to the channels reporting it.

        This is the channel's own reporting rather than a metadata dump, so it
        stays visible instead of going under a disclosure.
        """
        channel_links = self.channel_links(cluster)
        blocks: list[Block] = []
        for difference in cluster.diff:
            channel_ids = [
                channel_id
                for channel_id in difference.get("channel_ids", [])[:MAX_DIFF_CREDITS]
                if channel_id in channel_links
            ]
            text = (difference.get("text") or "").strip().rstrip(".")
            if not channel_ids or not text:
                continue
            credit = rich.join(
                [channel_links[channel_id] for channel_id in channel_ids], ", "
            )
            blocks.append(rich.blockquote(rich.paragraph(text), credit=credit))
        return blocks

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
        annotation_doc = cluster.annotation_doc
        return rich.footer(
            rich.join(
                [
                    rich.link(
                        annotation_doc.channel_title or annotation_doc.channel_id,
                        annotation_doc.url,
                    ),
                    f"👁 {self.views_to_str(cluster.views)}",
                ]
            )
        )

    def channel_links(self, cluster: Cluster) -> dict[str, RichText]:
        """One link per channel, pointing at that channel's earliest post."""
        links: dict[str, RichText] = dict()
        for doc in sorted(cluster.docs, key=lambda d: d.pub_time):
            if doc.channel_id in links:
                continue
            links[doc.channel_id] = rich.link(
                doc.channel_title or doc.channel_id, doc.url
            )
        return links

    # ---------------------------------------------------------------- legacy

    def render_legacy_cluster(
        self, cluster: Cluster, groups: list[tuple[str, list[Document]]]
    ) -> RenderedPost:
        emojis = {group: self.channels.group_emoji(group) for group, _ in groups}
        first_doc = copy.deepcopy(cluster.first_doc)
        first_doc.pub_time_dt = ts_to_dt(first_doc.pub_time, self.tz_name)

        text = self.cluster_template.render(
            annotation_doc=cluster.annotation_doc,
            diff=self.legacy_differences(cluster),
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

    def legacy_differences(self, cluster: Cluster) -> list[dict[str, str]]:
        """Differences with channel credits pre-rendered as HTML links.

        Only the legacy template needs markup in the data; the rich path keeps
        channel ids and builds links structurally.
        """
        channel_urls: dict[str, tuple[str, str]] = dict()
        for doc in sorted(cluster.docs, key=lambda d: d.pub_time):
            if doc.channel_id in channel_urls:
                continue
            channel_urls[doc.channel_id] = (
                doc.url,
                doc.channel_title or doc.channel_id,
            )

        result = []
        for difference in cluster.diff:
            channel_ids = [
                channel_id
                for channel_id in difference.get("channel_ids", [])[:MAX_DIFF_CREDITS]
                if channel_id in channel_urls
            ]
            text = (difference.get("text") or "").strip()
            if not channel_ids or not text:
                continue
            links = [
                '<a href="{}">{}</a>'.format(*channel_urls[channel_id])
                for channel_id in channel_ids
            ]
            result.append({"text": text, "channels": ", ".join(links)})
        return result

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
