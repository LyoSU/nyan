import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from nyan.util import Serializable, normalize_channel_id


# Reader-facing names of the trust groups. These live in code rather than only
# in channels.json because that file is per-deployment: a deployment with an
# older copy would otherwise show readers the internal group key ("blue").
#
# Every one of the three is a fact a reader can check for themselves, which is
# the point: a state body is legally answerable for what it says, a named owner
# can be looked up, and an anonymous channel is exactly the one where neither is
# possible. Anything about *quality* is a badge instead (see BADGE_NAMES), so
# that the claim carries the name of whoever actually made it.
DEFAULT_GROUP_NAMES = {
    "red": "Офіційні",
    "blue": "Медіа та автори",
    "grey": "Анонімні",
}

# Same reasoning for the emoji: a missing key used to render an empty string,
# which silently removed the trust marker the channel is built around.
DEFAULT_GROUP_EMOJIS = {
    "red": "🏛",
    "blue": "📰",
    "grey": "🎭",
}

DEFAULT_GROUP_COLORS = {
    "red": "#FF0000",
    "blue": "#0000FF",
    "grey": "#808080",
}

# Sections are printed in this order, worst-accountability last. Sorting by the
# group key instead — which is what the renderer used to do — is alphabetical,
# so "grey" landed between "blue" and "red" and the reader met the anonymous
# channels in the middle of the list.
GROUP_ORDER = ("red", "blue", "grey")

# "purple" was the middle of a four-tier scale that mixed accountability with
# our own opinion of a newsroom's quality. The tier is gone from channels.json,
# but every document already in Mongo carries it forever, and `Cluster.group`
# and the site both read the group off the document rather than off the
# registry. Reading it as "blue" keeps those documents interpretable.
LEGACY_GROUPS = {"purple": "blue"}


def normalize_group(group: str) -> str:
    """Map a retired group key onto the tier that replaced it."""
    return LEGACY_GROUPS.get(group, group)


# What kind of author is behind the channel. Media is the default and gets no
# emoji: a marker that appears on most rows stops being a marker. A channel in
# the "grey" tier has no kind at all, because not knowing who is speaking is
# precisely what puts it there.
DEFAULT_KIND_NAMES = {
    "media": "Медіа",
    "person": "Персона",
    "organization": "Організація",
}

DEFAULT_KIND_EMOJIS = {
    "media": "",
    "person": "👤",
    "organization": "👥",
}

# Badges say what an outside register says, never what we think. Each one has a
# public URL a reader can open, which is the whole reason they exist: we stopped
# calling channels "перевірені" on our own authority and started pointing at the
# body that assessed them.
DEFAULT_BADGE_NAMES = {
    "imi_white": "Білий список ІМІ",
    "investigated": "Є розслідування про власника",
    "sanctions": "Під санкціями РНБО",
}

DEFAULT_BADGE_EMOJIS = {
    "imi_white": "⚪",
    "investigated": "🔍",
    "sanctions": "⛔",
}

DEFAULT_BADGE_URLS = {
    "imi_white": "https://imi.org.ua/doslidzhennya-standartiv",
}


@dataclass
class Channel(Serializable):
    name: str
    groups: dict[str, str]
    alias: str = ""
    master: str | None = None
    disabled: bool = False
    emojis: dict[str, str] | None = None
    colors: dict[str, str] | None = None
    issue: str | None = None
    kind: str | None = None
    badges: list[str] = field(default_factory=list)
    #: Crawled and counted, never quoted. A sanctioned channel is worth
    #: measuring — whether it carried a story is itself the finding — but
    #: printing its words in a digest would be republishing them.
    monitor_only: bool = False


class Channels:
    def __init__(self, path: str) -> None:
        self.channels: dict[str, Channel] = dict()

        with open(path) as r:
            config = json.load(r)
        # A group is an accountability tier, stored as a colour name ("blue").
        # The emoji is what readers actually see; the title is what makes the
        # emoji legible instead of a private colour code. The file may override
        # any of them, but never has to define them.
        emojis = {**DEFAULT_GROUP_EMOJIS, **config.get("emojis", {})}
        colors = {**DEFAULT_GROUP_COLORS, **config.get("colors", {})}
        default_groups = config.get("default_groups", {})

        self.group_emojis: dict[str, str] = emojis
        self.group_names: dict[str, str] = {
            **DEFAULT_GROUP_NAMES,
            **config.get("group_names", {}),
        }
        self.kind_names: dict[str, str] = {
            **DEFAULT_KIND_NAMES,
            **config.get("kind_names", {}),
        }
        self.kind_emojis: dict[str, str] = {
            **DEFAULT_KIND_EMOJIS,
            **config.get("kind_emojis", {}),
        }
        self.badge_names: dict[str, str] = {
            **DEFAULT_BADGE_NAMES,
            **config.get("badge_names", {}),
        }
        self.badge_emojis: dict[str, str] = {
            **DEFAULT_BADGE_EMOJIS,
            **config.get("badge_emojis", {}),
        }
        self.badge_urls: dict[str, str] = {
            **DEFAULT_BADGE_URLS,
            **config.get("badge_urls", {}),
        }
        for record in config["channels"]:
            channel = Channel.fromdict(record)
            assert channel.groups
            assert channel.issue
            channel.groups = {
                issue: normalize_group(group) for issue, group in channel.groups.items()
            }
            for issue, group in default_groups.items():
                if issue not in channel.groups:
                    channel.groups[issue] = normalize_group(group)
            channel.emojis = {
                issue: emojis.get(group, "") for issue, group in channel.groups.items()
            }
            channel.colors = {
                issue: colors.get(group, "#808080") for issue, group in channel.groups.items()
            }
            self.add(channel)

    def group_title(self, group: str) -> str:
        """Human-readable name of an accountability group, falling back to its key."""
        return self.group_names.get(normalize_group(group), normalize_group(group))

    def group_emoji(self, group: str) -> str:
        return self.group_emojis.get(normalize_group(group), "")

    def kind_title(self, kind: str | None) -> str:
        return self.kind_names.get(kind or "", "")

    def kind_emoji(self, kind: str | None) -> str:
        return self.kind_emojis.get(kind or "", "")

    def badge_title(self, badge: str) -> str:
        return self.badge_names.get(badge, badge)

    def badge_emoji(self, badge: str) -> str:
        return self.badge_emojis.get(badge, "")

    def badge_url(self, badge: str) -> str:
        return self.badge_urls.get(badge, "")

    def marks(self, chid: str) -> str:
        """The glyphs that follow a channel's name: who it is, then its badges.

        Kept to one string because the two say different things and a reader
        reads them in that order — a person or an institution rather than a
        newsroom, and then whatever an outside register has recorded about it.
        """
        channel = self.channels.get(normalize_channel_id(chid))
        if channel is None:
            return ""
        glyphs = [self.kind_emoji(channel.kind)]
        glyphs += [self.badge_emoji(badge) for badge in channel.badges]
        return "".join(glyph for glyph in glyphs if glyph)

    def add(self, channel: Channel) -> None:
        self.channels[normalize_channel_id(channel.name)] = channel

    def __getitem__(self, chid: str) -> Channel:
        return self.channels[normalize_channel_id(chid)]

    def __contains__(self, chid: str) -> bool:
        return normalize_channel_id(chid) in self.channels

    def __iter__(self) -> Iterator[tuple[str, Channel]]:
        return iter(self.channels.items())
