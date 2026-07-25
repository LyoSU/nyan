import json
from collections.abc import Iterator
from dataclasses import dataclass

from nyan.util import Serializable, normalize_channel_id


# Reader-facing names of the trust groups. These live in code rather than only
# in channels.json because that file is per-deployment: a deployment with an
# older copy would otherwise show readers the internal group key ("blue").
DEFAULT_GROUP_NAMES = {
    "red": "Офіційні",
    "purple": "Перевірені медіа",
    "blue": "Новинні",
    "tech": "Технології",
    "economy": "Економіка",
    "other": "Інші",
}

# Same reasoning for the emoji: a missing key used to render an empty string,
# which silently removed the trust marker the channel is built around.
DEFAULT_GROUP_EMOJIS = {
    "red": "👑",
    "purple": "✅",
    "blue": "📰",
    "tech": "💻",
    "economy": "💰",
    "other": "🎙",
}

DEFAULT_GROUP_COLORS = {
    "red": "#FF0000",
    "purple": "#800080",
    "blue": "#0000FF",
    "tech": "#008000",
    "economy": "#FFD700",
    "other": "#808080",
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


class Channels:
    def __init__(self, path: str) -> None:
        self.channels: dict[str, Channel] = dict()

        with open(path) as r:
            config = json.load(r)
        # A group is a trust tier, stored as a colour name ("purple"). The
        # emoji is what readers actually see; the title is what makes the
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
        for channel in config["channels"]:
            channel = Channel.fromdict(channel)
            assert channel.groups
            assert channel.issue
            for issue, group in default_groups.items():
                if issue not in channel.groups:
                    channel.groups[issue] = group
            channel.emojis = {
                issue: emojis.get(group, "") for issue, group in channel.groups.items()
            }
            channel.colors = {
                issue: colors.get(group, "#808080") for issue, group in channel.groups.items()
            }
            self.add(channel)

    def group_title(self, group: str) -> str:
        """Human-readable name of a trust group, falling back to its key."""
        return self.group_names.get(group, group)

    def group_emoji(self, group: str) -> str:
        return self.group_emojis.get(group, "")

    def add(self, channel: Channel) -> None:
        self.channels[normalize_channel_id(channel.name)] = channel

    def __getitem__(self, chid: str) -> Channel:
        return self.channels[normalize_channel_id(chid)]

    def __contains__(self, chid: str) -> bool:
        return normalize_channel_id(chid) in self.channels

    def __iter__(self) -> Iterator[tuple[str, Channel]]:
        return iter(self.channels.items())
