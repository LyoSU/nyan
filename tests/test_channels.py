import json
from typing import Any

from nyan.channels import DEFAULT_GROUP_EMOJIS, DEFAULT_GROUP_NAMES, Channels


def write_channels(tmp_path: Any, config: dict[str, Any]) -> str:
    path = tmp_path / "channels.json"
    path.write_text(json.dumps(config, ensure_ascii=False))
    return str(path)


MINIMAL = {
    "channels": [{"name": "uanews", "groups": {"main": "blue"}, "issue": "main"}],
}


def test_group_names_do_not_depend_on_the_channel_file(tmp_path: Any) -> None:
    """A deployment mounts its own channels.json, which may predate any key.

    When it did, readers saw the internal group key ("blue") as a section title
    in the post instead of a name.
    """
    channels = Channels(write_channels(tmp_path, MINIMAL))

    assert channels.group_title("blue") == DEFAULT_GROUP_NAMES["blue"]
    assert channels.group_emoji("blue") == DEFAULT_GROUP_EMOJIS["blue"]


def test_the_channel_file_can_still_override_them(tmp_path: Any) -> None:
    config = dict(MINIMAL, group_names={"blue": "Свіжа назва"}, emojis={"blue": "🌊"})

    channels = Channels(write_channels(tmp_path, config))

    assert channels.group_title("blue") == "Свіжа назва"
    assert channels.group_emoji("blue") == "🌊"
    # Groups the file does not mention keep their defaults.
    assert channels.group_title("red") == DEFAULT_GROUP_NAMES["red"]


def test_an_unknown_group_falls_back_to_its_key(tmp_path: Any) -> None:
    channels = Channels(write_channels(tmp_path, MINIMAL))

    assert channels.group_title("chartreuse") == "chartreuse"
    assert channels.group_emoji("chartreuse") == ""


def test_default_groups_fill_in_missing_issues(tmp_path: Any) -> None:
    config = dict(MINIMAL, default_groups={"war": "purple"})

    channels = Channels(write_channels(tmp_path, config))

    assert channels["uanews"].groups == {"main": "blue", "war": "purple"}


def test_channels_are_found_by_any_form_of_their_id(tmp_path: Any) -> None:
    channels = Channels(write_channels(tmp_path, MINIMAL))

    assert "uanews" in channels
    assert "@UaNews" in channels
    assert "https://t.me/s/uanews" in channels
    assert "nosuchchannel" not in channels


def test_production_channel_file_matches_the_defaults() -> None:
    """The shipped file and the code should not drift apart silently.

    Only the trust groups the file actually puts channels in are checked. Its
    `group_names` also carries entries for issue names, which nothing looks up:
    the renderer titles a section by the channel's trust group, never by issue.
    """
    channels = Channels("channels.json")

    used = {group for _, channel in channels for group in channel.groups.values()}
    assert used, "No channel is in any trust group"
    for group in sorted(used):
        assert channels.group_title(group) == DEFAULT_GROUP_NAMES[group]
        assert channels.group_emoji(group) == DEFAULT_GROUP_EMOJIS[group]


def test_every_channel_can_render_its_own_issue() -> None:
    """A channel's `issue` routes its clusters; its `groups` decide what renders.

    When the two disagree the renderer keeps no documents for that issue and the
    story is dropped from every feed instead of appearing in one, so this has to
    hold for the shipped file.
    """
    channels = Channels("channels.json")

    mismatched = [
        (name, channel.issue, sorted(channel.groups))
        for name, channel in channels
        if channel.issue not in channel.groups
    ]
    assert not mismatched


def test_every_channel_is_in_the_main_feed() -> None:
    """The ranker falls back to "main" for issues it does not configure."""
    channels = Channels("channels.json")

    assert [name for name, channel in channels if "main" not in channel.groups] == []
