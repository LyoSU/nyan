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
    """The shipped file and the code should not drift apart silently."""
    channels = Channels("channels.json")

    for group, name in DEFAULT_GROUP_NAMES.items():
        assert channels.group_title(group) == name
