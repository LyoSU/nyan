import json
from typing import Any

from nyan.channels import (
    DEFAULT_GROUP_EMOJIS,
    DEFAULT_GROUP_NAMES,
    Channels,
    normalize_group,
)


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
    config = dict(MINIMAL, default_groups={"war": "red"})

    channels = Channels(write_channels(tmp_path, config))

    assert channels["uanews"].groups == {"main": "blue", "war": "red"}


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


def test_a_retired_group_is_read_as_the_tier_that_replaced_it(tmp_path: Any) -> None:
    """"purple" is gone from the file but lives on in every stored document.

    `Cluster.group` and the site both read the group off a document rather than
    off the registry, so a document written before the merge has to keep naming
    a tier that still exists.
    """
    config = dict(MINIMAL, channels=[{**MINIMAL["channels"][0], "groups": {"main": "purple"}}])

    channels = Channels(write_channels(tmp_path, config))

    assert normalize_group("purple") == "blue"
    assert channels["uanews"].groups == {"main": "blue"}
    assert channels.group_title("purple") == DEFAULT_GROUP_NAMES["blue"]


def test_marks_name_the_author_then_the_registers(tmp_path: Any) -> None:
    """Kind first, badges after: a reader asks who is speaking before asking
    what an outside body recorded about them."""
    config = dict(
        MINIMAL,
        channels=[
            {
                "name": "someone",
                "groups": {"main": "blue"},
                "issue": "main",
                "kind": "person",
                "badges": ["imi_white"],
            }
        ],
    )

    channels = Channels(write_channels(tmp_path, config))

    assert channels.marks("someone") == "👤⚪"


def test_media_is_the_unmarked_default(tmp_path: Any) -> None:
    """A marker on most rows is not a marker.

    Most channels are newsrooms, so only the departures from that get a glyph.
    """
    config = dict(
        MINIMAL,
        channels=[{"name": "outlet", "groups": {"main": "blue"}, "issue": "main", "kind": "media"}],
    )

    channels = Channels(write_channels(tmp_path, config))

    assert channels.marks("outlet") == ""


def test_an_anonymous_channel_claims_no_kind() -> None:
    """Not knowing who is behind a channel is what puts it in the grey tier, so
    asserting a kind for one would contradict the tier it is in."""
    channels = Channels("channels.json")

    disagreeing = [
        name
        for name, channel in channels
        if (channel.kind is None) != (channel.groups["main"] == "grey")
    ]
    assert not disagreeing


def test_a_badge_source_on_the_channel_wins_over_the_register(tmp_path: Any) -> None:
    """Two investigations cannot share one URL.

    NGL.media named the Труха network and Zheleznyak the five anonymous
    millionaires; a single per-badge URL attributed whichever claim it did not
    cover to an article that never made it.
    """
    config = dict(
        MINIMAL,
        channels=[
            {
                "name": "uanews",
                "groups": {"main": "grey"},
                "issue": "main",
                "badges": ["investigated"],
                "badge_sources": {"investigated": "https://example.org/who-owns-uanews"},
            }
        ],
        badge_urls={"investigated": "https://example.org/register"},
    )

    channels = Channels(write_channels(tmp_path, config))

    assert channels.badge_url("investigated", "uanews") == "https://example.org/who-owns-uanews"
    # Without a channel, and for a channel that records no source of its own,
    # the register-wide URL is still the answer.
    assert channels.badge_url("investigated") == "https://example.org/register"
    assert channels.badge_url("sanctions", "uanews") == ""


def test_every_badge_source_names_a_badge_the_channel_carries() -> None:
    """A source for a badge that is not on the channel documents nothing."""
    channels = Channels("channels.json")

    for name, channel in channels:
        for badge, url in channel.badge_sources.items():
            assert badge in channel.badges, f"{name}: source for a badge it lacks"
            assert url.startswith("https://"), f"{name}: {badge}"


def test_a_satellite_names_a_master_that_is_itself_a_source() -> None:
    """`master` collapses a clone network to one source when ranking.

    A master that is missing from the registry, or that is itself somebody's
    satellite, would make that collapse either crash or silently partial.
    """
    channels = Channels("channels.json")

    for name, channel in channels:
        if channel.master is None:
            continue
        assert channel.master in channels, f"{name}: master {channel.master} is not a channel"
        assert channels[channel.master].master is None, f"{name}: master is itself a satellite"


def test_only_registers_we_can_link_to_are_badges() -> None:
    """A badge exists to attribute a claim to somebody else, which it cannot do
    without a public URL for the register behind it."""
    channels = Channels("channels.json")

    used = {badge for _, channel in channels for badge in channel.badges}
    assert used, "No channel carries a badge"
    for badge in sorted(used):
        assert channels.badge_title(badge) != badge, badge
        assert channels.badge_url(badge).startswith("https://"), badge
