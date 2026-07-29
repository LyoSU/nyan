"""Tests for the channel graph: what counts as one voice, and what does not.

The thresholds themselves were set against the archive rather than by taste, so
what is pinned here is the arithmetic and the boundaries — a pair one second the
wrong side of the lag rule has to fall out, because the whole point of the
measure is that the site subtracts sources on the strength of it.
"""

from scripts.build_channel_graph import (
    MIN_TOGETHER,
    SAME_VOICE_LAG,
    SAME_VOICE_OVERLAP,
    build_pairs,
    build_records,
    count_cooccurrence,
    project,
)


def stories(*rows: dict[str, int]) -> list[dict[str, int]]:
    return list(rows)


def test_a_channel_is_counted_once_per_story() -> None:
    """Four follow-ups to one event are still one voice on it."""
    totals, together, lags = count_cooccurrence(
        stories({"a": 100, "b": 160}, {"a": 200, "b": 200, "c": 500})
    )

    assert totals == {"a": 2, "b": 2, "c": 1}
    assert together[("a", "b")] == 2
    assert together[("a", "c")] == 1
    assert lags[("a", "b")] == [60, 0]


def test_the_lag_is_measured_from_each_channels_first_post() -> None:
    totals, _, lags = count_cooccurrence(stories({"a": 100, "b": 40}))

    # Absolute: the direction of a gap belongs to the per-story timeline, not to
    # a months-long median.
    assert lags[("a", "b")] == [60]
    assert totals == {"a": 1, "b": 1}


def test_a_story_carried_alone_still_counts_towards_the_total() -> None:
    """It is the denominator, and leaving it out inflates every overlap.

    A channel publishing three stories, one of them alongside `b`, travels with
    `b` a third of the time. Counted over shared stories only it would be all of
    the time.
    """
    totals, together, _ = count_cooccurrence(
        stories({"a": 100, "b": 100}, {"a": 200}, {"a": 300})
    )

    assert totals == {"a": 3, "b": 1}
    assert together == {("a", "b"): 1}
    assert build_pairs(totals, together, {("a", "b"): [0]}) == []


def test_overlap_is_measured_against_the_rarer_channel() -> None:
    """Otherwise a small channel mirroring a huge one scores near zero.

    `b` here publishes ten stories and `a` carries every one of them plus ninety
    of its own. `b` is the mirror, and dividing by `a`'s total would hide that.
    """
    totals = {"a": 100, "b": 10}
    together = {("a", "b"): 10}
    lags = {("a", "b"): [5] * 10}

    pair = build_pairs(totals, together, lags)[0]

    assert pair["overlap"] == 1.0
    assert pair["median_lag"] == 5


def test_pairs_below_the_noise_floor_are_dropped() -> None:
    """Two channels that both posted three big events are not one voice."""
    totals = {"a": 3, "b": 3}
    together = {("a", "b"): MIN_TOGETHER - 1}
    lags = {("a", "b"): [1] * (MIN_TOGETHER - 1)}

    assert build_pairs(totals, together, lags) == []


def test_travelling_together_slowly_is_not_one_voice() -> None:
    """The overlap bar alone would merge a newsroom with its own beat rivals.

    Both pairs cover exactly the same stories. Only the simultaneous one is being
    posted by one hand; the other is two newsrooms watching the same ministry.
    """
    totals = {"fast": 20, "clone": 20, "rival": 20}
    together = {("clone", "fast"): 20, ("fast", "rival"): 20}
    lags = {
        ("clone", "fast"): [SAME_VOICE_LAG - 1] * 20,
        ("fast", "rival"): [SAME_VOICE_LAG + 1] * 20,
    }

    records = {
        r["channel_id"]: r
        for r in build_records(totals, build_pairs(totals, together, lags), {}, 60)
    }

    assert records["fast"]["same_voice"] == ["clone"]
    assert records["rival"]["same_voice"] == []
    # Both are still listed as neighbours: travelling together is worth showing
    # even when it is not worth subtracting a source for.
    assert {n["channel_id"] for n in records["fast"]["neighbours"]} == {"clone", "rival"}


def test_a_clone_network_is_never_truncated() -> None:
    """Neighbours are a display list and get cut; `same_voice` must not be.

    A network of a dozen mirrors is exactly the case where dropping the ninth
    would overstate independence — which is the number this whole thing exists
    to stop overstating.
    """
    size = 14
    channels = [f"clone{i}" for i in range(size)]
    totals = dict.fromkeys(channels, 30)
    pairs = {}
    lags = {}
    for i, left in enumerate(channels):
        for right in channels[i + 1 :]:
            pairs[(left, right)] = 30
            lags[(left, right)] = [1] * 30

    records = build_records(totals, build_pairs(totals, pairs, lags), {}, 60)

    assert all(len(r["same_voice"]) == size - 1 for r in records)
    assert all(len(r["neighbours"]) < size - 1 for r in records)


def test_overlap_exactly_on_the_bar_counts() -> None:
    """A boundary that reads as `>` in one place and `>=` in another is a bug."""
    totals = {"a": 100, "b": 100}
    together = {("a", "b"): int(100 * SAME_VOICE_OVERLAP)}
    lags = {("a", "b"): [1] * together[("a", "b")]}

    records = {
        r["channel_id"]: r
        for r in build_records(totals, build_pairs(totals, together, lags), {}, 60)
    }

    assert records["a"]["same_voice"] == ["b"]
    assert records["b"]["same_voice"] == ["a"]


def test_a_channel_with_no_shared_stories_does_not_break_the_projection() -> None:
    """Its row is all zeros, and normalizing it would divide every distance by 0."""
    totals = {"a": 5, "b": 5, "c": 5, "lonely": 5}
    together = {("a", "b"): 5, ("b", "c"): 5}

    coordinates = project(totals, together)

    assert set(coordinates) == set(totals)
    assert all(
        isinstance(value, float) and value == value  # not NaN
        for point in coordinates.values()
        for value in point
    )


def test_a_graph_too_small_to_lay_out_yields_no_coordinates() -> None:
    assert project({"a": 1, "b": 1}, {("a", "b"): 1}) == {}
