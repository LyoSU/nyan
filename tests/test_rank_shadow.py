import json
from typing import Any

from nyan.clusters import Cluster
from nyan.daemon import Daemon
from nyan.document import Document
from nyan.rank_shadow import RankShadow
from nyan.ranker import Ranker
from nyan.util import get_current_ts


def config(tmp_path: Any, min_channels: int = 6, views_percentile: int = 0) -> str:
    issue = {
        "issue_name": "main",
        "min_channels": min_channels,
        "max_age_minutes": 360,
        "views_percentile": views_percentile,
        "higher_views_percentile": 0,
        "higher_trigger_age_minutes": 10,
    }
    path = tmp_path / "ranker_config.json"
    path.write_text(json.dumps({"required_language": None, "issues": [issue]}))
    return str(path)


def cluster(
    channels: int,
    significance: float | None = None,
    urgent: float = 0.0,
    age: int = 3600,
    views: int = 100,
    name: str = "c",
) -> Cluster:
    result = Cluster()
    now = get_current_ts()
    for i in range(channels):
        student = {}
        if significance is not None:
            student = {"significance_mean": significance, "urgent": urgent}
        result.add(
            Document(
                url=f"https://t.me/{name}{i}/{i}",
                channel_id=f"{name}{i}",
                post_id=i,
                views=views,
                pub_time=now - age,
                fetch_time=now,
                text="Текст",
                patched_text="Текст",
                groups={"main": "purple"},
                issue="main",
                language="uk",
                embedding=[1.0, float(i), 0.0, 0.0],
                student=student,
            )
        )
    return result


def compare(path: str, clusters: list[Cluster]) -> list[dict[str, Any]]:
    ranked = Ranker(path)(clusters)
    return RankShadow(path, {}).compare(clusters, ranked)


def test_a_major_story_needs_half_the_newsrooms(tmp_path: Any) -> None:
    story = cluster(3, significance=4.5)

    records = compare(config(tmp_path), [story])

    assert [(r["change"], r["reasons"]) for r in records] == [("publish", ["major"])]
    assert records[0]["required"] == 3.0


def test_a_minor_story_needs_more_than_the_ranker_asks(tmp_path: Any) -> None:
    story = cluster(6, significance=1.2)

    records = compare(config(tmp_path), [story])

    assert [(r["change"], r["reasons"]) for r in records] == [("hold", ["minor"])]


def test_urgency_counts_only_while_the_story_is_young(tmp_path: Any) -> None:
    path = config(tmp_path)
    young = cluster(4, significance=3.0, urgent=0.9, age=300)
    old = cluster(4, significance=3.0, urgent=0.9, age=3 * 3600)

    assert [r["reasons"] for r in compare(path, [young])] == [["urgent"]]
    assert compare(path, [old]) == []


def test_without_the_students_answers_nothing_changes(tmp_path: Any) -> None:
    """A host without the model must rank exactly as before."""
    assert compare(config(tmp_path), [cluster(3), cluster(6)]) == []


def test_one_labelled_owner_is_not_enough_to_overrule_the_ranker(tmp_path: Any) -> None:
    story = cluster(3, significance=4.5)
    for doc in story.docs[1:]:
        doc.student = {}

    assert compare(config(tmp_path), [story]) == []


def test_the_shadow_leaves_the_heading_size_alone(tmp_path: Any) -> None:
    """Ranking marks young popular stories important; the shadow must not."""
    path = config(tmp_path, min_channels=2)
    story = cluster(3, significance=3.0, urgent=0.9, age=60)
    ranked = Ranker(path)([story])
    story.is_important = False

    RankShadow(path, {}).compare([story], ranked)

    assert story.is_important is False


def test_a_disagreement_is_recorded_once(tmp_path: Any) -> None:
    path = config(tmp_path)
    story = cluster(3, significance=4.5)
    shadow = RankShadow(path, {})
    ranked = Ranker(path)([story])

    assert len(shadow.compare([story], ranked)) == 1
    assert shadow.compare([story], ranked) == []


def test_a_restart_does_not_record_a_disagreement_again(tmp_path: Any) -> None:
    """What was recorded is in storage, and the process that recorded it is gone.

    Three deploys in one morning restarted the sender three times, and each
    restart recorded the same stories again.
    """
    path = config(tmp_path)
    story = cluster(3, significance=4.5)
    ranked = Ranker(path)([story])
    stored = RankShadow(path, {}).compare([story], ranked)

    restarted = RankShadow(path, {})
    restarted.remember(stored)

    assert restarted.restored
    assert restarted.compare([story], ranked) == []


class StoredRecords:
    """The part of a Mongo collection the daemon uses to keep shadow records."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = list(records)
        self.reads = 0

    def find(self, query: dict[str, Any], projection: dict[str, int]) -> list[dict[str, Any]]:
        self.reads += 1
        return [r for r in self.records if r["ts"] >= query["ts"]["$gte"]]

    def insert_many(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(records)


def test_the_daemon_reads_what_is_stored_once_per_start(tmp_path: Any, monkeypatch: Any) -> None:
    path = config(tmp_path)
    old, new = cluster(3, significance=4.5, name="old"), cluster(3, significance=4.5, name="new")
    stored = StoredRecords(RankShadow(path, {}).compare([old], Ranker(path)([old])))
    monkeypatch.setattr("nyan.daemon.get_rank_shadow_collection", lambda _: stored)
    daemon = object.__new__(Daemon)
    daemon.rank_shadow = RankShadow(path, {})

    for _ in range(2):
        daemon.shadow_rank([old, new], Ranker(path)([old, new]), "mongo_config.json")

    assert stored.reads == 1
    assert sorted(r["first_url"] for r in stored.records) == [
        new.docs[0].url,
        old.docs[0].url,
    ]


def test_the_stories_it_adds_do_not_raise_the_bar_for_the_rest(tmp_path: Any) -> None:
    """The views border is a percentile, so more candidates move it.

    Measured over the shadow's own, larger list, the loud major stories it lets
    in pushed quieter stories the real ranker publishes below the border, and
    the shadow "held" them — the Zelensky-at-the-UN post on the first evening
    in production — for no reason of its own.
    """
    path = config(tmp_path, views_percentile=50)
    ordinary = [cluster(6, views=100 * (k + 1), name=f"o{k}_") for k in range(4)]
    loud = [cluster(3, significance=4.5, views=10_000, name=f"l{k}_") for k in range(4)]

    records = compare(path, ordinary + loud)

    assert {r["change"] for r in records} == {"publish"}
    assert all(r["reasons"] == ["major"] for r in records)


def test_the_stories_it_adds_do_not_free_a_place_under_the_cap(tmp_path: Any) -> None:
    """The cap keeps the newest stories; the shadow's own must not move it.

    Widened by the stories it added, the cap let one more old ordinary story
    through whenever an added one was then dropped by the views border — two
    long-posted stories the first evening, "published" for no reason at all.
    """
    path = config(tmp_path, views_percentile=10)
    ordinary = [cluster(6, views=1000 + k, name=f"o{k}_") for k in range(13)]
    quiet = [cluster(3, significance=4.5, views=1, name=f"q{k}_") for k in range(2)]
    loud = cluster(3, significance=4.5, views=10_000, name="loud_")

    records = compare(path, ordinary + quiet + [loud])

    assert [(r["change"], r["reasons"]) for r in records] == [("publish", ["major"])]
