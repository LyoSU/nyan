import json
from typing import Any

from nyan.clusters import Cluster
from nyan.document import Document
from nyan.rank_shadow import RankShadow
from nyan.ranker import Ranker
from nyan.util import get_current_ts


def config(tmp_path: Any, min_channels: int = 6) -> str:
    issue = {
        "issue_name": "main",
        "min_channels": min_channels,
        "max_age_minutes": 360,
        "views_percentile": 0,
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
) -> Cluster:
    result = Cluster()
    now = get_current_ts()
    for i in range(channels):
        student = {}
        if significance is not None:
            student = {"significance_mean": significance, "urgent": urgent}
        result.add(
            Document(
                url=f"https://t.me/c{i}/{i}",
                channel_id=f"c{i}",
                post_id=i,
                views=100,
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
