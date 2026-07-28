import json
from typing import Any

from nyan.clusters import Cluster
from nyan.document import Document
from nyan.ranker import Ranker
from nyan.util import get_current_ts


def _write_config(tmp_path: Any, issues: list[dict[str, Any]]) -> str:
    path = tmp_path / "ranker_config.json"
    path.write_text(json.dumps({"required_language": None, "issues": issues}))
    return str(path)


def _issue(name: str, min_channels: int = 2) -> dict[str, Any]:
    return {
        "issue_name": name,
        "min_channels": min_channels,
        "max_age_minutes": 360,
        "views_percentile": 0,
        "higher_views_percentile": 0,
        "higher_trigger_age_minutes": 10,
    }


def _cluster(
    channel_ids: list[str],
    issue: str,
    group: str = "purple",
    master: str | None = None,
) -> Cluster:
    cluster = Cluster()
    now = get_current_ts()
    for i, channel_id in enumerate(channel_ids):
        cluster.add(
            Document(
                url=f"https://t.me/{channel_id}/{i}",
                channel_id=channel_id,
                post_id=i,
                views=100,
                pub_time=now - 60,
                fetch_time=now,
                text="Текст",
                patched_text="Текст",
                groups={"main": group, issue: group},
                master=master,
                issue=issue,
                language="uk",
                # Choosing the title compares documents by embedding, so a
                # cluster without them cannot even be logged.
                embedding=[1.0, float(i), 0.0, 0.0],
            )
        )
    return cluster


def test_a_cluster_whose_issue_is_not_configured_lands_in_main(tmp_path: Any) -> None:
    """channels.json may name an issue that ranker_config.json does not.

    The loop over issues only visits configured ones, so such a cluster used to
    be dropped from every feed instead of appearing in the main one. Whole groups
    of channels — local, economy, culture — published nothing at all.
    """
    config_path = _write_config(tmp_path, [_issue("main")])
    # Six of them because a local channel counts as a third of a newsroom, and
    # this test is about where the cluster is routed rather than whether it
    # clears the bar.
    cluster = _cluster([f"kyiv{i}" for i in range(6)], issue="local")

    ranked = Ranker(config_path)([cluster])

    assert ranked["main"] == [cluster]
    assert "local" not in ranked


def test_a_configured_issue_is_not_replaced_by_the_fallback(tmp_path: Any) -> None:
    config_path = _write_config(tmp_path, [_issue("main"), _issue("tech")])
    cluster = _cluster(["it1", "it2"], issue="tech")

    ranked = Ranker(config_path)([cluster])

    assert ranked["tech"] == [cluster]
    assert not ranked["main"]


def test_a_story_only_local_channels_carry_does_not_publish(tmp_path: Any) -> None:
    """Five Vinnytsia publics agreeing is not the country agreeing.

    `min_channels` counted every channel as one, and the roster now holds some
    seventy local sources, so a story no outlet outside one oblast had touched
    cleared the same bar as one the national press had.
    """
    config_path = _write_config(tmp_path, [_issue("main", min_channels=4)])
    cluster = _cluster(["vn1", "vn2", "vn3", "vn4", "vn5"], issue="local", group="grey")

    ranked = Ranker(config_path)([cluster])

    assert not ranked["main"]


def test_a_clone_network_cannot_carry_a_story_on_its_own(tmp_path: Any) -> None:
    """Ten Труха channels are one owner, not ten witnesses.

    The regional clones post the same text within minutes, so unweighted
    counting read a single newsroom as a national consensus.
    """
    config_path = _write_config(tmp_path, [_issue("main", min_channels=4)])
    clones = [f"truexa{i}" for i in range(10)]
    cluster = _cluster(clones, issue="local", group="grey")

    ranked = Ranker(config_path)([cluster])

    assert not ranked["main"]


def test_a_clone_network_counts_as_one_source(tmp_path: Any) -> None:
    """`master` collapses the network, so the discount is not what carries this.

    Blue and national on purpose: without the collapse these twenty would weigh
    twenty and publish outright, which is the failure the weights alone cannot
    catch — a network does not have to be anonymous or local to be one owner.
    """
    config_path = _write_config(tmp_path, [_issue("main", min_channels=4)])
    cluster = _cluster(
        [f"satellite{i}" for i in range(20)],
        issue="main",
        group="blue",
        master="flagship",
    )

    ranked = Ranker(config_path)([cluster])

    assert not ranked["main"]


def test_four_newsrooms_still_publish(tmp_path: Any) -> None:
    """The discount applies to anonymity and locality, not to everyone."""
    config_path = _write_config(tmp_path, [_issue("main", min_channels=4)])
    cluster = _cluster(["up", "suspilne", "babel", "liga"], issue="main", group="blue")

    ranked = Ranker(config_path)([cluster])

    assert ranked["main"] == [cluster]


def test_enough_anonymous_channels_still_publish(tmp_path: Any) -> None:
    """Discounted is not silenced: when the whole anonymous segment carries a
    story, that is itself a finding worth publishing."""
    config_path = _write_config(tmp_path, [_issue("main", min_channels=4)])
    cluster = _cluster([f"anon{i}" for i in range(12)], issue="main", group="grey")

    ranked = Ranker(config_path)([cluster])

    assert ranked["main"] == [cluster]


def test_the_national_press_lifts_a_local_story(tmp_path: Any) -> None:
    """A local story the national press picked up is no longer only local.

    The discount holds local channels back; it must not hold back a cluster they
    happen to be in once outlets outside the oblast carry the same story.
    """
    config_path = _write_config(tmp_path, [_issue("main", min_channels=2)])
    cluster = _cluster(["vn1", "vn2", "vn3"], issue="local", group="grey")
    for doc in _cluster(["suspilne", "up"], issue="main", group="blue").docs:
        cluster.add(doc)

    ranked = Ranker(config_path)([cluster])

    assert ranked["main"] == [cluster]


def test_the_production_config_configures_every_issue_it_ranks() -> None:
    """The fallback needs main, and the assert in Ranker guards that."""
    ranker = Ranker("configs/ranker_config.json")

    names = [issue["issue_name"] for issue in ranker.config["issues"]]
    assert "main" in names
