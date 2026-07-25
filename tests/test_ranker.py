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


def _cluster(channel_ids: list[str], issue: str) -> Cluster:
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
                groups={"main": "purple", issue: "purple"},
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
    cluster = _cluster(["kyiv1", "kyiv2"], issue="local")

    ranked = Ranker(config_path)([cluster])

    assert ranked["main"] == [cluster]
    assert "local" not in ranked


def test_a_configured_issue_is_not_replaced_by_the_fallback(tmp_path: Any) -> None:
    config_path = _write_config(tmp_path, [_issue("main"), _issue("tech")])
    cluster = _cluster(["it1", "it2"], issue="tech")

    ranked = Ranker(config_path)([cluster])

    assert ranked["tech"] == [cluster]
    assert not ranked["main"]


def test_the_production_config_configures_every_issue_it_ranks() -> None:
    """The fallback needs main, and the assert in Ranker guards that."""
    ranker = Ranker("configs/ranker_config.json")

    names = [issue["issue_name"] for issue in ranker.config["issues"]]
    assert "main" in names
