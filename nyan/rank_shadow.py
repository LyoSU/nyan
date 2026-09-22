"""The ranker as it would be if it listened to the student, run beside the real one.

`nyan/student.py` rates every post's significance (1-5) and urgency, and the
ranker hears neither: whether a story reaches a feed is decided by how many
newsrooms carried it and how much it is read. This shadow ranks the same
clusters again with two changes, and records every story the two disagree on
without acting on any of them:

- a significant story needs fewer newsrooms, and a minor one more;
- an urgent young story needs fewer newsrooms still, and passes on the ordinary
  views border instead of the higher one young stories otherwise have to clear.

Both read the cluster, not a post. Significance is the median over its owners,
urgency the share of them that called it urgent, so one channel in capitals
cannot lift a story alone. A cluster with fewer than `min_labelled` labelled
owners keeps the real ranker's verdict: too few answers to overrule it with.

What is recorded is read by `scripts/rank_shadow.py`, which answers the only
question that matters before this decides anything: would the feed have been
better — stories out sooner, nothing important missed, no noise let in.
"""

import logging
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from nyan.clusters import Cluster
from nyan.document import Document
from nyan.ranker import Ranker, independent_sources
from nyan.util import get_current_ts


# Longer than any issue's `max_age_minutes`, so a key is only forgotten once its
# cluster can no longer be ranked at all.
SEEN_TTL = 2 * 24 * 3600
SEEN_PRUNE_AT = 20000


@dataclass(frozen=True)
class RankShadowConfig:
    #: A cluster's median significance at or above which it counts as major.
    major: float = 3.8
    #: `min_channels` is multiplied by this for a major story: 6 becomes 3.
    major_ratio: float = 0.5
    #: At or below which it counts as minor.
    minor: float = 1.8
    minor_ratio: float = 1.5
    #: Share of the cluster's owners whose post the student called urgent.
    urgent_share: float = 0.5
    #: How young a story has to be for urgency to count, in minutes: past that
    #: it has had its chance to gather sources in the ordinary way.
    urgent_minutes: int = 30
    #: 6 becomes 4 (3.9): a young urgent story waits for two fewer newsrooms.
    urgent_ratio: float = 0.65
    #: Whatever the ratios say, no story gets through on one newsroom.
    floor: float = 2.0
    min_labelled: int = 2


@dataclass(frozen=True)
class Signals:
    significance: float
    urgent: float
    labelled: int


def signals(cluster: Cluster) -> Signals | None:
    """The student's view of a cluster, one vote per owner, or None if too few voted."""
    by_owner: dict[str, Document] = {}
    for doc in cluster.docs:
        if doc.monitor_only or not doc.student:
            continue
        by_owner.setdefault(doc.master or doc.channel_id, doc)
    votes = [doc.student for doc in by_owner.values() if "significance_mean" in doc.student]
    if not votes:
        return None
    return Signals(
        significance=float(statistics.median(v["significance_mean"] for v in votes)),
        urgent=sum(float(v.get("urgent", 0.0)) > 0.5 for v in votes) / len(votes),
        labelled=len(votes),
    )


class RankShadow(Ranker):
    def __init__(self, config_path: str, config: dict[str, Any]) -> None:
        super().__init__(config_path)
        self.shadow = RankShadowConfig(**config)
        #: What has been recorded already, and when. The daemon ranks every
        #: cluster again several times a second; a disagreement is news once.
        #: Forgotten after `SEEN_TTL`, past which no cluster is still ranked.
        self.seen: dict[tuple[str, str, str], int] = {}

    def usable(self, cluster: Cluster) -> Signals | None:
        found = signals(cluster)
        if found is None or found.labelled < self.shadow.min_labelled:
            return None
        return found

    def is_urgent(self, cluster: Cluster, found: Signals) -> bool:
        young = cluster.age < self.shadow.urgent_minutes * 60
        return young and found.urgent >= self.shadow.urgent_share

    def reasons(self, cluster: Cluster) -> list[str]:
        found = self.usable(cluster)
        if found is None:
            return []
        reasons = []
        if found.significance >= self.shadow.major:
            reasons.append("major")
        elif found.significance <= self.shadow.minor:
            reasons.append("minor")
        if self.is_urgent(cluster, found):
            reasons.append("urgent")
        return reasons

    def required_sources(self, cluster: Cluster, issue_config: dict[str, Any]) -> float:
        base = float(issue_config["min_channels"])
        ratio = 1.0
        reasons = self.reasons(cluster)
        if "major" in reasons:
            ratio *= self.shadow.major_ratio
        if "minor" in reasons:
            ratio *= self.shadow.minor_ratio
        if "urgent" in reasons:
            ratio *= self.shadow.urgent_ratio
        if ratio == 1.0:
            return base
        return max(self.shadow.floor, base * ratio) if ratio < 1.0 else base * ratio

    def is_breaking(self, cluster: Cluster) -> bool:
        return "urgent" in self.reasons(cluster)

    def compare(
        self, clusters: list[Cluster], ranked: dict[str, list[Cluster]]
    ) -> list[dict[str, Any]]:
        """Rank `clusters` again and describe each new story the two rankings split on.

        `ranked` is what the real ranker returned for the same clusters. The
        ranking runs on the same objects, so the one flag it sets on them —
        `is_important`, which sizes a post's heading — is put back afterwards:
        the shadow must not change how a real post looks.
        """
        flags = {id(cluster): cluster.is_important for cluster in clusters}
        try:
            shadow = self(clusters)
        finally:
            for cluster in clusters:
                cluster.is_important = flags[id(cluster)]

        records = []
        issues = {issue["issue_name"]: issue for issue in self.config["issues"]}
        for issue_name, issue_config in issues.items():
            real = {id(c) for c in ranked.get(issue_name, [])}
            ours = {id(c): c for c in shadow.get(issue_name, [])}
            theirs = {id(c): c for c in ranked.get(issue_name, [])}
            changes = [("publish", c) for i, c in ours.items() if i not in real]
            changes += [("hold", c) for i, c in theirs.items() if i not in ours]
            for change, cluster in changes:
                # Only stories not yet posted: a published one is updated in
                # place, and holding it back is not a choice the shadow has.
                if cluster.messages:
                    continue
                record = self.record(cluster, issue_name, issue_config, change)
                if record is not None:
                    records.append(record)
        return records

    def record(
        self, cluster: Cluster, issue_name: str, issue_config: dict[str, Any], change: str
    ) -> dict[str, Any] | None:
        first = min(cluster.docs, key=lambda d: d.pub_time)
        key = (first.url, issue_name, change)
        now = get_current_ts()
        if len(self.seen) > SEEN_PRUNE_AT:
            self.seen = {k: ts for k, ts in self.seen.items() if now - ts < SEEN_TTL}
        if key in self.seen:
            return None
        self.seen[key] = now
        found = self.usable(cluster)
        return {
            "ts": get_current_ts(),
            "issue": issue_name,
            "change": change,
            "reasons": self.reasons(cluster),
            "title": cluster.cropped_title,
            "first_url": first.url,
            "urls": [doc.url for doc in cluster.docs][:20],
            "age": cluster.age,
            "sources": round(independent_sources(cluster), 2),
            "min_channels": issue_config["min_channels"],
            "required": round(self.required_sources(cluster, issue_config), 2),
            "views_per_hour": cluster.views_per_hour,
            "signals": asdict(found) if found else None,
        }


def log_records(records: list[dict[str, Any]]) -> None:
    for r in records:
        logging.info(
            "Rank shadow would %s in %s (%s): %.1f/%.1f sources, %s",
            r["change"],
            r["issue"],
            ",".join(r["reasons"]) or "-",
            r["sources"],
            r["required"],
            r["title"],
        )
