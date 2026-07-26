import json
import logging
import os
from collections import defaultdict

from nyan.clusters import Cluster


# Below this many candidates an issue publishes everything it has: percentile
# filtering on a handful of clusters would cut the feed on noise.
MIN_CLUSTERS_TO_FILTER = 3

# Clusters kept per issue, taken from the most recent end.
MAX_CLUSTERS_PER_ISSUE = 10

# Accountability tiers whose view counts are balanced against each other in the
# main feed, so a single loud tier cannot set the bar for everyone.
#
# "grey" is deliberately absent. The anonymous channels are the largest audience
# in the roster — seven of them clear a million subscribers — so balancing them
# against newsrooms would let their reach set the threshold every newsroom then
# has to clear. They are here to be measured, not to decide what gets published.
BALANCED_GROUPS = ("blue", "red")

# Where a cluster goes when it belongs to no configured issue. Every channel is
# in the main feed (channels.json gives it as a default group), so this is the
# one issue that can always render.
FALLBACK_ISSUE = "main"


class Ranker:
    def __init__(self, config_path: str) -> None:
        assert os.path.exists(config_path)
        with open(config_path) as r:
            self.config = json.load(r)
        assert any(
            issue["issue_name"] == FALLBACK_ISSUE for issue in self.config["issues"]
        ), f"No '{FALLBACK_ISSUE}' issue configured for clusters to fall back to"

    def __call__(self, all_clusters: list[Cluster]) -> dict[str, list[Cluster]]:
        configured = {issue["issue_name"] for issue in self.config["issues"]}
        issues = defaultdict(list)
        for cluster in all_clusters:
            # A channel's issue in channels.json is not required to have an entry
            # here, and the loop below only visits configured issues — so without
            # this fallback a cluster of, say, purely local channels was dropped
            # from every feed rather than landing in the main one.
            cluster_issues = [issue for issue in cluster.issues if issue in configured]
            if not cluster_issues:
                # Logged without the title on purpose: rendering one picks an
                # annotation document, which is work this filter does not need.
                logging.info(
                    "No configured issue among %s, falling back to '%s'",
                    cluster.issues,
                    FALLBACK_ISSUE,
                )
                cluster_issues = [FALLBACK_ISSUE]
            for issue in cluster_issues:
                issues[issue].append(cluster)

        required_language = self.config.get("required_language", "uk")
        final_clusters = defaultdict(list)
        for issue_config in self.config["issues"]:
            issue_name = issue_config["issue_name"]
            min_channels = issue_config["min_channels"]
            max_age_minutes = issue_config["max_age_minutes"]

            clusters = issues[issue_name]
            filtered_clusters = []
            for cluster in clusters:
                unique_channels = {d.channel_id for d in cluster.docs}
                is_big_cluster = len(unique_channels) >= min_channels
                has_lang_doc = required_language is None or any(
                    doc.language == required_language for doc in cluster.docs
                )
                is_fresh = cluster.age < max_age_minutes * 60
                if is_big_cluster and has_lang_doc and is_fresh:
                    filtered_clusters.append(cluster)
            clusters = filtered_clusters

            logging.info(
                "Issue %s: %d clusters after the first filter", issue_name, len(clusters)
            )

            if len(clusters) <= MIN_CLUSTERS_TO_FILTER:
                final_clusters[issue_name].extend(clusters)
                for cluster in clusters:
                    logging.info(
                        "Added, no other candidates: %d %s",
                        cluster.views_per_hour,
                        cluster.cropped_title,
                    )
                continue

            clusters = self.filter_by_views(
                clusters,
                issue_name,
                issue_config["views_percentile"],
                issue_config["higher_views_percentile"],
                issue_config["higher_trigger_age_minutes"],
            )
            clusters.sort(key=lambda c: c.pub_time_percentile)
            clusters = clusters[-MAX_CLUSTERS_PER_ISSUE:]
            final_clusters[issue_name].extend(clusters)
        return final_clusters

    def calc_group_coefs(self, clusters: list[Cluster]) -> dict[str, float]:
        """Per-group multipliers that equalize total views between groups.

        Official and news channels have very different audience sizes, so
        without this the larger group would define the view threshold and
        crowd the other one out of the feed entirely.
        """
        group_views: dict[str, int] = defaultdict(int)
        for cluster in clusters:
            group_views[cluster.group] += cluster.views_per_hour

        max_views = max((group_views[group] for group in BALANCED_GROUPS), default=0)
        coefs: dict[str, float] = defaultdict(lambda: 1.0)
        for group in BALANCED_GROUPS:
            views = group_views[group]
            coefs[group] = (max_views / views) if views else 1.0
            logging.info("%s views coefficient: %.2f", group, coefs[group])
        return coefs

    def filter_by_views(
        self,
        clusters: list[Cluster],
        issue_name: str,
        views_percentile: int,
        higher_views_percentile: int,
        higher_trigger_age_minutes: int,
    ) -> list[Cluster]:
        coefs: dict[str, float] = defaultdict(lambda: 1.0)
        if issue_name == "main":
            coefs = self.calc_group_coefs(clusters)

        all_views_per_hour = sorted(
            int(cluster.views_per_hour * coefs[cluster.group]) for cluster in clusters
        )
        n = len(all_views_per_hour)

        border_index = max(0, min(n - 1, n * views_percentile // 100))
        border_views_per_hour = all_views_per_hour[border_index]

        higher_border_index = max(0, min(n - 1, n * higher_views_percentile // 100))
        higher_border_views_per_hour = all_views_per_hour[higher_border_index]

        logging.info(
            "Views border: %d, higher border: %d",
            border_views_per_hour,
            higher_border_views_per_hour,
        )

        hta = higher_trigger_age_minutes * 60
        filtered_clusters = []
        for cluster in clusters:
            views_per_hour = int(cluster.views_per_hour * coefs[cluster.group])
            cropped_title = cluster.cropped_title
            age = cluster.age
            if age > hta and views_per_hour >= border_views_per_hour:
                filtered_clusters.append(cluster)
                logging.info("Added by views: %d %s", views_per_hour, cropped_title)
            elif age < hta and views_per_hour >= higher_border_views_per_hour:
                # Young and already popular: the story is breaking, so it gets
                # a bigger heading in the post.
                cluster.is_important = True
                filtered_clusters.append(cluster)
                logging.info(
                    "Added by views (important): %d %s", views_per_hour, cropped_title
                )
            elif not cluster.messages:
                logging.info("Skipped by views: %d %s", views_per_hour, cropped_title)
        return filtered_clusters
