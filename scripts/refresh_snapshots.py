"""Regenerate the snapshots that tests/test_annotator.py and test_clusterer.py check.

The two are a chain, which is the reason this script updates both at once:

    input_docs -> annotator -> output_docs -> clusterer + ranker -> output_clusters

The clusterer test reads output_docs as its *input*, so refreshing the annotator
snapshot alone moves the ground under the clusterer snapshot and turns one red
test into another. That is not hypothetical — it is what happened the first time
these were regenerated.

Neither file can be edited by hand: one holds 347 annotated documents, the other
the clusters built from them. Until this script there was no other way to
produce them either, which is why a bump of `nyan.document.CURRENT_VERSION` or
an edit to tests/channels.json could leave the suite red for days.

Read the reported diff before committing. The snapshots exist so that an
unintended change to the pipeline shows up as a change nobody meant to make —
regenerating without looking throws that away.

    python -m scripts.refresh_snapshots --dry-run
    python -m scripts.refresh_snapshots
"""

import argparse
import logging
from collections import Counter

from nyan.annotator import Annotator
from nyan.channels import Channels
from nyan.clusterer import Clusterer
from nyan.clusters import Clusters
from nyan.document import Document, read_documents_file
from nyan.ranker import Ranker
from tests.conftest import (
    get_annotator_config_path,
    get_annotator_output_path,
    get_channels_info_path,
    get_clusterer_config_path,
    get_input_path,
    get_ranker_config_path,
    get_ranker_output_path,
)

# The issue the clusterer snapshot is taken from, as tests/test_clusterer.py
# reads it.
SNAPSHOT_ISSUE = "main"


def annotate() -> list[Document]:
    """The annotator snapshot, computed through the paths the test uses."""
    channels = Channels(get_channels_info_path())
    annotator = Annotator(get_annotator_config_path(), channels)
    docs = read_documents_file(get_input_path(), 0, 0)
    return annotator.postprocess(annotator(docs))


def cluster(docs: list[Document]) -> Clusters:
    """The clusterer snapshot: ranked clusters, in the order the test compares."""
    clusterer = Clusterer(get_clusterer_config_path())
    ranker = Ranker(get_ranker_config_path())

    clusters = Clusters()
    for ranked in ranker(clusterer(docs))[SNAPSHOT_ISSUE]:
        # Ids are assigned in ranked order, so that sorting the saved clusters
        # by id — which is what the test does — reproduces that same order.
        ranked.clid = None
        clusters.add(ranked)
    return clusters


def _same(one: object, other: object) -> bool:
    """Equality as the tests read it, not as Python's `==` does.

    A field defaulting to () is stored as [] and read back as a list, so a
    freshly computed document and its own saved copy differ on every such field
    while being the same document. Reporting those would drown the handful of
    changes worth looking at.
    """
    if isinstance(one, (list, tuple)) and isinstance(other, (list, tuple)):
        return list(one) == list(other)
    return one == other


def changed_fields(new: list[Document], old: list[Document]) -> Counter[str]:
    """Which fields moved, and in how many documents."""
    counts: Counter[str] = Counter()
    for fresh, stored in zip(new, old, strict=False):
        fresh_dict, stored_dict = fresh.asdict(), stored.asdict()
        for key in set(fresh_dict) | set(stored_dict):
            if not _same(fresh_dict.get(key), stored_dict.get(key)):
                counts[key] += 1
    return counts


def report(label: str, new: list[Document], old: list[Document]) -> None:
    if len(new) != len(old):
        logging.warning("%s: %d stored, %d now", label, len(old), len(new))
    counts = changed_fields(new, old)
    if not counts:
        logging.info("%s: no field changed", label)
        return
    logging.info("%s, fields that differ by document count:", label)
    for name, count in counts.most_common():
        logging.info("  %s: %d", name, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing anything",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs_path = get_annotator_output_path()
    clusters_path = get_ranker_output_path()

    fresh_docs = annotate()
    report("Annotator", fresh_docs, read_documents_file(docs_path, 0, 0))

    fresh_clusters = cluster(fresh_docs)
    stored_clusters = Clusters.load(clusters_path)
    report(
        "Clusterer",
        [c.annotation_doc for _, c in sorted(fresh_clusters.clid2cluster.items())],
        [c.annotation_doc for _, c in sorted(stored_clusters.clid2cluster.items())],
    )

    if args.dry_run:
        logging.info("Dry run, snapshots left alone")
        return

    with open(docs_path, "w") as w:
        for doc in fresh_docs:
            w.write(doc.serialize() + "\n")
    logging.info("Wrote %d documents to %s", len(fresh_docs), docs_path)

    fresh_clusters.save(clusters_path)
    logging.info("Wrote %d clusters to %s", len(fresh_clusters), clusters_path)


if __name__ == "__main__":
    main()
