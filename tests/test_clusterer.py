from collections.abc import Callable

from nyan.clusterer import Clusterer
from nyan.ranker import Ranker
from nyan.document import Document
from nyan.clusters import Clusters


def test_clusterer_and_ranker_on_snapshot(
    clusterer: Clusterer,
    ranker: Ranker,
    output_docs: list[Document],
    output_clusters: Clusters,
    compare_docs: Callable
):
    clusters = clusterer(output_docs)
    assert len(clusters) > 1

    filtered_clusters = ranker(clusters)["main"]
    assert len(filtered_clusters) >= 1

    canonical = sorted(output_clusters.clid2cluster.items())
    assert len(filtered_clusters) == len(canonical), "Different number of clusters"
    for pcl, (_, ccl) in zip(filtered_clusters, canonical, strict=True):
        compare_docs(pcl.annotation_doc, ccl.annotation_doc, is_short=True)
