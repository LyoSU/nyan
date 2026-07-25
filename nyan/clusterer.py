import json
import logging
import os
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore

from nyan.clusters import Cluster
from nyan.document import Document


MIN_DISTANCE = 0.0
MAX_DISTANCE = 1.0

# Images are only treated as evidence of the same story when both posts carry
# few enough of them that the match cannot be coincidental.
MAX_MATCHED_IMAGES = 2


class Clusterer:
    def __init__(self, config_path: str):
        assert os.path.exists(config_path)
        with open(config_path) as r:
            self.config: dict[str, Any] = json.load(r)

    def __call__(self, docs: list[Document]) -> list[Cluster]:
        assert docs, "No docs for clusterer"

        # AgglomerativeClustering requires at least two samples, and a lone
        # document has nothing to be grouped with anyway. Without this the whole
        # iteration fails whenever filtering leaves exactly one document.
        if len(docs) == 1:
            cluster = Cluster()
            cluster.add(docs[0])
            return [cluster]

        distances = self.calc_distances(docs)

        clustering = AgglomerativeClustering(**self.config["clustering"])
        labels = clustering.fit_predict(distances).tolist()

        indices: list[list[int]] = [[] for _ in range(max(labels) + 1)]
        for index, label in enumerate(labels):
            indices[label].append(index)

        clusters = []
        for doc_indices in indices:
            cluster = Cluster()
            for index in doc_indices:
                cluster.add(docs[index])
            clusters.append(cluster)
        return clusters

    def calc_distances(self, docs: list[Document]) -> NDArray[Any]:
        """Pairwise cosine distances, adjusted by the configured penalties.

        Written as whole-matrix numpy operations rather than a loop over pairs:
        a day of crawling is thousands of documents, and the pair count grows
        with the square of that, which the interpreter cannot keep up with.
        """
        config = self.config["distances"]
        same_channels_penalty = config.get("same_channels_penalty", 1.0)
        time_penalty_modifier = config.get("time_penalty_modifier", 1.0)
        time_shift_hours = config.get("time_shift_hours", 4)
        ntp_issues = set(config.get("no_time_penalty_issues", tuple()))
        image_bonus = config.get("image_bonus", 0.0)

        assert docs[0].embedding
        dim = len(docs[0].embedding)
        embeddings = np.zeros((len(docs), dim), dtype=np.float32)
        for i, doc in enumerate(docs):
            embeddings[i, :] = doc.embedding
        try:
            distances = pairwise_distances(
                embeddings, metric="cosine", ensure_all_finite=False
            )
        except TypeError:
            # scikit-learn renamed this argument in 1.6.
            distances = pairwise_distances(
                embeddings, metric="cosine", force_all_finite=False
            )

        # Pairs on the diagonal are a document against itself: always zero, and
        # never a candidate for any penalty.
        off_diagonal = ~np.eye(len(docs), dtype=bool)

        same_channel = self.same_channel_mask(docs) & off_diagonal
        if same_channels_penalty > 1.0:
            distances[same_channel] = np.minimum(
                MAX_DISTANCE, distances[same_channel] * same_channels_penalty
            )

        # Documents from one channel are already resolved above; the remaining
        # rules only apply to pairs from different channels.
        other_channel = off_diagonal & ~same_channel

        if image_bonus > 0.0:
            shared_images = self.shared_images_mask(docs) & other_channel
            distances[shared_images] = np.maximum(
                MIN_DISTANCE, distances[shared_images] * (1.0 - image_bonus)
            )

        if time_penalty_modifier > 1.0:
            penalized = self.time_penalized_mask(docs, ntp_issues) & other_channel
            penalty = self.time_penalty(
                docs, time_shift_hours, time_penalty_modifier
            )
            distances[penalized] = np.minimum(
                MAX_DISTANCE, distances[penalized] * penalty[penalized]
            )

        return cast(NDArray[Any], distances)

    @staticmethod
    def same_channel_mask(docs: list[Document]) -> NDArray[np.bool_]:
        channel_ids = np.array([doc.channel_id for doc in docs])
        return cast(
            NDArray[np.bool_], np.equal(channel_ids[:, None], channel_ids[None, :])
        )

    @staticmethod
    def time_penalized_mask(
        docs: list[Document], ntp_issues: Any
    ) -> NDArray[np.bool_]:
        """Pairs the time penalty applies to.

        An issue in `no_time_penalty_issues` publishes stories that stay
        relevant for days, so the penalty is skipped only when *both* documents
        belong to such an issue.
        """
        exempt = np.array([doc.issue in ntp_issues for doc in docs])
        return cast(NDArray[np.bool_], ~(exempt[:, None] & exempt[None, :]))

    @staticmethod
    def time_penalty(
        docs: list[Document], time_shift_hours: float, time_penalty_modifier: float
    ) -> NDArray[Any]:
        """A multiplier that grows smoothly with the gap between publications."""
        pub_times = np.array([doc.pub_time for doc in docs], dtype=np.float64)
        time_diff = np.abs(pub_times[:, None] - pub_times[None, :])
        hours_shifted = (time_diff / 3600) - time_shift_hours
        penalty = 1.0 + expit(hours_shifted) * (time_penalty_modifier - 1.0)
        return cast(NDArray[Any], penalty)

    def shared_images_mask(self, docs: list[Document]) -> NDArray[np.bool_]:
        """Pairs whose images were recognized as the same picture."""
        image_labels = self.find_image_duplicates(docs)
        labels = np.array(
            [image_labels.get(i, -1) for i in range(len(docs))], dtype=np.int64
        )
        # -1 means "no matched image", which must never match another -1.
        same_image = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)

        images_count = np.array(
            [len(doc.embedded_images) for doc in docs], dtype=np.int64
        )
        min_count = np.minimum(images_count[:, None], images_count[None, :])
        max_count = np.maximum(images_count[:, None], images_count[None, :])
        matched = same_image & (min_count >= 1) & (max_count <= MAX_MATCHED_IMAGES)
        return cast(NDArray[np.bool_], matched)

    def find_image_duplicates(self, docs: list[Document]) -> dict[int, int]:
        if len(docs) < 2:
            return dict()

        embeddings, image2doc = [], []
        for i, doc in enumerate(docs):
            for image in doc.embedded_images:
                embeddings.append(image["embedding"])
                image2doc.append(i)
        if len(embeddings) < 2:
            return dict()

        dim = len(embeddings[0])
        np_embeddings = np.zeros((len(image2doc), dim), dtype=np.float32)
        for i, embedding in enumerate(embeddings):
            np_embeddings[i, :] = embedding

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=0.02,
        )

        labels = clustering.fit_predict(np_embeddings).tolist()
        logging.info("%d images in %d groups", len(labels), len(set(labels)))
        return {image2doc[i]: label for i, label in enumerate(labels)}
