import os
import json
import logging
from collections import Counter
from time import sleep
from typing import Any, cast
from collections import Counter as CounterT

from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from nyan.annotator import Annotator
from nyan.client import TelegramClient
from nyan.clusters import Clusters, Cluster
from nyan.clusterer import Clusterer
from nyan.channels import Channels
from nyan.ranker import Ranker
from nyan.renderer import Renderer
from nyan.document import (
    read_documents_file,
    read_documents_mongo,
    Document,
    read_annotated_documents_mongo,
    write_annotated_documents_mongo,
)
from nyan.util import get_current_ts, ts_to_dt


# How long to wait before looking for documents again when there are none.
EMPTY_INPUT_SLEEP_SECONDS = 10


class Daemon:
    def __init__(
        self,
        client_config_path: str,
        annotator_config_path: str,
        clusterer_config_path: str,
        ranker_config_path: str,
        channels_info_path: str,
        renderer_config_path: str,
        daemon_config_path: str,
    ) -> None:
        self.client = TelegramClient(client_config_path)
        self.channels = Channels(channels_info_path)
        self.annotator = Annotator(annotator_config_path, self.channels)
        self.clusterer = Clusterer(clusterer_config_path)
        self.renderer = Renderer(renderer_config_path, self.channels)
        self.ranker = Ranker(ranker_config_path)

        assert os.path.exists(daemon_config_path)
        with open(daemon_config_path) as r:
            self.config: dict[str, Any] = json.load(r)

    def run(
        self,
        input_path: str | None,
        mongo_config_path: str | None,
        posted_clusters_path: str | None,
    ) -> None:
        while True:
            self.__call__(input_path, mongo_config_path, posted_clusters_path)

    def __call__(
        self,
        input_path: str | None,
        mongo_config_path: str | None,
        posted_clusters_path: str | None,
    ) -> None:
        assert (
            (input_path and not mongo_config_path) or (mongo_config_path and not input_path)
        )
        if input_path and not os.path.exists(input_path):
            logging.warning("No input documents at %s", input_path)
            return

        logging.info("===== New iteration =====")
        clusters_offset = self.config["clusters_offset"]
        posted_clusters = self.load_posted_clusters(
            mongo_config_path, posted_clusters_path, clusters_offset
        )

        documents_offset = self.config["documents_offset"]
        try:
            docs = self.read_documents(input_path, documents_offset, mongo_config_path)
        except Exception:
            logging.exception("Could not read documents, waiting")
            return
        if not docs:
            logging.info("No documents yet, waiting")
            sleep(EMPTY_INPUT_SLEEP_SECONDS)
            return
        self.log_bad_channels(docs)
        annotated_docs = self.annotate_documents(docs, mongo_config_path)

        updates_count = posted_clusters.update_documents(annotated_docs)
        logging.info("%d updated documents", updates_count)

        new_clusters: list[Cluster] = self.clusterer(annotated_docs)
        logging.info("%d clusters overall", len(new_clusters))

        ranked_clusters: dict[str, list[Cluster]] = self.ranker(new_clusters)
        num_clusters = sum(len(cl) for cl in ranked_clusters.values())
        logging.info("%d clusters in all issues after filtering", num_clusters)

        for issue, clusters in ranked_clusters.items():
            for cluster in clusters:
                self.send_cluster(
                    cluster,
                    issue,
                    posted_clusters,
                    posted_clusters_path,
                    mongo_config_path,
                )

        if posted_clusters_path:
            posted_clusters.save(posted_clusters_path)
            logging.info("%d clusters saved to file", len(posted_clusters))
        if mongo_config_path:
            saved_count = posted_clusters.save_to_mongo(mongo_config_path)
            logging.info("%d clusters saved to Mongo", saved_count)

    def load_posted_clusters(
        self,
        mongo_config_path: str | None,
        posted_clusters_path: str | None,
        clusters_offset: int,
    ) -> Clusters:
        posted_clusters = Clusters()
        if mongo_config_path:
            posted_clusters = Clusters.load_from_mongo(
                mongo_config_path, get_current_ts(), clusters_offset
            )
        elif posted_clusters_path and os.path.exists(posted_clusters_path):
            posted_clusters = Clusters.load(posted_clusters_path)
        logging.info("%d posted clusters loaded", len(posted_clusters))
        return posted_clusters

    def read_documents(
        self,
        input_path: str | None,
        documents_offset: int,
        mongo_config_path: str | None,
    ) -> list[Document]:
        if input_path and os.path.exists(input_path):
            docs = read_documents_file(input_path, get_current_ts(), documents_offset)
        elif mongo_config_path:
            docs = read_documents_mongo(
                mongo_config_path, get_current_ts(), documents_offset
            )
        else:
            raise AssertionError("Neither an input file nor a Mongo config was given")
        if not docs:
            return docs
        max_pub_time = ts_to_dt(max(d.pub_time for d in docs)).strftime("%d-%m-%y %H:%M")
        logging.info("%d docs loaded, last one at %s", len(docs), max_pub_time)
        return docs

    def log_bad_channels(self, docs: list[Document]) -> None:
        doc_channels_cnt: CounterT[str] = Counter()
        for doc in docs:
            doc_channels_cnt[doc.channel_id] += 1
        for channel_id, channel in self.channels:
            cnt = doc_channels_cnt.get(channel_id, 0)
            if cnt <= 1 and not channel.disabled and channel.issue == "main":
                logging.warning("Only %d docs from channel %s", cnt, channel_id)

    def annotate_documents(
        self, docs: list[Document], mongo_config_path: str | None
    ) -> list[Document]:
        all_annotated_docs: list[Document] = []
        remaining_docs = docs
        if mongo_config_path:
            all_annotated_docs, remaining_docs = read_annotated_documents_mongo(
                mongo_config_path, docs
            )
            logging.info(
                "%d docs already annotated, %d docs to annotate",
                len(all_annotated_docs),
                len(remaining_docs),
            )

        if remaining_docs:
            annotated_docs = self.annotator(remaining_docs)
            all_annotated_docs += annotated_docs
            if mongo_config_path:
                write_annotated_documents_mongo(mongo_config_path, annotated_docs)

        final_docs = self.annotator.postprocess(all_annotated_docs)
        logging.info("%d docs before clustering", len(final_docs))

        return final_docs

    def send_cluster(
        self,
        cluster: Cluster,
        issue_name: str,
        posted_clusters: Clusters,
        posted_clusters_path: str | None,
        mongo_config_path: str | None,
    ) -> None:
        sleep_time = self.config["sleep_time"]
        max_time_updated = self.config["max_time_updated"]

        posted_cluster = posted_clusters.find_similar(
            cluster,
            issue_name,
            min_intersection_ratio=self.config["similar_min_intersection_ratio"],
        )
        if posted_cluster:
            self.update_posted_cluster(
                cluster,
                posted_cluster,
                posted_clusters,
                issue_name,
                sleep_time,
                max_time_updated,
            )
            return

        # Looked up before rendering, not after: the post's text is written
        # lazily inside render_cluster, and the model has to know what the reader
        # already sees directly above this post.
        target = self.find_reply_target(cluster, posted_clusters, issue_name)
        reply_to = None
        if target is not None:
            parent, reply_to = target
            # The stored headline, never `parent.headline`: that property calls
            # the LLM on demand, so asking a neighbour for its title would pay
            # to rewrite a post that is already published.
            cluster.reply_to_headline = parent.stored_headline or ""

        post = self.renderer.render_cluster(cluster, issue_name)
        if post is None:
            logging.warning(
                "Skipping cluster, nothing to render: %s", cluster.cropped_title
            )
            return
        logging.info("New cluster in %s: %s", issue_name, cluster.cropped_title)

        self.client.update_discussion_mapping(issue_name)

        message = self.client.send_post(post, issue_name, reply_to=reply_to)
        if message is None:
            return

        cluster.create_time = get_current_ts()
        cluster.messages.append(message)
        posted_clusters.add(cluster)

        logging.info("Sent as message %d, saving", message.message_id)
        if posted_clusters_path:
            posted_clusters.save(posted_clusters_path)
        if mongo_config_path:
            posted_clusters.save_to_mongo(mongo_config_path)

        self.client.update_discussion_mapping(issue_name)
        discussion_message = self.client.get_discussion(message)
        for doc in cluster.docs:
            discussion_text = self.renderer.render_discussion_message(doc)
            self.client.send_discussion_message(discussion_text, discussion_message)
            sleep(sleep_time)

    def update_posted_cluster(
        self,
        cluster: Cluster,
        posted_cluster: Cluster,
        posted_clusters: Clusters,
        issue_name: str,
        sleep_time: float,
        max_time_updated: int,
    ) -> None:
        """Mirror new documents into the discussion, then refresh the post."""
        message = posted_cluster.get_issue_message(issue_name)
        assert message
        discussion_message = self.client.get_discussion(message)

        new_docs = [doc for doc in cluster.docs if not posted_cluster.has(doc)]
        for doc in new_docs:
            posted_cluster.add(doc)
            discussion_text = self.renderer.render_discussion_message(doc)
            self.client.send_discussion_message(discussion_text, discussion_message)
            sleep(sleep_time)
        if new_docs:
            logging.info(
                "%d new docs in cluster %d", len(new_docs), message.message_id
            )
            posted_clusters.invalidate_caches()

        time_diff = abs(get_current_ts() - posted_cluster.pub_time_percentile)
        if time_diff >= max_time_updated or not posted_cluster.changed():
            logging.info(
                "Same cluster %d at %s: %s",
                message.message_id,
                message.issue,
                posted_cluster.cropped_title,
            )
            return

        # In the format the message was sent in, not the configured one: an
        # older message may be media with a caption.
        post = self.renderer.render_cluster(
            posted_cluster, issue_name, post_format=message.post_format or None
        )
        if post is None:
            logging.warning(
                "Skipping update, nothing to render: %s", posted_cluster.cropped_title
            )
            return
        logging.info(
            "Updating message %d at %s: %s",
            message.message_id,
            message.issue,
            posted_cluster.cropped_title,
        )
        self.client.update_post(message, post)

    def find_reply_target(
        self, cluster: Cluster, posted_clusters: Clusters, issue_name: str
    ) -> tuple[Cluster, int] | None:
        """The published post this one belongs under, and its message id.

        Both halves come from one lookup because both are needed at once: the
        message id threads the new post under the old one in Telegram, and the
        neighbour itself carries the headline the prompt needs so that two posts
        standing next to each other do not say the same thing twice.

        A neighbour is not the same story — an identical one is found by
        `Clusters.find_similar` and edited in place instead. This is merely the
        closest one above the configured similarity, so it may equally be the
        previous stage of one event or a separate event on the same topic.
        """
        threshold = float(self.config["related_threshold"])

        current_ts = get_current_ts()
        clusters = posted_clusters.get_embedded_clusters(current_ts, issue_name)
        if not clusters:
            return None

        pivot_embedding = [cluster.annotation_doc.embedding]
        embeddings = [cl.embedding for cl in clusters]
        sims = cosine_similarity(pivot_embedding, embeddings)[0]

        max_index = sims.argmax()
        max_sim = sims[max_index]
        best_cluster = clusters[max_index]
        logging.info(
            "Closest cluster to '%s' is '%s' at %.3f",
            cluster.cropped_title,
            best_cluster.cropped_title,
            max_sim,
        )

        if best_cluster.pub_time_percentile > cluster.pub_time_percentile:
            return None

        if max_sim < threshold:
            return None

        for m in best_cluster.messages:
            if m.issue == issue_name:
                return best_cluster, cast(int, m.message_id)
        return None
