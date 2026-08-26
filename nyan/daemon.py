import os
import json
import logging
from collections import Counter
from time import sleep
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from collections import Counter as CounterT

from nyan.annotator import Annotator
from nyan.client import MessageId, TelegramClient
from nyan.clusters import Clusters, Cluster
from nyan.clusterer import Clusterer
from nyan.channels import Channels
from nyan.logs import log_new_iteration
from nyan.ranker import Ranker
from nyan.relation import SAME, Relation, judge_relation, nearest_clusters
from nyan.renderer import Renderer
from nyan.document import (
    read_documents_file,
    read_documents_mongo,
    Document,
    prune_annotated_documents_mongo,
    read_annotated_documents_mongo,
    write_annotated_documents_mongo,
)
from nyan.util import get_current_ts, normalize_url, ts_to_dt


# How long to wait before looking for documents again when there are none.
EMPTY_INPUT_SLEEP_SECONDS = 10

# How close a lone document has to be to a published post before it is worth
# asking whether it belongs there. Measured over a day of production: some forty
# documents a day clear it, seven clear 0.96, and thousands clear the 0.86 used
# between clusters.
DEFAULT_ATTACH_THRESHOLD = 0.94

# And how far apart in time the two may be.
ATTACH_WINDOW_SECONDS = 6 * 3600

# How long a send whose answer never arrived keeps its story from being sent
# again. Long enough to cover the whole life of a story rather than a single
# iteration: a short window would only postpone the duplicate, since the same
# documents come back around on every pass. A story that really did fail to
# post is not lost by this — it returns as soon as enough new sources have
# joined it for the overlap with the held attempt to fall below
# `similar_min_intersection_ratio`, and a story that never gathers those was
# not worth a second attempt.
DEFAULT_PENDING_SEND_TTL = 3 * 3600


def _unit_rows(embeddings: list[list[float]]) -> NDArray[np.float32]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return cast("NDArray[np.float32]", matrix / np.where(norms == 0.0, 1.0, norms))


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

        log_new_iteration()
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

        # Before clustering rather than after: a document that joins a published
        # post here is held with that post by the assignments handed down below,
        # instead of being free to start a piece of its own all over again.
        attached = self.attach_loose_documents(annotated_docs, posted_clusters)
        logging.info("%d documents joined a post they had missed", attached)

        new_clusters: list[Cluster] = self.clusterer(
            annotated_docs, posted_clusters.published_documents()
        )
        logging.info("%d clusters overall", len(new_clusters))

        ranked_clusters: dict[str, list[Cluster]] = self.ranker(new_clusters)
        num_clusters = sum(len(cl) for cl in ranked_clusters.values())
        logging.info("%d clusters in all issues after filtering", num_clusters)

        for issue, clusters in self.drop_unpostable_issues(ranked_clusters).items():
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
                mongo_config_path, docs, self.channels
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

        if mongo_config_path:
            # After writing, not before: an iteration that adds annotations should
            # be the one that pays back the space, so the collection cannot grow
            # across a run where pruning happened to fail.
            #
            # Failure here is never fatal. This is housekeeping on a cache, and
            # the iteration it runs in has already produced the annotations the
            # digest needs — losing a pass costs disk, while raising would cost
            # the news.
            try:
                pruned = prune_annotated_documents_mongo(
                    mongo_config_path, get_current_ts()
                )
                if pruned:
                    logging.info("Pruned %d annotations past their window", pruned)
            except Exception:
                logging.exception("Could not prune annotations; continuing")

        final_docs = self.annotator.postprocess(all_annotated_docs)
        logging.info("%d docs before clustering", len(final_docs))

        return final_docs

    def drop_unpostable_issues(
        self, ranked_clusters: dict[str, list[Cluster]]
    ) -> dict[str, list[Cluster]]:
        """Issues the client has no channel for, dropped before anything renders.

        The ranker groups clusters by issue regardless of what the deployment
        publishes, so a host configured for `main` alone still ranks `war` and
        `tech`. Every send for those was refused anyway — but only after
        `render_cluster` had written the post, which is where the LLM is called.
        Answered once per issue instead of twice per cluster.
        """
        postable: dict[str, list[Cluster]] = dict()
        for issue, clusters in ranked_clusters.items():
            if self.client.has_issue(issue):
                postable[issue] = clusters
                continue
            logging.warning(
                "No channel for issue '%s' in the client config, dropping %d clusters",
                issue,
                len(clusters),
            )
        return postable

    def send_cluster(
        self,
        cluster: Cluster,
        issue_name: str,
        posted_clusters: Clusters,
        posted_clusters_path: str | None,
        mongo_config_path: str | None,
    ) -> None:
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
                max_time_updated,
            )
            return

        # An earlier send of this same story whose answer never came back. The
        # post may well be in the channel — that is what cannot be known — so it
        # is held rather than sent again. Checked before rendering: an answer of
        # "already told" should not cost an LLM call.
        pending = posted_clusters.find_pending(
            cluster,
            issue_name,
            min_intersection_ratio=self.config["similar_min_intersection_ratio"],
            current_ts=get_current_ts(),
            ttl=self.config.get("pending_send_ttl", DEFAULT_PENDING_SEND_TTL),
        )
        if pending is not None:
            logging.warning(
                "Unconfirmed send %ds ago, holding: %s",
                get_current_ts() - (pending.pending_since or 0),
                cluster.cropped_title,
            )
            return

        # Looked up before rendering, not after: the post's text is written
        # lazily inside render_cluster, and the model has to know what the reader
        # already sees directly above this post.
        relation = self.find_relation(cluster, posted_clusters, issue_name)
        parent = relation.cluster

        # The same story as one already published: fold this in rather than
        # sending a second message. `find_similar` missed it because the
        # clusterer had cut the story into pieces that share too few source
        # URLs to be recognized by their URLs alone. Past the editing window
        # the documents are still taken in and nothing is sent — "same" means
        # the reader has already been told this, and an evening story re-told
        # by the morning wave of channels was absorbed exactly so whenever the
        # clusterer happened to merge the waves itself. Only a post this issue
        # never carried is excused: folding the story into it would tell these
        # readers nothing.
        if (
            relation.verdict == SAME
            and parent is not None
            and parent.get_issue_message(issue_name) is not None
        ):
            self.update_posted_cluster(
                cluster, parent, posted_clusters, issue_name, max_time_updated
            )
            return

        # Everything else stands under the post it belongs to, if there is one:
        # a development of it, or the same story told where this issue's readers
        # cannot see it.
        reply_to = None
        if parent is not None:
            message = parent.get_issue_message(issue_name)
            if message is not None:
                reply_to = message.message_id
                # The stored headline and summary, never `parent.headline`:
                # that property calls the LLM on demand, so asking a neighbour
                # for its title would pay to rewrite a post that is already
                # published.
                cluster.reply_to_headline = parent.stored_headline or ""
                cluster.reply_to_text = parent.stored_summary.as_text()

        post = self.renderer.render_cluster(cluster, issue_name)
        if post is None:
            logging.warning(
                "Skipping cluster, nothing to render: %s", cluster.cropped_title
            )
            return
        logging.info("New cluster in %s: %s", issue_name, cluster.cropped_title)

        if self.sends_docs_to_discussion:
            self.client.update_discussion_mapping(issue_name)

        # On record before Telegram is asked, and persisted right away: what
        # has to survive is the send whose answer never comes back, including
        # the one that takes the process down with it.
        posted_clusters.mark_pending(cluster, issue_name, get_current_ts())
        if posted_clusters_path:
            posted_clusters.save(posted_clusters_path)
        if mongo_config_path:
            posted_clusters.save_one_to_mongo(mongo_config_path, cluster)

        message = self.client.send_post(post, issue_name, reply_to=reply_to)
        if message is None:
            # Whether the post exists is exactly what is unknown here, so the
            # attempt stays on record and the next iteration reads it.
            logging.warning(
                "No answer for %s, holding the story back", cluster.cropped_title
            )
            return

        cluster.create_time = get_current_ts()
        posted_clusters.confirm_pending(cluster, message)

        logging.info("Sent as message %d, saving", message.message_id)
        if posted_clusters_path:
            posted_clusters.save(posted_clusters_path)
        if mongo_config_path:
            posted_clusters.save_to_mongo(mongo_config_path)

        self.send_docs_to_discussion(cluster.docs, message, refresh_mapping=True)

    @property
    def sends_docs_to_discussion(self) -> bool:
        """Whether every source post is mirrored into the comments.

        Off: it repeated the whole channel a second time under each post. Kept
        behind `send_docs_to_discussion` in the daemon config in case we want it
        back.
        """
        return bool(self.config.get("send_docs_to_discussion", False))

    def send_docs_to_discussion(
        self,
        docs: list[Document],
        message: MessageId,
        refresh_mapping: bool = False,
    ) -> None:
        if not docs or not self.sends_docs_to_discussion:
            return
        sleep_time = self.config["sleep_time"]
        # A freshly published post has no mirror in the discussion group yet,
        # so its mapping has to be re-read; an older one is already in there.
        if refresh_mapping:
            self.client.update_discussion_mapping(message.issue)
        discussion_message = self.client.get_discussion(message)
        for doc in docs:
            discussion_text = self.renderer.render_discussion_message(doc)
            self.client.send_discussion_message(discussion_text, discussion_message)
            sleep(sleep_time)

    def update_posted_cluster(
        self,
        cluster: Cluster,
        posted_cluster: Cluster,
        posted_clusters: Clusters,
        issue_name: str,
        max_time_updated: int,
    ) -> None:
        """Take in the new documents, then refresh the post."""
        message = posted_cluster.get_issue_message(issue_name)
        assert message

        new_docs = [doc for doc in cluster.docs if not posted_cluster.has(doc)]
        for doc in new_docs:
            posted_cluster.add(doc)
        self.send_docs_to_discussion(new_docs, message)
        if new_docs:
            logging.info(
                "%d new docs in cluster %d", len(new_docs), message.message_id
            )
            posted_clusters.invalidate_caches()

        self.refresh_post(posted_cluster, issue_name, max_time_updated)

    def refresh_post(
        self, posted_cluster: Cluster, issue_name: str, max_time_updated: int
    ) -> None:
        """Rewrite a published message around what its cluster now holds."""
        message = posted_cluster.get_issue_message(issue_name)
        if message is None:
            return

        if not posted_cluster.accepts_updates(max_time_updated):
            logging.info(
                "Past editing, message %d at %s: %s",
                message.message_id,
                message.issue,
                posted_cluster.cropped_title,
            )
            return
        if not posted_cluster.changed():
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

    def attach_loose_documents(
        self, docs: list[Document], posted_clusters: Clusters
    ) -> int:
        """Put documents that carried a published story into the post that told it.

        The other two halves of this cannot reach these. Holding published
        documents in place across iterations keeps what a post already has, and
        the judge on the publish boundary compares clusters that got that far —
        while a channel whose post never gathered four independent sources of
        its own is filtered out by the ranker and reaches neither. Over a day of
        production that is some forty documents sitting closer than 0.94 to a
        post they are not in, the closest of them being one newsroom writing
        "У Києві чути вибухи" against a post whose first line is that sentence.

        The floor is far above the one used between clusters because a single
        document carries far less evidence than a cluster of them: at 0.86 this
        would be thousands of questions a day rather than forty.
        """
        threshold = float(self.config.get("attach_threshold", DEFAULT_ATTACH_THRESHOLD))
        max_time_updated = self.config["max_time_updated"]

        published = [
            cluster
            for cluster in posted_clusters.clid2cluster.values()
            if cluster.messages and cluster.embedding
        ]
        taken = posted_clusters.published_documents()
        loose = [
            doc
            for doc in docs
            if doc.embedding and normalize_url(doc.url) not in taken
        ]
        if not published or not loose:
            return 0

        # One width at a time. Annotations are cached, so an encoder swap leaves
        # both widths in flight for a while, and comparing across them is
        # meaningless. Attaching is an improvement rather than a duty, so a
        # minority width simply waits for the next iteration.
        width = Counter(len(doc.embedding or ()) for doc in loose).most_common(1)[0][0]
        published = [c for c in published if len(c.embedding or ()) == width]
        loose = [doc for doc in loose if len(doc.embedding or ()) == width]
        if not published or not loose:
            return 0

        similarity = _unit_rows(
            [doc.embedding or [] for doc in loose]
        ) @ _unit_rows([cluster.embedding or [] for cluster in published]).T

        # Time is the one thing the text cannot say: "вибухи в Києві" is written
        # the same way on every night it happens, so a document is only offered
        # to a post published around its own hour.
        doc_times = np.array([doc.pub_time for doc in loose])
        post_times = np.array([cluster.pub_time_percentile for cluster in published])
        near = np.abs(doc_times[:, None] - post_times[None, :]) <= ATTACH_WINDOW_SECONDS
        similarity = np.where(near, similarity, -1.0)

        attached = 0
        touched: dict[int, Cluster] = dict()
        for index, doc in enumerate(loose):
            best = int(similarity[index].argmax())
            if similarity[index][best] < threshold:
                continue
            candidate = published[best]
            story = Cluster()
            story.add(doc)
            if judge_relation(story, [candidate]).verdict != SAME:
                continue
            candidate.add(doc)
            attached += 1
            if candidate.clid is not None:
                touched[candidate.clid] = candidate
            logging.info(
                "Document from %s joins message %s",
                doc.channel_id,
                [m.message_id for m in candidate.messages],
            )

        if attached:
            posted_clusters.invalidate_caches()
        for cluster in touched.values():
            for message in cluster.messages:
                self.refresh_post(cluster, message.issue, max_time_updated)
        return attached

    def find_relation(
        self, cluster: Cluster, posted_clusters: Clusters, issue_name: str
    ) -> Relation:
        """What this story is to the posts already published: same, next, or new.

        Two stages, because neither half can do the job alone. The cosine
        narrows hundreds of published posts to a couple of plausible ones, which
        is what it is good for; it cannot make the decision, and the numbers say
        so plainly. Over a week of production, pairs of separately published
        posts that turned out to be the same event and pairs that merely share a
        template overlap almost completely — an AUC of 0.70, and 74% the best
        accuracy any single threshold could reach. The most similar pair of the
        whole week, 0.984, was two different nights of explosions over Kyiv.

        `related_threshold` is now that narrowing floor rather than a verdict,
        which is why it sits well below where a verdict would have to.
        """
        candidates = nearest_clusters(
            cluster,
            posted_clusters.get_embedded_clusters(get_current_ts(), issue_name),
            threshold=float(self.config["related_threshold"]),
        )
        return judge_relation(cluster, candidates)
