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
from nyan.publish import notify_published
from nyan.ranker import Ranker
from nyan.jev import JevRelationShadow
from nyan.mongo import get_rank_shadow_collection, get_relation_shadow_collection
from nyan.rank_shadow import SEEN_TTL, RankShadow, log_records
from nyan.relation import (
    FOLLOW_UP,
    SAME,
    Relation,
    judge_relation,
    nearest_clusters,
)
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
# again. Long enough to cover most of a story's life rather than a single
# iteration, since the same documents come back around on every pass. Within
# it, a story that really did fail returns early only once enough new sources
# have joined it for the overlap with the held attempt to fall below
# `similar_min_intersection_ratio`. Past it, `find_pending` stops holding the
# story and it is published anyway: a possible duplicate hours later is
# preferred to a story that never ran.
DEFAULT_PENDING_SEND_TTL = 3 * 3600

# How long after a post went out a document has to appear before it may be
# news the post does not carry. The clusterer holds a published post's
# documents together, so a development hours later — the toll confirmed, the
# payment made — lands in the same cluster as the post and is recognized by
# its URLs as the post itself. Earlier than this it is the first wave still
# arriving, and folding it in is right.
DEFAULT_FOLLOW_UP_GAP = 3600

# How long those late documents are held out of the post while they are too
# few to stand as a story of their own. A development is reported by several
# channels within the hour; held, they can gather into something the judge is
# asked about, where taken in one at a time they never would. Past this they
# are folded in, as everything was before.
DEFAULT_FOLLOW_UP_HOLD = 3600


def _cluster_of(docs: list[Document]) -> Cluster:
    cluster = Cluster()
    for doc in docs:
        cluster.add(doc)
    return cluster


def _unit_rows(embeddings: list[list[float]]) -> NDArray[np.float32]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return cast("NDArray[np.float32]", matrix / np.where(norms == 0.0, 1.0, norms))


class Daemon:
    #: Declared here as well as set in `__init__`, so a daemon assembled without
    #: it — as the tests do, to exercise one method — simply has no shadow.
    relation_shadow: JevRelationShadow | None = None
    rank_shadow: RankShadow | None = None
    mongo_config_path: str | None = None

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
        if "shadow" in self.ranker.config:
            self.rank_shadow = RankShadow(ranker_config_path, self.ranker.config["shadow"])

        assert os.path.exists(daemon_config_path)
        with open(daemon_config_path) as r:
            self.config: dict[str, Any] = json.load(r)

        if "jev_relation_shadow" in self.config:
            self.relation_shadow = JevRelationShadow(self.config["jev_relation_shadow"])
        # Where the relation shadow writes is set per iteration by `__call__`,
        # because the path arrives with the call rather than the constructor.

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
        self.mongo_config_path = mongo_config_path
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
        attached = self.attach_loose_documents(
            annotated_docs, posted_clusters, mongo_config_path
        )
        logging.info("%d documents joined a post they had missed", attached)

        new_clusters: list[Cluster] = self.clusterer(
            annotated_docs, posted_clusters.published_documents()
        )
        logging.info("%d clusters overall", len(new_clusters))

        ranked_clusters: dict[str, list[Cluster]] = self.ranker(new_clusters)
        num_clusters = sum(len(cl) for cl in ranked_clusters.values())
        logging.info("%d clusters in all issues after filtering", num_clusters)
        self.shadow_rank(new_clusters, ranked_clusters, mongo_config_path)

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

    def shadow_rank(
        self,
        clusters: list[Cluster],
        ranked: dict[str, list[Cluster]],
        mongo_config_path: str | None,
    ) -> None:
        """What the student-aware ranker would have done differently, recorded.

        Never raises into the daemon, and changes nothing it goes on to send:
        see `RankShadow.compare`.
        """
        if self.rank_shadow is None:
            return
        try:
            if mongo_config_path and not self.rank_shadow.restored:
                # Once per start: what an earlier process recorded is recorded.
                self.rank_shadow.remember(
                    get_rank_shadow_collection(mongo_config_path).find(
                        {"ts": {"$gte": get_current_ts() - SEEN_TTL}},
                        {"first_url": 1, "issue": 1, "change": 1, "ts": 1},
                    )
                )
            records = self.rank_shadow.compare(clusters, ranked)
            log_records(records)
            if records and mongo_config_path:
                get_rank_shadow_collection(mongo_config_path).insert_many(records)
        except Exception:
            logging.exception("Rank shadow failed")

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
            absorbed, follow_up = self.split_off_follow_up(
                cluster, posted_cluster, posted_clusters, issue_name
            )
            self.update_posted_cluster(
                absorbed,
                posted_cluster,
                posted_clusters,
                issue_name,
                max_time_updated,
            )
            if follow_up is not None:
                self.publish(
                    follow_up,
                    issue_name,
                    posted_clusters,
                    posted_clusters_path,
                    mongo_config_path,
                    parent=posted_cluster,
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
        self.publish(
            cluster,
            issue_name,
            posted_clusters,
            posted_clusters_path,
            mongo_config_path,
            parent=parent,
        )

    def publish(
        self,
        cluster: Cluster,
        issue_name: str,
        posted_clusters: Clusters,
        posted_clusters_path: str | None,
        mongo_config_path: str | None,
        parent: Cluster | None = None,
    ) -> None:
        """Send a story as a new message, under `parent`'s post if it has one."""
        reply_to = None
        if parent is not None:
            # Kept whether or not the parent has a message in this issue: the
            # story is a follow-up either way, and the thread on the site is not
            # tied to which feed happened to carry the parent.
            cluster.reply_to_clid = parent.clid
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

        # After the save, not before: the site answers by reading the story
        # from Mongo, and a ping ahead of the record would find nothing.
        assert cluster.clid is not None
        notify_published([cluster.clid])

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

    def split_off_follow_up(
        self,
        cluster: Cluster,
        posted_cluster: Cluster,
        posted_clusters: Clusters,
        issue_name: str,
    ) -> tuple[Cluster, Cluster | None]:
        """What of `cluster` the post takes in, and what goes out on its own.

        `find_similar` recognizes a post by its URLs, and the clusterer keeps a
        post's documents together — so a development reported hours later
        arrives glued to the post it follows and never reaches the judge in
        `find_relation`. Folded in, it was lost: past the editing window nothing
        is sent, and inside it the text stays as written until coverage grows a
        whole generation.

        So documents that appeared well after the post went out are asked
        about, as a story of their own against the post, once they are enough
        sources to be published as one. The judge is the same as on the
        publish boundary and runs once per batch: a `same` batch is taken in and
        is not new to the post again. Every other answer, a failed call
        included, takes them in as before.

        Documents another post already carries are neither taken in nor asked
        about: they belong to that post, typically a follow-up split off here
        earlier and now clustered beside its parent.
        """
        published = posted_clusters.urls2messages[issue_name]
        fresh = [
            doc
            for doc in cluster.docs
            if not posted_cluster.has(doc) and normalize_url(doc.url) not in published
        ]
        since = (posted_cluster.create_time or posted_cluster.pub_time_percentile) + int(
            self.config.get("follow_up_gap", DEFAULT_FOLLOW_UP_GAP)
        )
        late = [doc for doc in fresh if doc.pub_time >= since]
        if not late:
            return _cluster_of(fresh), None

        late_cluster = _cluster_of(late)
        early = [doc for doc in fresh if doc.pub_time < since]
        if not self.ranker.stands_alone(late_cluster, issue_name):
            hold = int(self.config.get("follow_up_hold", DEFAULT_FOLLOW_UP_HOLD))
            now = get_current_ts()
            ripe = [doc for doc in late if now - doc.pub_time >= hold]
            return _cluster_of(early + ripe), None

        # A follow-up already sent without an answer: held like any other, and
        # without asking the judge again on every pass while it is.
        pending = posted_clusters.find_pending(
            late_cluster,
            issue_name,
            min_intersection_ratio=self.config["similar_min_intersection_ratio"],
            current_ts=get_current_ts(),
            ttl=self.config.get("pending_send_ttl", DEFAULT_PENDING_SEND_TTL),
        )
        if pending is not None:
            return _cluster_of(early), None

        relation = self.judge(late_cluster, [posted_cluster], "late")
        if relation.verdict != FOLLOW_UP:
            return _cluster_of(fresh), None
        logging.info(
            "%d late docs follow up cluster %s: %s",
            len(late),
            posted_cluster.clid,
            late_cluster.cropped_title,
        )
        return _cluster_of(early), late_cluster

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
        if self.client.update_post(message, post):
            posted_cluster.saved_hash = posted_cluster.hash

    def attach_loose_documents(
        self,
        docs: list[Document],
        posted_clusters: Clusters,
        mongo_config_path: str | None = None,
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

        Each pair is asked about once and the answer is kept. A "yes" keeps
        itself: the document joins the post and is a published document from
        then on. A "no" used to keep nothing, so the same question went to the
        model on every iteration until the document aged out of
        `documents_offset` a day later — some forty documents a day, asked
        about thirty to fifty times an hour each, which came to 90% of
        everything this feed sent the model.
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
        refused: dict[int, Cluster] = dict()
        for index, doc in enumerate(loose):
            best = int(similarity[index].argmax())
            if similarity[index][best] < threshold:
                continue
            candidate = published[best]
            # Already asked, and the answer was no. Neither text has changed
            # since, so the model would be paid to write it out again. The
            # same holds for another channel's word-for-word copy of it.
            if candidate.refuses(doc):
                continue
            story = Cluster()
            story.add(doc)
            # A copy of words the post already carries needs no question: the
            # post has told the reader exactly this.
            if (
                not candidate.holds_text_of(doc)
                and self.judge(story, [candidate], "attach").verdict != SAME
            ):
                candidate.refuse(doc)
                if candidate.clid is not None:
                    refused[candidate.clid] = candidate
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
        # Written here rather than left to the save at the end of the iteration:
        # that one skips clusters whose newest document is over a day old, and a
        # refusal that never reached storage is a question asked all over again
        # the next time the container restarts.
        if mongo_config_path:
            for cluster in refused.values():
                posted_clusters.save_one_to_mongo(mongo_config_path, cluster)
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
        return self.judge(cluster, candidates, "publish")

    def judge(self, story: Cluster, candidates: list[Cluster], site: str) -> Relation:
        """The LLM judge's verdict, which is the one acted on, with Jev's recorded beside it.

        `site` names which of the three questions this is — the publish
        boundary, late documents under a post, a loose document joining one —
        because they differ in what a wrong answer costs, and the comparison
        has to be read per site to mean anything.

        The shadow never raises into the daemon: a failure to ask or to write
        leaves the LLM's verdict standing and costs one log line.
        """
        relation = judge_relation(story, candidates)
        if self.relation_shadow is None or not self.relation_shadow.enabled:
            return relation
        try:
            record = self.relation_shadow(story, candidates, relation, site)
            if record is not None and self.mongo_config_path:
                get_relation_shadow_collection(self.mongo_config_path).insert_one(record)
        except Exception:
            logging.exception("Jev relation shadow failed for '%s'", story.cropped_title)
        return relation
