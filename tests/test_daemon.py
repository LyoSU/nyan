from typing import Any
from collections.abc import Sequence

from nyan.client import MessageId
from nyan.clusters import Cluster, Clusters
from nyan.daemon import Daemon
from nyan.document import Document
from nyan.util import get_current_ts


def _daemon(**config: Any) -> Daemon:
    """A Daemon carrying nothing but its config.

    `__init__` loads the embedder, the classifier and a Telegram client. Nothing
    tested here needs any of them, and building them would turn a unit test into
    a model download.
    """
    daemon = object.__new__(Daemon)
    daemon.config = {
        "related_threshold": 0.89,
        "similar_min_intersection_ratio": 0.25,
        "max_time_updated": 3600,
        "sleep_time": 0,
        **config,
    }
    return daemon


def _cluster(
    embedding: list[float],
    message_id: int | None = None,
    headline: str = "",
    age_seconds: int = 600,
) -> Cluster:
    """One cluster of one document. Published when given a message id."""
    cluster = Cluster()
    now = get_current_ts()
    post_id = message_id if message_id is not None else 0
    cluster.add(
        Document(
            url=f"https://t.me/uanews/{post_id}",
            channel_id="uanews",
            post_id=post_id,
            views=100,
            pub_time=now - age_seconds,
            fetch_time=now,
            text="Текст",
            patched_text="Текст",
            groups={"main": "blue"},
            issue="main",
            language="uk",
            embedding=embedding,
        )
    )
    if message_id is not None:
        cluster.messages.append(MessageId(message_id=message_id, issue="main"))
    if headline:
        # As a cluster loaded from Mongo carries it: already written and paid
        # for, so reading it asks nothing of the LLM.
        cluster.saved_analysis = {"headline": headline, "generation": "0-"}
    return cluster


def _posted(*clusters: Cluster) -> Clusters:
    posted = Clusters()
    for cluster in clusters:
        posted.add(cluster)
    return posted


def test_the_reply_target_carries_the_neighbour_and_its_message() -> None:
    parent = _cluster(
        [1.0, 0.0], message_id=11, headline="Росія вдарила по Запоріжжю", age_seconds=3600
    )
    cluster = _cluster([1.0, 0.02])

    target = _daemon().find_reply_target(cluster, _posted(parent), "main")

    assert target is not None
    neighbour, reply_to = target
    assert reply_to == 11
    assert neighbour.stored_headline == "Росія вдарила по Запоріжжю"


def test_a_distant_neighbour_is_not_a_reply_target() -> None:
    parent = _cluster([1.0, 0.0], message_id=11, headline="Курс гривні", age_seconds=3600)
    cluster = _cluster([0.0, 1.0])

    assert _daemon().find_reply_target(cluster, _posted(parent), "main") is None


class _FakeRenderer:
    def __init__(self) -> None:
        self.rendered_under: list[str] = []

    def render_cluster(
        self, cluster: Cluster, issue_name: str, post_format: str | None = None
    ) -> str:
        self.rendered_under.append(cluster.reply_to_headline)
        return "post"

    def render_discussion_message(self, doc: Document) -> str:
        return "discussion"


class _FakeClient:
    def __init__(self, issues: Sequence[str] = ("main",)) -> None:
        self.reply_to: int | None = None
        self.issues = set(issues)
        self.mapping_updates = 0
        self.discussion_messages: list[str] = []
        self.updated: list[int] = []

    def has_issue(self, issue_name: str) -> bool:
        return issue_name in self.issues

    def update_discussion_mapping(self, issue_name: str) -> None:
        self.mapping_updates += 1

    def send_post(self, post: Any, issue_name: str, reply_to: int | None = None) -> MessageId:
        self.reply_to = reply_to
        return MessageId(message_id=99, issue=issue_name)

    def get_discussion(self, message: MessageId) -> MessageId:
        return MessageId(
            message_id=message.message_id + 1000,
            issue=message.issue,
            from_discussion=True,
        )

    def send_discussion_message(self, text: str, message: Any) -> None:
        self.discussion_messages.append(text)

    def update_post(self, message: MessageId, post: Any) -> None:
        self.updated.append(message.message_id)


def test_the_post_is_written_knowing_what_it_stands_under() -> None:
    """The text of a post is produced lazily, inside render_cluster.

    So the neighbour has to be found before rendering rather than just before
    sending: while the lookup came after it, every reply was written without
    knowing that the reader can see a nearly identical headline right above it.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster(
        [1.0, 0.0], message_id=11, headline="Росія вдарила по Запоріжжю", age_seconds=3600
    )
    posted = _posted(parent)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", posted, None, None)

    renderer: Any = daemon.renderer
    assert renderer.rendered_under == ["Росія вдарила по Запоріжжю"]
    client: Any = daemon.client
    assert client.reply_to == 11


def test_a_post_with_no_neighbour_stands_under_nothing() -> None:
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", Clusters(), None, None)

    renderer: Any = daemon.renderer
    assert renderer.rendered_under == [""]
    client: Any = daemon.client
    assert client.reply_to is None


def test_the_sources_are_not_repeated_in_the_comments() -> None:
    """Mirroring every source post posted the whole channel a second time.

    Off unless `send_docs_to_discussion` says otherwise — and with it off, the
    discussion mapping is not worth a getUpdates either.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]

    daemon.send_cluster(_cluster([1.0, 0.0]), "main", Clusters(), None, None)

    client: Any = daemon.client
    assert client.discussion_messages == []
    assert client.mapping_updates == 0


def test_the_toggle_brings_the_comments_back() -> None:
    daemon = _daemon(send_docs_to_discussion=True)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]

    daemon.send_cluster(_cluster([1.0, 0.0]), "main", Clusters(), None, None)

    client: Any = daemon.client
    assert client.discussion_messages == ["discussion"]
    assert client.mapping_updates == 2


def test_new_docs_still_join_a_posted_cluster_with_the_comments_off() -> None:
    """Taking a document in is not the same thing as commenting with it.

    Both used to happen in one loop, so silencing the comments must not stop the
    cluster from growing — the post itself is rewritten from those documents.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    posted_cluster = _cluster([1.0, 0.0], message_id=11)
    incoming = _cluster([1.0, 0.0], message_id=12)
    incoming.messages.clear()

    daemon.update_posted_cluster(
        incoming, posted_cluster, _posted(posted_cluster), "main", 3600
    )

    assert posted_cluster.has(incoming.docs[0])
    client: Any = daemon.client
    assert client.discussion_messages == []
    assert client.updated == [11]


def test_an_issue_with_no_channel_is_dropped_before_rendering(caplog: Any) -> None:
    """An issue absent from the client config has nowhere to post.

    Discovering that inside the send meant rendering every one of its clusters
    first — and the post's text is written inside render_cluster, by the LLM.
    Both warnings then arrived per cluster; this one arrives per issue.
    """
    daemon = _daemon()
    daemon.client = _FakeClient(issues=("main",))  # type: ignore[assignment]

    postable = daemon.drop_unpostable_issues(
        {
            "main": [_cluster([1.0, 0.0])],
            "war": [_cluster([0.0, 1.0]), _cluster([1.0, 1.0])],
        }
    )

    assert list(postable) == ["main"]
    assert "war" in caplog.text
    assert "2 clusters" in caplog.text


def test_every_configured_issue_survives_the_check() -> None:
    daemon = _daemon()
    daemon.client = _FakeClient(issues=("main", "war"))  # type: ignore[assignment]
    ranked = {"main": [_cluster([1.0, 0.0])], "war": [_cluster([0.0, 1.0])]}

    assert daemon.drop_unpostable_issues(ranked) == ranked
