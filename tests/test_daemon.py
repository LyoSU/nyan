from typing import Any
from collections.abc import Sequence

from nyan.client import MessageId
from nyan.clusters import Cluster, Clusters
from nyan.daemon import Daemon
from nyan.document import Document
from nyan.relation import FOLLOW_UP, SAME, UNRELATED, Relation
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
    message_issue: str = "main",
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
        cluster.messages.append(MessageId(message_id=message_id, issue=message_issue))
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


def _patch_judge(monkeypatch: Any, verdict: str, index: int = 0) -> list[Any]:
    """Answer for the judge, and the candidate lists it was given."""
    seen: list[Any] = []

    def fake_judge(cluster: Cluster, candidates: Sequence[Cluster]) -> Relation:
        seen.append(candidates)
        if verdict == UNRELATED or not candidates:
            return Relation(UNRELATED)
        return Relation(verdict, candidates[index])

    monkeypatch.setattr("nyan.daemon.judge_relation", fake_judge)
    return seen


def test_the_relation_carries_the_neighbour_it_is_about(monkeypatch: Any) -> None:
    parent = _cluster(
        [1.0, 0.0], message_id=11, headline="Росія вдарила по Запоріжжю", age_seconds=3600
    )
    _patch_judge(monkeypatch, FOLLOW_UP)

    relation = _daemon().find_relation(_cluster([1.0, 0.02]), _posted(parent), "main")

    assert relation.verdict == FOLLOW_UP
    assert relation.cluster is parent


def test_a_distant_neighbour_is_never_asked_about(monkeypatch: Any) -> None:
    """The cosine still does the narrowing, so the judge runs at most once.

    Measured over a week of production this leaves 1.7 candidates per post and
    none at all for 45% of them.
    """
    parent = _cluster([1.0, 0.0], message_id=11, headline="Курс гривні", age_seconds=3600)
    seen = _patch_judge(monkeypatch, SAME)

    relation = _daemon().find_relation(_cluster([0.0, 1.0]), _posted(parent), "main")

    assert relation.verdict == UNRELATED
    assert list(seen[0]) == []


class _FakeRenderer:
    def __init__(self) -> None:
        self.rendered_under: list[str] = []
        self.rendered_knowing: list[str] = []

    def render_cluster(
        self, cluster: Cluster, issue_name: str, post_format: str | None = None
    ) -> str:
        self.rendered_under.append(cluster.reply_to_headline)
        self.rendered_knowing.append(cluster.reply_to_text)
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
        self.sent = 0

    def has_issue(self, issue_name: str) -> bool:
        return issue_name in self.issues

    def update_discussion_mapping(self, issue_name: str) -> None:
        self.mapping_updates += 1

    def send_post(self, post: Any, issue_name: str, reply_to: int | None = None) -> MessageId:
        self.reply_to = reply_to
        self.sent += 1
        return MessageId(message_id=99, issue=issue_name)

    def get_discussion(self, message: MessageId) -> MessageId:
        return MessageId(
            message_id=message.message_id + 1000,
            issue=message.issue,
            from_discussion=True,
        )

    def send_discussion_message(self, text: str, message: Any) -> None:
        self.discussion_messages.append(text)

    def update_post(self, message: MessageId, post: Any) -> bool:
        self.updated.append(message.message_id)
        return True


def test_the_post_is_written_knowing_what_it_stands_under(monkeypatch: Any) -> None:
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
    _patch_judge(monkeypatch, FOLLOW_UP)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", _posted(parent), None, None)

    renderer: Any = daemon.renderer
    assert renderer.rendered_under == ["Росія вдарила по Запоріжжю"]
    client: Any = daemon.client
    assert client.reply_to == 11


def test_a_follow_up_remembers_which_story_it_answers(monkeypatch: Any) -> None:
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster(
        [1.0, 0.0], message_id=11, headline="Росія вдарила по Запоріжжю", age_seconds=3600
    )
    parent.clid = 64099
    child = _cluster([1.0, 0.02])
    _patch_judge(monkeypatch, FOLLOW_UP)

    daemon.send_cluster(child, "main", _posted(parent), None, None)

    assert child.reply_to_clid == 64099


def test_a_post_with_no_neighbour_stands_under_nothing(monkeypatch: Any) -> None:
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    _patch_judge(monkeypatch, UNRELATED)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", Clusters(), None, None)

    renderer: Any = daemon.renderer
    assert renderer.rendered_under == [""]
    client: Any = daemon.client
    assert client.reply_to is None


def test_the_same_story_joins_the_post_that_already_told_it(monkeypatch: Any) -> None:
    """The duplicate the whole change is about.

    The clusterer cut this story away from the post it belongs to — measured on
    production, 47% of published clusters no longer survive re-clustering as one
    piece — and the documents share too few URLs with it for `find_similar` to
    recognize. Recognized by what it says instead, it is folded in and the
    published post is edited rather than a second one being sent.
    """
    daemon = _daemon(max_time_updated=10800)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибух", age_seconds=3600)
    _patch_judge(monkeypatch, SAME)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", _posted(parent), None, None)

    client: Any = daemon.client
    assert client.reply_to is None, "no second post"
    assert client.updated == [11], "the published one was edited"
    assert len(parent.docs) == 2, "the new documents joined it"


def test_the_same_story_too_late_to_edit_is_absorbed_without_a_second_post(
    monkeypatch: Any,
) -> None:
    """The same story resurfacing after the post stopped being re-rendered.

    An evening story is re-told by the morning wave of channels, always past
    the editing window — measured over a month of production, one such pair
    every two days. Replying with a full post read as a duplicate, while the
    very same wave was absorbed silently whenever the clusterer happened to
    merge it on its own. "Same" means the reader has already been told this,
    so the documents are taken in and nothing is sent: `refresh_post` declines
    the edit by itself, and a genuine development is a `follow_up`, which
    still goes out under the post.
    """
    daemon = _daemon(max_time_updated=600)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибух", age_seconds=7200)
    _patch_judge(monkeypatch, SAME)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", _posted(parent), None, None)

    client: Any = daemon.client
    assert client.sent == 0, "the reader has already been told this"
    assert client.updated == [], "past editing, the post stays as it was sent"
    assert len(parent.docs) == 2, "but the documents still join it"


def test_the_same_story_this_issue_never_saw_is_still_published(
    monkeypatch: Any,
) -> None:
    """"Same" is same for a reader — and this issue's readers never saw it.

    A cluster's documents can belong to several issues while its post went out
    in only one of them. Folding the story into that post would tell this
    issue's readers nothing, so it goes out on its own here.
    """
    daemon = _daemon(max_time_updated=600)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster(
        [1.0, 0.0],
        message_id=11,
        headline="Вибух",
        age_seconds=7200,
        message_issue="war",
    )
    _patch_judge(monkeypatch, SAME)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", _posted(parent), None, None)

    client: Any = daemon.client
    assert client.sent == 1
    assert client.reply_to is None, "nothing of this story stands in this issue"
    assert len(parent.docs) == 1, "the other issue's post is left alone"


def test_a_reply_is_written_knowing_the_text_it_stands_under(
    monkeypatch: Any,
) -> None:
    """The neighbour's headline alone was too thin to stop the re-telling.

    The reasons behind a decision rarely fit its headline, so a reply written
    against the headline honestly took them for news and re-told the whole
    story. The model gets the text the reader sees above, not just its title.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster(
        [1.0, 0.0], message_id=11, headline="Трамп скоротив навчання", age_seconds=3600
    )
    assert parent.saved_analysis is not None
    parent.saved_analysis["summary"] = {
        "headline": "Трамп скоротив навчання",
        "blocks": [{"type": "text", "text": "Причина — стосунки з Кімом."}],
    }
    _patch_judge(monkeypatch, FOLLOW_UP)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", _posted(parent), None, None)

    renderer: Any = daemon.renderer
    assert renderer.rendered_knowing == ["Причина — стосунки з Кімом."]


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


def _loose_doc(
    embedding: list[float],
    age_seconds: int = 600,
    post_id: int = 500,
    text: str = "Текст іншого каналу",
    channel_id: str = "other",
) -> Document:
    now = get_current_ts()
    return Document(
        url=f"https://t.me/{channel_id}/{post_id}",
        channel_id=channel_id,
        post_id=post_id,
        views=100,
        pub_time=now - age_seconds,
        fetch_time=now,
        text=text,
        patched_text=text,
        groups={"main": "blue"},
        issue="main",
        language="uk",
        embedding=embedding,
    )


def test_a_lone_document_joins_the_post_it_belongs_to(monkeypatch: Any) -> None:
    """A channel that carried the story but never reached a cluster of its own.

    Neither of the other two changes can see this one: holding published
    documents in place keeps what a post already has, and the judge at the
    publish boundary compares clusters that got that far. A single document that
    never gathered four independent sources gets to neither. Measured over a
    day of production, 41 documents sit closer than 0.94 to a post they are not
    in — the closest being Суспільне writing "У Києві чути вибухи" against a
    post whose first line is the same sentence.
    """
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    _patch_judge(monkeypatch, SAME)

    attached = daemon.attach_loose_documents([_loose_doc([1.0, 0.02])], _posted(parent))

    assert attached == 1
    assert len(parent.docs) == 2
    client: Any = daemon.client
    assert client.updated == [11], "the post was re-rendered with the new source"


def test_a_lone_document_the_judge_refuses_stays_out(monkeypatch: Any) -> None:
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    _patch_judge(monkeypatch, UNRELATED)

    assert daemon.attach_loose_documents([_loose_doc([1.0, 0.02])], _posted(parent)) == 0
    assert len(parent.docs) == 1


def test_a_refused_document_is_never_asked_about_twice(monkeypatch: Any) -> None:
    """The same pair went to the model on every iteration for a whole day.

    A document that sits close to a post it does not belong to is offered again
    on the next pass, and the next, until it ages out of `documents_offset` —
    with the same two texts, so the request is byte for byte the one already
    answered. Over two days of production that was 29,178 of 32,179 requests.
    """
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, UNRELATED)
    doc = _loose_doc([1.0, 0.02])

    assert daemon.attach_loose_documents([doc], _posted(parent)) == 0
    assert daemon.attach_loose_documents([doc], _posted(parent)) == 0

    assert len(seen) == 1, "the answer was already known on the second pass"
    assert len(parent.docs) == 1


def test_a_refusal_outlives_the_process(monkeypatch: Any) -> None:
    """Containers restart, so an answer held in memory is an answer paid for again."""
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, UNRELATED)
    doc = _loose_doc([1.0, 0.02])

    assert daemon.attach_loose_documents([doc], _posted(parent)) == 0
    reloaded = Cluster.deserialize(parent.serialize())

    assert daemon.attach_loose_documents([doc], _posted(reloaded)) == 0
    assert len(seen) == 1


def test_a_copy_of_a_refused_text_is_not_asked_about_again(monkeypatch: Any) -> None:
    """Channels repost each other word for word, and each copy is a document.

    Nine copies of "Від ранку триває ліквідація наслідків" were offered to one
    post within a minute, and the judge was paid nine times for one answer —
    then again for every copy that arrived after a restart.
    """
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, FOLLOW_UP)
    copies = [
        _loose_doc([1.0, 0.02], post_id=i, channel_id=f"repost{i}", text="Від ранку триває")
        for i in range(3)
    ]

    assert daemon.attach_loose_documents(copies[:2], _posted(parent)) == 0
    reloaded = Cluster.deserialize(parent.serialize())
    assert daemon.attach_loose_documents(copies[2:], _posted(reloaded)) == 0

    assert len(seen) == 1


def test_a_copy_of_words_the_post_carries_joins_without_asking(monkeypatch: Any) -> None:
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, UNRELATED)
    copy = _loose_doc([1.0, 0.02], text=parent.docs[0].patched_text or "")

    assert daemon.attach_loose_documents([copy], _posted(parent)) == 1
    assert seen == []
    assert parent.has(copy)


def test_a_document_refused_by_one_post_is_still_offered_to_another(
    monkeypatch: Any,
) -> None:
    """The refusal is about a pair, not about the document.

    A post published after the refusal may be the one the document belongs to,
    and nothing that was asked before covers it.
    """
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    refuser = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    doc = _loose_doc([1.0, 0.02])
    _patch_judge(monkeypatch, UNRELATED)
    assert daemon.attach_loose_documents([doc], _posted(refuser)) == 0

    later = _cluster([1.0, 0.02], message_id=12, headline="Вибухи вдруге", age_seconds=600)
    seen = _patch_judge(monkeypatch, SAME)

    assert daemon.attach_loose_documents([doc], _posted(refuser, later)) == 1
    assert len(seen) == 1
    assert later.has(doc)


def test_a_document_far_from_every_post_is_never_asked_about(monkeypatch: Any) -> None:
    """The floor is far above the one used between clusters.

    A single document carries much less evidence than a cluster does, so it has
    to look almost identical before it is worth a question. At 0.94 that is some
    forty documents a day; at 0.86, the floor used between clusters, it would be
    thousands.
    """
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, SAME)

    assert daemon.attach_loose_documents([_loose_doc([0.9, 0.44])], _posted(parent)) == 0
    assert seen == []


def test_a_document_already_in_a_post_is_left_alone(monkeypatch: Any) -> None:
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, SAME)

    attached = daemon.attach_loose_documents(list(parent.docs), _posted(parent))

    assert attached == 0
    assert seen == []


def test_a_document_from_another_hour_is_not_offered(monkeypatch: Any) -> None:
    """Time is the one thing the text cannot say.

    "Вибухи в Києві" is written the same way on every night it happens, so a
    document is only ever offered to a post published around its own time.
    """
    daemon = _daemon(attach_threshold=0.94)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    parent = _cluster([1.0, 0.0], message_id=11, headline="Вибухи", age_seconds=1800)
    seen = _patch_judge(monkeypatch, SAME)

    attached = daemon.attach_loose_documents(
        [_loose_doc([1.0, 0.02], age_seconds=1800 + 12 * 3600)], _posted(parent)
    )

    assert attached == 0
    assert seen == []


class _LosesTheAnswerClient(_FakeClient):
    """Telegram takes the post; the answer never comes back.

    What the daemon sees — `None` — is what it sees when nothing was published
    at all, and the two are indistinguishable from here. This is the client of
    the night of 26-08-2026, when 39955 and 39956 carried the same story.
    """

    def send_post(
        self, post: Any, issue_name: str, reply_to: int | None = None
    ) -> MessageId | None:
        self.sent += 1
        return None


def test_a_post_whose_answer_was_lost_is_not_sent_a_second_time(
    monkeypatch: Any,
) -> None:
    """The story goes out once even when the confirmation does not arrive.

    Nothing was saved for a send that returned `None`, so the next iteration met
    the same documents as a story never told and published them again. The post
    the reader already had was invisible to every check: `find_similar` and the
    judge both read `messages`, which an unconfirmed post has none of.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _LosesTheAnswerClient()  # type: ignore[assignment]
    _patch_judge(monkeypatch, UNRELATED)
    posted = Clusters()

    daemon.send_cluster(_cluster([1.0, 0.0]), "main", posted, None, None)
    daemon.send_cluster(_cluster([1.0, 0.0]), "main", posted, None, None)

    client: Any = daemon.client
    assert client.sent == 1, "the story went out once"


def test_an_unconfirmed_send_survives_the_process_that_made_it() -> None:
    """The attempt is on record in storage, not only in the running daemon.

    An exception on the way out — a read timeout is the one that happens —
    reaches `run`, which does not catch it, and the process ends. Held in
    memory alone the attempt would die with it, and the daemon that comes back
    up would meet the story as one never told.
    """
    cluster = _cluster([1.0, 0.0])
    posted = Clusters()
    posted.mark_pending(cluster, "main", get_current_ts())

    after_restart = Clusters()
    after_restart.add(Cluster.fromdict(cluster.asdict()))

    assert (
        after_restart.find_pending(
            _cluster([1.0, 0.0]),
            "main",
            min_intersection_ratio=0.15,
            current_ts=get_current_ts(),
            ttl=3600,
        )
        is not None
    )


def test_a_story_held_long_enough_is_published_after_all(monkeypatch: Any) -> None:
    """Holding is not losing.

    A send that really did fail leaves the same record as one that quietly
    succeeded, so the hold has to end: past the window the story is published,
    on the reading that a duplicate hours later costs less than a story the
    feed never carried.
    """
    daemon = _daemon(pending_send_ttl=3600)
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    _patch_judge(monkeypatch, UNRELATED)
    held = _cluster([1.0, 0.0])
    posted = Clusters()
    posted.mark_pending(held, "main", get_current_ts() - 7200)

    daemon.send_cluster(_cluster([1.0, 0.0]), "main", posted, None, None)

    client: Any = daemon.client
    assert client.sent == 1


def _record_pings(monkeypatch: Any) -> list[list[int]]:
    pings: list[list[int]] = []
    monkeypatch.setattr("nyan.daemon.notify_published", lambda clids: pings.append(clids))
    return pings


def test_a_confirmed_post_announces_its_story_to_the_site(monkeypatch: Any) -> None:
    """The site drops its caches for the story and submits it to IndexNow.

    One ping per post, naming the story by its clid, as soon as Telegram has
    confirmed the message.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    _patch_judge(monkeypatch, UNRELATED)
    pings = _record_pings(monkeypatch)
    cluster = _cluster([1.0, 0.02])

    daemon.send_cluster(cluster, "main", Clusters(), None, None)

    assert cluster.clid is not None
    assert pings == [[cluster.clid]]


def test_an_unanswered_send_tells_the_site_nothing(monkeypatch: Any) -> None:
    """Whether the post exists is unknown, so there is nothing to announce yet."""
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _LosesTheAnswerClient()  # type: ignore[assignment]
    _patch_judge(monkeypatch, UNRELATED)
    pings = _record_pings(monkeypatch)

    daemon.send_cluster(_cluster([1.0, 0.02]), "main", Clusters(), None, None)

    assert pings == []


class _RefusingClient(_FakeClient):
    def update_post(self, message: MessageId, post: Any) -> bool:
        self.updated.append(message.message_id)
        return False


def _grown_post() -> Cluster:
    """A published post that has just taken in a channel it did not show."""
    posted = _cluster([1.0, 0.0], message_id=11)
    posted.saved_hash = posted.hash
    newcomer = _cluster([1.0, 0.0], message_id=12).docs[0]
    newcomer.channel_id = "othernews"
    posted.add(newcomer)
    return posted


def test_a_rejected_edit_is_tried_again_on_the_next_pass() -> None:
    """Stored as done, a refused edit left the post stale for good: the next
    pass read the new hash back, saw nothing changed, and never edited again."""
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _RefusingClient()  # type: ignore[assignment]
    posted = _grown_post()
    shown = posted.saved_hash

    daemon.refresh_post(posted, "main", 3600)

    assert posted.changed()
    assert posted.asdict()["hash"] == shown


def test_a_confirmed_edit_is_what_gets_stored() -> None:
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    posted = _grown_post()

    daemon.refresh_post(posted, "main", 3600)

    assert not posted.changed()
    assert posted.asdict()["hash"] == posted.hash


def test_a_new_post_is_stored_as_shown() -> None:
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    cluster = _cluster([1.0, 0.0])

    daemon.send_cluster(cluster, "main", Clusters(), None, None)

    assert cluster.messages
    assert not cluster.changed()


# ------------------------------------------ developments glued to their post


class _TwoSourceRanker:
    """Two channels are a story; one is not."""

    def stands_alone(self, cluster: Cluster, issue_name: str) -> bool:
        return len({doc.channel_id for doc in cluster.docs}) >= 2


def _doc(channel_id: str, post_id: int, age_seconds: int) -> Document:
    now = get_current_ts()
    return Document(
        url=f"https://t.me/{channel_id}/{post_id}",
        channel_id=channel_id,
        post_id=post_id,
        views=100,
        pub_time=now - age_seconds,
        fetch_time=now,
        text="Оплату підтвердили",
        patched_text="Оплату підтвердили",
        groups={"main": "blue"},
        issue="main",
        language="uk",
        embedding=[1.0, 0.0],
    )


def _glued(*late: Document) -> tuple[Daemon, Cluster, Clusters, Cluster]:
    """A post four hours old, and the clusterer's cluster holding it plus `late`.

    The shape `hold_published_together` produces: the post's own document is
    in the cluster, so `find_similar` recognizes the post by it.
    """
    daemon = _daemon()
    daemon.renderer = _FakeRenderer()  # type: ignore[assignment]
    daemon.client = _FakeClient()  # type: ignore[assignment]
    daemon.ranker = _TwoSourceRanker()  # type: ignore[assignment]
    post = _cluster([1.0, 0.0], message_id=11, headline="Обіцяли виплату", age_seconds=4 * 3600)
    post.create_time = get_current_ts() - 4 * 3600 + 60
    posted = _posted(post)
    incoming = Cluster()
    incoming.add(post.docs[0])
    for doc in late:
        incoming.add(doc)
    return daemon, post, posted, incoming


def test_a_development_glued_to_its_post_goes_out_as_a_reply(monkeypatch: Any) -> None:
    """Hours later, the payment made: news the post does not carry.

    Recognized as the post by its URLs, it used to be folded in — and past the
    editing window that meant it was never told at all.
    """
    seen = _patch_judge(monkeypatch, FOLLOW_UP)
    late = [_doc("a", 1, 600), _doc("b", 1, 600)]
    daemon, post, posted, incoming = _glued(*late)

    daemon.send_cluster(incoming, "main", posted, None, None)

    client: Any = daemon.client
    assert [list(candidates) for candidates in seen] == [[post]]
    assert client.sent == 1
    assert client.reply_to == 11
    assert not any(post.has(doc) for doc in late)


def test_a_late_wave_of_the_same_event_is_folded_in_once(monkeypatch: Any) -> None:
    seen = _patch_judge(monkeypatch, SAME)
    late = [_doc("a", 1, 600), _doc("b", 1, 600)]
    daemon, post, posted, incoming = _glued(*late)

    daemon.send_cluster(incoming, "main", posted, None, None)
    daemon.send_cluster(incoming, "main", posted, None, None)

    client: Any = daemon.client
    assert len(seen) == 1
    assert client.sent == 0
    assert all(post.has(doc) for doc in late)


def test_one_late_channel_is_held_while_others_may_join(monkeypatch: Any) -> None:
    """Taken in one at a time, a development would never be enough to ask about."""
    seen = _patch_judge(monkeypatch, FOLLOW_UP)
    young = _doc("a", 1, 600)
    daemon, post, posted, incoming = _glued(young)

    daemon.send_cluster(incoming, "main", posted, None, None)

    assert not seen
    assert not post.has(young)


def test_a_late_channel_no_one_joined_is_folded_in(monkeypatch: Any) -> None:
    seen = _patch_judge(monkeypatch, FOLLOW_UP)
    ripe = _doc("a", 1, 2 * 3600)
    daemon, post, posted, incoming = _glued(ripe)

    daemon.send_cluster(incoming, "main", posted, None, None)

    assert not seen
    assert post.has(ripe)


def test_the_first_wave_is_folded_in_without_asking(monkeypatch: Any) -> None:
    seen = _patch_judge(monkeypatch, FOLLOW_UP)
    early = [_doc("a", 1, 4 * 3600 - 600), _doc("b", 1, 4 * 3600 - 600)]
    daemon, post, posted, incoming = _glued(*early)

    daemon.send_cluster(incoming, "main", posted, None, None)

    assert not seen
    assert all(post.has(doc) for doc in early)


def test_a_follow_up_clustered_beside_its_parent_stays_its_own(
    monkeypatch: Any,
) -> None:
    """Once sent, its documents belong to it, not to the post it follows."""
    seen = _patch_judge(monkeypatch, FOLLOW_UP)
    late = [_doc("a", 1, 600), _doc("b", 1, 600)]
    daemon, post, posted, incoming = _glued(*late)
    daemon.send_cluster(incoming, "main", posted, None, None)

    daemon.send_cluster(incoming, "main", posted, None, None)

    client: Any = daemon.client
    assert len(seen) == 1
    assert client.sent == 1
    assert not any(post.has(doc) for doc in late)
