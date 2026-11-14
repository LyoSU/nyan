"""The editor's one decision per story, and what the daemon does with it."""

import json
from typing import Any

from nyan import editor
from nyan.client import MessageId
from nyan.clusters import Cluster, Clusters
from nyan.daemon import Daemon
from nyan.document import Document
from nyan.util import get_current_ts

NOW = 1794470400  # 12.11.2026, 10:00 Kyiv


def _doc(
    url: str, text: str, minutes_ago: int, embedding: list[float], channel: str = ""
) -> Document:
    channel_id = channel or url.split("/")[-2]
    return Document(
        url=url,
        channel_id=channel_id,
        post_id=int(url.split("/")[-1]),
        views=1000,
        pub_time=NOW - minutes_ago * 60,
        fetch_time=NOW,
        text=text,
        patched_text=text,
        groups={"main": "blue"},
        issue="main",
        language="uk",
        embedding=embedding,
    )


def _post(
    clid: int, headline: str, hours_ago: float, embedding: list[float], storyline: str | None = None
) -> Cluster:
    post = Cluster()
    post.add(_doc(f"https://t.me/src{clid}/{clid}", headline, int(hours_ago * 60) + 20, embedding))
    post.clid = clid
    post.create_time = NOW - int(hours_ago * 3600)
    post.messages.append(MessageId(message_id=clid - 60000, issue="main"))
    post.saved_analysis = {"headline": headline, "generation": post.generation}
    post.saved_hash = post.hash
    post.storyline = storyline
    return post


def _story(texts: list[str], embedding: list[float]) -> Cluster:
    story = Cluster()
    for index, text in enumerate(texts):
        story.add(_doc(f"https://t.me/ch{index}/{100 + index}", text, 30 - index, embedding))
    return story


def _patch(monkeypatch: Any, answer: dict[str, Any]) -> list[Any]:
    calls: list[Any] = []

    def fake(**kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps(answer, ensure_ascii=False)

    monkeypatch.setattr("nyan.editor.openai_completion", fake)
    return calls


# The week's own posts, as the editor would have been shown them.
KYIV_DRONE = "У Києві після падіння БпЛА загорілося авто"
KYIV_DEBRIS = "Уламки БпЛА впали на дорогу в Києві, загорілося авто"
KHARKIV_NIGHT_ONE = "Дрон влучив у дах дев’ятиповерхового будинку в Харкові"


def test_the_editor_sees_the_channel_and_the_closest_post_in_full(monkeypatch: Any) -> None:
    near = _post(66378, KYIV_DRONE, 0.1, [1.0, 0.0])
    far = _post(66300, "Бензин дешевшає, автогаз дорожчає", 5, [0.0, 1.0])
    calls = _patch(monkeypatch, {"relation": "same", "post": 1, "decision": "publish"})

    decision = editor.decide(_story([KYIV_DEBRIS], [1.0, 0.01]), [far, near], NOW)

    user = calls[0]["messages"][1]["content"]
    assert "#1 12.11" in user and KYIV_DRONE in user
    assert "Бензин дешевшає" in user, "the whole day's headlines, not only the close ones"
    assert user.index("ОПУБЛІКОВАНО #1") < user.index("НОВА ІСТОРІЯ")
    assert decision.relation == editor.SAME
    assert decision.post is near


def test_the_rules_come_first_and_never_change(monkeypatch: Any) -> None:
    """The system half is the same bytes on every call, and the channel's history
    opens the user half, so consecutive calls share the longest prefix."""
    calls = _patch(monkeypatch, {"relation": "new", "post": None, "decision": "skip"})
    posts = [_post(66378, KYIV_DRONE, 0.1, [1.0, 0.0])]
    editor.decide(_story([KYIV_DEBRIS], [1.0, 0.0]), posts, NOW)
    editor.decide(_story(["Інша новина"], [0.0, 1.0]), posts, NOW)

    assert calls[0]["messages"][0] == calls[1]["messages"][0]
    first, second = (call["messages"][1]["content"] for call in calls)
    head = first.split("<ДЖЕРЕЛА>")[0]
    assert second.startswith(head)


def test_a_link_to_no_post_is_no_answer(monkeypatch: Any) -> None:
    """Acted on, "same as nothing" would publish a repeat or drop a story."""
    _patch(monkeypatch, {"relation": "update", "post": 7, "decision": "publish"})

    decision = editor.decide(_story([KYIV_DEBRIS], [1.0, 0.0]), [], NOW)

    assert decision.failed


def test_a_failed_call_is_no_answer(monkeypatch: Any) -> None:
    def explode(**kwargs: Any) -> str:
        raise RuntimeError("gateway is down")

    monkeypatch.setattr("nyan.editor.openai_completion", explode)

    assert editor.decide(_story([KYIV_DEBRIS], [1.0, 0.0]), [], NOW).failed


def test_a_storyline_keeps_the_name_the_channel_already_uses(monkeypatch: Any) -> None:
    """The next hit on Kharkiv is another event of a storyline the editor named before."""
    earlier = _post(66119, KHARKIV_NIGHT_ONE, 23, [1.0, 0.0], storyline="Атаки на Харків")
    _patch(
        monkeypatch, {"relation": "storyline", "post": 1, "decision": "publish", "storyline": ""}
    )

    decision = editor.decide(
        _story(["Влучання ударного дрона в Київському районі Харкова"], [1.0, 0.02]), [earlier], NOW
    )

    assert decision.relation == editor.STORYLINE
    assert decision.storyline == "Атаки на Харків"


def test_the_story_is_shown_once_per_text(monkeypatch: Any) -> None:
    """Channels copy each other word for word; the copies are not more evidence to read."""
    calls = _patch(monkeypatch, {"relation": "new", "post": None, "decision": "publish"})
    story = _story(
        [KYIV_DEBRIS, KYIV_DEBRIS, KYIV_DEBRIS, "Рятувальники загасили авто на дорозі"], [1.0, 0.0]
    )

    editor.decide(story, [], NOW)

    assert calls[0]["messages"][1]["content"].count(KYIV_DEBRIS) == 1


# What the daemon does with the answer.


class _Renderer:
    def __init__(self) -> None:
        self.rendered: list[Cluster] = []

    def render_cluster(
        self, cluster: Cluster, issue_name: str, post_format: str | None = None
    ) -> str:
        self.rendered.append(cluster)
        return "post"


class _Client:
    def __init__(self) -> None:
        self.sent: list[int | None] = []
        self.updated: list[int] = []

    def send_post(self, post: Any, issue_name: str, reply_to: int | None = None) -> MessageId:
        self.sent.append(reply_to)
        return MessageId(message_id=999, issue=issue_name)

    def update_post(self, message: MessageId, post: Any) -> bool:
        self.updated.append(message.message_id)
        return True

    def update_discussion_mapping(self, issue_name: str) -> None:
        pass


def _daemon() -> Daemon:
    daemon = object.__new__(Daemon)
    daemon.config = {
        "related_threshold": 0.86,
        "similar_min_intersection_ratio": 0.25,
        "max_time_updated": 10800,
        "sleep_time": 0,
        "editor": {"enabled": True, "full_sources": 0},
    }
    daemon.renderer = _Renderer()  # type: ignore[assignment]
    daemon.client = _Client()  # type: ignore[assignment]
    return daemon


def _posted(*posts: Cluster) -> Clusters:
    clusters = Clusters()
    for post in posts:
        clusters.add(post)
    return clusters


def _answer(monkeypatch: Any, answer: dict[str, Any]) -> list[Any]:
    monkeypatch.setattr("nyan.daemon.get_current_ts", lambda: NOW)
    monkeypatch.setattr("nyan.daemon.notify_published", lambda clids: None)
    return _patch(monkeypatch, answer)


def test_new_numbers_rewrite_the_post_instead_of_adding_one(monkeypatch: Any) -> None:
    """One strike on Kyiv gave five posts in ninety minutes, each a new count."""
    daemon = _daemon()
    parent = _post(66360, "БпЛА влучив у багатоповерхівку в Києві", 1, [1.0, 0.0])
    _answer(monkeypatch, {"relation": "update", "post": 1, "decision": "publish"})
    monkeypatch.setattr("nyan.clusters.get_current_ts", lambda: NOW)

    daemon.send_cluster(
        _story(["Кількість постраждалих у Києві зросла до п’яти"], [1.0, 0.0]),
        "main",
        _posted(parent),
        None,
        None,
    )

    client: Any = daemon.client
    assert client.sent == []
    assert client.updated == [parent.messages[0].message_id]
    assert len(parent.docs) == 2


def test_new_numbers_past_the_editing_window_go_out_as_a_reply(monkeypatch: Any) -> None:
    daemon = _daemon()
    parent = _post(66360, "БпЛА влучив у багатоповерхівку в Києві", 5, [1.0, 0.0])
    _answer(monkeypatch, {"relation": "update", "post": 1, "decision": "publish"})
    monkeypatch.setattr("nyan.clusters.get_current_ts", lambda: NOW)

    daemon.send_cluster(
        _story(["Кількість постраждалих у Києві зросла до п’яти"], [1.0, 0.0]),
        "main",
        _posted(parent),
        None,
        None,
    )

    client: Any = daemon.client
    assert client.sent == [parent.messages[0].message_id]


def test_a_storyline_is_its_own_post_and_remembers_its_neighbour(monkeypatch: Any) -> None:
    daemon = _daemon()
    earlier = _post(66119, KHARKIV_NIGHT_ONE, 23, [1.0, 0.0])
    _answer(
        monkeypatch,
        {
            "relation": "storyline",
            "post": 1,
            "decision": "publish",
            "storyline": "Атаки на Харків",
            "level": "normal",
        },
    )
    story = _story(["Влучання ударного дрона в Київському районі Харкова"], [1.0, 0.02])

    daemon.send_cluster(story, "main", _posted(earlier), None, None)

    client: Any = daemon.client
    assert client.sent == [None], "not a reply"
    assert story.storyline == "Атаки на Харків"
    assert story.storyline_clid == 66119


def test_a_story_held_back_is_not_asked_about_again_at_once(monkeypatch: Any) -> None:
    """A story that gathers channels one at a time would cost a call per channel."""
    daemon = _daemon()
    calls = _answer(monkeypatch, {"relation": "new", "post": None, "decision": "digest"})
    texts = ["Сніг засипав дорогу на Криворіжжі"]

    daemon.send_cluster(_story(texts, [0.0, 1.0]), "main", _posted(), None, None)
    daemon.send_cluster(_story(texts, [0.0, 1.0]), "main", _posted(), None, None)
    daemon.send_cluster(
        _story([*texts, "На Криворіжжі сніг"], [0.0, 1.0]), "main", _posted(), None, None
    )

    client: Any = daemon.client
    assert client.sent == []
    assert len(calls) == 1


def test_an_editor_that_did_not_answer_holds_the_story(monkeypatch: Any) -> None:
    daemon = _daemon()
    _answer(monkeypatch, {"verdict": "same"})

    daemon.send_cluster(_story([KYIV_DEBRIS], [1.0, 0.0]), "main", _posted(), None, None)

    client: Any = daemon.client
    assert client.sent == []
    assert list(daemon.held_now.values()) == ["editor"]


def test_the_rules_name_the_editing_window_the_daemon_uses() -> None:
    """The editor is told a post older than three hours can no longer be edited,
    so an update to it costs a message; that has to stay the daemon's window."""
    with open("configs/daemon_config.json") as r:
        window = json.load(r)["max_time_updated"]
    with open("nyan/prompts/editor.txt") as r:
        rules = r.read()

    assert window == 3 * 3600
    assert "старший за 3 години" in rules


def test_without_the_editor_nothing_changes() -> None:
    daemon = _daemon()
    daemon.config["editor"] = {"enabled": False}

    assert daemon.editor_config is None
    assert get_current_ts() > 0


def test_an_update_leaves_a_line_for_the_reader(monkeypatch: Any) -> None:
    daemon = _daemon()
    parent = _post(66360, "БпЛА влучив у багатоповерхівку в Києві", 1, [1.0, 0.0])
    _answer(
        monkeypatch,
        {"relation": "update", "post": 1, "decision": "publish", "change": "постраждалих п’ятеро"},
    )
    monkeypatch.setattr("nyan.clusters.get_current_ts", lambda: NOW)

    daemon.send_cluster(
        _story(["Кількість постраждалих у Києві зросла до п’яти"], [1.0, 0.0]),
        "main",
        _posted(parent),
        None,
        None,
    )

    assert parent.updates == [{"time": NOW, "text": "постраждалих п’ятеро"}]


def test_a_late_retelling_of_an_old_post_is_folded_not_replied(monkeypatch: Any) -> None:
    """11 November: the night strike on Saltivka reached the editor at 13:50 from
    sources of 07:56-08:20, and went out as a reply to the 03:35 post."""
    daemon = _daemon()
    parent = _post(66300, "Росія завдала подвійного ракетного удару по Харкову", 10, [1.0, 0.0])
    _answer(monkeypatch, {"relation": "update", "post": 1, "decision": "publish"})
    monkeypatch.setattr("nyan.clusters.get_current_ts", lambda: NOW)
    story = Cluster()
    for index in range(3):
        story.add(
            _doc(
                f"https://t.me/kh{index}/{index + 1}",
                "Вночі росіяни вдарили по Салтівському району, горіли 38 гаражів",
                330 + index * 5,
                [1.0, 0.0],
            )
        )

    daemon.send_cluster(story, "main", _posted(parent), None, None)

    client: Any = daemon.client
    assert client.sent == []
    assert len(parent.docs) == 4
    assert parent.updates == []


def test_a_storyline_post_lists_the_earlier_ones_of_its_channel(monkeypatch: Any) -> None:
    daemon = _daemon()
    first = _post(66119, KHARKIV_NIGHT_ONE, 23, [1.0, 0.0])
    labelled = _post(
        66200, "Удар по Холодногірському району Харкова", 12, [0.9, 0.1], "Атаки на Харків"
    )
    other = _post(66201, "Бензин дешевшає", 2, [0.0, 1.0], "Ціни на пальне")
    _answer(
        monkeypatch,
        {"relation": "storyline", "post": 2, "decision": "publish", "storyline": "Атаки на Харків"},
    )
    story = _story(["Влучання ударного дрона в Київському районі Харкова"], [1.0, 0.02])

    daemon.send_cluster(story, "main", _posted(first, labelled, other), None, None)

    # The unlabelled 66119 is not found: posts from before the editor carry no
    # label until the editor names one of them as a storyline's nearest.
    assert [post["clid"] for post in story.storyline_posts] == [66200]
    assert all(post["issue"] == "main" for post in story.storyline_posts)
    assert story.storyline_posts[0]["message_id"] == labelled.messages[0].message_id


def test_on_a_host_with_a_channel_per_rubric_the_editor_sees_that_channel(
    monkeypatch: Any,
) -> None:
    """War stories may move to a channel of their own; the editor of the war
    channel is shown what that channel published, not the main feed."""
    daemon = _daemon()
    in_main = _post(66300, "Бензин дешевшає, автогаз дорожчає", 2, [0.0, 1.0])
    in_war = _post(66301, "Дрон влучив у дах дев’ятиповерхівки в Харкові", 3, [1.0, 0.0])
    in_war.messages = [MessageId(message_id=777, issue="war")]
    calls = _answer(monkeypatch, {"relation": "new", "post": None, "decision": "skip"})

    daemon.send_cluster(
        _story(["Влучання ударного дрона в Харкові"], [1.0, 0.0]),
        "war",
        _posted(in_main, in_war),
        None,
        None,
    )

    shown = calls[0]["messages"][1]["content"]
    assert "Дрон влучив у дах" in shown
    assert "Бензин дешевшає" not in shown
