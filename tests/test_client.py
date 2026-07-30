"""Tests for the Telegram client that need no network.

Everything here drives the client through a fake transport, so the payloads it
builds and the way it reacts to Telegram's errors are checked directly.
"""

import json
from typing import Any

import pytest

from nyan.client import FORMAT_LEGACY, FORMAT_RICH, MessageId, TelegramClient
from nyan.media import MEDIA_PHOTO, MEDIA_VIDEO, MediaItem, SentMedia
from nyan.rich import RenderedPost, heading, paragraph
from nyan.rich import photo as rich_photo


CLIENT_CONFIG = {
    "issues": [
        {
            "name": "main",
            "channel_id": -100,
            "discussion_id": -200,
            "bot_token": "token",
        }
    ]
}


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self.payload = payload if payload is not None else {"result": {"message_id": 1}}

    def json(self) -> Any:
        return self.payload

    @property
    def text(self) -> str:
        return json.dumps(self.payload)


def caption_only_error() -> FakeResponse:
    """What Telegram answers when editMessageText hits a media message."""
    return FakeResponse(
        400,
        {
            "ok": False,
            "error_code": 400,
            "description": "Bad Request: there is no text in the message to edit",
        },
    )


@pytest.fixture
def client(tmp_path: Any, monkeypatch: Any) -> TelegramClient:
    config = tmp_path / "client_config.json"
    config.write_text(json.dumps(CLIENT_CONFIG))
    # Skip the getUpdates calls the constructor makes.
    monkeypatch.setattr(TelegramClient, "update_discussion_mapping", lambda *_: None)
    return TelegramClient(str(config))


def record_calls(client: TelegramClient, responses: list[FakeResponse]) -> list[Any]:
    """Replace the transport, returning the list that records every request."""
    calls: list[Any] = []
    queue = list(responses)

    def fake_post(url: str, params: dict[str, Any]) -> FakeResponse:
        calls.append((url, params))
        return queue.pop(0) if queue else FakeResponse()

    client._post = fake_post  # type: ignore[method-assign]
    return calls


def rich_post() -> RenderedPost:
    return RenderedPost(blocks=[heading("Заголовок"), paragraph("Текст")])


def test_a_sent_post_remembers_its_format(client: TelegramClient) -> None:
    """An update has to edit a message the same way it was sent."""
    record_calls(client, [FakeResponse()])

    message = client.send_post(rich_post(), "main")

    assert message is not None
    assert message.post_format == FORMAT_RICH


def test_a_legacy_post_remembers_its_format(client: TelegramClient) -> None:
    record_calls(client, [FakeResponse()])

    message = client.send_post(RenderedPost(text="Текст"), "main")

    assert message is not None
    assert message.post_format == FORMAT_LEGACY


def test_rich_blocks_are_sent_as_json(client: TelegramClient) -> None:
    calls = record_calls(client, [FakeResponse()])

    client.send_post(rich_post(), "main")

    url, params = calls[0]
    assert url.endswith("/sendRichMessage")
    assert json.loads(params["rich_message"])["blocks"][0]["type"] == "heading"
    assert params["chat_id"] == -100


def test_a_reply_uses_reply_parameters(client: TelegramClient) -> None:
    """sendRichMessage takes an object, not the flat reply_to_message_id."""
    calls = record_calls(client, [FakeResponse()])

    client.send_post(rich_post(), "main", reply_to=42)

    _, params = calls[0]
    assert "reply_to_message_id" not in params
    assert json.loads(params["reply_parameters"])["message_id"] == 42


def test_a_media_message_is_remembered_as_legacy_after_telegram_says_so(
    client: TelegramClient, caplog: Any
) -> None:
    """Posts sent before the rich format carry a caption, not text.

    editMessageText fails on them, and without recording that, every iteration
    retried the same doomed edit and logged an error.
    """
    record_calls(client, [caption_only_error()])
    message = MessageId(message_id=38104, issue="main")

    client.update_post(message, rich_post())

    assert message.post_format == FORMAT_LEGACY
    assert "predates the rich format" in caplog.text
    # Not an error: it is an expected consequence of the migration.
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_other_update_failures_are_still_errors(
    client: TelegramClient, caplog: Any
) -> None:
    record_calls(
        client, [FakeResponse(400, {"description": "Bad Request: message not found"})]
    )
    message = MessageId(message_id=1, issue="main", post_format=FORMAT_RICH)

    client.update_post(message, rich_post())

    assert [r for r in caplog.records if r.levelname == "ERROR"]
    assert message.post_format == FORMAT_RICH


def test_the_format_survives_serialization() -> None:
    message = MessageId(message_id=1, issue="main", post_format=FORMAT_RICH)

    restored = MessageId.fromdict(message.asdict())

    assert restored.post_format == FORMAT_RICH


def test_messages_stored_before_the_field_existed_load_fine() -> None:
    restored = MessageId.fromdict({"message_id": 7, "issue": "main"})

    assert restored.post_format == ""


def test_missing_issues_are_reported_not_raised(
    client: TelegramClient, caplog: Any
) -> None:
    assert client.send_post(rich_post(), "nosuchissue") is None
    assert "Missing issue" in caplog.text


def test_has_issue_answers_before_anything_is_sent(client: TelegramClient) -> None:
    """Asked by the daemon, so an unpostable issue costs nothing to discover."""
    assert client.has_issue("main")
    assert not client.has_issue("war")


def test_message_equality_tolerates_other_types() -> None:
    message = MessageId(message_id=1, issue="main")

    assert message != None  # noqa: E711
    assert message != "not a message"
    assert message == MessageId(message_id=1, issue="main")
    # The format is metadata, not identity: the same post either way.
    assert message == MessageId(message_id=1, issue="main", post_format=FORMAT_RICH)


# ------------------------------------------------------------------- file ids


def photo_response(file_id: str = "full", message_id: int = 1) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "result": {
                "message_id": message_id,
                "photo": [
                    {"file_id": "thumb", "width": 90, "file_size": 1000},
                    {"file_id": file_id, "width": 1280, "file_size": 200000},
                ],
            }
        },
    )


def legacy_post(*media: MediaItem) -> RenderedPost:
    return RenderedPost(text="Текст", media=media)


def photo(url: str) -> MediaItem:
    return MediaItem(type=MEDIA_PHOTO, url=url)


def video(url: str) -> MediaItem:
    return MediaItem(type=MEDIA_VIDEO, url=url)


def test_a_sent_photo_remembers_its_file_id(client: TelegramClient) -> None:
    """The CDN URL rots; the file_id does not, and every edit can reuse it."""
    record_calls(client, [photo_response()])

    message = client.send_post(legacy_post(photo("a.jpg")), "main")

    assert message is not None
    assert [(m.url, m.file_id) for m in message.media] == [("a.jpg", "full")]


def test_a_group_remembers_a_file_id_and_a_message_id_per_attachment(
    client: TelegramClient,
) -> None:
    """A media group is one message per attachment, and editing the second
    needs the second message's id."""
    record_calls(
        client,
        [
            FakeResponse(
                200,
                {
                    "result": [
                        {"message_id": 10, "photo": [{"file_id": "p", "width": 800}]},
                        {"message_id": 11, "video": {"file_id": "v"}},
                    ]
                },
            )
        ],
    )

    message = client.send_post(legacy_post(photo("a.jpg"), video("b.mp4")), "main")

    assert message is not None
    assert [(m.url, m.file_id, m.message_id) for m in message.media] == [
        ("a.jpg", "p", 10),
        ("b.mp4", "v", 11),
    ]


def test_photos_and_videos_go_out_as_one_group(client: TelegramClient) -> None:
    """One slideshow, in the order chosen — not photos in one format and a
    video in the other, which is what deciding per format used to produce."""
    calls = record_calls(client, [FakeResponse()])

    client.send_post(legacy_post(video("b.mp4"), photo("a.jpg")), "main")

    url, params = calls[0]
    assert url.endswith("/sendMediaGroup")
    assert [item["type"] for item in json.loads(params["media"])] == ["video", "photo"]


def test_a_rich_update_sends_the_file_id_instead_of_the_url(
    client: TelegramClient,
) -> None:
    """The whole point of storing it: the edit references the file Telegram
    already has rather than asking it to fetch a URL that may be gone."""
    calls = record_calls(client, [FakeResponse()])
    message = MessageId(
        message_id=1,
        issue="main",
        post_format=FORMAT_RICH,
        media=[SentMedia(type=MEDIA_PHOTO, file_id="known", url="a.jpg")],
    )

    client.update_post(message, RenderedPost(blocks=[rich_photo("a.jpg")]))

    _, params = calls[0]
    assert json.loads(params["rich_message"])["blocks"][0]["photo"]["media"] == "known"


def test_media_an_update_adds_gets_its_file_id_stored(client: TelegramClient) -> None:
    """A cluster grows, a second photo joins the post, and the next edit has to
    be able to reference that one too."""
    record_calls(
        client,
        [
            FakeResponse(
                200,
                {
                    "result": {
                        "message_id": 1,
                        "blocks": [
                            {"photo": [{"file_id": "known", "width": 800}]},
                            {"photo": [{"file_id": "fresh", "width": 800}]},
                        ],
                    }
                },
            )
        ],
    )
    message = MessageId(
        message_id=1,
        issue="main",
        post_format=FORMAT_RICH,
        media=[SentMedia(type=MEDIA_PHOTO, file_id="known", url="a.jpg")],
    )

    client.update_post(
        message, RenderedPost(blocks=[rich_photo("a.jpg"), rich_photo("b.jpg")])
    )

    assert [(m.url, m.file_id) for m in message.media] == [
        ("a.jpg", "known"),
        ("b.jpg", "fresh"),
    ]


def test_a_post_sent_without_media_is_still_updated_as_text(
    client: TelegramClient,
) -> None:
    """How the message was sent decides, not what the cluster looks like now.

    A cluster that gained photos after publication used to be edited with
    editMessageCaption — on a message that has no caption — so Telegram refused
    and the post stopped updating for good.
    """
    calls = record_calls(client, [FakeResponse()])
    message = MessageId(message_id=1, issue="main", post_format=FORMAT_LEGACY)

    client.update_post(message, legacy_post(photo("a.jpg")))

    url, _ = calls[0]
    assert url.endswith("/editMessageText")


def test_a_post_sent_with_media_is_updated_as_a_caption(
    client: TelegramClient,
) -> None:
    """And the same rule the other way: media was sent, so the text is a caption
    even on an iteration where the cluster has no media left to show."""
    calls = record_calls(client, [FakeResponse()])
    message = MessageId(
        message_id=1,
        issue="main",
        post_format=FORMAT_LEGACY,
        media=[SentMedia(type=MEDIA_PHOTO, file_id="known", url="a.jpg")],
    )

    client.update_post(message, RenderedPost(text="Новий текст"))

    url, _ = calls[0]
    assert url.endswith("/editMessageCaption")


def test_file_ids_survive_serialization() -> None:
    """They are stored with the cluster, so the site can serve media by file_id
    and an edit after a restart still has them."""
    message = MessageId(
        message_id=1,
        issue="main",
        media=[SentMedia(type=MEDIA_PHOTO, file_id="known", url="a.jpg", message_id=1)],
    )

    restored = MessageId.fromdict(message.asdict())

    assert restored.media == message.media
    assert restored.media[0].file_id == "known"
