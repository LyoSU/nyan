"""Tests for the Telegram client that need no network.

Everything here drives the client through a fake transport, so the payloads it
builds and the way it reacts to Telegram's errors are checked directly.
"""

import json
from typing import Any

import pytest

from nyan.client import FORMAT_LEGACY, FORMAT_RICH, MessageId, TelegramClient
from nyan.rich import RenderedPost, heading, paragraph


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


def test_a_legacy_post_is_updated_as_a_caption(client: TelegramClient) -> None:
    calls = record_calls(client, [FakeResponse()])
    message = MessageId(message_id=1, issue="main", post_format=FORMAT_LEGACY)

    client.update_post(message, RenderedPost(text="Новий текст", photos=("a",)))

    url, params = calls[0]
    assert url.endswith("/editMessageCaption")
    assert params["caption"] == "Новий текст"


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
