"""Tests for the one place the bot fetches a file itself.

Every URL here came out of a crawled channel page, so this module is the
project's only spot where an outside-controlled string turns into an outbound
request. What it refuses matters as much as what it fetches.
"""

from typing import Any

from nyan.download import fetch_media, is_allowed_media_url


def test_the_telegram_cdn_is_allowed() -> None:
    """Every video in the database comes from exactly these hosts."""
    assert is_allowed_media_url("https://cdn4.telesco.pe/file/a.mov?token=x")
    assert is_allowed_media_url("https://cdn1.telesco.pe/file/a.mp4")
    # What fix_media_url rewrites those into.
    assert is_allowed_media_url("https://cdn4.cdn-telegram.org/file/a.mov")


def test_anything_but_the_cdn_is_refused() -> None:
    """A crawled page is outside input, and this is where it becomes a request.

    Without the check, a URL in a channel's markup could point the server at
    cloud metadata or an internal address, and the response would be handed
    straight to Telegram to publish.
    """
    assert not is_allowed_media_url("https://169.254.169.254/latest/meta-data/")
    assert not is_allowed_media_url("http://127.0.0.1:27017/")
    assert not is_allowed_media_url("https://evil.example.com/payload.mp4")
    # Suffix matching must not be fooled by a lookalike domain.
    assert not is_allowed_media_url("https://telesco.pe.evil.com/a.mp4")


def test_plain_http_is_refused() -> None:
    """Every URL on record is https; downgrading is nobody's legitimate need."""
    assert not is_allowed_media_url("http://cdn4.telesco.pe/file/a.mp4")
    assert not is_allowed_media_url("file:///etc/passwd")


def test_a_refused_url_is_never_requested(monkeypatch: Any) -> None:
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a refused url must not reach the network")

    monkeypatch.setattr("nyan.download.requests.get", explode)

    assert fetch_media("https://evil.example.com/payload.mp4", 1024) is None


class FakeStream:
    def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None):
        self.status_code = 200
        self.headers = headers or {}
        self._chunks = chunks
        self.kwargs: dict[str, Any] = {}

    def iter_content(self, size: int) -> list[bytes]:
        return self._chunks

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_redirects_are_not_followed(monkeypatch: Any) -> None:
    """An allowed host that redirects elsewhere would walk around the check."""
    seen: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> FakeStream:
        seen.update(kwargs)
        return FakeStream([b"data"])

    monkeypatch.setattr("nyan.download.requests.get", fake_get)

    fetch_media("https://cdn4.telesco.pe/file/a.mp4", 1024)

    assert seen["allow_redirects"] is False


def test_a_file_over_the_limit_is_not_kept(monkeypatch: Any) -> None:
    """Checked while reading: a missing or lying content-length is not a promise."""
    monkeypatch.setattr(
        "nyan.download.requests.get",
        lambda url, **kwargs: FakeStream([b"x" * 600, b"x" * 600]),
    )

    assert fetch_media("https://cdn4.telesco.pe/file/a.mp4", 1000) is None


def test_a_file_within_the_limit_comes_back_whole(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "nyan.download.requests.get",
        lambda url, **kwargs: FakeStream([b"abc", b"def"]),
    )

    assert fetch_media("https://cdn4.telesco.pe/file/a.mp4", 1000) == b"abcdef"
