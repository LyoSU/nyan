import json
from typing import Any
from collections.abc import Callable

import httpx

from nyan import publish


def _transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.MockTransport(record), seen


def _ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


def test_without_a_token_the_site_is_left_alone(monkeypatch: Any) -> None:
    """An unset token is how dev and tests run: nothing must reach production."""
    monkeypatch.setattr(publish, "PUBLISH_URL", "https://site.test/api/publish")
    monkeypatch.setattr(publish, "PUBLISH_TOKEN", "")
    transport, seen = _transport(_ok)

    assert publish.notify_published([62652], transport=transport) is False
    assert seen == []


def test_without_a_url_the_ping_is_off_too(monkeypatch: Any) -> None:
    """No address is baked in: where the site lives is deployment config."""
    monkeypatch.setattr(publish, "PUBLISH_URL", "")
    monkeypatch.setattr(publish, "PUBLISH_TOKEN", "secret")
    transport, seen = _transport(_ok)

    assert publish.notify_published([62652], transport=transport) is False
    assert seen == []


def test_the_story_is_announced_by_its_clid_with_the_bearer_token(monkeypatch: Any) -> None:
    """The site's precise entry: POST with the cluster ids, not a blind scan."""
    monkeypatch.setattr(publish, "PUBLISH_URL", "https://site.test/api/publish")
    monkeypatch.setattr(publish, "PUBLISH_TOKEN", "secret")
    transport, seen = _transport(_ok)

    assert publish.notify_published([62652], transport=transport) is True

    assert [str(request.url) for request in seen] == ["https://site.test/api/publish"]
    assert seen[0].method == "POST"
    assert seen[0].headers["Authorization"] == "Bearer secret"
    assert seen[0].headers["Content-Type"] == "application/json"
    assert json.loads(seen[0].content) == {"clids": [62652]}


def test_a_refusal_from_the_site_is_reported_not_raised(monkeypatch: Any) -> None:
    monkeypatch.setattr(publish, "PUBLISH_URL", "https://site.test/api/publish")
    monkeypatch.setattr(publish, "PUBLISH_TOKEN", "secret")
    transport, _ = _transport(lambda _: httpx.Response(500))

    assert publish.notify_published([62652], transport=transport) is False


def test_an_unreachable_site_is_reported_not_raised(monkeypatch: Any) -> None:
    """The ping runs right after a post went out; it must never take the send down."""
    monkeypatch.setattr(publish, "PUBLISH_URL", "https://site.test/api/publish")
    monkeypatch.setattr(publish, "PUBLISH_TOKEN", "secret")

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    transport, _ = _transport(unreachable)

    assert publish.notify_published([62652], transport=transport) is False
