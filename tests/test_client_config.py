"""The generator that turns CLIENT_CONFIG_JSON into configs/client_config.json.

`configs/` ships inside the image, so the two files that cannot be committed —
the Mongo credentials and the publishing tokens — are written at container
start from the environment instead. This is the second of the two, and the
tests below are about the one thing that separates it from a plain `echo`:
a malformed value has to be rejected while the container is still starting.
"""

import json

import pytest

from create_client_config import build_client_config, ClientConfigError


VALID = {
    "issues": [
        {
            "name": "main",
            "channel_id": -1001946191033,
            "discussion_id": -1001927777221,
            "bot_token": "7542680000:AAH-not-a-real-token",
        }
    ]
}


def test_a_valid_value_is_passed_through() -> None:
    assert build_client_config(json.dumps(VALID)) == VALID


def test_several_issues_are_kept() -> None:
    """`issues` is a list because a deployment may publish to more than one
    channel. The generator must not flatten it to the first one."""
    value = {"issues": [VALID["issues"][0], {**VALID["issues"][0], "name": "tech"}]}

    assert [i["name"] for i in build_client_config(json.dumps(value))["issues"]] == [
        "main",
        "tech",
    ]


def test_an_unset_value_is_refused() -> None:
    with pytest.raises(ClientConfigError, match="CLIENT_CONFIG_JSON"):
        build_client_config(None)


def test_whitespace_is_treated_as_unset() -> None:
    """Compose passes an undefined variable as the empty string, and a value
    pasted into a web form arrives with the newline the browser added."""
    with pytest.raises(ClientConfigError, match="CLIENT_CONFIG_JSON"):
        build_client_config("  \n ")


def test_broken_json_is_refused_while_the_container_starts() -> None:
    """The likeliest way to break this: a value truncated on paste."""
    with pytest.raises(ClientConfigError, match="не є коректним JSON"):
        build_client_config('{"issues": [{"name": "main"')


def test_a_json_list_is_refused() -> None:
    """Valid JSON, wrong shape: the issues list pasted without its wrapper."""
    with pytest.raises(ClientConfigError, match="об'єкт"):
        build_client_config(json.dumps(VALID["issues"]))


def test_a_config_without_issues_is_refused() -> None:
    with pytest.raises(ClientConfigError, match="issues"):
        build_client_config(json.dumps({"channel_id": -1}))


def test_an_empty_issues_list_is_refused() -> None:
    """Syntactically fine and completely useless: nothing could be published."""
    with pytest.raises(ClientConfigError, match="issues"):
        build_client_config(json.dumps({"issues": []}))


@pytest.mark.parametrize("missing", ["name", "channel_id", "bot_token"])
def test_an_issue_missing_a_required_field_is_refused(missing: str) -> None:
    """Caught here rather than in the client, because the client discovers it
    only when it has a post to publish — which is hours later and in a loop."""
    issue = {k: v for k, v in VALID["issues"][0].items() if k != missing}

    with pytest.raises(ClientConfigError, match=missing):
        build_client_config(json.dumps({"issues": [issue]}))


def test_discussion_id_is_optional() -> None:
    """A channel with comments turned off has no discussion group."""
    issue = {k: v for k, v in VALID["issues"][0].items() if k != "discussion_id"}

    assert build_client_config(json.dumps({"issues": [issue]}))["issues"][0]["name"] == "main"


def test_the_error_never_quotes_the_token() -> None:
    """The message goes to container logs, which are not a secret store."""
    token = "7542680000:AAH-not-a-real-token"

    with pytest.raises(ClientConfigError) as caught:
        build_client_config(json.dumps({"issues": [{"bot_token": token}]}))

    assert token not in str(caught.value)
