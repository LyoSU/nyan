"""Tests for reading Telegram's own file ids out of a send response.

A file_id is the only handle on an attachment that stays valid: the CDN URL the
crawler found expires, which is what `fix_media_url` has been working around,
while a file_id can be sent back to Telegram indefinitely. So every send and
every edit is read for them, and later edits reference the file instead of
asking Telegram to fetch a URL again.
"""

from nyan.media import (
    MEDIA_PHOTO,
    MEDIA_VIDEO,
    attach_urls,
    extract_sent_media,
    largest_file_id,
)


def photo_sizes() -> list[dict[str, object]]:
    """How Telegram reports a photo: every rendition, smallest first."""
    return [
        {"file_id": "thumb", "width": 90, "height": 60, "file_size": 1000},
        {"file_id": "full", "width": 1280, "height": 853, "file_size": 200000},
    ]


def test_the_biggest_rendition_of_a_photo_is_kept() -> None:
    """A thumbnail's file_id would silently downgrade the post on the next edit."""
    assert largest_file_id(photo_sizes()) == "full"


def test_a_video_reports_one_file_id() -> None:
    assert largest_file_id({"file_id": "vid"}) == "vid"


def test_a_photo_message_yields_its_file_id() -> None:
    media = extract_sent_media({"message_id": 5, "photo": photo_sizes()})

    assert [(item.type, item.file_id, item.message_id) for item in media] == [
        (MEDIA_PHOTO, "full", 5)
    ]


def test_a_media_group_yields_one_item_per_message() -> None:
    """A group is not one message but one per attachment, and editing the third
    photo of a slideshow needs the third message's id."""
    media = extract_sent_media(
        [
            {"message_id": 5, "photo": photo_sizes()},
            {"message_id": 6, "video": {"file_id": "vid"}},
        ]
    )

    assert [(item.type, item.file_id, item.message_id) for item in media] == [
        (MEDIA_PHOTO, "full", 5),
        (MEDIA_VIDEO, "vid", 6),
    ]


def test_a_videos_thumbnail_is_not_a_second_attachment() -> None:
    """Only keys that name a media type are followed."""
    media = extract_sent_media(
        {
            "message_id": 5,
            "video": {"file_id": "vid", "thumbnail": {"file_id": "thumb"}},
        }
    )

    assert [item.file_id for item in media] == ["vid"]


def test_media_nested_in_a_block_tree_is_found() -> None:
    """A rich message carries its attachments inside blocks, not at the top."""
    media = extract_sent_media(
        {
            "message_id": 5,
            "blocks": [
                {"type": "heading"},
                {"type": "slideshow", "blocks": [{"photo": photo_sizes()}]},
            ],
        }
    )

    assert [item.file_id for item in media] == ["full"]


def test_a_response_without_media_yields_nothing() -> None:
    assert extract_sent_media({"message_id": 5, "text": "Текст"}) == []


def test_urls_are_paired_with_file_ids_by_position() -> None:
    """Position is the only link: Telegram answers in the order it was sent."""
    media = extract_sent_media(
        [
            {"message_id": 5, "photo": photo_sizes()},
            {"message_id": 6, "video": {"file_id": "vid"}},
        ]
    )

    attach_urls(media, ["a.jpg", "b.mp4"])

    assert [(item.url, item.file_id) for item in media] == [
        ("a.jpg", "full"),
        ("b.mp4", "vid"),
    ]


def test_a_short_response_does_not_shift_urls_onto_the_wrong_file() -> None:
    """One attachment came back for two sent: the pairing stops, it does not slide."""
    media = extract_sent_media([{"message_id": 5, "photo": photo_sizes()}])

    attach_urls(media, ["a.jpg", "b.jpg"])

    assert [(item.url, item.file_id) for item in media] == [("a.jpg", "full")]
