from nyan import rich
from nyan.media import MEDIA_PHOTO, MediaItem


def test_join_drops_empty_parts_so_separators_never_double() -> None:
    assert rich.join(["a", None, "", [], "b"]) == ["a", " · ", "b"]


def test_join_of_single_part_has_no_separator() -> None:
    assert rich.join([None, "only"]) == ["only"]


def test_join_of_nothing_is_empty() -> None:
    assert rich.join([None, "", []]) == []


def test_link_needs_no_escaping_of_the_title() -> None:
    # The title that would break the HTML template goes through untouched.
    block = rich.paragraph(rich.link("Кіно & Театр <18>", "https://t.me/c/1"))
    assert block["text"]["text"] == "Кіно & Театр <18>"


def test_date_time_keeps_a_standalone_fallback() -> None:
    entity = rich.date_time("14:23", 1647531900)
    assert entity["text"] == "14:23"
    assert entity["unix_time"] == 1647531900
    assert entity["date_time_format"] == "t"


def test_photo_rewrites_unreachable_media_host() -> None:
    block = rich.photo("https://cdn4.telesco.pe/file/x.jpg")
    assert block["photo"]["media"] == "https://cdn4.cdn-telegram.org/file/x.jpg"


def test_media_block_has_no_caption_key_when_there_is_no_caption() -> None:
    assert "caption" not in rich.photo("https://example.com/x.jpg")


def test_photo_caption_carries_credit() -> None:
    block = rich.photo("https://example.com/x.jpg", credit="Суспільне")
    assert block["caption"] == {"text": "", "credit": "Суспільне"}


def test_blockquote_omits_credit_when_absent() -> None:
    assert "credit" not in rich.blockquote(rich.paragraph("text"))


def test_details_is_collapsed_by_default() -> None:
    assert "is_open" not in rich.details("summary", rich.paragraph("text"))
    assert rich.details("s", is_open=True)["is_open"] is True


def test_bullet_list_wraps_every_item_in_blocks() -> None:
    block = rich.bullet_list([rich.paragraph("one")], [rich.paragraph("two")])
    assert block["type"] == "list"
    assert [item["blocks"][0]["text"] for item in block["items"]] == ["one", "two"]


def test_heading_size_must_be_in_range() -> None:
    import pytest

    with pytest.raises(AssertionError):
        rich.heading("too small", size=7)


def test_count_blocks_counts_nested_and_list_items() -> None:
    blocks = [
        rich.heading("h"),
        rich.details(
            "s",
            rich.bullet_list([rich.paragraph("a")], [rich.paragraph("b")]),
            rich.divider(),
        ),
    ]
    # heading + details + list + 2 items + divider
    assert rich.count_blocks(blocks) == 6


def test_rendered_post_knows_its_format() -> None:
    assert rich.RenderedPost(blocks=[]).is_rich
    assert not rich.RenderedPost(text="plain").is_rich
    assert not rich.RenderedPost(text="plain").has_media
    media = (MediaItem(type=MEDIA_PHOTO, url="u"),)
    assert rich.RenderedPost(text="plain", media=media).has_media
