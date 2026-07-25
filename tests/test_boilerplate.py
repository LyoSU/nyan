
from nyan.annotator import Annotator, boilerplate_key
from nyan.document import Document


def make_doc(channel_id: str, text: str, url: str) -> Document:
    return Document(
        url=url,
        channel_id=channel_id,
        post_id=1,
        views=100,
        pub_time=1700000000,
        issue="main",
        groups={"main": "blue"},
        patched_text=text,
    )


def channel_docs(channel_id: str, bodies: list[str], footer: str = "") -> list[Document]:
    docs = []
    for i, body in enumerate(bodies):
        text = f"{body}\n{footer}" if footer else body
        docs.append(make_doc(channel_id, text, f"https://t.me/{channel_id}/{i}"))
    return docs


def test_key_ignores_case_and_whitespace() -> None:
    assert boilerplate_key("  Підпишись   на Канал ") == boilerplate_key(
        "підпишись на канал"
    )


def test_repeated_footer_is_removed(annotator: Annotator) -> None:
    bodies = [f"Унікальна новина номер {i} з подробицями." for i in range(6)]
    docs = channel_docs("uanews", bodies, footer="Підпишись на Канал | Facebook | X.")

    annotator.strip_boilerplate(docs)

    for doc, body in zip(docs, bodies, strict=True):
        assert doc.patched_text == body


def test_a_channel_with_too_few_posts_is_left_alone(annotator: Annotator) -> None:
    """Below the minimum, a repeated line is not yet evidence of a template."""
    bodies = [f"Новина {i} з подробицями." for i in range(3)]
    footer = "Підпишись на Канал."
    docs = channel_docs("small", bodies, footer=footer)

    annotator.strip_boilerplate(docs)

    for doc in docs:
        assert doc.patched_text is not None
        assert footer in doc.patched_text


def test_a_line_in_a_minority_of_posts_is_kept(annotator: Annotator) -> None:
    """Occasional repetition is a recurring topic, not boilerplate."""
    docs = channel_docs("mixed", [f"Новина {i} з подробицями." for i in range(8)])
    docs[0].patched_text = "Новина 0 з подробицями.\nПовітряна тривога."
    docs[1].patched_text = "Новина 1 з подробицями.\nПовітряна тривога."

    annotator.strip_boilerplate(docs)

    assert docs[0].patched_text is not None
    assert "Повітряна тривога." in docs[0].patched_text


def test_channels_are_treated_separately(annotator: Annotator) -> None:
    shared = "Читайте нас у Telegram."
    noisy = channel_docs("noisy", [f"Новина {i} тут." for i in range(6)], footer=shared)
    quiet = channel_docs("quiet", [f"Інша новина {i} тут." for i in range(6)])
    quiet[0].patched_text = f"Інша новина 0 тут.\n{shared}"

    annotator.strip_boilerplate(noisy + quiet)

    assert noisy[0].patched_text == "Новина 0 тут."
    # The same line is rare for the other channel, so it stays.
    assert quiet[0].patched_text is not None
    assert shared in quiet[0].patched_text


def test_a_post_that_is_only_boilerplate_becomes_empty(annotator: Annotator) -> None:
    footer = "Підпишись на Канал."
    docs = channel_docs("promo", [f"Новина {i} з подробицями." for i in range(6)], footer)
    docs.append(make_doc("promo", footer, "https://t.me/promo/99"))

    kept = annotator.postprocess(docs)

    # is_discarded() drops it, which is what should happen to a post with
    # nothing but a subscribe line in it.
    assert "https://t.me/promo/99" not in {doc.url for doc in kept}
    assert len(kept) == 6
