from collections.abc import Callable

from nyan.annotator import Annotator
from nyan.document import Document


def test_annotator_on_snapshot(
    annotator: Annotator,
    input_docs: list[Document],
    output_docs: list[Document],
    compare_docs: Callable
):
    docs = annotator(input_docs)
    docs = annotator.postprocess(docs)
    assert len(docs) == len(output_docs), "Different number of documents"
    for predicted_doc, canonical_doc in zip(docs, output_docs, strict=True):
        compare_docs(predicted_doc, canonical_doc)


class FakeImageProcessor:
    """Embeds every url handed to it, except the ones that "fail to fetch"."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, urls: list[str]) -> list[dict[str, object]]:
        self.seen = list(urls)
        return [
            {"url": url, "embedding": [1.0, 0.0]} for url in urls if "dead" not in url
        ]


def video_annotator() -> tuple[Annotator, FakeImageProcessor]:
    """An annotator with nothing but the image step, which is all this needs.

    Built without `__init__` on purpose: loading the real encoders to check a
    dictionary remap would make this test take a minute and need the models on
    disk.
    """
    annotator = object.__new__(Annotator)
    processor = FakeImageProcessor()
    annotator.image_processor = processor  # type: ignore[assignment]
    return annotator, processor


def video_doc(videos: tuple[str, ...], thumbs: tuple[str, ...]) -> Document:
    return Document(
        url="https://t.me/a/1",
        channel_id="a",
        post_id=1,
        views=1,
        pub_time=100,
        videos=videos,
        video_thumbs=thumbs,
    )


def test_a_videos_vector_is_filed_under_the_video_not_the_still() -> None:
    """`Cluster.media` has the video url in hand and nothing else.

    What is embedded is the still, so the result has to be keyed back to the
    video, or the lookup finds nothing and the clip is compared by url again.
    """
    annotator, processor = video_annotator()

    embedded = annotator.process_video_thumbs(
        video_doc(("a.mp4", "b.mp4"), ("thumb-a.jpg", "thumb-b.jpg"))
    )

    assert processor.seen == ["thumb-a.jpg", "thumb-b.jpg"]
    assert [item["url"] for item in embedded] == ["a.mp4", "b.mp4"]


def test_a_video_without_a_still_is_not_embedded() -> None:
    """Telegram renders none for some, and a missing still is not an error."""
    annotator, _ = video_annotator()

    embedded = annotator.process_video_thumbs(video_doc(("a.mp4", "b.mp4"), ("", "")))

    assert embedded == []


def test_a_still_that_could_not_be_fetched_is_skipped() -> None:
    """Its video then falls back to url identity, which is the old behaviour."""
    annotator, _ = video_annotator()

    embedded = annotator.process_video_thumbs(
        video_doc(("a.mp4", "b.mp4"), ("dead-thumb.jpg", "thumb-b.jpg"))
    )

    assert [item["url"] for item in embedded] == ["b.mp4"]
