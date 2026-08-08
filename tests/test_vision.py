from collections.abc import Callable
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from nyan.vision import VisionEmbedder


def test_vision_matches_texts_to_images(image_data):
    texts = [r["en_text"] for r in image_data]
    images = [r["image"] for r in image_data]
    embedder = VisionEmbedder()
    images = embedder.fetch_images(images)
    text_embeddings = embedder.embed_texts(texts)
    image_embeddings = embedder.embed_images([i["content"] for i in images])
    similarity = cosine_similarity(text_embeddings, image_embeddings)
    assert len(images) == len(texts), "Some images could not be fetched"
    for i, (text, image) in enumerate(zip(texts, images, strict=True)):
        best_index = similarity[i].argmax()
        assert best_index == i, \
            f"{text} vs {image} mismatch, matching image: {images[best_index]}"


def test_image_processor(image_data, annotator):
    images = [r["image"] for r in image_data]

    embedder = VisionEmbedder()
    fetched_images = embedder.fetch_images(images)
    image_embeddings = embedder.embed_images([i["content"] for i in fetched_images])

    embedded_images = annotator.image_processor(images)
    for embedded_image, embedding in zip(embedded_images, image_embeddings, strict=True):
        assert embedded_image["embedding"] == embedding.tolist()


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.status_code = 200
        self.content = content
        self.raw = BytesIO(content)


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def fetcher(
    bodies: dict[str, bytes]
) -> tuple[VisionEmbedder, Callable[..., FakeResponse]]:
    """A `VisionEmbedder` with no encoder, serving canned bodies.

    Built without `__init__` on purpose: what a 200 with a broken body does is
    decided before any model is touched, and loading SigLIP to find out would
    make this test take a minute and need the weights on disk.
    """
    embedder: VisionEmbedder = object.__new__(VisionEmbedder)
    return embedder, lambda url, **kwargs: FakeResponse(bodies[url])


@pytest.mark.parametrize(
    "body", [b"<html>file not found</html>", b"", png_bytes()[:20]],
    ids=["error page", "empty", "cut off"],
)
def test_a_200_that_is_not_an_image_is_skipped(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    """Telegram serves all three for a still whose link has gone stale.

    Unhandled, any of them ends the daemon mid-annotation, which restarts onto
    the very same still — the post is never annotated, so it is never done with.
    """
    url = "https://cdn.telegram/stale.jpg"
    embedder, get = fetcher({url: body})
    monkeypatch.setattr(requests, "get", get)

    assert embedder.fetch_images([url]) == []


def test_the_readable_stills_around_a_broken_one_survive(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreadable still costs its own video's vector and nothing else."""
    good, bad = "https://cdn.telegram/a.png", "https://cdn.telegram/b.jpg"
    embedder, get = fetcher({good: png_bytes(), bad: b"<html>gone</html>"})
    monkeypatch.setattr(requests, "get", get)

    fetched = embedder.fetch_images([bad, good])

    assert [image["url"] for image in fetched] == [good]
    assert fetched[0]["content"].size == (4, 4)


def test_an_annotated_image_carries_what_picks_between_copies(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hashes, the quality and the signature, computed at fetch time.

    All three describe the picture rather than its meaning, and all three are
    needed when a post is assembled — long after the crawl, from a document
    read out of Mongo. Recomputing them then would mean fetching every copy of
    every candidate again, inside `Cluster.media`, which the daemon calls for
    every cluster on every iteration.
    """
    from nyan.image import ImageProcessor
    from nyan.picture import SIGNATURE_SIDE

    url = "https://cdn.telegram/photo.png"
    embedder, get = fetcher({url: png_bytes()})
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(
        embedder, "embed_images", lambda images: np.zeros((len(images), 4)), raising=False
    )
    processor: ImageProcessor = object.__new__(ImageProcessor)
    processor.vision_embedder = embedder

    (annotated,) = processor([url])

    assert annotated["url"] == url
    assert len(annotated["hashes"]) == 2
    assert 0.0 <= annotated["quality"] <= 1.0
    assert len(annotated["signature"]) == SIGNATURE_SIDE * SIGNATURE_SIDE
