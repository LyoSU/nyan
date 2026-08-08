from typing import Any

from nyan.picture import perceptual_hashes, picture_quality, thumbnail_signature
from nyan.vision import VisionEmbedder, DEFAULT_MODEL_PATH


class ImageProcessor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.vision_embedder = VisionEmbedder(
            config.get("model_name", DEFAULT_MODEL_PATH)
        )

    def __call__(self, images: list[str]) -> list[dict[str, Any]]:
        """What is worth knowing about each picture, from one fetch of it.

        The embedding says what the picture is about; the rest say which
        rendition of it this is. Both are needed when a post is assembled, and
        that happens long after the crawl, from documents read out of Mongo —
        so computing the second kind then would mean fetching every copy of
        every candidate again, inside `Cluster.media`, which the daemon calls
        for every cluster on every iteration. Here the picture is already
        decoded and in hand, and all three readings are arithmetic over it.
        """
        fetched_images = self.vision_embedder.fetch_images(images)
        if not fetched_images:
            return []
        contents = [i["content"] for i in fetched_images]
        embeddings = self.vision_embedder.embed_images(contents)
        return [
            {
                "url": image["url"],
                "embedding": embedding.tolist(),
                "hashes": list(perceptual_hashes(image["content"])),
                "quality": picture_quality(image["content"]).score,
                "signature": list(thumbnail_signature(image["content"])),
            }
            for image, embedding in zip(fetched_images, embeddings, strict=True)
        ]
