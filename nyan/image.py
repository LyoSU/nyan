from typing import Any

from nyan.vision import VisionEmbedder, DEFAULT_MODEL_PATH


class ImageProcessor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.vision_embedder = VisionEmbedder(
            config.get("model_name", DEFAULT_MODEL_PATH)
        )

    def __call__(self, images: list[str]) -> list[dict[str, Any]]:
        fetched_images = self.vision_embedder.fetch_images(images)
        if not fetched_images:
            return []
        contents = [i["content"] for i in fetched_images]
        embeddings = self.vision_embedder.embed_images(contents)
        return [
            {"url": image["url"], "embedding": embedding.tolist()}
            for image, embedding in zip(fetched_images, embeddings, strict=True)
        ]
