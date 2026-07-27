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
