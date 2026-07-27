from typing import TypeVar, Any, cast
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
import requests
import torch
from transformers import AutoModel, AutoProcessor
from tqdm.auto import tqdm
from PIL import Image

from nyan.util import gen_batch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SigLIP 2 in place of the original CLIP. Same idea — one shared space for
# images and texts, compared by cosine — but trained with a sigmoid loss, which
# separates near-identical images much better. That is precisely the job here:
# recognising the same photo reposted by several channels, and a channel's own
# recurring boilerplate image. The thresholds that consume these similarities
# are calibrated per model, so changing this id means re-checking them.
# The NaFlex variant, not the fixed-224 one: it keeps a picture's native
# aspect ratio instead of squashing it to a square, which matters for the
# screenshots and wide news photos that make up much of the feed. It is also
# the variant that loads cleanly — the fixed-resolution siglip2 checkpoints
# report `model_type: siglip`, for which transformers 5.2 has a null tokenizer
# mapping and raises AttributeError inside AutoProcessor.
DEFAULT_MODEL_PATH = "google/siglip2-base-patch16-naflex"

# SigLIP was trained with every caption padded to a fixed length, and the text
# tower expects that shape. Padding to the longest item in the batch instead
# would make one text's embedding depend on what else happened to be batched
# with it — a difference that is invisible until retrieval quality drifts.
TEXT_MAX_LENGTH = 64

T = TypeVar("T")


class VisionEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_PATH,
        normalize: bool = True,
        image_batch_size: int = 16,
        text_batch_size: int = 32,
        device: str = DEVICE,
        enable_tqdm: bool = False,
    ):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.normalize = normalize
        self.enable_tqdm = enable_tqdm

    def fetch_images(self, urls: list[str]) -> list[dict[str, Any]]:
        images = []
        for url in urls:
            if not url.startswith("http://") and not url.startswith("https://"):
                continue
            try:
                response = requests.get(url, stream=True)
            except Exception:
                continue
            if response.status_code != 200:
                continue
            images.append({"url": url, "content": Image.open(response.raw)})  # type: ignore[arg-type]
        return images

    def embed_images(self, images: list[Image.Image]) -> NDArray[np.float32]:
        return self._calc_embeddings(
            func=self._process_images_batch,
            inputs=images,
            batch_size=self.image_batch_size,
            desc="Image embeddings",
        )

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        return self._calc_embeddings(
            func=self._process_texts_batch,
            inputs=texts,
            batch_size=self.text_batch_size,
            desc="Text embeddings",
        )

    def _calc_embeddings(
        self,
        func: Callable[[list[T]], torch.Tensor],
        inputs: list[T],
        batch_size: int,
        desc: str,
    ) -> NDArray[np.float32]:
        # The width of the output is whatever the model returns, discovered from
        # the batches themselves rather than read off a config attribute:
        # `projection_dim` exists on CLIP and not on SigLIP, and hard-coding
        # either one is how this breaks again at the next encoder swap.
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)
        batches: list[torch.Tensor] = []
        total = (len(inputs) + batch_size - 1) // batch_size
        gen = gen_batch(inputs, batch_size)
        for batch in tqdm(gen, total=total, desc=desc, disable=not self.enable_tqdm):
            with torch.no_grad():
                batches.append(func(batch).cpu())
        embeddings = torch.cat(batches, dim=0)
        if self.normalize:
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return cast(NDArray[np.float32], embeddings.numpy())

    def _process_images_batch(self, images: list[Image.Image]) -> torch.Tensor:
        inputs: dict[str, torch.Tensor] = self.processor(
            images=images, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        result = self.model.get_image_features(**inputs)
        if not isinstance(result, torch.Tensor):
            result = result.pooler_output
        return cast(torch.Tensor, result)

    def _process_texts_batch(self, texts: list[str]) -> torch.Tensor:
        inputs: dict[str, torch.Tensor] = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        result = self.model.get_text_features(**inputs)
        if not isinstance(result, torch.Tensor):
            result = result.pooler_output
        return cast(torch.Tensor, result)
