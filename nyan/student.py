"""The distilled post classifier: one ONNX model, every label the feed uses.

A multilingual-e5-base fine-tuned on 40k posts labelled by an LLM teacher,
with a head per question — category, topic, scope, region, role, significance
and three yes/no scores. On the blind 500 (`scripts/eval_jev.py`) its category
is right on 94-95% of posts against 72% for the joblib head it replaces, and it
runs on the CPU with no API behind it. The training and export scripts live
outside the repository, in `data/distill/scripts`; what ships is the model
directory they write: `model_emb8.onnx`, the tokenizer and `meta.json`, which
names the labels of every head so this file does not have to.

Its encoder is its own. The shared `embedding` stays with `Embedder`, whose
vectors the clusterer's thresholds were tuned on: a fine-tuned encoder places
posts by what they are about, not by which event they report, and swapping it
in would quietly change which posts are merged into one story.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nyan.document import Document


@dataclass(frozen=True)
class StudentConfig:
    path: str
    model_file: str = "model_emb8.onnx"
    batch_size: int = 16
    #: How sure the model has to be before a post leaves the feed. Below it the
    #: runner-up category wins instead. On the blind 500, 0.7 kept three of the
    #: nine news posts the plain argmax dropped and cost three of 116 caught
    #: not-news posts, at the same accuracy: a missed story costs more than a
    #: stray one.
    not_news_threshold: float = 0.0
    #: Enough for a batch of new posts to take seconds, and few enough to leave
    #: cores to the crawler and the image encoder sharing the machine.
    threads: int = 4


class Student:
    """Labels documents in batches; `load` returns None when the model is absent."""

    def __init__(self, config: StudentConfig) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        self.config = config
        with open(os.path.join(config.path, "meta.json")) as r:
            meta: dict[str, Any] = json.load(r)
        self.prefix: str = meta["prefix"]
        self.max_length: int = meta["max_length"]
        self.max_chars: int = meta["max_chars"]
        self.choices: dict[str, list[Any]] = meta["choices"]
        self.binary: list[str] = meta["binary"]
        self.outputs: list[str] = list(self.choices) + self.binary
        # Which training run, from the export; the directory name stays the same
        # across retrains, so it cannot tell a v1 answer from a v3 one in Mongo.
        self.name: str = meta.get("name") or os.path.basename(os.path.normpath(config.path))

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            os.path.join(config.path, config.model_file),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.path)

    @classmethod
    def load(cls, config: dict[str, Any]) -> "Student | None":
        """The model, or None and a warning when its files are not on disk.

        Models are not in the image but on a mounted volume, and this one is
        not in the release archive `download_models.sh` fetches. A host that
        has not been given it keeps annotating with the older category head
        instead of failing to start.
        """
        student_config = StudentConfig(**config)
        meta = os.path.join(student_config.path, "meta.json")
        model = os.path.join(student_config.path, student_config.model_file)
        if not (os.path.exists(meta) and os.path.exists(model)):
            logging.warning(
                "Student model not found at %s, keeping the old category head",
                student_config.path,
            )
            return None
        return cls(student_config)

    def predict(self, texts: list[str]) -> list[dict[str, NDArray[np.float32]]]:
        """Every head's probabilities for each text, in the order given."""
        inputs = [self.prefix + text[: self.max_chars] for text in texts]
        # Batched by length, so a batch pads to its longest post and not to the
        # longest post of the whole call.
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        results: list[dict[str, NDArray[np.float32]]] = [{} for _ in inputs]
        for start in range(0, len(order), self.config.batch_size):
            indices = order[start : start + self.config.batch_size]
            batch = self.tokenizer(
                [inputs[i] for i in indices],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            outputs = self.session.run(
                self.outputs,
                {
                    "input_ids": batch["input_ids"].astype(np.int64),
                    "attention_mask": batch["attention_mask"].astype(np.int64),
                },
            )
            for column, index in enumerate(indices):
                results[index] = {
                    name: np.asarray(value[column])
                    for name, value in zip(self.outputs, outputs, strict=True)
                }
        return results

    def record(self, probs: dict[str, NDArray[np.float32]]) -> dict[str, Any]:
        """What is kept of one post's answers besides its category."""
        record: dict[str, Any] = {"model": self.name}
        for head, labels in self.choices.items():
            if head == "category":
                continue
            best = int(np.argmax(probs[head]))
            record[head] = labels[best]
            record[f"{head}_p"] = round(float(probs[head][best]), 3)
        # Significance is an ordinal scale, and its expectation is steadier than
        # its argmax: a post torn between 3 and 4 reads as 3.5, not as either.
        scale = np.asarray(self.choices["significance"], dtype=float)
        record["significance_mean"] = round(float(probs["significance"] @ scale), 2)
        for head in self.binary:
            record[head] = round(float(probs[head]), 3)
        return record

    def pick(self, scores: NDArray[np.float32]) -> str:
        categories = self.choices["category"]
        ranked = [categories[i] for i in np.argsort(-scores)]
        not_news = categories.index("not_news")
        if ranked[0] == "not_news" and scores[not_news] < self.config.not_news_threshold:
            return str(ranked[1])
        return str(ranked[0])

    def __call__(self, docs: list[Document]) -> list[Document]:
        wanted = [doc for doc in docs if doc.patched_text]
        if not wanted:
            return docs
        predictions = self.predict([doc.patched_text or "" for doc in wanted])
        categories = self.choices["category"]
        for doc, probs in zip(wanted, predictions, strict=True):
            scores = probs["category"]
            doc.category = self.pick(scores)
            doc.category_scores = {
                label: float(score) for label, score in zip(categories, scores, strict=True)
            }
            doc.student = self.record(probs)
        return docs
