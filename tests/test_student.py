import json

import numpy as np
import pytest

from nyan.annotator import Annotator
from nyan.document import Document
from nyan.rubrics import RubricDetector
from nyan.student import Student, StudentConfig

CONFIG_PATH = "configs/annotator_config.json"


def student_config() -> dict[str, object]:
    with open(CONFIG_PATH) as r:
        config: dict[str, object] = json.load(r)["student"]
    return config


def doc(post_id: int, text: str | None) -> Document:
    return Document(
        url=f"https://t.me/a/{post_id}",
        channel_id="a",
        post_id=post_id,
        views=1,
        pub_time=100,
        patched_text=text,
    )


def test_a_host_without_the_model_keeps_the_old_head() -> None:
    """The model is not in the release archive, so its absence is ordinary."""
    assert Student.load({"path": "models/no_such_student"}) is None


needs_model = pytest.mark.skipif(
    Student.load(student_config()) is None, reason="no student model on this machine"
)


@needs_model
def test_every_post_with_text_gets_a_category_and_the_other_heads() -> None:
    student = Student.load(student_config())
    assert student is not None
    docs = [
        doc(1, "Сили ППО збили 40 дронів над Київщиною, уламки впали на житловий будинок"),
        doc(2, "НБУ залишив облікову ставку без змін — 15,5%"),
        doc(3, None),
    ]

    student(docs)

    assert docs[0].category == "war"
    assert docs[1].category == "economy"
    assert docs[2].category is None and docs[2].student == {}
    for labelled in docs[:2]:
        assert labelled.category_scores[labelled.category] == max(labelled.category_scores.values())
        assert sum(labelled.category_scores.values()) == pytest.approx(1.0, abs=1e-4)
        assert {"topic", "scope", "region", "significance_mean", "urgent"} <= set(labelled.student)
        assert labelled.student["model"] == student.name


@needs_model
def test_a_batch_answers_as_each_post_would_alone() -> None:
    """Length-sorted batching must hand every answer back to its own post."""
    student = Student.load(student_config())
    assert student is not None
    texts = [
        "Коротко: у Львові відключення світла.",
        "Довший текст " * 40 + "про вибори до парламенту Молдови.",
    ]

    together = student.predict(texts)
    alone = [student.predict([text])[0] for text in texts]

    for a, b in zip(together, alone, strict=True):
        assert abs(a["category"] - b["category"]).max() < 1e-3


class FakeStudent:
    """Answers as the model would, without the model: CI has no ONNX file to load."""

    def __init__(self, category: str) -> None:
        self.category = category

    def __call__(self, docs: list[Document]) -> list[Document]:
        for labelled in docs:
            if labelled.patched_text:
                labelled.category = self.category
                labelled.category_scores = {self.category: 0.9, "other": 0.1}
                labelled.student = {"model": "fake", "topic": f"{self.category}.front"}
        return docs


class OldHead:
    """The joblib head; it must stay silent once a student has answered."""

    def __call__(self, embedding: list[float], embedding_key: str) -> tuple[str, dict[str, float]]:
        return "sports", {"sports": 1.0}


def annotator_after_embeddings(student: FakeStudent | None) -> Annotator:
    """An annotator with only the steps that decide the category.

    Built without `__init__`, like the one in `test_annotator.py`: the encoders
    take a minute to load and have nothing to say about which head wins.
    """
    annotator = object.__new__(Annotator)
    annotator.student = student  # type: ignore[assignment]
    annotator.cat_detector = OldHead()  # type: ignore[assignment]
    annotator.rubric_detector = RubricDetector({"patterns": ["^реклама\\b"]})
    return annotator


def decide(annotator: Annotator, docs: list[Document]) -> list[Document]:
    """The category steps of `Annotator.__call__`, in its order."""
    if annotator.student is not None:
        docs = annotator.student(docs)
    return [annotator.detect_rubrics(annotator.predict_category(d)) for d in docs]


def embedded(post_id: int, text: str) -> Document:
    document = doc(post_id, text)
    document.embedding = [0.1, 0.2]
    document.embedding_key = "multilingual_e5_base"
    return document


def test_the_students_category_is_the_one_the_annotator_keeps() -> None:
    """With both heads loaded, the older one must not write over the student."""
    annotator = annotator_after_embeddings(FakeStudent("war"))

    (labelled,) = decide(annotator, [embedded(1, "Сили оборони відбили штурм")])

    assert labelled.category == "war"
    assert labelled.student["model"] == "fake"


def test_a_rubric_still_overrules_the_student() -> None:
    annotator = annotator_after_embeddings(FakeStudent("economy"))

    (labelled,) = decide(annotator, [embedded(1, "Реклама нового банку")])

    assert labelled.category == "not_news"


def test_without_a_student_the_older_head_answers_and_leaves_no_record() -> None:
    """What a host without the model does, and what `student_health` looks for."""
    annotator = annotator_after_embeddings(None)

    (labelled,) = decide(annotator, [embedded(1, "Збірна виграла матч")])

    assert labelled.category == "sports"
    assert labelled.student == {}


def bare_student(threshold: float) -> Student:
    """The picking and recording logic over made-up probabilities, without a session."""
    student = object.__new__(Student)
    student.config = StudentConfig(path="-", not_news_threshold=threshold)
    student.name = "bare"
    student.choices = {
        "category": ["not_news", "war", "politics"],
        "significance": [1, 2, 3, 4, 5],
        "topic": ["war.front", "politics.world"],
    }
    student.binary = ["urgent"]
    return student


def test_an_unsure_not_news_gives_way_to_the_runner_up() -> None:
    """A post leaves the feed only when the model is sure: a missed story costs more."""
    student = bare_student(0.7)

    assert student.pick(np.array([0.6, 0.3, 0.1], dtype=np.float32)) == "war"
    assert student.pick(np.array([0.8, 0.15, 0.05], dtype=np.float32)) == "not_news"
    assert student.pick(np.array([0.2, 0.3, 0.5], dtype=np.float32)) == "politics"


def test_significance_is_recorded_as_an_expectation() -> None:
    student = bare_student(0.7)

    record = student.record(
        {
            "significance": np.array([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32),
            "topic": np.array([0.9, 0.1], dtype=np.float32),
            "urgent": np.float32(0.25),
        }
    )

    assert record["significance_mean"] == 3.5
    assert record["topic"] == "war.front" and record["topic_p"] == 0.9
    assert record["urgent"] == 0.25 and record["model"] == "bare"
