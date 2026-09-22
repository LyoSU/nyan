import json
import os

import pytest

from nyan.document import Document
from nyan.student import Student

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
        assert labelled.student["model"] == os.path.basename(str(student_config()["path"]))


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
