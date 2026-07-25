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
