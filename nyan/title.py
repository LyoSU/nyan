from typing import List
from statistics import mean

from scipy.spatial.distance import cosine  # type: ignore

from nyan.document import Document
from nyan.util import normalize_url


def filter_uk_only(doc: Document) -> bool:
    return doc.language == "uk"


def filter_not_obscene(doc: Document) -> bool:
    return not doc.has_obscene


def filter_not_long(doc: Document) -> bool:
    if not doc.text:
        return False
    return len(doc.text) < 500


def filter_fresh(doc: Document) -> bool:
    if not doc.fetch_time or not doc.pub_time:
        return False
    return abs(doc.fetch_time - doc.pub_time) < 3600


def filter_purple(doc: Document) -> bool:
    if not doc.groups:
        return False
    return doc.groups.get("main") == "purple"


def choose_title(docs: List[Document], issues: List[str]) -> Document:
    assert docs

    avg_distances = dict()
    for doc1 in docs:
        distances = [cosine(doc1.embedding, doc2.embedding) for doc2 in docs]
        avg_distances[normalize_url(doc1.url)] = mean(distances)

    hard_filters = (filter_uk_only, filter_not_obscene, filter_fresh)
    for flt in hard_filters:
        filtered_docs = list(filter(flt, docs))
        if filtered_docs:
            docs = filtered_docs

    # Choosing documents specific for issues
    issue_filters = []
    first_doc_groups = docs[0].groups if docs[0].groups else {}
    possible_issues = set(first_doc_groups.keys())
    for issue in issues:
        if issue == "main":
            continue
        if issue not in possible_issues:
            continue
        # Double lambda to capture "issue" properly
        issue_filter = (lambda x: lambda doc: doc.groups.get(x) == x)(issue)
        issue_filters.append(issue_filter)

    soft_filters = [filter_not_long] + issue_filters + [filter_purple]

    for f in soft_filters:
        if not f:
            continue
        filtered_docs = list(filter(f, docs))
        if len(filtered_docs) >= 2:
            docs = filtered_docs

    return min(docs, key=lambda x: avg_distances[normalize_url(x.url)])
