from collections.abc import Callable
from statistics import mean

from scipy.spatial.distance import cosine  # type: ignore

from nyan.document import Document
from nyan.util import normalize_url


# A title longer than this reads as an article rather than a headline.
MAX_TITLE_LENGTH = 500

# A document is "fresh" when it was fetched close to when it was published, so
# its view count and text still reflect the original post.
MAX_FETCH_DELAY_SECONDS = 3600

DocumentFilter = Callable[[Document], bool]


def filter_uk_only(doc: Document) -> bool:
    return doc.language == "uk"


def filter_not_obscene(doc: Document) -> bool:
    return not doc.has_obscene


def filter_not_long(doc: Document) -> bool:
    if not doc.text:
        return False
    return len(doc.text) < MAX_TITLE_LENGTH


def filter_fresh(doc: Document) -> bool:
    if not doc.fetch_time or not doc.pub_time:
        return False
    return abs(doc.fetch_time - doc.pub_time) < MAX_FETCH_DELAY_SECONDS


def filter_purple(doc: Document) -> bool:
    if not doc.groups:
        return False
    return doc.groups.get("main") == "purple"


def make_issue_filter(issue: str) -> DocumentFilter:
    """Accepts documents from channels that cover `issue`."""

    def flt(doc: Document) -> bool:
        return issue in doc.groups

    return flt


def choose_title(docs: list[Document], issues: list[str]) -> Document:
    assert docs

    avg_distances = dict()
    for doc1 in docs:
        distances = [cosine(doc1.embedding, doc2.embedding) for doc2 in docs]
        avg_distances[normalize_url(doc1.url)] = mean(distances)

    hard_filters: tuple[DocumentFilter, ...] = (
        filter_uk_only,
        filter_not_obscene,
        filter_fresh,
    )
    for flt in hard_filters:
        filtered_docs = list(filter(flt, docs))
        if filtered_docs:
            docs = filtered_docs

    # Prefer channels that actually cover this issue, so a war story gets its
    # title from a war channel rather than from whichever channel happened to
    # post it. `doc.groups` maps an issue to the channel's trust group, so
    # membership — not the group's value — is what marks the channel as
    # covering it.
    issue_filters = [
        make_issue_filter(issue)
        for issue in issues
        if issue != "main" and issue in docs[0].groups
    ]

    soft_filters: list[DocumentFilter] = [
        filter_not_long,
        *issue_filters,
        filter_purple,
    ]

    for flt in soft_filters:
        filtered_docs = list(filter(flt, docs))
        if len(filtered_docs) >= 2:
            docs = filtered_docs

    return min(docs, key=lambda x: avg_distances[normalize_url(x.url)])
