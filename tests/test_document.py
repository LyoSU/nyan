from nyan.document import CURRENT_VERSION, Document


def make_doc(**kwargs) -> Document:
    fields = dict(
        url="https://t.me/uniannet/1",
        channel_id="uniannet",
        post_id=1,
        views=100,
        pub_time=1000,
        text="Якийсь достатньо довгий текст новини",
    )
    fields.update(kwargs)
    return Document(**fields)


def test_reannotation_not_needed_when_nothing_changed():
    stored = make_doc(issue="main", groups={"main": "blue"}, version=CURRENT_VERSION)
    assert not stored.is_reannotation_needed(make_doc(), is_known_channel=True)


def test_reannotation_needed_when_text_changed():
    stored = make_doc(issue="main", groups={"main": "blue"}, version=CURRENT_VERSION)
    assert stored.is_reannotation_needed(make_doc(text="Інший текст"), is_known_channel=True)


def test_reannotation_needed_when_annotation_lost_its_channel():
    """A stored annotation of a configured channel that carries no channel data.

    This is what the crawler wrote during the 2026-07-25 incident: the version
    and the text match, so the old check saw nothing wrong, and the document
    stayed unusable forever — `is_discarded()` drops anything without an issue.
    """
    stored = make_doc(issue=None, groups={}, version=CURRENT_VERSION)
    assert stored.is_reannotation_needed(make_doc(), is_known_channel=True)


def test_no_reannotation_loop_for_channels_absent_from_config():
    """The same empty annotation is correct when the channel is not configured.

    Re-annotating it would recompute an embedding on every iteration and still
    produce nothing, because there is no channel to take an issue from.
    """
    stored = make_doc(issue=None, groups={}, version=CURRENT_VERSION)
    assert not stored.is_reannotation_needed(make_doc(), is_known_channel=False)
