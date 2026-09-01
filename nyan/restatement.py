"""Whether a sentence says the headline again.

A post opens with its headline in bold and the first paragraph right under it.
On a site those two are meant to overlap — the headline sells, the lede tells —
but in a chat the reader sees one bold line and then the same line in plain
weight, and reads it as a stutter. The prompt forbids it; the model still does
it often enough that the renderer has to be able to tell.

The measure is lexical, on stems rather than words, because Ukrainian declines
everything: "мобілізацію" in the headline is "мобілізації" in the lede. Five
letters is where the stem of most content words ends and the ending begins.
Function words are dropped, since sharing "після" says nothing about sharing a
fact, and numbers are kept whole, since a figure is a fact by itself.

Two thresholds, because two kinds of sentence get measured. A lede the model
wrote is ours to cut, and a sentence that is the headline plus padding is
exactly the violation the prompt names. A sentence quoted from a channel is
theirs, and is either shown whole or not at all — so it goes only when it is
nearly nothing but the headline, and stays when it carries a reason the
headline had no room for.
"""

import re

from nyan.markup import strip_markup


# How much of the headline a sentence has to contain before it can count as a
# restatement at all. Below this the sentence is about the same story but says
# something else, and its length is beside the point.
HEADLINE_COVERED = 0.75

# Share of a sentence that is the headline again, above which it is dropped.
# The lede value admits sentences that pad the headline with a clause of
# filler; the quoted value admits only sentences that barely leave the
# headline's words, because dropping a channel's sentence drops whatever fact
# it added along the way.
LEDE_RESTATED = 0.4
QUOTED_RESTATED = 0.6

STEM_LENGTH = 5
# Shorter tokens are endings and particles more often than words; a number is
# kept at any length.
MIN_WORD_LENGTH = 3

# Prepositions, conjunctions, pronouns and auxiliaries: words two sentences
# share whether or not they share a fact.
STOPWORDS = frozenset(
    {
        # Conjunctions and particles.
        "і", "й", "та", "а", "але", "або", "чи", "що", "щоб", "як", "не", "ні", "так",
        "бо", "тому", "вже", "уже", "ще", "же", "ж", "також", "теж", "лише", "тільки",
        "навіть", "майже", "понад", "близько", "коли", "де", "куди", "там", "тут",
        # Prepositions.
        "у", "в", "на", "з", "із", "зі", "до", "від", "для", "при", "під", "над", "про",
        "після", "через", "між", "серед", "без", "за", "по", "щодо", "проти", "поза",
        "попри",
        # Pronouns and demonstratives, in the forms that occur in news prose.
        "це", "цей", "ця", "ці", "цього", "цієї", "цих", "цим", "цією", "той", "те",
        "ті", "того", "тієї", "тих", "тим", "тією", "який", "яка", "які", "яке", "якого",
        "якій", "яких", "його", "її", "їх", "їхній", "він", "вона", "вони", "воно", "ми",
        "ви", "я", "ти", "свій", "своя", "свої", "своє",
        # To be.
        "є", "був", "була", "були", "було", "буде", "бути",
    }
)

# Letters and digits, with apostrophes and hyphens kept inside a word so that
# "дев'ятеро" and "58-річна" stay whole.
_WORD = re.compile(r"[^\W_]+(?:['’ʼ-][^\W_]+)*")

# A sentence ends at .!? followed by whitespace, or at a line break — Telegram
# posts put their lead on its own line, often with no trailing punctuation.
_SENTENCE_END_CHARS = ".!?"


def stems(text: str) -> set[str]:
    """The content words of `text`, cut to their stems, markup removed."""
    result: set[str] = set()
    for token in _WORD.findall(strip_markup(text).lower()):
        if token in STOPWORDS:
            continue
        if len(token) < MIN_WORD_LENGTH and not any(c.isdigit() for c in token):
            continue
        result.add(token[:STEM_LENGTH])
    return result


def restatement(headline: str, sentence: str) -> float:
    """How much of `sentence` is `headline` said again, from 0.0 to 1.0.

    Zero unless the sentence contains nearly the whole headline: a sentence
    that shares half the headline's words is telling the story further, and how
    much else it says is then irrelevant. Past that gate, the score is the
    share of the sentence's own words that the headline already had — one for a
    verbatim copy, lower the more the sentence adds.
    """
    said = stems(headline)
    again = stems(sentence)
    if not said or not again:
        return 0.0
    shared = said & again
    if len(shared) / len(said) < HEADLINE_COVERED:
        return 0.0
    return len(shared) / len(again)


def find_sentence_end(text: str) -> int | None:
    """Index just past the first sentence, or None if there is only one."""
    for index, char in enumerate(text):
        if char == "\n":
            return index
        if char not in _SENTENCE_END_CHARS:
            continue
        following = text[index + 1 : index + 2]
        # End of text is not a split: there is no second sentence.
        if following and following.isspace():
            return index + 1
    return None
