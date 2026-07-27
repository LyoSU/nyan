from typing import Any

from fasttext import load_model as ft_load_model  # type: ignore


# The rest of the pipeline speaks two-letter codes: `ranker.py` keeps a cluster
# only if some document in it is `uk`, and `title.py` prefers a `uk` document
# for the headline.
#
# fastText's lid.176, the model in use, already emits those codes and passes
# through this table untouched. The table is here so the model can be swapped
# without touching anything downstream: the newer fastText language models
# (OpenLID, GlotLID) label in FLORES style — `ukr_Cyrl`, an ISO 639-3 code plus
# a script — and this translates them on the way out. Only the languages a
# Ukrainian news feed actually meets are listed; anything else keeps its
# three-letter code, which is enough for the one thing the rest of the code
# does with a foreign language: notice that it is not `uk`.
FLORES_TO_ISO_639_1 = {
    "ukr_Cyrl": "uk",
    "rus_Cyrl": "ru",
    "eng_Latn": "en",
    "bel_Cyrl": "be",
    "pol_Latn": "pl",
    "deu_Latn": "de",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "ita_Latn": "it",
    "ron_Latn": "ro",
    "hun_Latn": "hu",
    "ces_Latn": "cs",
    "slk_Latn": "sk",
    "bul_Cyrl": "bg",
    "srp_Cyrl": "sr",
    "hrv_Latn": "hr",
    "slv_Latn": "sl",
    "mkd_Cyrl": "mk",
    "lit_Latn": "lt",
    "lvs_Latn": "lv",
    "est_Latn": "et",
    "ell_Grek": "el",
    "tur_Latn": "tr",
    "kat_Geor": "ka",
    "hye_Armn": "hy",
    "azj_Latn": "az",
    "kaz_Cyrl": "kk",
    "heb_Hebr": "he",
    "arb_Arab": "ar",
    "pes_Arab": "fa",
    "zho_Hans": "zh",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "nld_Latn": "nl",
    "por_Latn": "pt",
    "swe_Latn": "sv",
    "dan_Latn": "da",
    "fin_Latn": "fi",
    "nob_Latn": "no",
}

LABEL_PREFIX = "__label__"

# Enough words to judge by, not so many that a long post costs more than it
# has to: language is settled by the first sentence or two.
MAX_TOKENS = 50

# Letters that exist in only one of the two languages this feed has to tell
# apart. Ukrainian writes і, ї, є, ґ, which Russian does not have at all;
# Russian writes ы, э, ъ, which Ukrainian does not.
UKRAINIAN_ONLY_LETTERS = frozenset("іїєґІЇЄҐ")
RUSSIAN_ONLY_LETTERS = frozenset("ыэъЫЭЪ")


def count_script_markers(text: str) -> tuple[int, int]:
    """How many Ukrainian-only and Russian-only letters the text contains."""
    ukrainian = sum(char in UKRAINIAN_ONLY_LETTERS for char in text)
    russian = sum(char in RUSSIAN_ONLY_LETTERS for char in text)
    return ukrainian, russian


def resolve_uk_ru(
    language: str,
    probability: float,
    text: str,
    min_probability: float,
) -> str | None:
    """Decide the final label for a post the model read as Ukrainian or Russian.

    The two mistakes do not cost the same. A Ukrainian post filed as `ru` can
    cost the feed a whole story, because `ranker.py` keeps a cluster only when
    some document in it is `uk`. A Russian post filed as `uk` costs at most an
    off-language headline on a story that still gets published.

    `count_script_markers` supplies evidence the model does not use directly:
    і/ї/є/ґ occur only in Ukrainian and ы/э/ъ only in Russian, so where they
    appear they are close to decisive — and they are absent from short posts,
    where the model's own probability is all there is to go on.

    TODO(you): implement the policy. Worth deciding:
      - how many markers count as evidence — one stray letter, or several?
      - may markers override a confident model, or only a shaky one?
      - what to return below `min_probability` with no markers either way:
        `None` (honest, but the cluster loses its Ukrainian document) or a
        fallback to `uk` (keeps the story, at the price of mislabelling
        Russian posts)
    """
    if probability < min_probability:
        return None
    return language


class LanguageDetector:
    def __init__(self, config: dict[str, Any] | str):
        # A bare path is how this was configured before the switch to
        # OpenLID-v3, and deployments still carry that form in their configs.
        if isinstance(config, str):
            config = {"path": config}
        self.model = ft_load_model(config["path"])
        self.min_probability: float = config.get("min_probability", 0.0)
        self.lower: bool = config.get("lower", False)
        self.max_tokens: int = config.get("max_tokens", MAX_TOKENS)

    def __call__(self, text: str) -> tuple[str | None, float]:
        text = text.replace("\xa0", " ").strip()
        text = " ".join(text.split())

        if self.lower:
            text = text.lower()

        sample = " ".join(text.split()[: self.max_tokens])
        if not sample:
            return None, 0.0

        (raw_label,), (probability,) = self.model.predict(sample, k=1)
        flores_label = raw_label[len(LABEL_PREFIX):]
        # Unlisted languages keep their ISO 639-3 code, minus the script tag.
        language: str | None = FLORES_TO_ISO_639_1.get(
            flores_label, flores_label.split("_")[0]
        )

        if language in ("uk", "ru"):
            assert language is not None
            language = resolve_uk_ru(
                language, probability, text, self.min_probability
            )
        elif probability < self.min_probability:
            language = None

        return language, probability
