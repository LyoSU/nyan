import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any
from collections import Counter as CounterT
from urllib.parse import unquote, urlparse

from tqdm import tqdm

from nyan.channels import Channels
from nyan.document import Document
from nyan.fasttext_clf import FasttextClassifier
from nyan.classifier import ClassifierHead
from nyan.embedder import Embedder
from nyan.text import TextProcessor
from nyan.image import ImageProcessor
from nyan.rubrics import RubricDetector
from nyan.tokenizer import Tokenizer
from nyan.util import normalize_channel_id


# Defaults for repeated-line detection. A channel needs a few posts before its
# habits are visible, and a line has to appear in a substantial share of them
# before it counts as a template rather than a recurring topic.
DEFAULT_BOILERPLATE_MIN_DOCS = 5
DEFAULT_BOILERPLATE_MIN_RATIO = 0.5


def boilerplate_key(line: str) -> str:
    """Comparison key for a line, tolerating whitespace and case drift."""
    return " ".join(line.lower().split())


class Annotator:
    def __init__(self, config_path: str, channels: Channels):
        assert isinstance(channels, Channels), "Wrong channels argument in Annotator"
        with open(config_path) as r:
            config = json.load(r)

        self.embedder = Embedder(**config["embedder"])
        self.text_processor = TextProcessor(config["text_processor"])
        self.tokenizer = Tokenizer(**config.get("tokenizer", {}))

        self.image_processor = None
        if "image_processor" in config:
            self.image_processor = ImageProcessor(config["image_processor"])

        self.lang_detector = None
        if "lang_detector" in config:
            self.lang_detector = FasttextClassifier(config["lang_detector"])

        self.cat_detector = None
        if "cat_detector" in config:
            self.cat_detector = ClassifierHead(config["cat_detector"])

        self.rubric_detector = None
        if "rubric_detector" in config:
            self.rubric_detector = RubricDetector(config["rubric_detector"])

        boilerplate_config: dict[str, Any] = config.get("boilerplate", {})
        self.boilerplate_min_docs = boilerplate_config.get(
            "min_docs", DEFAULT_BOILERPLATE_MIN_DOCS
        )
        self.boilerplate_min_ratio = boilerplate_config.get(
            "min_ratio", DEFAULT_BOILERPLATE_MIN_RATIO
        )

        self.channels = channels

    def __call__(self, docs: list[Document]) -> list[Document]:
        logging.info("Annotating %d documents", len(docs))
        pre_pipeline = (
            self.process_channels_info,
            self.clean_text,
            self.tokenize,
            self.normalize_links,
            self.has_obscene,
            self.predict_language,
            self.process_images,
        )
        processed_docs = list()
        for doc in tqdm(docs, desc="Annotator pre-embeddings pipeline"):
            for step in pre_pipeline:
                doc = step(doc)
            processed_docs.append(doc)
        docs = processed_docs

        if self.embedder is not None:
            docs = self.calc_embeddings(docs)
            logging.info("Embeddings calculated for %d documents", len(docs))

        # The rubric check runs after the model, and overrides it: the model has
        # no way to know that a funeral notice is routine rather than news, so
        # whatever category it picked for one is wrong.
        post_pipeline = (self.predict_category, self.detect_rubrics)
        processed_docs = list()
        for doc in tqdm(docs, desc="Annotator post-embeddings pipeline"):
            for step in post_pipeline:
                doc = step(doc)
            processed_docs.append(doc)
        logging.info("Annotated %d documents", len(processed_docs))
        return processed_docs

    def postprocess(self, docs: list[Document]) -> list[Document]:
        docs = self.strip_boilerplate(docs)
        return [doc for doc in docs if not doc.is_discarded()]

    def find_boilerplate(self, docs: list[Document]) -> dict[str, set[str]]:
        """Lines each channel repeats across most of its posts.

        A subscribe footer says nothing about the story and is identical
        everywhere, so its own repetition gives it away. Deriving this from the
        documents needs no per-channel configuration to keep up to date, which
        a hand-written list of substrings across a hundred channels would.
        """
        line_counts: dict[str, CounterT[str]] = defaultdict(Counter)
        doc_counts: CounterT[str] = Counter()
        for doc in docs:
            if not doc.patched_text:
                continue
            doc_counts[doc.channel_id] += 1
            # Counted once per document: a line repeated inside a single post
            # says nothing about the channel's habits.
            for key in {boilerplate_key(line) for line in doc.patched_text.split("\n")}:
                line_counts[doc.channel_id][key] += 1

        boilerplate: dict[str, set[str]] = dict()
        for channel_id, counts in line_counts.items():
            doc_count = doc_counts[channel_id]
            if doc_count < self.boilerplate_min_docs:
                continue
            repeated = {
                key
                for key, count in counts.items()
                if key and count / doc_count >= self.boilerplate_min_ratio
            }
            if repeated:
                boilerplate[channel_id] = repeated
        return boilerplate

    def strip_boilerplate(self, docs: list[Document]) -> list[Document]:
        boilerplate = self.find_boilerplate(docs)
        if not boilerplate:
            return docs

        removed_lines = 0
        for doc in docs:
            channel_boilerplate = boilerplate.get(doc.channel_id)
            if not channel_boilerplate or not doc.patched_text:
                continue
            lines = doc.patched_text.split("\n")
            kept = [
                line for line in lines if boilerplate_key(line) not in channel_boilerplate
            ]
            if len(kept) == len(lines):
                continue
            removed_lines += len(lines) - len(kept)
            # A post that was nothing but boilerplate becomes empty here, and
            # is_discarded() drops it, which is the correct outcome.
            doc.patched_text = "\n".join(kept).strip()

        logging.info(
            "Removed %d repeated lines from %d channels",
            removed_lines,
            len(boilerplate),
        )
        return docs

    def process_channels_info(self, doc: Document) -> Document:
        channel_id = normalize_channel_id(doc.channel_id)
        doc.channel_id = channel_id
        if channel_id not in self.channels:
            return doc

        channel_info = self.channels[channel_id]
        doc.groups = channel_info.groups
        doc.badges = list(channel_info.badges)
        doc.monitor_only = channel_info.monitor_only
        doc.issue = channel_info.issue

        channel_alias = channel_info.alias
        if channel_alias:
            doc.channel_title = channel_alias
        return doc

    def clean_text(self, doc: Document) -> Document:
        if not doc.text:
            return doc
        doc.patched_text = self.text_processor(doc.text)
        return doc

    def tokenize(self, doc: Document) -> Document:
        if not doc.patched_text:
            return doc
        tokens = self.tokenizer(doc.patched_text)
        tokens = [
            "{}_{}".format(t.lemma.lower().replace("_", ""), t.pos) for t in tokens
        ]
        doc.tokens = " ".join(tokens)
        return doc

    def normalize_links(self, doc: Document) -> Document:
        def has_cyrillic(text: str) -> bool:
            return bool(re.search("[а-яА-Я]", text))

        fixed_links = []
        for link in doc.links:
            decoded_link = unquote(link)
            parsed_link = urlparse(decoded_link)
            host = parsed_link.netloc
            if not host:
                continue
            if has_cyrillic(host) and host.split(".")[-1] != "рф":
                continue
            fixed_links.append(decoded_link)
        doc.links = fixed_links
        return doc

    def has_obscene(self, doc: Document) -> Document:
        if not doc.patched_text:
            return doc
        doc.has_obscene = self.text_processor.has_obscene(doc.patched_text)
        return doc

    def calc_embeddings(self, docs: list[Document]) -> list[Document]:
        ready_docs = [d for d in docs if d.patched_text is not None]
        texts = [d.patched_text for d in ready_docs if d.patched_text is not None]
        embeddings = self.embedder(texts)
        for d, embedding in zip(ready_docs, embeddings, strict=True):
            d.embedding = embedding.numpy().tolist()
        return ready_docs

    def predict_language(self, doc: Document) -> Document:
        if not self.lang_detector:
            return doc
        if not doc.patched_text:
            return doc
        language, _probability = self.lang_detector(doc.patched_text)
        doc.language = language
        return doc

    def predict_category(self, doc: Document) -> Document:
        if not self.cat_detector:
            return doc
        if not doc.patched_text:
            return doc
        if not doc.embedding:
            return doc
        category, scores = self.cat_detector(doc.embedding, doc.embedding_key)
        doc.category_scores = scores
        doc.category = category
        return doc

    def detect_rubrics(self, doc: Document) -> Document:
        if not self.rubric_detector or not doc.patched_text:
            return doc
        if self.rubric_detector(doc.patched_text):
            doc.category = "not_news"
        return doc

    def process_images(self, doc: Document) -> Document:
        if not self.image_processor:
            return doc
        doc.embedded_images = self.image_processor(list(doc.images))
        return doc
