"""Shared text preprocessing pipeline (Section 3.11).

This is the single implementation used by both training (run_pipeline.py)
and inference (the FastAPI app). Train/inference preprocessing drift is the
most common silent cause of degraded production performance — do not fork
this logic for one call site or the other.

Preprocessing differs by model family:

    Stage               Classical (NB, SVM, BiLSTM)   Transformer (BERT)
    HTML/URL/whitespace  strip                         strip
    Lowercase            yes                           yes (uncased)
    Punctuation          remove                        keep
    Stop words           remove                        keep
    Lemmatise            yes (spaCy)                   no
    Tokenise             word-level (downstream)       WordPiece (downstream)
    Length               n/a                           256 tokens (downstream)

Punctuation and stop words are kept for BERT because sensationalist
punctuation and function-word patterns carry signal. Tokenisation itself
(TF-IDF vectorisation / vocab lookup for classical, WordPiece + the 256-token
cap for BERT) is left to the caller, since it is model-specific.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

ModelFamily = Literal["classical", "transformer"]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def strip_html_tags(text: str) -> str:
    """Strip HTML/script tags only. Used at submission time (Section 3.7.2)
    ahead of Jinja2's autoescaping on render."""
    return _HTML_TAG_RE.sub(" ", str(text))


def strip_html_url_whitespace(text: str) -> str:
    """Strip HTML/script tags and URLs, collapse whitespace, trim ends."""
    text = strip_html_tags(text)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _spacy_pipeline():
    """Lazily load spaCy's small English model, tagger+lemmatiser only."""
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def _clean_for_lemmatisation(text: str) -> str:
    text = strip_html_url_whitespace(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _lemmatise_doc(doc) -> str:
    return " ".join(tok.lemma_.strip() for tok in doc if not tok.is_stop and tok.lemma_.strip())


def preprocess_classical(text: str) -> str:
    """Classical pipeline: lowercase, strip punctuation, remove stop words,
    lemmatise. Used for Multinomial Naive Bayes, Linear SVM and the
    Bidirectional LSTM (all word-level models). One document at a time —
    fine for a single inference request, too slow for bulk training use
    (see preprocess_classical_batch)."""
    cleaned = _clean_for_lemmatisation(text)
    if not cleaned:
        return ""
    return _lemmatise_doc(_spacy_pipeline()(cleaned))


def preprocess_classical_batch(texts: list[str], batch_size: int = 200) -> list[str]:
    """Same pipeline as preprocess_classical, batched via spaCy's nlp.pipe()
    for bulk training-time use (run_pipeline.py, thousands of documents) —
    dramatically faster than calling nlp() once per document."""
    cleaned = [_clean_for_lemmatisation(t) for t in texts]
    nlp = _spacy_pipeline()
    return [_lemmatise_doc(doc) for doc in nlp.pipe(cleaned, batch_size=batch_size)]


def preprocess_transformer(text: str) -> str:
    """BERT pipeline: strip HTML/URL/whitespace and lowercase only. Keeps
    punctuation and stop words; WordPiece tokenisation and the 256-token cap
    are applied by the Hugging Face tokenizer at the call site."""
    return strip_html_url_whitespace(text).lower()


def preprocess(text: str, family: ModelFamily) -> str:
    """Dispatch to the pipeline for the given model family."""
    if family == "classical":
        return preprocess_classical(text)
    if family == "transformer":
        return preprocess_transformer(text)
    raise ValueError(f"Unknown model family: {family!r}")
