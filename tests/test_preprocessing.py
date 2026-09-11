import pytest

from app.preprocessing import (
    preprocess,
    preprocess_classical,
    preprocess_transformer,
    strip_html_tags,
    strip_html_url_whitespace,
)

try:
    import spacy

    spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

requires_spacy = pytest.mark.skipif(
    not _SPACY_AVAILABLE, reason="spaCy en_core_web_sm model not installed"
)


def test_strip_html_tags_removes_tags_only():
    assert strip_html_tags("<b>Bold</b> text") == " Bold  text"


def test_strip_html_url_whitespace_removes_tags_urls_and_collapses_whitespace():
    # Strips markup only, not a script tag's inner text — that's inert once
    # Jinja2 escapes it on render, and stripping tag *content* generically
    # would risk eating legitimate article text that happens to look tag-like.
    text = "Check   this <b>bad</b> out http://example.com/page now"
    assert strip_html_url_whitespace(text) == "Check this bad out now"


def test_strip_html_url_whitespace_handles_www_urls():
    assert strip_html_url_whitespace("Visit www.example.com today") == "Visit today"


def test_preprocess_transformer_lowercases_but_keeps_punctuation_and_stopwords():
    text = "BREAKING: This is NOT a drill!!!"
    assert preprocess_transformer(text) == "breaking: this is not a drill!!!"


def test_preprocess_transformer_strips_html_and_urls():
    text = "<p>See http://example.com NOW</p>"
    assert preprocess_transformer(text) == "see now"


@requires_spacy
def test_preprocess_classical_lowercases_lemmatises_and_removes_stopwords_punctuation():
    text = "The Cats were RUNNING quickly, and they jumped!"
    assert preprocess_classical(text) == "cat run quickly jump"


@requires_spacy
def test_preprocess_classical_strips_html_and_urls_first():
    text = "<p>Visit http://example.com</p> and the cats ran."
    assert preprocess_classical(text) == "visit cat run"


@requires_spacy
def test_preprocess_dispatches_by_family():
    text = "Running Fast!"
    assert preprocess(text, "classical") == preprocess_classical(text)
    assert preprocess(text, "transformer") == preprocess_transformer(text)


def test_preprocess_rejects_unknown_family():
    with pytest.raises(ValueError):
        preprocess("text", "unknown")  # type: ignore[arg-type]
