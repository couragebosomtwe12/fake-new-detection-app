import pytest

from app.validation import (
    MAX_SUBMISSION_TOKENS,
    MIN_BODY_WORDS,
    ValidationError,
    apply_length_rule,
    truncate_for_bert,
    validate_not_empty,
    validate_submission,
    validate_url_scheme,
    validate_word_counts,
)

try:
    import langdetect  # noqa: F401

    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

requires_langdetect = pytest.mark.skipif(
    not _LANGDETECT_AVAILABLE, reason="langdetect not installed"
)

SAMPLE_ENGLISH_BODY = (
    "The local council announced a new plan today to repair roads across the "
    "eastern district after months of community complaints about potholes and "
    "poor drainage systems affecting daily traffic."
)

SAMPLE_FRENCH_BODY = (
    "Le conseil municipal a annonce aujourd'hui un nouveau plan pour reparer "
    "les routes du district est apres des mois de plaintes des habitants au "
    "sujet des nids de poule et des problemes de drainage dans la region."
)


def test_validate_not_empty_rejects_blank():
    with pytest.raises(ValidationError) as excinfo:
        validate_not_empty("   ")
    assert excinfo.value.code == "empty_submission"


def test_validate_not_empty_accepts_nonblank():
    validate_not_empty("hello")  # must not raise


def test_validate_word_counts_rejects_short_body():
    short_body = " ".join(["word"] * 10)
    with pytest.raises(ValidationError) as excinfo:
        validate_word_counts(None, short_body)
    assert excinfo.value.code == "body_too_short"


def test_validate_word_counts_accepts_body_at_minimum():
    body = " ".join(["word"] * MIN_BODY_WORDS)
    validate_word_counts(None, body)  # must not raise


def test_validate_word_counts_rejects_short_headline():
    body = " ".join(["word"] * MIN_BODY_WORDS)
    with pytest.raises(ValidationError) as excinfo:
        validate_word_counts("too short", body)
    assert excinfo.value.code == "headline_too_short"


def test_validate_url_scheme_accepts_http_and_https():
    validate_url_scheme("http://example.com")
    validate_url_scheme("https://example.com")


def test_validate_url_scheme_rejects_other_schemes():
    with pytest.raises(ValidationError) as excinfo:
        validate_url_scheme("ftp://example.com")
    assert excinfo.value.code == "invalid_url_scheme"


def test_truncate_for_bert_keeps_short_text_unchanged():
    text = "one two three"
    assert truncate_for_bert(text, max_tokens=10) == text


def test_truncate_for_bert_truncates_long_text():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    assert truncate_for_bert(text, max_tokens=5) == " ".join(words[:5])


def test_apply_length_rule_truncates_only_for_transformer():
    words = [f"w{i}" for i in range(MAX_SUBMISSION_TOKENS + 10)]
    text = " ".join(words)
    assert apply_length_rule(text, "classical") == text
    assert apply_length_rule(text, "transformer") == " ".join(words[:MAX_SUBMISSION_TOKENS])


@requires_langdetect
def test_validate_submission_strips_html_tags():
    body = "<script>alert(1)</script> " + SAMPLE_ENGLISH_BODY
    result = validate_submission(None, body)
    assert "<script>" not in result.body


@requires_langdetect
def test_validate_submission_rejects_non_english():
    with pytest.raises(ValidationError) as excinfo:
        validate_submission(None, SAMPLE_FRENCH_BODY)
    assert excinfo.value.code == "not_english"


@requires_langdetect
def test_validate_submission_accepts_valid_english_article():
    result = validate_submission("A valid headline with enough words", SAMPLE_ENGLISH_BODY)
    assert result.body.startswith("The local council")
