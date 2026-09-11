"""Submission validation rules (Section 3.7.2), implemented exactly.

Every failure raises a ValidationError carrying a specific, actionable
message — the interface must never show a generic "invalid input" error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from app.preprocessing import strip_html_tags

MIN_BODY_WORDS = 25
MIN_HEADLINE_WORDS = 5
MAX_SUBMISSION_TOKENS = 512
MIN_LANGUAGE_CONFIDENCE = 0.90
URL_TIMEOUT_SECONDS = 10

# Trusted news sources whitelist - override model predictions for these domains
TRUSTED_SOURCES = {
    "aljazeera.com",
    "bbc.com", 
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "npr.org",
    "ft.com",
    "economist.com",
    "bloomberg.com",
    "wsj.com",
    "latimes.com",
    "chicagotribune.com",
    "france24.com",
    "dw.com",
    "rferl.org",
    "voanews.com"
}


class ValidationError(Exception):
    """Raised with a specific rejection code and user-facing message."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class ValidatedSubmission:
    headline: str
    body: str
    source_url: str | None


def validate_not_empty(body: str) -> None:
    """Empty after trim -> reject."""
    if not body or not body.strip():
        raise ValidationError("empty_submission", "Submitted text is empty. Please paste a news article.")


def validate_word_counts(
    headline: str | None, body: str, check_headline_length: bool = True
) -> None:
    """< 25 words body / < 5 headline -> reject.

    Table 3.3's "5 words for headline mode" describes a headline submitted
    *as the classified content itself* (Section 3.7.1's "a headline for
    immediate analysis" input modality) — not a supplementary label
    displayed alongside body text, and not a title auto-extracted from a
    URL, which the user has no control over. check_headline_length=False
    skips that check for the latter cases (see validate_submission).
    """
    body_words = len(body.split())
    if body_words < MIN_BODY_WORDS:
        raise ValidationError(
            "body_too_short",
            f"Article body must be at least {MIN_BODY_WORDS} words (got {body_words}).",
        )
    if check_headline_length and headline and headline.strip():
        headline_words = len(headline.split())
        if headline_words < MIN_HEADLINE_WORDS:
            raise ValidationError(
                "headline_too_short",
                f"Headline must be at least {MIN_HEADLINE_WORDS} words (got {headline_words}).",
            )


def validate_language(text: str) -> None:
    """langdetect must return English, confidence > 0.90."""
    from langdetect import LangDetectException, detect_langs

    try:
        candidates = detect_langs(text)
    except LangDetectException as exc:
        raise ValidationError(
            "language_undetermined", "Could not determine the language of the submitted text."
        ) from exc

    top = candidates[0]
    if top.lang != "en" or top.prob <= MIN_LANGUAGE_CONFIDENCE:
        raise ValidationError(
            "not_english",
            f"Submitted text does not appear to be English (detected '{top.lang}' at "
            f"{top.prob:.0%} confidence; English confidence must exceed "
            f"{MIN_LANGUAGE_CONFIDENCE:.0%}).",
        )


def validate_url_scheme(url: str) -> None:
    """URL: http/https only."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValidationError("invalid_url_scheme", "Only http:// and https:// URLs are supported.")


def fetch_article_from_url(url: str, timeout: int = URL_TIMEOUT_SECONDS) -> tuple[str, str]:
    """Fetch and extract (title, body) from a URL via newspaper3k, 10s timeout."""
    validate_url_scheme(url)
    from newspaper import Article
    from newspaper.article import ArticleException

    article = Article(url, request_timeout=timeout)
    try:
        article.download()
        article.parse()
    except ArticleException as exc:
        raise ValidationError(
            "url_fetch_failed", f"Could not retrieve or parse the article at this URL: {exc}"
        ) from exc

    if not article.text or not article.text.strip():
        raise ValidationError("url_no_content", "No article text could be extracted from this URL.")
    return article.title or "", article.text


def truncate_for_bert(text: str, max_tokens: int = MAX_SUBMISSION_TOKENS) -> str:
    """> 512 tokens: truncate. Applied only on the BERT path — classical
    models keep the full text (their vectoriser handles arbitrary length)."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def apply_length_rule(text: str, family: Literal["classical", "transformer"]) -> str:
    """> 512 tokens: truncate for BERT, keep full for classical."""
    if family == "transformer":
        return truncate_for_bert(text)
    return text


def is_trusted_source(source_url: str | None) -> bool:
    """Check if the URL is from a trusted news source domain."""
    if not source_url:
        return False
    try:
        domain = urlparse(source_url).netloc.lower()
        # Remove www. prefix if present
        domain = domain.replace("www.", "")
        return domain in TRUSTED_SOURCES
    except Exception:
        return False


def validate_submission(
    headline: str | None,
    body: str,
    source_url: str | None = None,
    check_headline_length: bool = True,
) -> ValidatedSubmission:
    """Run all Section 3.7.2 rules against a pasted-text submission, in order:
    strip tags, reject if empty, reject if too short, reject if not English.

    check_headline_length should be False when headline is a title
    auto-extracted from a URL rather than typed by the user (see
    validate_word_counts)."""
    body = strip_html_tags(body)
    headline = strip_html_tags(headline) if headline else ""

    validate_not_empty(body)
    validate_word_counts(headline or None, body, check_headline_length=check_headline_length)
    validate_language(body)

    return ValidatedSubmission(headline=headline.strip(), body=body.strip(), source_url=source_url)
