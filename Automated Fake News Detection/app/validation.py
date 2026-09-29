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

# Trusted sources whitelist - override model predictions for these domains.
# International news outlets, Ghanaian news sites (Feedspot top-20 Ghana list),
# and Ghanaian university domains. Subdomains match (e.g. site.gctu.edu.gh).
TRUSTED_SOURCES = {
    # --- International news ---
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
    "voanews.com",
    # --- Ghanaian news (Feedspot top-20 Ghana news websites) ---
    "pulse.com.gh",
    "modernghana.com",
    "myjoyonline.com",
    "citinewsroom.com",
    "ghanaweb.com",
    "ghanaiantimes.com.gh",
    "peacefmonline.com",
    "adomonline.com",
    "graphic.com.gh",
    "theheraldghana.com",
    "accramail.com",
    "jbklutse.com",
    "mfidie.com",
    "ghanasummary.com",
    "gbcghana.com",
    "tv3.com.gh",
    "3news.com",
    "starrfm.com.gh",
    "classfmonline.com",
    "gna.org.gh",
    # --- Ghanaian universities ---
    "gctu.edu.gh",
    "ug.edu.gh",
    "knust.edu.gh",
    "ucc.edu.gh",
    "uds.edu.gh",
    "umat.edu.gh",
    "uew.edu.gh",
    "gimpa.edu.gh",
    "upsa.edu.gh",
    "uhas.edu.gh",
    "uenr.edu.gh",
    "upsamail.edu.gh",
    "ashesi.edu.gh",
    "central.edu.gh",
    "vvu.edu.gh",
    "atu.edu.gh",
    "ktu.edu.gh",
    "ttu.edu.gh",
    "cctu.edu.gh",
    "stu.edu.gh",
}

TRUSTED_SOURCE_NAMES = {
    "aljazeera.com": "Al Jazeera",
    "bbc.com": "BBC News",
    "bbc.co.uk": "BBC News",
    "reuters.com": "Reuters",
    "apnews.com": "AP News",
    "theguardian.com": "The Guardian",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "cnn.com": "CNN",
    "npr.org": "NPR",
    "ft.com": "Financial Times",
    "economist.com": "The Economist",
    "bloomberg.com": "Bloomberg",
    "wsj.com": "The Wall Street Journal",
    "latimes.com": "Los Angeles Times",
    "chicagotribune.com": "Chicago Tribune",
    "france24.com": "France 24",
    "dw.com": "Deutsche Welle",
    "rferl.org": "Radio Free Europe/Radio Liberty",
    "voanews.com": "Voice of America",
    "pulse.com.gh": "Pulse Ghana",
    "modernghana.com": "Modern Ghana",
    "myjoyonline.com": "MyJoyOnline",
    "citinewsroom.com": "Citi Newsroom",
    "ghanaweb.com": "GhanaWeb",
    "ghanaiantimes.com.gh": "Ghanaian Times",
    "peacefmonline.com": "Peace FM Online",
    "adomonline.com": "Adom Online",
    "graphic.com.gh": "Graphic Online",
    "theheraldghana.com": "The Herald Ghana",
    "accramail.com": "Accra Mail",
    "jbklutse.com": "JBKlutse",
    "mfidie.com": "Mfidie",
    "ghanasummary.com": "GhanaSummary",
    "gbcghana.com": "GBC Ghana",
    "tv3.com.gh": "TV3 Ghana",
    "3news.com": "3News",
    "starrfm.com.gh": "Starr FM",
    "classfmonline.com": "Class FM",
    "gna.org.gh": "Ghana News Agency",
    "gctu.edu.gh": "Ghana Communication Technology University",
    "ug.edu.gh": "University of Ghana",
    "knust.edu.gh": "KNUST",
    "ucc.edu.gh": "University of Cape Coast",
    "uds.edu.gh": "University for Development Studies",
    "umat.edu.gh": "University of Mines and Technology",
    "uew.edu.gh": "University of Education, Winneba",
    "gimpa.edu.gh": "GIMPA",
    "upsa.edu.gh": "UPSA",
    "uhas.edu.gh": "University of Health and Allied Sciences",
    "uenr.edu.gh": "University of Energy and Natural Resources",
    "upsamail.edu.gh": "UPSA",
    "ashesi.edu.gh": "Ashesi University",
    "central.edu.gh": "Central University",
    "vvu.edu.gh": "Valley View University",
    "atu.edu.gh": "Accra Technical University",
    "ktu.edu.gh": "Koforidua Technical University",
    "ttu.edu.gh": "Takoradi Technical University",
    "cctu.edu.gh": "Cape Coast Technical University",
    "stu.edu.gh": "Sunyani Technical University",
}


def _match_trusted_domain(source_url: str | None) -> str | None:
    """Return the trusted base domain matching the URL's host (exact or
    subdomain match), or None."""
    if not source_url:
        return None
    try:
        host = urlparse(source_url).netloc.lower()
        host = host.removeprefix("www.")
        for trusted in TRUSTED_SOURCES:
            if host == trusted or host.endswith("." + trusted):
                return trusted
    except Exception:
        pass
    return None


def trusted_source_name(source_url: str | None) -> str | None:
    """Return the display name of the trusted source for a URL, or None."""
    domain = _match_trusted_domain(source_url)
    return TRUSTED_SOURCE_NAMES.get(domain) if domain else None


def is_internal_source(source_url: str | None) -> bool:
    """University sources publish internal institutional news — external
    fact-checkers don't cover them, so the check is marked 'Internal'."""
    domain = _match_trusted_domain(source_url)
    return bool(domain and ".edu." in domain)


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
    """Fetch and extract (title, body) from a URL. Trafilatura is tried
    first — its boilerplate stripping handles liveblogs and JS-heavy pages
    that newspaper3k truncates — with newspaper3k as fallback."""
    validate_url_scheme(url)

    html_source = None
    try:
        import requests
        import trafilatura

        # Download via requests so the fetch is bounded by our timeout —
        # trafilatura's own fetch_url has no enforced cap and can stall on
        # heavy pages (liveblogs) for tens of seconds.
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"},
        )
        if resp.ok:
            html_source = resp.text
            body = trafilatura.extract(
                html_source,
                include_comments=False,
                include_tables=False,
                favor_recall=True,
            )
            if body and body.strip():
                metadata = trafilatura.extract_metadata(html_source)
                title = metadata.title if metadata and metadata.title else ""
                return title, body
    except ImportError:
        pass
    except Exception:
        pass  # fall through to newspaper3k

    from newspaper import Article
    from newspaper.article import ArticleException

    article = Article(url, request_timeout=timeout)
    try:
        if html_source:
            article.set_html(html_source)
        else:
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
    """Check if the URL is from a trusted source domain (exact or subdomain)."""
    return _match_trusted_domain(source_url) is not None


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
