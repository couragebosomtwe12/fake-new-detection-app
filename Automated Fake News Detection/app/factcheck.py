"""External fact-check layer: queries the Google Fact Check Tools API for
professional fact-checks matching the submitted article's claim.

This is a verification signal independent of the statistical classifier —
it looks up whether fact-checking organisations (Snopes, PolitiFact, AFP,
Dubawa, etc.) have already published a review of a similar claim. Results
are displayed for context; they do not silently override the model's
verdict, because coverage is uneven (strong on viral international claims,
thinner on local news).

Requires a free Google Cloud API key in the GOOGLE_FACTCHECK_API_KEY
environment variable. If unset or the request fails, the layer is skipped
silently — the classification result is unaffected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

API_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY_ENV = "GOOGLE_FACTCHECK_API_KEY"
MAX_RESULTS = 5
TIMEOUT_SECONDS = 8


@dataclass
class FactCheckMatch:
    claim: str
    claimant: str | None
    publisher: str
    rating: str  # e.g. "False", "Mostly true", "Mixture"
    review_url: str
    review_date: str | None

    @property
    def tone(self) -> str:
        """'false' | 'true' | 'mixed' — used for badge colouring."""
        if _rating_is_false(self.rating):
            return "false"
        if _rating_is_true(self.rating):
            return "true"
        return "mixed"


def _rating_is_false(rating: str) -> bool:
    """Heuristic: does this rating indicate the claim is false/misleading?"""
    r = rating.lower()
    return any(
        k in r
        for k in ("false", "pants on fire", "fake", "incorrect", "misleading",
                  "scam", "hoax", "unproven", "mostly false", "wrong")
    )


def _rating_is_true(rating: str) -> bool:
    r = rating.lower()
    return any(
        k in r
        for k in ("true", "correct", "accurate", "mostly true", "verified")
    ) and "not true" not in r and "untrue" not in r


def search_fact_checks(query: str, max_results: int = MAX_RESULTS) -> list[FactCheckMatch]:
    """Search published fact-checks matching a claim query.

    Returns [] when no API key is configured, the request fails, or no
    fact-checks match — the feature degrades silently."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key or not query.strip():
        return []

    import requests

    try:
        resp = requests.get(
            API_ENDPOINT,
            params={
                "query": query.strip()[:500],
                "languageCode": "en",
                "pageSize": max_results,
                "key": api_key,
            },
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return []
        claims = resp.json().get("claims", [])
    except Exception:
        return []

    matches: list[FactCheckMatch] = []
    for claim in claims:
        for review in claim.get("claimReview", []):
            pub = review.get("publisher", {})
            matches.append(
                FactCheckMatch(
                    claim=claim.get("text", ""),
                    claimant=claim.get("claimant"),
                    publisher=pub.get("name", "Unknown publisher"),
                    rating=review.get("textualRating", ""),
                    review_url=review.get("url", ""),
                    review_date=(review.get("reviewDate") or "")[:10] or None,
                )
            )
        if len(matches) >= max_results:
            break
    return matches[:max_results]


def fact_check_verdict(matches: list[FactCheckMatch]) -> str | None:
    """Summarise matched reviews: 'false' if any review rates the claim
    false, 'true' if any rates it true and none false, else None."""
    if not matches:
        return None
    ratings = [m.rating for m in matches]
    if any(_rating_is_false(r) for r in ratings):
        return "false"
    if any(_rating_is_true(r) for r in ratings):
        return "true"
    return "mixed"


def build_query(headline: str | None, body: str) -> str:
    """Best claim text for lookup: the headline if present (headlines carry
    the central claim), otherwise the opening of the body."""
    if headline and headline.strip():
        return headline.strip()
    return " ".join(body.split()[:40])
