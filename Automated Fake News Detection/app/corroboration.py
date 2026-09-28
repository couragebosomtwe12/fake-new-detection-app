from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from app.validation import is_trusted_source, trusted_source_name


API_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
API_KEY_ENV = "GOOGLE_CSE_API_KEY"
FALLBACK_API_KEY_ENV = "GOOGLE_FACTCHECK_API_KEY"
SEARCH_ENGINE_ENV = "GOOGLE_CSE_ID"
MAX_SEARCH_RESULTS = 10
MAX_REPORTS = 5
TIMEOUT_SECONDS = 8

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with", "news", "report", "reports", "says", "said",
}
_NEGATIONS = {"no", "not", "never", "false", "denies", "denied", "incorrect", "without"}
_UNCERTAINTY = {"may", "might", "could", "alleged", "reportedly", "possible", "possibly"}
_OFFICIAL_DOMAINS = {
    "worldathletics.org",
    "who.int",
    "imf.org",
    "worldbank.org",
    "un.org",
    "bog.gov.gh",
    "statsghana.gov.gh",
    "ec.gov.gh",
    "ghana.gov.gh",
    "moh.gov.gh",
    "ghs.gov.gh",
}


@dataclass(frozen=True)
class CorroboratingReport:
    title: str
    publisher: str
    url: str
    domain: str
    snippet: str
    stance: Literal["supports", "conflicts", "related"]
    source_kind: Literal["official", "reputable_news"]


@dataclass(frozen=True)
class CorroborationResult:
    status: Literal["not_configured", "success", "failed"]
    reports: list[CorroboratingReport]

    @property
    def supporting(self) -> list[CorroboratingReport]:
        return [report for report in self.reports if report.stance == "supports"]

    @property
    def conflicting(self) -> list[CorroboratingReport]:
        return [report for report in self.reports if report.stance == "conflicts"]


def _host(url: str | None) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower().split(":", 1)[0].removeprefix("www.")


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _is_official(host: str) -> bool:
    return (
        host.endswith(".gov")
        or host.endswith(".gov.gh")
        or host.endswith(".int")
        or any(_matches_domain(host, domain) for domain in _OFFICIAL_DOMAINS)
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", html.unescape(text).lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _is_relevant(query: str, title: str, snippet: str) -> bool:
    query_tokens = _tokens(query)
    result_tokens = _tokens(f"{title} {snippet}")
    if not query_tokens or not result_tokens:
        return False
    shared = query_tokens & result_tokens
    return len(shared) >= 3 or (
        len(shared) >= 2 and len(shared) / min(len(query_tokens), 8) >= 0.35
    )


def _stance(query: str, title: str, snippet: str) -> Literal["supports", "conflicts", "related"]:
    title_tokens = _tokens(title)
    if "?" in title or title_tokens & _UNCERTAINTY:
        return "related"
    query_negative = bool(_tokens(query) & _NEGATIONS)
    result_negative = bool(_tokens(f"{title} {snippet}") & _NEGATIONS)
    return "conflicts" if query_negative != result_negative else "supports"


def _publisher(item: dict, url: str) -> str:
    known_name = trusted_source_name(url)
    if known_name:
        return known_name
    return item.get("displayLink") or _host(url)


def _source_key(url: str) -> str:
    known_name = trusted_source_name(url)
    if known_name:
        return known_name.lower()
    return _host(url)


def search_corroboration(
    query: str,
    original_url: str | None = None,
    max_reports: int = MAX_REPORTS,
) -> CorroborationResult:
    api_key = os.environ.get(API_KEY_ENV) or os.environ.get(FALLBACK_API_KEY_ENV)
    search_engine_id = os.environ.get(SEARCH_ENGINE_ENV)
    if not api_key or not search_engine_id:
        return CorroborationResult(status="not_configured", reports=[])
    if not query.strip():
        return CorroborationResult(status="success", reports=[])

    import requests

    try:
        response = requests.get(
            API_ENDPOINT,
            params={
                "key": api_key,
                "cx": search_engine_id,
                "q": query.strip()[:300],
                "num": MAX_SEARCH_RESULTS,
                "safe": "active",
            },
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return CorroborationResult(status="failed", reports=[])
        items = response.json().get("items", [])
    except (requests.RequestException, ValueError):
        return CorroborationResult(status="failed", reports=[])

    original_source = _source_key(original_url or "")
    seen_sources: set[str] = set()
    seen_titles: set[frozenset[str]] = set()
    reports: list[CorroboratingReport] = []
    for item in items:
        url = item.get("link", "")
        domain = _host(url)
        source_key = _source_key(url)
        if not domain or source_key == original_source or source_key in seen_sources:
            continue
        official = _is_official(domain)
        if not official and not is_trusted_source(url):
            continue
        title = html.unescape(item.get("title", "")).strip()
        snippet = html.unescape(item.get("snippet", "")).strip()
        if not _is_relevant(query, title, snippet):
            continue
        title_fingerprint = frozenset(_tokens(title))
        if title_fingerprint in seen_titles:
            continue
        reports.append(
            CorroboratingReport(
                title=title,
                publisher=_publisher(item, url),
                url=url,
                domain=domain,
                snippet=snippet,
                stance=_stance(query, title, snippet),
                source_kind="official" if official else "reputable_news",
            )
        )
        seen_sources.add(source_key)
        seen_titles.add(title_fingerprint)
        if len(reports) >= max_reports:
            break
    return CorroborationResult(status="success", reports=reports)
