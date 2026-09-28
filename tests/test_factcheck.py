import requests

from app.factcheck import fact_check_verdict, search_fact_checks


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


def test_lookup_reports_not_configured_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    result = search_fact_checks("A central claim")
    assert result.status == "not_configured"
    assert result.matches == []


def test_lookup_reports_http_failure(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(403))
    result = search_fact_checks("A central claim")
    assert result.status == "failed"


def test_lookup_distinguishes_successful_empty_result(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload={"claims": []}))
    result = search_fact_checks("A central claim")
    assert result.status == "success"
    assert result.matches == []


def test_lookup_parses_published_review(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    payload = {
        "claims": [
            {
                "text": "The reviewed claim",
                "claimant": "A claimant",
                "claimReview": [
                    {
                        "publisher": {"name": "GhanaFact"},
                        "textualRating": "False",
                        "url": "https://ghanafact.com/review",
                        "reviewDate": "2026-01-02T10:00:00Z",
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload=payload))
    result = search_fact_checks("The reviewed claim")
    assert result.status == "success"
    assert len(result.matches) == 1
    assert result.matches[0].publisher == "GhanaFact"
    assert result.matches[0].review_date == "2026-01-02"
    assert fact_check_verdict(result.matches) == "false"
