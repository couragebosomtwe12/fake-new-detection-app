import requests

from app.corroboration import search_corroboration


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


def configure(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test-engine")


def test_search_is_not_configured_without_engine_id(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    result = search_corroboration("A valid central claim")
    assert result.status == "not_configured"
    assert result.reports == []


def test_non_success_response_is_reported_as_failure(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(429))
    result = search_corroboration("A valid central claim")
    assert result.status == "failed"


def test_search_filters_original_unknown_and_duplicate_domains(monkeypatch):
    configure(monkeypatch)
    payload = {
        "items": [
            {
                "title": "Eliud Kipchoge record breaking Nike shoes banned",
                "link": "https://cnn.com/original",
                "displayLink": "cnn.com",
                "snippet": "The original report about the shoes.",
            },
            {
                "title": "Eliud Kipchoge record breaking Nike shoes banned under new rules",
                "link": "https://bbc.com/sport/report",
                "displayLink": "bbc.com",
                "snippet": "Nike shoes are covered by the athletics rules.",
            },
            {
                "title": "Another Eliud Kipchoge Nike shoes banned report",
                "link": "https://bbc.co.uk/sport/duplicate",
                "displayLink": "bbc.co.uk",
                "snippet": "A duplicate publisher result.",
            },
            {
                "title": "Eliud Kipchoge record breaking Nike shoes banned",
                "link": "https://unknown-blog.example/post",
                "displayLink": "unknown-blog.example",
                "snippet": "An unknown website repeats the report.",
            },
            {
                "title": "Competition rules cover Eliud Kipchoge record breaking Nike shoes",
                "link": "https://worldathletics.org/news/rules",
                "displayLink": "worldathletics.org",
                "snippet": "Official regulations explain why the shoes were banned.",
            },
        ]
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload=payload))

    result = search_corroboration(
        "Eliud Kipchoge record breaking Nike shoes banned",
        "https://cnn.com/article",
    )

    assert result.status == "success"
    assert [item.domain for item in result.reports] == ["bbc.com", "worldathletics.org"]
    assert result.reports[0].source_kind == "reputable_news"
    assert result.reports[1].source_kind == "official"


def test_explicit_negation_is_marked_as_conflicting(monkeypatch):
    configure(monkeypatch)
    payload = {
        "items": [
            {
                "title": "Eliud Kipchoge Nike shoes were not banned",
                "link": "https://reuters.com/sports/shoes",
                "displayLink": "reuters.com",
                "snippet": "The report says the shoes remain permitted.",
            }
        ]
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload=payload))
    result = search_corroboration("Eliud Kipchoge Nike shoes banned")
    assert len(result.reports) == 1
    assert result.reports[0].stance == "conflicts"
