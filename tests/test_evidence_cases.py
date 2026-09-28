import json
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


FIXTURE_PATH = Path(__file__).with_name("evidence_cases.json")
VALID_ASSESSMENTS = {
    "likely_real",
    "likely_fake",
    "misleading",
    "partly_true",
    "corrected",
    "retracted",
}
RATING_TO_ASSESSMENT = {
    "true": "likely_real",
    "false": "likely_fake",
    "misleading": "misleading",
    "partly_true": "partly_true",
    "corrected": "corrected",
    "retracted": "retracted",
}
REQUIRED_CASE_FIELDS = {
    "id",
    "country",
    "topic",
    "claim",
    "claimant",
    "review_publisher",
    "review_date",
    "published_rating",
    "expected_assessment",
    "review_type",
    "evidence_basis",
    "review_url",
    "correction_detected",
    "correction_url",
    "public_reason",
}


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_evidence_fixture_has_expected_schema():
    fixture = load_fixture()
    assert fixture["schema_version"] == 1
    assert date.fromisoformat(fixture["collected_on"])
    assert set(fixture["assessment_definitions"]) == VALID_ASSESSMENTS
    assert isinstance(fixture["cases"], list)
    assert fixture["cases"]


def test_case_ids_are_unique_and_fields_are_complete():
    cases = load_fixture()["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert set(case) == REQUIRED_CASE_FIELDS
        assert case["country"] == "Ghana"
        assert case["claim"].strip()
        assert case["claimant"].strip()
        assert case["evidence_basis"]


def test_ratings_map_to_expected_assessments():
    for case in load_fixture()["cases"]:
        assert case["published_rating"] in RATING_TO_ASSESSMENT
        assert case["expected_assessment"] == RATING_TO_ASSESSMENT[case["published_rating"]]


def test_review_dates_and_urls_are_traceable():
    fixture = load_fixture()
    collected_on = date.fromisoformat(fixture["collected_on"])
    for case in fixture["cases"]:
        assert date.fromisoformat(case["review_date"]) <= collected_on
        parsed = urlparse(case["review_url"])
        assert parsed.scheme == "https"
        assert parsed.netloc
        assert parsed.path not in {"", "/"}


def test_corrections_include_traceable_urls():
    for case in load_fixture()["cases"]:
        if case["correction_detected"]:
            assert case["correction_url"]
            assert urlparse(case["correction_url"]).scheme == "https"
        else:
            assert case["correction_url"] is None
        if case["expected_assessment"] in {"corrected", "retracted"}:
            assert case["correction_detected"] is True


def test_public_reasons_describe_evidence_without_model_jargon():
    prohibited_terms = {"model", "svm", "algorithm", "training data"}
    for case in load_fixture()["cases"]:
        reason = case["public_reason"].strip()
        assert len(reason) >= 60
        assert reason.endswith(".")
        assert not any(term in reason.lower() for term in prohibited_terms)


def test_starter_collection_is_balanced_across_assessment_types():
    cases = load_fixture()["cases"]
    counts = Counter(case["expected_assessment"] for case in cases)
    assert counts["likely_real"] >= 4
    assert counts["likely_fake"] >= 4
    assert counts["misleading"] >= 2
    assert counts["partly_true"] >= 2
    assert counts["corrected"] + counts["retracted"] >= 4
    assert len({case["review_publisher"] for case in cases}) >= 4
