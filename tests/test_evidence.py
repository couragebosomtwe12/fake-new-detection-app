from app.evidence import build_evidence_assessment, detect_correction
from app.factcheck import FactCheckMatch


def match(rating, publisher="GhanaFact"):
    return FactCheckMatch(
        claim="The central claim under review",
        claimant=None,
        publisher=publisher,
        rating=rating,
        review_url="https://example.com/review",
        review_date="2026-01-01",
    )


def assess(matches=None, **overrides):
    values = {
        "headline": "A sufficiently descriptive news headline",
        "body": "This article reports a central public claim with enough surrounding text for assessment.",
        "source_trusted": False,
        "internal_source": False,
        "fact_check_enabled": True,
        "fact_check_matches": matches or [],
    }
    values.update(overrides)
    return build_evidence_assessment(**values)


def test_detect_correction_from_headline_and_opening_notice():
    assert detect_correction("Correction: Earlier report amended", "Article body") == "correction"
    assert detect_correction(None, "Editor's note: this report contained an error.") == "correction"
    assert detect_correction("Retraction and apology", "Article body") == "retraction"


def test_false_fact_check_outweighs_trusted_source_reputation():
    result = assess([match("False")], source_trusted=True)
    assert result.source_reputation == "TRUSTED"
    assert result.final_label == "LIKELY FAKE"
    assert result.final_tone == "fake"
    assert "GhanaFact" in result.reason


def test_true_fact_check_supports_likely_real_assessment():
    result = assess([match("True", "Dubawa Ghana")])
    assert result.final_label == "LIKELY REAL"
    assert result.final_tone == "real"
    assert "Dubawa Ghana" in result.reason


def test_mixed_fact_check_produces_disputed_assessment():
    result = assess([match("Mixture")])
    assert result.final_label == "DISPUTED"
    assert result.final_tone == "mixed"


def test_contradictory_fact_checks_produce_disputed_assessment():
    result = assess([match("True"), match("False")])
    assert result.final_label == "DISPUTED"
    assert result.final_tone == "mixed"


def test_correction_notice_takes_precedence_over_fact_check():
    result = assess(
        [match("True")],
        headline="Correction: Publisher amends inaccurate report",
    )
    assert result.correction_status == "CORRECTION DETECTED"
    assert result.final_label == "CORRECTED"


def test_trusted_source_without_independent_evidence_remains_unverified():
    result = assess(source_trusted=True)
    assert result.source_reputation == "TRUSTED"
    assert result.final_label == "UNVERIFIED"
    assert "source reputation does not verify" in result.reason


def test_missing_api_key_is_distinct_from_no_match():
    unavailable = assess(fact_check_enabled=False)
    no_match = assess(fact_check_enabled=True)
    assert unavailable.fact_check_status == "NOT CONFIGURED"
    assert no_match.fact_check_status == "NO MATCH FOUND"


def test_internal_source_is_reported_without_external_fact_check_claim():
    result = assess(internal_source=True, source_trusted=True)
    assert result.source_reputation == "INTERNAL"
    assert result.fact_check_status == "INTERNAL SOURCE"
    assert result.final_label == "UNVERIFIED"


def test_corroboration_is_not_claimed_without_a_provider():
    result = assess([match("True")])
    assert result.corroboration_status == "NOT CHECKED"
