from app.corroboration import CorroboratingReport, CorroborationResult
from app.evidence import build_evidence_assessment, detect_correction
from app.factcheck import FactCheckMatch, FactCheckResult


def match(rating, publisher="GhanaFact"):
    return FactCheckMatch(
        claim="The central claim under review",
        claimant=None,
        publisher=publisher,
        rating=rating,
        review_url="https://example.com/review",
        review_date="2026-01-01",
    )


def report(publisher, stance="supports"):
    domain = publisher.lower().replace(" ", "") + ".com"
    return CorroboratingReport(
        title="Independent report about the same central claim",
        publisher=publisher,
        url=f"https://{domain}/report",
        domain=domain,
        snippet="The report describes the same event.",
        stance=stance,
        source_kind="reputable_news",
    )


def assess(matches=None, reports=None, **overrides):
    values = {
        "headline": "A sufficiently descriptive news headline",
        "body": "This article reports a central public claim with enough surrounding text for assessment.",
        "source_trusted": False,
        "internal_source": False,
        "fact_check_result": FactCheckResult(status="success", matches=matches or []),
        "corroboration_result": CorroborationResult(status="success", reports=reports or []),
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


def test_correction_notice_takes_precedence_over_other_evidence():
    result = assess(
        [match("True")],
        [report("Reuters"), report("BBC News")],
        headline="Correction: Publisher amends inaccurate report",
    )
    assert result.correction_status == "CORRECTION DETECTED"
    assert result.final_label == "CORRECTED"


def test_two_independent_supporting_reports_produce_likely_real():
    result = assess(reports=[report("Reuters"), report("BBC News")])
    assert result.corroboration_status == "FOUND — 2 SUPPORTING SOURCES"
    assert result.final_label == "LIKELY REAL"
    assert "Reuters and BBC News" in result.reason


def test_one_supporting_report_is_not_enough_for_final_confirmation():
    result = assess(reports=[report("Reuters")])
    assert result.final_label == "UNVERIFIED"
    assert "one independent source is insufficient" in result.reason


def test_conflicting_report_produces_disputed_assessment():
    result = assess(reports=[report("Reuters"), report("BBC News", "conflicts")])
    assert result.final_label == "DISPUTED"
    assert result.corroboration_status == "CONFLICTING — 1 SUPPORTING, 1 OPPOSING"


def test_trusted_source_without_independent_evidence_remains_unverified():
    result = assess(source_trusted=True)
    assert result.source_reputation == "TRUSTED"
    assert result.final_label == "UNVERIFIED"
    assert "source reputation does not verify" in result.reason


def test_fact_check_statuses_are_distinct():
    unavailable = assess(
        fact_check_result=FactCheckResult(status="not_configured", matches=[])
    )
    failed = assess(fact_check_result=FactCheckResult(status="failed", matches=[]))
    no_match = assess()
    assert unavailable.fact_check_status == "NOT CONFIGURED"
    assert failed.fact_check_status == "LOOKUP FAILED"
    assert no_match.fact_check_status == "NO MATCH FOUND"


def test_corroboration_statuses_are_distinct():
    unavailable = assess(
        corroboration_result=CorroborationResult(status="not_configured", reports=[])
    )
    failed = assess(corroboration_result=CorroborationResult(status="failed", reports=[]))
    no_match = assess()
    assert unavailable.corroboration_status == "NOT CONFIGURED"
    assert failed.corroboration_status == "LOOKUP FAILED"
    assert no_match.corroboration_status == "NO SUPPORTING REPORTS"


def test_internal_source_is_reported_without_external_fact_check_claim():
    result = assess(internal_source=True, source_trusted=True)
    assert result.source_reputation == "INTERNAL"
    assert result.fact_check_status == "INTERNAL SOURCE"
    assert result.final_label == "UNVERIFIED"
