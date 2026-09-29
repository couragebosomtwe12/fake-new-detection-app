from __future__ import annotations

import re
from dataclasses import dataclass

from app.factcheck import FactCheckMatch, FactCheckResult, fact_check_verdict


_RETRACTION_RE = re.compile(
    r"\b(retraction|retracted|withdrawn in full|withdraws? (?:the|this) (?:article|report|claim|publication))\b",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(correction|corrected|editor['’]s note|apology)\b",
    re.IGNORECASE,
)
_OPENING_RETRACTION_RE = re.compile(
    r"(?:^|\n)\s*(?:retraction|we (?:hereby )?retract|this (?:article|report|publication) has been (?:retracted|withdrawn))\b",
    re.IGNORECASE,
)
_OPENING_CORRECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:correction|editor['’]s note|update|we (?:wish to )?(?:correct|apologise|apologize)|this (?:article|report|publication) has been (?:corrected|updated|amended))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    source_reputation: str
    fact_check_status: str
    correction_status: str
    final_label: str
    final_tone: str
    reason: str


def detect_correction(headline: str | None, body: str) -> str | None:
    headline_text = headline or ""
    opening = " ".join(body.split()[:120])
    if _RETRACTION_RE.search(headline_text) or _OPENING_RETRACTION_RE.search(opening):
        return "retraction"
    if _CORRECTION_RE.search(headline_text) or _OPENING_CORRECTION_RE.search(opening):
        return "correction"
    return None


def _matching_review(matches: list[FactCheckMatch], tone: str) -> FactCheckMatch | None:
    return next((match for match in matches if match.tone == tone), None)


def _fact_check_status(internal_source: bool, result: FactCheckResult) -> str:
    if internal_source:
        return "INTERNAL SOURCE"
    if result.status == "not_configured":
        return "NOT CONFIGURED"
    if result.status == "failed":
        return "LOOKUP FAILED"
    if result.matches:
        return "MATCH FOUND"
    return "NO MATCH FOUND"


def build_evidence_assessment(
    *,
    headline: str | None,
    body: str,
    source_trusted: bool,
    internal_source: bool,
    fact_check_result: FactCheckResult,
) -> EvidenceAssessment:
    correction = detect_correction(headline, body)
    source_reputation = "INTERNAL" if internal_source else "TRUSTED" if source_trusted else "UNKNOWN"
    fact_check_status = _fact_check_status(internal_source, fact_check_result)

    if correction == "retraction":
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status="RETRACTION DETECTED IN ARTICLE",
            final_label="RETRACTED",
            final_tone="fake",
            reason=(
                "The submitted article contains a retraction notice indicating that the publisher "
                "withdrew the original report or claim."
            ),
        )

    if correction == "correction":
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status="CORRECTION DETECTED IN ARTICLE",
            final_label="CORRECTED",
            final_tone="mixed",
            reason=(
                "The submitted article contains a correction, amendment or apology notice showing "
                "that material in the original publication was changed."
            ),
        )

    correction_status = "NOT DETECTED IN ARTICLE"
    verdict = fact_check_verdict(fact_check_result.matches)
    if verdict == "false":
        review = _matching_review(fact_check_result.matches, "false")
        publisher = review.publisher if review else "A professional fact-checker"
        rating = review.rating if review else "false or misleading"
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=f"MATCH FOUND — {rating.upper()}",
            correction_status=correction_status,
            final_label="LIKELY FAKE",
            final_tone="fake",
            reason=(
                f"{publisher} published a review rating the central claim “{rating}”. "
                "Open the linked review to examine the evidence behind that finding."
            ),
        )

    if verdict == "true":
        review = _matching_review(fact_check_result.matches, "true")
        publisher = review.publisher if review else "A professional fact-checker"
        rating = review.rating if review else "true"
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=f"MATCH FOUND — {rating.upper()}",
            correction_status=correction_status,
            final_label="LIKELY REAL",
            final_tone="real",
            reason=(
                f"{publisher} published a review rating the central claim “{rating}”. "
                "Open the linked review to examine the supporting evidence."
            ),
        )

    if verdict == "mixed":
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status="MATCH FOUND — MIXED",
            correction_status=correction_status,
            final_label="DISPUTED",
            final_tone="mixed",
            reason=(
                "Published fact-check information does not support a clear true-or-false conclusion; "
                "the linked reviews should be examined before accepting or sharing the claim."
            ),
        )

    if fact_check_result.status == "failed":
        reason = (
            "The professional fact-check lookup failed, and no correction or retraction notice was "
            "found in the submitted article, so its central claim remains unverified."
        )
    elif fact_check_result.status == "not_configured":
        reason = (
            "Professional fact-check lookup is not configured, and no correction or retraction "
            "notice was found in the submitted article, so its central claim remains unverified."
        )
    elif source_trusted:
        reason = (
            "The publisher is generally reputable, but source reputation does not verify this "
            "individual report, and no matching published fact-check was found."
        )
    else:
        reason = (
            "No matching published fact-check or correction notice was found, so the article's "
            "central claim cannot currently be confirmed as true or false."
        )
    return EvidenceAssessment(
        source_reputation=source_reputation,
        fact_check_status=fact_check_status,
        correction_status=correction_status,
        final_label="UNVERIFIED",
        final_tone="mixed",
        reason=reason,
    )
