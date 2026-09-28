from __future__ import annotations

import re
from dataclasses import dataclass

from app.corroboration import CorroborationResult
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
    corroboration_status: str
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


def _corroboration_status(result: CorroborationResult) -> str:
    if result.status == "not_configured":
        return "NOT CONFIGURED"
    if result.status == "failed":
        return "LOOKUP FAILED"
    supporting = len(result.supporting)
    conflicting = len(result.conflicting)
    if supporting and conflicting:
        return f"CONFLICTING — {supporting} SUPPORTING, {conflicting} OPPOSING"
    if conflicting:
        return f"CONFLICTING — {conflicting} OPPOSING SOURCE{'S' if conflicting != 1 else ''}"
    if supporting:
        return f"FOUND — {supporting} SUPPORTING SOURCE{'S' if supporting != 1 else ''}"
    if result.reports:
        return "RELATED REPORTS FOUND"
    return "NO SUPPORTING REPORTS"


def _publisher_list(matches) -> str:
    publishers = list(dict.fromkeys(item.publisher for item in matches))
    if len(publishers) == 1:
        return publishers[0]
    return ", ".join(publishers[:-1]) + f" and {publishers[-1]}"


def build_evidence_assessment(
    *,
    headline: str | None,
    body: str,
    source_trusted: bool,
    internal_source: bool,
    fact_check_result: FactCheckResult,
    corroboration_result: CorroborationResult,
) -> EvidenceAssessment:
    correction = detect_correction(headline, body)
    source_reputation = "INTERNAL" if internal_source else "TRUSTED" if source_trusted else "UNKNOWN"
    fact_check_status = _fact_check_status(internal_source, fact_check_result)
    corroboration_status = _corroboration_status(corroboration_result)

    if correction == "retraction":
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status="RETRACTION DETECTED",
            corroboration_status=corroboration_status,
            final_label="RETRACTED",
            final_tone="fake",
            reason=(
                "The article contains a retraction notice indicating that the publisher withdrew "
                "the original report or claim; the notice should be read for the exact correction."
            ),
        )

    if correction == "correction":
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status="CORRECTION DETECTED",
            corroboration_status=corroboration_status,
            final_label="CORRECTED",
            final_tone="mixed",
            reason=(
                "The article contains a correction or apology notice showing that material in the "
                "original publication was changed; consult the notice for the corrected facts."
            ),
        )

    correction_status = "NOT DETECTED"
    verdict = fact_check_verdict(fact_check_result.matches)
    if verdict == "false":
        review = _matching_review(fact_check_result.matches, "false")
        publisher = review.publisher if review else "A professional fact-checker"
        rating = review.rating if review else "false or misleading"
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=f"MATCH FOUND — {rating.upper()}",
            correction_status=correction_status,
            corroboration_status=corroboration_status,
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
            corroboration_status=corroboration_status,
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
            corroboration_status=corroboration_status,
            final_label="DISPUTED",
            final_tone="mixed",
            reason=(
                "Published fact-check information does not support a clear true-or-false conclusion; "
                "the linked reviews should be examined before accepting or sharing the claim."
            ),
        )

    supporting = corroboration_result.supporting
    conflicting = corroboration_result.conflicting
    if conflicting:
        publishers = _publisher_list(conflicting)
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status=correction_status,
            corroboration_status=corroboration_status,
            final_label="DISPUTED",
            final_tone="mixed",
            reason=(
                f"Reports from {publishers} contain wording that conflicts with the submitted "
                "claim; review the linked sources before drawing a conclusion."
            ),
        )

    if len(supporting) >= 2:
        publishers = _publisher_list(supporting[:3])
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status=correction_status,
            corroboration_status=corroboration_status,
            final_label="LIKELY REAL",
            final_tone="real",
            reason=(
                f"Independent reports from {publishers} describe the same central claim, and no "
                "correction, retraction or contradictory professional fact-check was found."
            ),
        )

    if len(supporting) == 1:
        return EvidenceAssessment(
            source_reputation=source_reputation,
            fact_check_status=fact_check_status,
            correction_status=correction_status,
            corroboration_status=corroboration_status,
            final_label="UNVERIFIED",
            final_tone="mixed",
            reason=(
                f"A related report from {supporting[0].publisher} supports the claim, but one "
                "independent source is insufficient for a corroborated conclusion."
            ),
        )

    if corroboration_result.status == "failed":
        reason = (
            "The independent news search failed, so the article's central claim could not be "
            "corroborated during this analysis."
        )
    elif corroboration_result.status == "not_configured":
        reason = (
            "Independent news search is not configured, so the article's central claim could not "
            "be compared with other reports."
        )
    elif source_trusted:
        reason = (
            "The publisher is generally reputable, but source reputation does not verify this "
            "individual report, and no matching fact-check or independent supporting report was found."
        )
    else:
        reason = (
            "No matching published fact-check or independent supporting report was found, so the "
            "article's central claim cannot currently be confirmed as true or false."
        )
    return EvidenceAssessment(
        source_reputation=source_reputation,
        fact_check_status=fact_check_status,
        correction_status=correction_status,
        corroboration_status=corroboration_status,
        final_label="UNVERIFIED",
        final_tone="mixed",
        reason=reason,
    )
