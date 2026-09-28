from __future__ import annotations

import re
from dataclasses import dataclass

from app.factcheck import FactCheckMatch, fact_check_verdict


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


def build_evidence_assessment(
    *,
    headline: str | None,
    body: str,
    source_trusted: bool,
    internal_source: bool,
    fact_check_enabled: bool,
    fact_check_matches: list[FactCheckMatch],
) -> EvidenceAssessment:
    correction = detect_correction(headline, body)
    source_reputation = "INTERNAL" if internal_source else "TRUSTED" if source_trusted else "UNKNOWN"
    corroboration_status = "NOT CHECKED"

    if internal_source:
        fact_check_status = "INTERNAL SOURCE"
    elif not fact_check_enabled:
        fact_check_status = "NOT CONFIGURED"
    elif fact_check_matches:
        fact_check_status = "MATCH FOUND"
    else:
        fact_check_status = "NO MATCH FOUND"

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
    verdict = fact_check_verdict(fact_check_matches)
    if verdict == "false":
        review = _matching_review(fact_check_matches, "false")
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
        review = _matching_review(fact_check_matches, "true")
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

    if source_trusted:
        reason = (
            "The publisher is generally reputable, but source reputation does not verify this "
            "individual report, and no matching published fact-check or independent corroboration was available."
        )
    else:
        reason = (
            "No matching published fact-check or independent corroboration was available, so the "
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
