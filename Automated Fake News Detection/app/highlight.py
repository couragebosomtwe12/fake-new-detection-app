"""Renders submitted text with influential tokens shaded by direction and
magnitude (Section 3.8, output requirement 4). Produces escaped HTML safe to
mark as |safe in the template — all non-matched text is escaped, and matched
text is escaped before being wrapped in a <mark> span."""
from __future__ import annotations

import html
import re

from app.explain import TokenWeight

FAKE_RGB = "220, 38, 38"   # red: pushes toward Fake
REAL_RGB = "22, 163, 74"   # green: pushes toward Real
MIN_ALPHA = 0.15
MAX_ALPHA = 0.70


def build_highlighted_html(text: str, tokens: list[TokenWeight]) -> str:
    if not tokens:
        return html.escape(text)

    max_weight = max(abs(t.weight) for t in tokens) or 1.0
    weight_by_token = {t.token.lower(): t.weight for t in tokens}
    # Longest tokens first so overlapping matches don't get partially shadowed.
    ordered = sorted(tokens, key=lambda t: len(t.token), reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t.token) for t in ordered) + r")\b", re.IGNORECASE
    )

    def repl(match: re.Match) -> str:
        matched = match.group(0)
        weight = weight_by_token.get(matched.lower(), 0.0)
        intensity = min(abs(weight) / max_weight, 1.0)
        alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * intensity
        colour = FAKE_RGB if weight > 0 else REAL_RGB
        direction = "pushes toward Fake" if weight > 0 else "pushes toward Real"
        return (
            f'<mark style="background-color: rgba({colour}, {alpha:.2f});" '
            f'title="{direction} ({weight:+.3f})">{html.escape(matched)}</mark>'
        )

    parts, last_end = [], 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[last_end:match.start()]))
        parts.append(repl(match))
        last_end = match.end()
    parts.append(html.escape(text[last_end:]))
    return "".join(parts)
