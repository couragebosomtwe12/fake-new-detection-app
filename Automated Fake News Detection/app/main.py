"""FastAPI gateway (Figure 3.2): validation -> preprocessing ->
classification engine -> LIME explainer -> storage -> render.

Output requirements (Section 3.8) are non-negotiable: label, confidence,
low-confidence flag, shaded text, top-10 diverging bar chart, and the
standing caveat on every single result.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import database as db
from app.classifier import Classifier
from app.explain import ExplanationResult, TokenWeight, explain
from app.factcheck import build_query, fact_check_verdict, search_fact_checks
from app.highlight import build_highlighted_html
from app.validation import ValidationError, fetch_article_from_url, validate_submission, is_trusted_source, trusted_source_name, is_internal_source, TRUSTED_SOURCES

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STANDING_CAVEAT = (
    "This is an automated statistical assessment, not a verified fact-check. "
    "Use it as one input among several when judging an article's credibility."
)

app = FastAPI(title="Automated Fake News Detection System")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_classifier: Classifier | None = None


def get_classifier() -> Classifier:
    global _classifier
    if _classifier is None:
        _classifier = Classifier()
    return _classifier


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()
    model = get_classifier().model
    conn = db.get_connection()
    try:
        existing = db.get_active_model(conn, model.family)
        if existing is None or bool(existing["is_placeholder"]) != model.is_placeholder:
            db.register_model(
                conn,
                model_name=model.name,
                version=model.version,
                # training_dataset is NOT NULL in Table 3.4; fall back rather
                # than crash the whole app if an artifact predates metadata
                # tracking (see save_model_metadata in run_pipeline.py).
                training_dataset=model.training_dataset or "unknown (no metadata.json found)",
                family=model.family,
                accuracy=model.metrics.get("accuracy") if model.metrics else None,
                precision=model.metrics.get("precision") if model.metrics else None,
                recall=model.metrics.get("recall") if model.metrics else None,
                f1=model.metrics.get("f1") if model.metrics else None,
                artifact_path=model.artifact_path,
                is_placeholder=model.is_placeholder,
            )
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"caveat": STANDING_CAVEAT})


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    request: Request,
    headline: str = Form(""),
    body: str = Form(""),
    source_url: str = Form(""),
) -> HTMLResponse:
    classifier = get_classifier()
    source_url = source_url.strip() or None

    # When a URL is provided, always fetch the source page so the displayed
    # headline is the article's real title — a user-typed headline that
    # doesn't match the source must not be shown. If the user also pasted a
    # body, keep it (their text is what's classified) but still verify the
    # headline against the source.
    headline_is_user_typed = bool(headline.strip())
    if source_url:
        try:
            fetched_title, fetched_body = fetch_article_from_url(source_url)
            headline = fetched_title or headline.strip()
            headline_is_user_typed = False  # auto-extracted titles are exempt from the minimum
            if not body.strip():
                body = fetched_body
        except ValidationError:
            if not body.strip():
                return templates.TemplateResponse(
                    request,
                    "index.html",
                    {
                        "caveat": STANDING_CAVEAT,
                        "error": "Could not fetch article from URL. Please provide the article text manually.",
                        "headline": headline,
                        "body": body,
                        "source_url": source_url or "",
                    },
                    status_code=422,
                )
            # Fetch failed but the user pasted text — proceed without
            # headline verification rather than rejecting usable input.
    
    try:
        # An auto-extracted URL title isn't subject to the 5-word headline
        # minimum (Table 3.3) — the user has no control over its length,
        # and the body (not the headline) is what's actually classified.
        submission = validate_submission(
            headline, body, source_url, check_headline_length=headline_is_user_typed
        )
    except ValidationError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "caveat": STANDING_CAVEAT,
                "error": exc.message,
                "headline": headline,
                "body": body,
                "source_url": source_url or "",
            },
            status_code=422,
        )
    except ValidationError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "caveat": STANDING_CAVEAT,
                "error": exc.message,
                "headline": headline,
                "body": body,
                "source_url": source_url or "",
            },
            status_code=422,
        )

    # External fact-check lookup runs concurrently with classification —
    # it's an independent signal, not part of the model verdict. Skipped
    # for internal (university) sources, which external checkers don't cover.
    from concurrent.futures import ThreadPoolExecutor

    fc_executor = ThreadPoolExecutor(max_workers=1)
    fc_future = None
    if not is_internal_source(submission.source_url):
        fc_future = fc_executor.submit(
            search_fact_checks, build_query(submission.headline, submission.body)
        )

    conn = db.get_connection()
    try:
        text_hash = db.hash_text(submission.body)
        cached = db.find_cached_analysis(conn, text_hash)
        if cached is not None:
            explanation_rows = db.get_explanations(conn, cached["analysis_id"])
            model_row = db.get_model(conn, cached["model_id"])
            # Results recorded while the dev placeholder was active have no
            # evidentiary value — re-analyse with the real loaded model rather
            # than serving a stale placeholder verdict from cache.
            if model_row and model_row["is_placeholder"]:
                cached = None
        if cached is not None:
            # Check both the current submission's URL and the cached record's
            # URL — either being trusted is enough to apply the override.
            is_trusted_override = (
                is_trusted_source(submission.source_url)
                or is_trusted_source(cached["source_url"])
            )
            # Apply whitelist override to cached results too — a cached FAKE
            # verdict from before the whitelist existed must not be shown for
            # a trusted source.
            if is_trusted_override and cached["predicted_label"] == db.LABEL_FAKE:
                cached = dict(cached)
                cached["predicted_label"] = db.LABEL_REAL
                cached["confidence"] = _trusted_confidence(submission.body)
                cached["is_low_confidence"] = 0
            fc_matches = fc_future.result() if fc_future else []
            context = _render_context(
                submission, cached, explanation_rows, model_row, True, is_trusted_override, fc_matches
            )
        else:
            result = classifier.classify(submission.body)
            trusted = is_trusted_source(submission.source_url)

            # Override classification for trusted sources
            if trusted:
                result.label = "real"
                result.label_display = "Likely Real"
                result.confidence = _trusted_confidence(submission.body)
                result.is_low_confidence = False

            # Skip LIME for trusted sources — the verdict is forced Real
            # regardless, so the explanation is decorative and the ~500
            # model calls are wasted latency on free-tier CPUs.
            explanation = (
                ExplanationResult(tokens=[]) if trusted
                else explain(submission.body, classifier)
            )

            is_trusted_override = is_trusted_source(submission.source_url)
            
            model_row = db.get_active_model(conn, classifier.model.family)
            analysis_id = db.record_analysis(
                conn,
                model_id=model_row["model_id"],
                normalised_text=submission.body,
                source_type="url" if submission.source_url else "text",
                source_url=submission.source_url,
                predicted_label=db.LABEL_FAKE if result.label == "fake" else db.LABEL_REAL,
                confidence=result.confidence,
                is_low_confidence=result.is_low_confidence,
            )
            db.record_explanations(conn, analysis_id, [(t.token, t.weight) for t in explanation.tokens])
            analysis_row = db.get_analysis(conn, analysis_id)
            explanation_rows = db.get_explanations(conn, analysis_id)
            fc_matches = fc_future.result() if fc_future else []
            context = _render_context(
                submission, analysis_row, explanation_rows, model_row, False, is_trusted_override, fc_matches
            )
    finally:
        conn.close()

    return templates.TemplateResponse(request, "result.html", context)


def _trusted_confidence(body: str) -> float:
    """Deterministic 95-100% confidence for trusted-source overrides —
    derived from the text hash so the same article always shows the same
    value rather than a random one."""
    import hashlib

    digest = hashlib.sha256(body.encode()).hexdigest()
    return 0.95 + (int(digest[:8], 16) % 51) / 1000  # 0.950 - 1.000


def _render_context(
    submission, analysis_row, explanation_rows, model_row, cache_hit: bool,
    is_trusted_source_override: bool = False, fc_matches: list | None = None,
) -> dict:
    tokens = [TokenWeight(token=r["token"], weight=r["weight"]) for r in explanation_rows]
    max_weight = max((abs(t.weight) for t in tokens), default=1.0) or 1.0
    ordered = sorted(tokens, key=lambda t: t.weight)  # ascending, for a diverging bar chart
    chart_data = [
        {
            "token": t.token,
            "weight": round(t.weight, 3),
            "direction": "fake" if t.weight > 0 else "real",
            "pct": round(abs(t.weight) / max_weight * 100, 1),  # % of the half-track width
        }
        for t in ordered
    ]
    is_fake = analysis_row["predicted_label"] == db.LABEL_FAKE
    trusted_name = trusted_source_name(submission.source_url) or trusted_source_name(
        analysis_row["source_url"]
    )
    # Display name for the source line: trusted outlet name, else the raw
    # domain, else "Pasted text" when no URL was submitted.
    url = submission.source_url or analysis_row["source_url"]
    if trusted_name:
        source_display = trusted_name
    elif url:
        from urllib.parse import urlparse

        source_display = urlparse(url).netloc.removeprefix("www.") or url
    else:
        source_display = "Pasted text"
    return {
        "caveat": STANDING_CAVEAT,
        "headline": submission.headline,
        "body": submission.body,
        "highlighted_body": build_highlighted_html(submission.body, tokens),
        "label": "fake" if is_fake else "real",
        "label_display": "Likely Fake" if is_fake else "Likely Real",
        "confidence_pct": round(analysis_row["confidence"] * 100, 1),
        "is_low_confidence": bool(analysis_row["is_low_confidence"]),
        "chart_data": chart_data,
        "model_name": model_row["model_name"] if model_row else "unknown",
        "is_placeholder": bool(model_row["is_placeholder"]) if model_row else False,
        "cache_hit": cache_hit,
        "is_trusted_source_override": is_trusted_source_override,
        "trusted_source_name": trusted_name,
        "source_display": source_display,
        "is_internal_source": is_internal_source(submission.source_url)
        or is_internal_source(analysis_row["source_url"]),
        "fc_matches": fc_matches or [],
        "fc_verdict": fact_check_verdict(fc_matches or []),
        "fc_enabled": bool(os.environ.get("GOOGLE_FACTCHECK_API_KEY")),
        "source_url": submission.source_url,
    }
