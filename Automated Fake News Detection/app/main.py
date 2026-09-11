"""FastAPI gateway (Figure 3.2): validation -> preprocessing ->
classification engine -> LIME explainer -> storage -> render.

Output requirements (Section 3.8) are non-negotiable: label, confidence,
low-confidence flag, shaded text, top-10 diverging bar chart, and the
standing caveat on every single result.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import database as db
from app.classifier import Classifier
from app.explain import ExplanationResult, TokenWeight, explain
from app.highlight import build_highlighted_html
from app.validation import ValidationError, fetch_article_from_url, validate_submission, is_trusted_source, TRUSTED_SOURCES

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

    headline_is_user_typed = bool(headline.strip())
    
    # If URL is provided and body is empty, try to fetch from URL
    if source_url and not body.strip():
        try:
            fetched_title, fetched_body = fetch_article_from_url(source_url)
            headline = headline.strip() or fetched_title
            body = fetched_body
        except ValidationError:
            # If URL fetch fails, return error since we have no text to analyze
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
    
    # If both URL and body are provided, use the provided body with URL for whitelist
    # (don't try to fetch from URL if body is already provided)
    
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

    conn = db.get_connection()
    try:
        text_hash = db.hash_text(submission.body)
        cached = db.find_cached_analysis(conn, text_hash)
        if cached is not None:
            explanation_rows = db.get_explanations(conn, cached["analysis_id"])
            model_row = db.get_model(conn, cached["model_id"])
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
                cached["confidence"] = 0.95
                cached["is_low_confidence"] = 0
            context = _render_context(
                submission, cached, explanation_rows, model_row, True, is_trusted_override
            )
        else:
            result = classifier.classify(submission.body)
            
            # Override classification for trusted sources
            if is_trusted_source(submission.source_url):
                result.label = "real"
                result.label_display = "Likely Real"
                result.confidence = 0.95  # High confidence for trusted sources
                result.is_low_confidence = False
            
            explanation = explain(submission.body, classifier)

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
            context = _render_context(
                submission, analysis_row, explanation_rows, model_row, False, is_trusted_override
            )
    finally:
        conn.close()

    return templates.TemplateResponse(request, "result.html", context)


def _render_context(
    submission, analysis_row, explanation_rows, model_row, cache_hit: bool, is_trusted_source_override: bool = False
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
        "source_url": submission.source_url,
    }
