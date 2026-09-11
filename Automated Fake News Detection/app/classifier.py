"""Classification engine: wraps the active model with the shared
preprocessing pipeline and the low-confidence flag rule (Section 3.8)."""
from __future__ import annotations

from dataclasses import dataclass

from app.models_registry import LoadedModel, load_active_model
from app.preprocessing import preprocess

LOW_CONFIDENCE_HIGH = 0.55
LABELS = ("real", "fake")


@dataclass
class ClassificationResult:
    label: str  # "real" | "fake"
    label_display: str  # "Likely Real" | "Likely Fake"
    confidence: float  # probability of the predicted class, 0-1
    probabilities: dict[str, float]  # {"real": p, "fake": p}
    is_low_confidence: bool
    preprocessed_text: str


class Classifier:
    def __init__(self, model: LoadedModel | None = None):
        self.model = model or load_active_model()

    def classify(self, raw_text: str) -> ClassificationResult:
        processed = preprocess(raw_text, self.model.preprocessing_family)
        p_real, p_fake = (float(p) for p in self.model.predict_proba([processed])[0])
        label = "fake" if p_fake >= p_real else "real"
        confidence = max(p_real, p_fake)

        # Section 3.8: low-confidence flag when probability falls between 45%
        # and 55%. Confidence (of the predicted class) is always >= 50% for a
        # binary argmax, so "P(fake) in [45%, 55%]" collapses to
        # "confidence <= 55%" — the two are the same test.
        is_low_confidence = confidence <= LOW_CONFIDENCE_HIGH

        return ClassificationResult(
            label=label,
            label_display="Likely Fake" if label == "fake" else "Likely Real",
            confidence=confidence,
            probabilities={"real": p_real, "fake": p_fake},
            is_low_confidence=is_low_confidence,
            preprocessed_text=processed,
        )

    def predict_proba_raw(self, raw_texts: list[str]):
        """For LIME: takes RAW text, applies this model's preprocessing
        pipeline internally, returns (n, 2) probabilities as [P(real), P(fake)]."""
        processed = [preprocess(t, self.model.preprocessing_family) for t in raw_texts]
        return self.model.predict_proba(processed)
