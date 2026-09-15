"""LIME explainability wrapper (Section 3.15).

The explainer issues repeated queries back to the classification engine's
own predict_proba_raw — that loop is why explanation is slower than
classification. Do not substitute a cheaper proxy model here; the model
explained must be the model that classified (Figure 3.2).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.classifier import Classifier

NUM_SAMPLES = 500  # halved for faster explanations on CPU-only hosting;
                   # still stable enough for a demo (see Section 3.15)
NUM_FEATURES = 10
CLASS_NAMES = ["Real", "Fake"]
RANDOM_STATE = 42


@dataclass
class TokenWeight:
    token: str
    weight: float  # signed: positive pushes toward Fake, negative toward Real


@dataclass
class ExplanationResult:
    tokens: list[TokenWeight]  # top-10, ranked by |weight| descending


def explain(raw_text: str, classifier: Classifier) -> ExplanationResult:
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES, random_state=RANDOM_STATE, bow=classifier.model.bow
    )
    exp = explainer.explain_instance(
        raw_text,
        classifier.predict_proba_raw,
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
        labels=(1,),  # explain the "Fake" class so weight sign reads directly
                      # as push-toward-fake (+) / push-toward-real (-)
    )
    tokens = [TokenWeight(token=w, weight=v) for w, v in exp.as_list(label=1)]
    tokens.sort(key=lambda tw: abs(tw.weight), reverse=True)
    return ExplanationResult(tokens=tokens)
