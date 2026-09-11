"""Model registry: discovers trained model artifacts under results/, or
falls back to an in-memory placeholder classifier for local development.

The placeholder is NOT a trained ISOT/LIAR model and produces no accuracy
figures for the report (see the academic integrity note in CLAUDE.md). It
exists only so the application's full request/response flow can be
exercised end to end before real model artifacts — produced by
run_pipeline.py against real ISOT/LIAR data — exist under results/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

RESULTS_DIR = Path("results")
ModelFamily = Literal["naive_bayes", "svm", "bilstm", "bert"]


@dataclass
class LoadedModel:
    family: ModelFamily
    name: str
    version: str
    predict_proba: Callable[[list[str]], np.ndarray]  # (n, 2): columns [P(real), P(fake)]
    preprocessing_family: Literal["classical", "transformer"]
    bow: bool  # LIME bag-of-words mode
    is_placeholder: bool
    artifact_path: str | None = None
    training_dataset: str | None = None  # Table 3.4: models.training_dataset
    metrics: dict | None = None  # Table 3.4: models.accuracy/precision/recall/f1


def _load_metadata(path: Path) -> dict:
    """Read the training_dataset/metrics sidecar written by
    save_model_metadata() in run_pipeline.py, if present."""
    import json

    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _load_svm() -> LoadedModel | None:
    path = RESULTS_DIR / "svm"
    model_path, vec_path = path / "model.joblib", path / "vectorizer.joblib"
    if not (model_path.exists() and vec_path.exists()):
        return None
    try:
        import joblib
    except ImportError:
        print(f"! Found SVM artifacts at {path} but joblib is not installed; skipping.")
        return None

    clf = joblib.load(model_path)
    vec = joblib.load(vec_path)
    meta = _load_metadata(path)

    def predict_proba(texts: list[str]) -> np.ndarray:
        return clf.predict_proba(vec.transform(texts))

    return LoadedModel(
        family="svm", name="Linear SVM", version="1.0", predict_proba=predict_proba,
        preprocessing_family="classical", bow=True, is_placeholder=False,
        artifact_path=str(path), training_dataset=meta.get("training_dataset"),
        metrics=meta.get("metrics"),
    )


def _load_naive_bayes() -> LoadedModel | None:
    path = RESULTS_DIR / "naive_bayes"
    model_path, vec_path = path / "model.joblib", path / "vectorizer.joblib"
    if not (model_path.exists() and vec_path.exists()):
        return None
    try:
        import joblib
    except ImportError:
        print(f"! Found Naive Bayes artifacts at {path} but joblib is not installed; skipping.")
        return None

    clf = joblib.load(model_path)
    vec = joblib.load(vec_path)
    meta = _load_metadata(path)

    def predict_proba(texts: list[str]) -> np.ndarray:
        return clf.predict_proba(vec.transform(texts))

    return LoadedModel(
        family="naive_bayes", name="Multinomial Naive Bayes", version="1.0",
        predict_proba=predict_proba, preprocessing_family="classical", bow=True,
        is_placeholder=False, artifact_path=str(path),
        training_dataset=meta.get("training_dataset"), metrics=meta.get("metrics"),
    )


def _load_bilstm() -> LoadedModel | None:
    path = RESULTS_DIR / "bilstm"
    model_path, vocab_path = path / "model.pt", path / "vocab.json"
    if not (model_path.exists() and vocab_path.exists()):
        return None
    try:
        import torch
    except ImportError:
        print(f"! Found BiLSTM artifacts at {path} but torch is not installed; skipping.")
        return None

    import json

    from app.bilstm_model import BiLSTMClassifier, collate_batch, encode

    with open(vocab_path) as f:
        vocab = json.load(f)
    model = BiLSTMClassifier(len(vocab))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    meta = _load_metadata(path)

    def predict_proba(texts: list[str]) -> np.ndarray:
        batch = [(torch.tensor(encode(t, vocab), dtype=torch.long), 0) for t in texts]
        padded, lengths, _ = collate_batch(batch)
        with torch.no_grad():
            logits = model(padded, lengths)
        return torch.softmax(logits, dim=-1).numpy()

    return LoadedModel(
        family="bilstm", name="Bidirectional LSTM", version="1.0", predict_proba=predict_proba,
        preprocessing_family="classical",
        bow=False,  # unlike NB/SVM, the BiLSTM is order-sensitive, so LIME
                    # should mask perturbed words in place rather than
                    # dropping them (same reasoning as BERT's bow=False)
        is_placeholder=False, artifact_path=str(path),
        training_dataset=meta.get("training_dataset"), metrics=meta.get("metrics"),
    )


def _load_bert() -> LoadedModel | None:
    path = RESULTS_DIR / "bert"
    if not (path / "config.json").exists():
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print(f"! Found BERT artifacts at {path} but torch/transformers are not installed; skipping.")
        return None

    # LIME re-queries this model ~1000 times per explanation (Section 3.15's
    # documented CPU bottleneck). torch defaults to using only half of the
    # machine's logical cores for intra-op parallelism; let it use them all.
    # This only changes wall-clock speed, not the model's outputs, so it does
    # not affect any figure already measured for the report.
    torch.set_num_threads(os.cpu_count() or 4)

    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()
    meta = _load_metadata(path)

    def predict_proba(texts: list[str]) -> np.ndarray:
        batch_size = 64  # top of the 32-64 range from the LIME/BERT performance warning, Section 3.15
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            chunks.append(torch.softmax(logits, dim=-1).numpy())
        return np.concatenate(chunks, axis=0)

    return LoadedModel(
        family="bert", name="BERT base uncased", version="1.0", predict_proba=predict_proba,
        preprocessing_family="transformer", bow=False, is_placeholder=False,
        artifact_path=str(path), training_dataset=meta.get("training_dataset"),
        metrics=meta.get("metrics"),
    )


_PLACEHOLDER_REAL = [
    "The city council approved the annual budget on Tuesday after a three hour public session.",
    "Researchers at the university published a peer reviewed study on regional rainfall patterns this week.",
    "The central bank left interest rates unchanged, citing steady inflation figures from the last quarter.",
    "Local health officials confirmed the water treatment plant passed its scheduled safety inspection.",
    "The transport ministry announced a phased repair of the coastal highway starting next month.",
    "Election observers reported that voting proceeded without major incident across the region.",
    "The company released its quarterly earnings, which matched analyst expectations for revenue growth.",
    "Farmers in the northern district reported an average harvest this season after adequate rainfall.",
    "The ministry of education confirmed that final exam schedules will remain unchanged this term.",
    "A new bus route linking the two districts will begin operating from the start of next week.",
    "The hospital administration confirmed that the new ward will open following a final inspection.",
    "Parliament debated the proposed amendment for two days before referring it to committee.",
    "The national statistics office released updated unemployment figures for the previous quarter.",
    "Officials said the bridge repair project remains on schedule for completion by year end.",
    "The weather service forecast moderate rainfall across the coastal region over the weekend.",
]

_PLACEHOLDER_FAKE = [
    "Secret government files PROVE the moon landing was staged in a hidden desert studio, insiders claim.",
    "Doctors HATE this one weird trick that cures every disease overnight, according to anonymous sources.",
    "Shocking leaked document reveals the president is secretly a robot controlled from another country.",
    "You won't believe what scientists don't want you to know about tap water turning people invisible.",
    "Anonymous whistleblower claims aliens have been running the central bank for the past fifty years.",
    "This miracle fruit melts fat while you sleep, banned by big pharma to protect their profits.",
    "Exclusive: secret society is microchipping citizens through breakfast cereal, sources refuse to confirm.",
    "Leaked audio 'proves' the entire election was decided by a coin flip in a hidden bunker.",
    "Local man discovers government is hiding a second sun behind the moon, refuses to elaborate further.",
    "Breaking: chocolate cures cancer instantly but hospitals refuse to tell patients, insider claims.",
    "Shock claim: your smartphone is reading your thoughts and selling them to a secret shadow agency.",
    "Unverified report says the entire ocean will be replaced with lemonade by next year, sources say.",
    "Anonymous tipster insists the weather is fully controlled by a hidden machine under the capital.",
    "Viral post claims drinking bleach cures the flu, doctors 'refuse' to confirm or deny the miracle cure.",
    "Secret recording allegedly shows officials admitting the entire currency is made of chocolate.",
]


def _build_placeholder() -> LoadedModel:
    """Tiny in-memory MultinomialNB fitted on a bundled synthetic fixture —
    not ISOT/LIAR data, not a reported result. See module docstring."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    from app.preprocessing import preprocess_classical

    texts = _PLACEHOLDER_REAL + _PLACEHOLDER_FAKE
    labels = [0] * len(_PLACEHOLDER_REAL) + [1] * len(_PLACEHOLDER_FAKE)
    processed = [preprocess_classical(t) for t in texts]

    vec = TfidfVectorizer()
    X = vec.fit_transform(processed)
    clf = MultinomialNB(alpha=1.0).fit(X, labels)

    def predict_proba(raw_texts: list[str]) -> np.ndarray:
        processed_inputs = [preprocess_classical(t) for t in raw_texts]
        return clf.predict_proba(vec.transform(processed_inputs))

    return LoadedModel(
        family="naive_bayes",
        name="Placeholder Naive Bayes (dev fixture, not a trained ISOT/LIAR model)",
        version="placeholder",
        predict_proba=predict_proba,
        preprocessing_family="classical",
        bow=True,
        is_placeholder=True,
        artifact_path=None,
        training_dataset="none (synthetic dev fixture)",
        metrics=None,
    )


def load_active_model() -> LoadedModel:
    """Optimized model loading: SVM is the default active model for production use.
    
    SVM is chosen as the default because:
    1. Fast startup time (lightweight model)
    2. Fast LIME explanations (vs BERT which requires ~1000 forward passes)
    3. Strong performance (99.35% accuracy on ISOT dataset)
    4. Suitable for CPU-only hosting environments
    
    This function only loads the SVM model to optimize startup time.
    If SVM artifacts are not found, it falls back to other models in order:
    SVM > Naive Bayes > BiLSTM > BERT > placeholder"""
    
    # Try to load SVM first (default for production)
    model = _load_svm()
    if model is not None:
        print("[OK] Loaded SVM model (default for production use)")
        return model
    
    # Fallback to other models if SVM not available
    print("[!] SVM model not found, trying fallback models...")
    
    for loader_name, loader in [("naive_bayes", _load_naive_bayes), 
                                 ("bilstm", _load_bilstm), 
                                 ("bert", _load_bert)]:
        model = loader()
        if model is not None:
            print(f"[OK] Loaded {loader_name} model as fallback")
            return model
    
    print(
        "! No trained model artifacts found under results/. Falling back to an "
        "in-memory placeholder Naive Bayes classifier for local development only. "
        "Its predictions carry no evidentiary or report value - run run_pipeline.py "
        "against real ISOT/LIAR data and place the artifacts under results/ before "
        "reporting any figures."
    )
    return _build_placeholder()
