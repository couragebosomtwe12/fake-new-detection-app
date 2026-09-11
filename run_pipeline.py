#!/usr/bin/env python3
"""
run_pipeline.py — Automated Fake News Detection System
Produces every outstanding value for Chapter Three and the results tables
for Chapter Four.

DESIGNED FOR GOOGLE COLAB (Runtime > Change runtime type > T4 GPU).

Setup cell to run first in Colab:
    !pip install -q transformers datasets lime scikit-learn pandas joblib spacy
    !python -m spacy download en_core_web_sm

This script imports app/preprocessing.py (Section 3.11) so training and the
FastAPI app's inference path share one implementation — run it from the repo
root so the app package is importable.

Data:
    ISOT  - download True.csv and Fake.csv from Kaggle:
            clmentbisaillon/fake-and-real-news-dataset
            Place both in ./data/
    LIAR  - loaded automatically from Hugging Face (`liar`), or place
            train.tsv / valid.tsv / test.tsv in ./data/liar/

Run:
    python run_pipeline.py            # everything
    python run_pipeline.py --stage svm    # just the SVM C sweep (2 min)

Every number this prints is one you measured. Do not edit them by hand.
"""

import argparse, json, os, re, time, random, sys
from pathlib import Path

# app/ lives inside "Automated Fake News Detection/" (moved there for
# organisation) rather than directly under the repo root, so it needs to be
# added to sys.path explicitly before the `from app...` imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent / "Automated Fake News Detection"))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from app.bilstm_model import BiLSTMClassifier, build_vocab, collate_batch, encode
from app.preprocessing import (preprocess_classical, preprocess_classical_batch,
                               preprocess_transformer, strip_html_url_whitespace)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
DATA = Path("data"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
RESULTS = {}


def rule(t=""):
    print("\n" + "=" * 74)
    if t: print(t); print("=" * 74)


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def clean(t):
    """Thin alias onto the shared preprocessing module (app/preprocessing.py)
    so training and inference never drift apart (Section 3.11)."""
    return strip_html_url_whitespace(t)


def load_isot():
    tp, fp = DATA / "True.csv", DATA / "Fake.csv"
    if not tp.exists():
        sys.exit(f"Missing {tp}. Download the ISOT dataset from Kaggle into ./data/")
    real = pd.read_csv(tp); fake = pd.read_csv(fp)
    real["label"] = 0; fake["label"] = 1
    df = pd.concat([real, fake], ignore_index=True)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean)
    df = df[df["text"].str.split().str.len() >= 25]
    return df[["text", "label"]].sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_liar():
    """Six-way labels collapsed to binary, per Section 3.10.1."""
    FAKE = {"pants-fire", "false", "barely-true"}
    local = DATA / "liar"
    if local.exists():
        cols = ["id", "label", "statement", "subject", "speaker", "job", "state",
                "party", "bt", "f", "ht", "mt", "pof", "context"]
        parts = [pd.read_csv(local / f, sep="\t", header=None, names=cols)
                 for f in ("train.tsv", "valid.tsv", "test.tsv") if (local / f).exists()]
        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns={"statement": "text"})
    else:
        from datasets import load_dataset
        ds = load_dataset("liar")
        names = ds["train"].features["label"].names
        df = pd.concat([pd.DataFrame(ds[s]) for s in ("train", "validation", "test")],
                       ignore_index=True)
        df["label"] = df["label"].map(lambda i: names[i])
        df = df.rename(columns={"statement": "text"})
    df["text"] = df["text"].map(clean)
    df["label"] = df["label"].str.lower().str.strip().map(lambda l: 1 if l in FAKE else 0)
    df = df[df["text"].str.split().str.len() >= 5]
    return df[["text", "label"]].sample(frac=1, random_state=SEED).reset_index(drop=True)


def split(df):
    """70 / 15 / 15 stratified, fixed seed — Section 3.14.1."""
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def metrics(y, p):
    return dict(
        accuracy=accuracy_score(y, p),
        precision=precision_score(y, p, zero_division=0),
        recall=recall_score(y, p, zero_division=0),
        f1=f1_score(y, p, zero_division=0),
        macro_f1=f1_score(y, p, average="macro", zero_division=0),
        confusion=confusion_matrix(y, p).tolist(),
    )


def show(name, m):
    print(f"  {name:<26} acc {m['accuracy']:.4f}  P {m['precision']:.4f}  "
          f"R {m['recall']:.4f}  F1 {m['f1']:.4f}  macroF1 {m['macro_f1']:.4f}")


def save_model_metadata(out_dir: Path, dataset: str, m: dict) -> None:
    """Record which dataset trained this artifact and its held-out metrics,
    so app/models_registry.py can populate the models table (Table 3.4:
    training_dataset, accuracy, precision, recall, f1) instead of leaving
    those columns null. Written alongside model.joblib/model.pt each time a
    stage saves an artifact — otherwise overwriting results/svm/ (say) with
    a run on a different dataset would silently lose this provenance.
    """
    meta = {
        "training_dataset": dataset,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {k: m[k] for k in ("accuracy", "precision", "recall", "f1") if k in m},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------
# 2. SVM C SWEEP  -> fills Table 3.7
# ----------------------------------------------------------------------
def stage_svm(tr, va):
    rule("SVM HYPERPARAMETER SWEEP  (Table 3.7)")
    # Section 3.11 classical pipeline: lowercase, strip punctuation, remove
    # stop words, lemmatise (spaCy) — done once here via app.preprocessing,
    # so the vectoriser itself no longer needs stop_words/lowercase. Batched
    # (spaCy nlp.pipe) since this runs over thousands of documents.
    tr_text = preprocess_classical_batch(list(tr.text))
    va_text = preprocess_classical_batch(list(va.text))
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
    Xtr = vec.fit_transform(tr_text); Xva = vec.transform(va_text)

    scores = {}
    for C in (0.1, 1, 10):
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
        clf.fit(Xtr, tr.label)
        s = f1_score(va.label, clf.predict(Xva), average="macro")
        scores[C] = s
        print(f"  C = {C:<6} validation macro-F1 = {s:.4f}")

    best = max(scores, key=scores.get)
    print(f"\n  >>> SELECTED C = {best}   (put this in Table 3.7)")
    RESULTS["svm_C_sweep"] = scores
    RESULTS["svm_C_selected"] = best
    return vec, best


# ----------------------------------------------------------------------
# 3. CLASSICAL MODELS
# ----------------------------------------------------------------------
def stage_classical(tr, va, te, C, dataset):
    rule("CLASSICAL MODELS")
    tr_text = preprocess_classical_batch(list(tr.text))
    te_text = preprocess_classical_batch(list(te.text))
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
    Xtr = vec.fit_transform(tr_text); Xte = vec.transform(te_text)

    nb = MultinomialNB(alpha=1.0).fit(Xtr, tr.label)
    m_nb = metrics(te.label, nb.predict(Xte)); show("Multinomial Naive Bayes", m_nb)

    sv = CalibratedClassifierCV(
        LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=SEED), cv=3)
    sv.fit(Xtr, tr.label)
    m_sv = metrics(te.label, sv.predict(Xte)); show("Linear SVM", m_sv)

    RESULTS["naive_bayes"] = m_nb; RESULTS["svm"] = m_sv

    # Persist artifacts so the FastAPI app's model registry (app/models_registry.py)
    # can load real trained models instead of its dev-only placeholder.
    for subdir, clf, m in (("naive_bayes", nb, m_nb), ("svm", sv, m_sv)):
        out_dir = OUT / subdir
        out_dir.mkdir(exist_ok=True)
        joblib.dump(clf, out_dir / "model.joblib")
        joblib.dump(vec, out_dir / "vectorizer.joblib")
        save_model_metadata(out_dir, dataset, m)
    print(f"\n  Saved classical model artifacts under {OUT}/naive_bayes/ and {OUT}/svm/")

    return vec, nb, sv


# ----------------------------------------------------------------------
# 4. BIDIRECTIONAL LSTM
# ----------------------------------------------------------------------
def stage_bilstm(tr, va, te, dataset):
    rule("BIDIRECTIONAL LSTM  (Section 3.13)")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(SEED)  # weight init + DataLoader(shuffle=True) both draw
                             # from torch's own RNG, not random/numpy's
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {dev}")

    # Same word-level pipeline as NB/SVM (Section 3.11): lowercase, strip
    # punctuation, remove stop words, lemmatise via spaCy. Batched for speed.
    tr_text = preprocess_classical_batch(list(tr.text))
    va_text = preprocess_classical_batch(list(va.text))
    te_text = preprocess_classical_batch(list(te.text))

    vocab = build_vocab(tr_text)
    print(f"  vocabulary size: {len(vocab)}")

    class SeqDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return torch.tensor(encode(self.texts[i], vocab), dtype=torch.long), self.labels[i]

    dl_tr = DataLoader(SeqDataset(tr_text, tr.label), batch_size=64, shuffle=True, collate_fn=collate_batch)
    dl_va = DataLoader(SeqDataset(va_text, va.label), batch_size=64, collate_fn=collate_batch)
    dl_te = DataLoader(SeqDataset(te_text, te.label), batch_size=64, collate_fn=collate_batch)

    model = BiLSTMClassifier(len(vocab)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss, bad_epochs, best_state = float("inf"), 0, None
    patience = 3
    for epoch in range(20):  # max 20 epochs, early stopping patience 3 on val loss
        model.train(); train_loss = 0.0
        for x, lengths, y in dl_tr:
            x, lengths, y = x.to(dev), lengths.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(dl_tr)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, lengths, y in dl_va:
                x, lengths, y = x.to(dev), lengths.to(dev), y.to(dev)
                val_loss += loss_fn(model(x, lengths), y).item()
        val_loss /= len(dl_va)
        print(f"  epoch {epoch + 1:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch + 1} (patience {patience})")
                break

    model.load_state_dict(best_state)
    model.eval(); preds = []
    with torch.no_grad():
        for x, lengths, y in dl_te:
            x, lengths = x.to(dev), lengths.to(dev)
            preds += model(x, lengths).argmax(-1).cpu().tolist()

    m = metrics(te.label, preds); show("Bidirectional LSTM", m)
    RESULTS["bilstm"] = m

    out_dir = OUT / "bilstm"; out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)
    save_model_metadata(out_dir, dataset, m)
    print(f"\n  Saved BiLSTM artifacts under {out_dir}/")

    return model, vocab


# ----------------------------------------------------------------------
# 5. BERT
# ----------------------------------------------------------------------
def stage_bert(tr, va, te, dataset, epochs=3):
    rule("BERT FINE-TUNING  (Section 3.14.3)")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {dev}")
    if dev == "cpu":
        print("  ! No GPU detected. This will take hours. Enable a GPU runtime.")

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2).to(dev)

    class DS(Dataset):
        def __init__(s, df):
            s.t = [preprocess_transformer(t) for t in df.text]; s.y = list(df.label)
        def __len__(s): return len(s.y)
        def __getitem__(s, i):
            e = tok(s.t[i], truncation=True, padding="max_length",
                    max_length=256, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in e.items()} | {"labels": torch.tensor(s.y[i])}

    dl_tr = DataLoader(DS(tr), batch_size=16, shuffle=True)
    dl_va = DataLoader(DS(va), batch_size=32)
    dl_te = DataLoader(DS(te), batch_size=32)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total = len(dl_tr) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    # Section 3.14.3 reports "validation loss ceased to improve" as the
    # reason for stopping at 3 epochs — that claim has to be measured, not
    # assumed, so validation loss is tracked every epoch alongside training.
    val_losses = []
    for ep in range(epochs):
        model.train(); tot = 0
        for i, b in enumerate(dl_tr):
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(**b); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += out.loss.item()
            if i % 200 == 0:
                print(f"    epoch {ep+1} step {i}/{len(dl_tr)} loss {out.loss.item():.4f}")
        train_loss = tot / len(dl_tr)

        model.eval(); val_tot = 0
        with torch.no_grad():
            for b in dl_va:
                b = {k: v.to(dev) for k, v in b.items()}
                val_tot += model(**b).loss.item()
        val_loss = val_tot / len(dl_va)
        val_losses.append(val_loss)
        print(f"  epoch {ep+1} mean train loss {train_loss:.4f}  val loss {val_loss:.4f}")

    if len(val_losses) >= 2 and val_losses[-1] >= val_losses[-2]:
        print(f"  validation loss did not improve on the final epoch "
              f"({val_losses[-2]:.4f} -> {val_losses[-1]:.4f}) — supports stopping here.")
    elif len(val_losses) >= 2:
        print(f"  ! validation loss was STILL improving at epoch {epochs} "
              f"({val_losses[-2]:.4f} -> {val_losses[-1]:.4f}). \"Ceased to improve\" is not "
              f"true at this epoch count — consider more epochs or rephrasing Section 3.14.3.")

    model.eval(); preds = []
    with torch.no_grad():
        for b in dl_te:
            b = {k: v.to(dev) for k, v in b.items()}
            preds += model(**b).logits.argmax(-1).cpu().tolist()

    m = metrics(te.label, preds); show("BERT base uncased", m)
    RESULTS["bert"] = m
    RESULTS["bert_val_losses"] = val_losses
    model.save_pretrained("results/bert"); tok.save_pretrained("results/bert")
    save_model_metadata(OUT / "bert", dataset, m)
    return model, tok


# ----------------------------------------------------------------------
# 6. CROSS-DATASET EVALUATION  -> Table 3.6 protocol, Chapter 4 results
# ----------------------------------------------------------------------
def stage_cross(isot, liar):
    rule("CROSS-DATASET EVALUATION  (Section 3.14.2)")
    existing = _load_existing_results()
    out = {}
    for name_a, a, name_b, b in (("ISOT", isot, "LIAR", liar),
                                 ("LIAR", liar, "ISOT", isot)):
        tr, va, te = split(a)
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
        Xtr = vec.fit_transform(preprocess_classical_batch(list(tr.text)))
        # Use this dataset's own previously-selected C (Table 3.7), not a
        # value selected for the other dataset.
        prior = existing.get(name_a.lower(), {})
        C = prior.get("svm_C_selected", 1)
        if "svm_C_selected" not in prior:
            print(f"  ! No prior SVM C sweep found for {name_a} — defaulting to C=1. "
                  f"Run --dataset {name_a.lower()} --stage svm first to select one.")
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
        clf.fit(Xtr, tr.label)

        m_in = metrics(te.label, clf.predict(vec.transform(preprocess_classical_batch(list(te.text)))))
        m_x = metrics(b.label, clf.predict(vec.transform(preprocess_classical_batch(list(b.text)))))
        gap = m_in["macro_f1"] - m_x["macro_f1"]

        print(f"\n  Trained on {name_a}")
        show(f"  in-distribution ({name_a})", m_in)
        show(f"  cross-dataset ({name_b})", m_x)
        print(f"    generalisation gap (macro-F1): {gap:+.4f}")
        out[f"{name_a}->{name_b}"] = dict(in_dist=m_in, cross=m_x, gap=gap)
    RESULTS["cross_dataset"] = out


def stage_cross_bert(isot, liar):
    """Cross-dataset evaluation for the fine-tuned BERT checkpoint under
    results/bert/ (Section 3.14.2) — loads the already-trained model rather
    than retraining (fine-tuning is Colab-only; evaluation is a single
    forward pass per example and runs fine on CPU), and reports the same
    in-distribution vs cross-dataset generalisation gap as stage_cross does
    for the SVM."""
    rule("CROSS-DATASET EVALUATION — BERT  (Section 3.14.2)")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = OUT / "bert"
    if not (path / "config.json").exists():
        print("  ! No BERT checkpoint found under results/bert/ — run --stage bert first.")
        return

    training_dataset = "isot"
    meta_path = path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            training_dataset = json.load(f).get("training_dataset", "isot")

    datasets = {"isot": isot, "liar": liar}
    if training_dataset not in datasets:
        print(f"  ! Unknown training_dataset '{training_dataset}' in BERT metadata; skipping.")
        return

    name_a = training_dataset.upper()
    a = datasets[training_dataset]
    name_b, b = next((n.upper(), d) for n, d in datasets.items() if n != training_dataset)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {dev}")
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(dev)
    model.eval()

    def predict_labels(texts, batch_size=32):
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = [preprocess_transformer(t) for t in texts[i:i + batch_size]]
            enc = tok(batch, truncation=True, padding=True, max_length=256,
                      return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc).logits
            preds += logits.argmax(-1).cpu().tolist()
        return preds

    _, _, te = split(a)  # same fixed-seed split (Section 3.14.1) used when BERT was trained
    m_in = metrics(te.label, predict_labels(list(te.text)))
    m_x = metrics(b.label, predict_labels(list(b.text)))
    gap = m_in["macro_f1"] - m_x["macro_f1"]

    print(f"\n  BERT trained on {name_a}")
    show(f"  in-distribution ({name_a})", m_in)
    show(f"  cross-dataset ({name_b})", m_x)
    print(f"    generalisation gap (macro-F1): {gap:+.4f}")

    RESULTS["cross_dataset_bert"] = {f"{name_a}->{name_b}": dict(in_dist=m_in, cross=m_x, gap=gap)}


# ----------------------------------------------------------------------
# 7. LIME STABILITY + FAITHFULNESS  -> Section 3.15.4
# ----------------------------------------------------------------------
def _lime_eval(predict, te, result_key, bow, n_items=50, n_runs=5, n_samples=1000):
    """Shared LIME stability + faithfulness core (Section 3.15.4). `predict`
    takes a list of RAW strings and returns (n, 2) probabilities — the
    caller is responsible for applying that model's own preprocessing
    inside `predict` (classical vs transformer), same contract as
    app/classifier.py's predict_proba_raw. `bow` must match the model
    family per Table 3.8 (True for classical, False for transformers)."""
    rule(f"LIME STABILITY AND FAITHFULNESS — {result_key}  (Section 3.15.4)")
    from lime.lime_text import LimeTextExplainer

    sample = te.sample(min(n_items, len(te)), random_state=SEED).reset_index(drop=True)

    jac, lat = [], []
    for i, row in sample.iterrows():
        sets = []
        for r in range(n_runs):
            ex = LimeTextExplainer(class_names=["Real", "Fake"], random_state=r, bow=bow)
            t0 = time.time()
            e = ex.explain_instance(row.text, predict, num_features=10,
                                    num_samples=n_samples)
            if r == 0: lat.append(time.time() - t0)
            sets.append({w for w, _ in e.as_list()})
        pair = [len(a & b) / len(a | b) for x, a in enumerate(sets)
                for b in sets[x + 1:]]
        jac.append(np.mean(pair))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sample)} explained")

    print(f"\n  Mean Jaccard similarity (top-10, {n_runs} runs): {np.mean(jac):.3f} "
          f"(sd {np.std(jac):.3f})")
    print(f"  Mean explanation latency: {np.mean(lat):.2f}s  "
          f"(95th pct {np.percentile(lat,95):.2f}s)")

    # faithfulness: delete top-5 positive tokens, does the label flip?
    flips, drops = 0, []
    for _, row in sample.iterrows():
        ex = LimeTextExplainer(class_names=["Real", "Fake"], random_state=SEED, bow=bow)
        e = ex.explain_instance(row.text, predict, num_features=10, num_samples=n_samples)
        top = [w for w, v in e.as_list() if v > 0][:5]
        before = predict([row.text])[0]
        stripped = re.sub(r"\b(" + "|".join(map(re.escape, top)) + r")\b", " ",
                          row.text, flags=re.I) if top else row.text
        after = predict([stripped])[0]
        if before.argmax() != after.argmax(): flips += 1
        drops.append(before.max() - after[before.argmax()])

    print(f"  Label flipped after removing top-5 features: {flips}/{len(sample)} "
          f"({100*flips/len(sample):.1f}%)")
    print(f"  Mean confidence drop: {np.mean(drops):.4f}")

    RESULTS[result_key] = dict(jaccard_mean=float(np.mean(jac)),
                               jaccard_sd=float(np.std(jac)),
                               latency_mean=float(np.mean(lat)),
                               latency_p95=float(np.percentile(lat, 95)),
                               flip_rate=flips / len(sample),
                               mean_conf_drop=float(np.mean(drops)),
                               n_items=len(sample), n_runs=n_runs, n_samples=n_samples)


def stage_lime(vec, clf, te, n_items=50, n_runs=5, n_samples=1000):
    """LIME evaluation for the classical SVM (bow=True per Table 3.8)."""
    # predict_fn takes RAW text (LIME perturbs the raw string) and applies the
    # same classical preprocessing pipeline used at training time (Section
    # 3.11). Batched — LIME calls this with hundreds of perturbed samples
    # per explain_instance call.
    predict = lambda xs: clf.predict_proba(vec.transform(preprocess_classical_batch(list(xs))))
    _lime_eval(predict, te, result_key="lime", bow=True,
               n_items=n_items, n_runs=n_runs, n_samples=n_samples)


def stage_lime_bert(te, n_items=50, n_runs=5, n_samples=1000, feasibility_budget_s=300):
    """LIME evaluation for the fine-tuned BERT checkpoint under results/bert/
    (bow=False per Table 3.8 — BERT is order-sensitive). This is the slow
    path Section 3.15 warns about (up to 1000 forward passes per
    explanation): run on a GPU (Colab) if it's impractically slow on CPU,
    or reduce n_items/n_samples and report that reduction as a limitation —
    both are explicitly sanctioned by CLAUDE.md rather than silently
    guessed at.

    Times a single explanation first. If it doesn't finish within
    `feasibility_budget_s` seconds, the full n_items*n_runs + n_items run is
    abandoned rather than left to run for hours, and the measured bound is
    recorded as the evidence for reporting this as a limitation (never
    fabricated — see CLAUDE.md's academic integrity note).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from lime.lime_text import LimeTextExplainer

    path = OUT / "bert"
    if not (path / "config.json").exists():
        print("  ! No BERT checkpoint found under results/bert/ — run --stage bert first.")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {dev}")
    if dev == "cpu":
        print("  ! No GPU detected — this will be slow (Section 3.15's own performance "
              "warning). Timing one explanation before committing to the full run.")

    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(dev)
    model.eval()

    def predict(xs, batch_size=32):
        texts = [preprocess_transformer(x) for x in xs]
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=256,
                      return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    probe_text = te.sample(1, random_state=SEED).iloc[0].text
    ex = LimeTextExplainer(class_names=["Real", "Fake"], random_state=0, bow=False)
    t0 = time.time()
    e = ex.explain_instance(probe_text, predict, num_features=10, num_samples=n_samples)
    probe_s = time.time() - t0
    n_explanations = n_items * n_runs + n_items
    projected_minutes = probe_s * n_explanations / 60
    print(f"  Single explanation: {probe_s:.1f}s  ->  projected {n_explanations} "
          f"explanations = {projected_minutes:.0f} min on this machine")

    if probe_s > feasibility_budget_s or projected_minutes > 120:
        print(f"  ! Single explanation exceeded the {feasibility_budget_s}s feasibility "
              f"budget (or projected full run > 2h). Not running the full "
              f"{n_explanations}-explanation stability/faithfulness protocol on CPU — "
              f"recording this as a measured limitation instead of fabricating a result.")
        RESULTS["lime_bert"] = dict(
            status="not_run_infeasible_on_cpu",
            device=dev,
            single_explanation_seconds=float(probe_s),
            n_samples=n_samples,
            projected_full_run_minutes=float(projected_minutes),
            n_items=n_items, n_runs=n_runs,
            note=("Section 3.15's own performance warning confirmed empirically: a single "
                  "LIME explanation over bert-base-uncased on CPU took the above measured "
                  "time, making the full 50-item x 5-run stability + 50-item faithfulness "
                  "protocol impractical without a GPU. Recommend re-running on Colab GPU "
                  "(as BERT training itself was) if this figure is required."),
        )
        return

    _lime_eval(predict, te, result_key="lime_bert", bow=False,
               n_items=n_items, n_runs=n_runs, n_samples=n_samples)


# ----------------------------------------------------------------------
# 8. LATENCY  -> Section 3.17.2
# ----------------------------------------------------------------------
def stage_latency(vec, clf, te, n=100):
    rule("END-TO-END LATENCY  (Section 3.17.2)")
    xs = te.text.sample(min(n, len(te)), random_state=SEED).tolist()
    ts = []
    for t in xs:
        t0 = time.time()
        clf.predict(vec.transform([preprocess_classical(t)]))
        ts.append(time.time() - t0)
    print(f"  Preprocessing + classification — mean {np.mean(ts)*1000:.1f} ms, "
          f"95th pct {np.percentile(ts,95)*1000:.1f} ms  (n={len(xs)})")
    RESULTS["latency_classify_ms"] = dict(mean=float(np.mean(ts) * 1000),
                                          p95=float(np.percentile(ts, 95) * 1000))


# ----------------------------------------------------------------------
# 9. SUMMARY
# ----------------------------------------------------------------------
# Keys that come out of a single-dataset run (tr/va/te derived from
# --dataset isot|liar) and must be nested under that dataset's name in
# results.json so a run on one dataset never clobbers another dataset's
# already-measured numbers. "cross_dataset"/"cross_dataset_bert" are the
# exceptions: they always load both datasets themselves, so they're
# shared/top-level rather than nested under one dataset.
_DATASET_SCOPED_KEYS = (
    "svm_C_sweep", "svm_C_selected", "naive_bayes", "svm", "bilstm", "bert",
    "bert_val_losses", "lime", "lime_bert", "latency_classify_ms",
)


def _load_existing_results() -> dict:
    path = OUT / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def summary(dataset: str):
    rule("VALUES FOR CHAPTER THREE")
    if "svm_C_selected" in RESULTS:
        print(f"  Table 3.7, SVM row  ->  selected value {RESULTS['svm_C_selected']} (dataset: {dataset})")
    if "lime" in RESULTS:
        L = RESULTS["lime"]
        print(f"  Section 3.15.4      ->  Jaccard {L['jaccard_mean']:.2f}, "
              f"flip rate {100*L['flip_rate']:.0f}%")
    if "latency_classify_ms" in RESULTS:
        print(f"  Section 3.17.2      ->  mean {RESULTS['latency_classify_ms']['mean']:.0f} ms")

    rule("RESULTS TABLE FOR CHAPTER FOUR")
    print(f"  {'Model':<26}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'MacroF1':>10}")
    for k, n in (("naive_bayes", "Multinomial Naive Bayes"), ("svm", "Linear SVM"),
                 ("bilstm", "Bidirectional LSTM"), ("bert", "BERT base")):
        if k in RESULTS:
            m = RESULTS[k]
            print(f"  {n:<26}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
                  f"{m['recall']:>8.4f}{m['f1']:>8.4f}{m['macro_f1']:>10.4f}")

    # Merge into any existing results.json rather than overwriting it, so a
    # run on one --stage/--dataset combination never destroys numbers
    # already measured by an earlier, separate invocation.
    merged = _load_existing_results()
    for key, value in RESULTS.items():
        if key in _DATASET_SCOPED_KEYS:
            merged.setdefault(dataset, {})[key] = value
        else:
            merged[key] = value  # e.g. cross_dataset: shared, not dataset-scoped

    with open(OUT / "results.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n  Full results written to {OUT/'results.json'} (merged, dataset='{dataset}')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "svm", "classical", "bilstm", "bert", "cross", "cross_bert",
                             "lime", "lime_bert", "latency"])
    ap.add_argument("--dataset", default="isot", choices=["isot", "liar"])
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    df = load_isot() if a.dataset == "isot" else load_liar()
    print(f"Loaded {a.dataset.upper()}: {len(df)} items "
          f"({(df.label==1).sum()} fake / {(df.label==0).sum()} real)")
    tr, va, te = split(df)
    print(f"Split: train {len(tr)} / val {len(va)} / test {len(te)}  (seed {SEED})")

    if a.stage in ("all", "svm"):
        _, C = stage_svm(tr, va)
    else:
        # Reuse whatever C an earlier, separate `--stage svm` run already
        # selected for this dataset, instead of silently defaulting to an
        # unselected value.
        prior = _load_existing_results().get(a.dataset, {})
        C = prior.get("svm_C_selected", 1)
        if "svm_C_selected" not in prior:
            print(f"  ! No prior SVM C sweep found for dataset='{a.dataset}' — "
                  f"defaulting to C=1. Run --stage svm first to select one.")

    if a.stage in ("all", "classical", "lime", "latency"):
        vec, nb, sv = stage_classical(tr, va, te, C, a.dataset)
    if a.stage in ("all", "bilstm"):
        stage_bilstm(tr, va, te, a.dataset)
    if a.stage in ("all", "cross"):
        stage_cross(load_isot(), load_liar())
    if a.stage in ("all", "cross_bert"):
        stage_cross_bert(load_isot(), load_liar())
    if a.stage in ("all", "bert"):
        stage_bert(tr, va, te, a.dataset, a.epochs)
    if a.stage in ("all", "lime"):
        stage_lime(vec, sv, te)
    if a.stage in ("all", "lime_bert"):
        stage_lime_bert(te)
    if a.stage in ("all", "latency"):
        stage_latency(vec, sv, te)

    summary(a.dataset)


if __name__ == "__main__":
    main()
