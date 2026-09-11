"""SQLite database layer (Section 3.9.1), fields per report Table 3.4.

Three tables: models, analyses, explanations. One model has many analyses;
one analysis has many explanation rows.

A few columns beyond Table 3.4's literal field list are added because the
model registry (app/models_registry.py) needs them to function — they're
called out below wherever they appear so it's clear which fields are the
report's contract and which are implementation necessities.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path("app.db")
EXCERPT_MAX_CHARS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    training_dataset TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    trained_at TEXT,
    -- implementation additions (not in Table 3.4): needed so the model
    -- registry can find "the active model for family X" and distinguish
    -- the dev-only placeholder from a real trained artifact.
    family TEXT NOT NULL CHECK (family IN ('naive_bayes', 'svm', 'bilstm', 'bert')),
    artifact_path TEXT,
    is_placeholder INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_hash TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('text', 'url')),
    source_url TEXT,
    text_excerpt TEXT NOT NULL,
    predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),  -- 0 = real, 1 = fake
    confidence REAL NOT NULL,
    model_id INTEGER NOT NULL REFERENCES models(model_id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- implementation addition (not in Table 3.4): Section 3.8 requires a
    -- low-confidence flag on every result; stored rather than recomputed
    -- so a cached result renders identically to the original.
    is_low_confidence INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_analyses_hash ON analyses(input_hash);

CREATE TABLE IF NOT EXISTS explanations (
    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(analysis_id),
    token TEXT NOT NULL,
    weight REAL NOT NULL,
    -- implementation addition (not in Table 3.4): preserves top-10 display
    -- order without relying on re-sorting signed weights at read time.
    rank INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_explanations_analysis ON explanations(analysis_id);
"""

LABEL_REAL = 0
LABEL_FAKE = 1


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def hash_text(normalised_text: str) -> str:
    """SHA-256 of normalised text, for duplicate detection (input_hash)."""
    return hashlib.sha256(normalised_text.encode("utf-8")).hexdigest()


def excerpt(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Only a 500-character excerpt is stored, never the full third-party
    article (copyright)."""
    return text[:max_chars]


def register_model(
    conn: sqlite3.Connection,
    model_name: str,
    version: str,
    training_dataset: str,
    family: str,
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    artifact_path: str | None = None,
    is_placeholder: bool = False,
    trained_at: str | None = None,
    activate: bool = True,
) -> int:
    """Insert a model row. If activate, deactivates other models in the same
    family first so at most one model per family is active."""
    if activate:
        conn.execute("UPDATE models SET is_active = 0 WHERE family = ?", (family,))
    cur = conn.execute(
        """INSERT INTO models (model_name, version, training_dataset, accuracy, precision,
                                recall, f1, trained_at, family, artifact_path, is_placeholder,
                                is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_name, version, training_dataset, accuracy, precision, recall, f1, trained_at,
         family, artifact_path, int(is_placeholder), int(activate)),
    )
    conn.commit()
    return cur.lastrowid


def get_active_model(conn: sqlite3.Connection, family: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM models WHERE family = ? AND is_active = 1 ORDER BY model_id DESC LIMIT 1",
        (family,),
    ).fetchone()


def get_model(conn: sqlite3.Connection, model_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()


def find_cached_analysis(conn: sqlite3.Connection, input_hash: str) -> sqlite3.Row | None:
    """Return the most recent cached analysis for this text hash, if any."""
    return conn.execute(
        "SELECT * FROM analyses WHERE input_hash = ? ORDER BY analysis_id DESC LIMIT 1",
        (input_hash,),
    ).fetchone()


def record_analysis(
    conn: sqlite3.Connection,
    model_id: int,
    normalised_text: str,
    source_type: str,
    source_url: str | None,
    predicted_label: int,
    confidence: float,
    is_low_confidence: bool,
) -> int:
    """Insert an analysis row. predicted_label is an integer: 0 = real, 1 =
    fake. Every prediction is traceable to model_id."""
    if source_type not in ("text", "url"):
        raise ValueError(f"source_type must be 'text' or 'url', got {source_type!r}")
    if predicted_label not in (LABEL_REAL, LABEL_FAKE):
        raise ValueError(f"predicted_label must be 0 (real) or 1 (fake), got {predicted_label!r}")

    cur = conn.execute(
        """INSERT INTO analyses (input_hash, source_type, source_url, text_excerpt,
                                  predicted_label, confidence, model_id, is_low_confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            hash_text(normalised_text),
            source_type,
            source_url,
            excerpt(normalised_text),
            predicted_label,
            confidence,
            model_id,
            int(is_low_confidence),
        ),
    )
    conn.commit()
    return cur.lastrowid


def record_explanations(
    conn: sqlite3.Connection, analysis_id: int, tokens_weights: list[tuple[str, float]]
) -> None:
    conn.executemany(
        "INSERT INTO explanations (analysis_id, token, weight, rank) VALUES (?, ?, ?, ?)",
        [(analysis_id, token, weight, rank) for rank, (token, weight) in enumerate(tokens_weights)],
    )
    conn.commit()


def get_explanations(conn: sqlite3.Connection, analysis_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM explanations WHERE analysis_id = ? ORDER BY rank", (analysis_id,)
    ).fetchall()


def get_analysis(conn: sqlite3.Connection, analysis_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
