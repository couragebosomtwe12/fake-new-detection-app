import pytest

from app import database as db


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    connection = db.get_connection(db_path)
    yield connection
    connection.close()


def test_register_model_and_get_active_model(conn):
    model_id = db.register_model(
        conn, model_name="Linear SVM", version="1.0", training_dataset="isot", family="svm"
    )
    row = db.get_active_model(conn, "svm")
    assert row is not None
    assert row["model_id"] == model_id
    assert row["is_active"] == 1
    assert row["training_dataset"] == "isot"


def test_registering_new_active_model_deactivates_previous(conn):
    first_id = db.register_model(
        conn, model_name="SVM v1", version="1.0", training_dataset="isot", family="svm"
    )
    second_id = db.register_model(
        conn, model_name="SVM v2", version="2.0", training_dataset="isot", family="svm"
    )

    active = db.get_active_model(conn, "svm")
    assert active["model_id"] == second_id

    first = conn.execute(
        "SELECT is_active FROM models WHERE model_id = ?", (first_id,)
    ).fetchone()
    assert first["is_active"] == 0


def test_register_model_stores_held_out_metrics(conn):
    model_id = db.register_model(
        conn, model_name="NB", version="1.0", training_dataset="liar", family="naive_bayes",
        accuracy=0.612, precision=0.592, recall=0.388, f1=0.469,
    )
    row = db.get_model(conn, model_id)
    assert row["accuracy"] == pytest.approx(0.612)
    assert row["f1"] == pytest.approx(0.469)


def test_record_analysis_stores_500_char_excerpt_and_is_traceable_to_model(conn):
    model_id = db.register_model(
        conn, model_name="NB", version="1.0", training_dataset="isot", family="naive_bayes"
    )
    text = "a" * 600  # longer than the excerpt cap

    analysis_id = db.record_analysis(
        conn, model_id=model_id, normalised_text=text, source_type="text", source_url=None,
        predicted_label=db.LABEL_FAKE, confidence=0.83, is_low_confidence=False,
    )
    row = db.get_analysis(conn, analysis_id)
    assert row["model_id"] == model_id
    assert row["predicted_label"] == db.LABEL_FAKE
    assert row["source_type"] == "text"
    assert len(row["text_excerpt"]) == 500


def test_record_analysis_rejects_invalid_source_type(conn):
    model_id = db.register_model(
        conn, model_name="NB", version="1.0", training_dataset="isot", family="naive_bayes"
    )
    with pytest.raises(ValueError):
        db.record_analysis(
            conn, model_id=model_id, normalised_text="text", source_type="carrier-pigeon",
            source_url=None, predicted_label=db.LABEL_REAL, confidence=0.9,
            is_low_confidence=False,
        )


def test_find_cached_analysis_matches_on_text_hash(conn):
    model_id = db.register_model(
        conn, model_name="NB", version="1.0", training_dataset="isot", family="naive_bayes"
    )
    text = "duplicate detection sample text"
    analysis_id = db.record_analysis(
        conn, model_id=model_id, normalised_text=text, source_type="text", source_url=None,
        predicted_label=db.LABEL_REAL, confidence=0.7, is_low_confidence=False,
    )
    cached = db.find_cached_analysis(conn, db.hash_text(text))
    assert cached is not None
    assert cached["analysis_id"] == analysis_id


def test_find_cached_analysis_returns_none_for_unseen_text(conn):
    assert db.find_cached_analysis(conn, db.hash_text("never seen before")) is None


def test_record_explanations_preserves_rank_order(conn):
    model_id = db.register_model(
        conn, model_name="NB", version="1.0", training_dataset="isot", family="naive_bayes"
    )
    analysis_id = db.record_analysis(
        conn, model_id=model_id, normalised_text="sample text", source_type="text",
        source_url=None, predicted_label=db.LABEL_REAL, confidence=0.9,
        is_low_confidence=False,
    )
    db.record_explanations(conn, analysis_id, [("miracle", 0.5), ("council", -0.3)])

    rows = db.get_explanations(conn, analysis_id)
    assert [r["token"] for r in rows] == ["miracle", "council"]
    assert [r["rank"] for r in rows] == [0, 1]
