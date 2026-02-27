import csv
from pathlib import Path

from fraud_pipeline import (
    confusion_matrix,
    fraud_detection_rate,
    load_data,
    optimize_threshold,
    precision_recall_f1,
    preprocess_data,
    train_advanced_model,
    train_baseline,
)


def _make_dataset(path: Path, rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["amount", "time", "merchant", "country", "Class"])
        for i in range(rows):
            # Imbalanced classes with stronger fraud signal.
            fraud = 1 if i % 12 == 0 else 0
            amount = 500 + (i % 50) * 12 if fraud else 10 + (i % 20) * 2
            time = i % 24
            merchant = "electronics" if fraud else "grocery"
            country = "high_risk" if fraud else "local"
            if i % 33 == 0:
                merchant = ""  # missing categorical
            if i % 40 == 0:
                amount = ""  # missing numeric
            writer.writerow([amount, time, merchant, country, fraud])


def test_data_loading(tmp_path: Path):
    data_file = tmp_path / "transactions.csv"
    _make_dataset(data_file)

    rows = load_data(str(data_file), target_column="Class")
    assert data_file.exists()
    assert len(rows) > 0


def test_preprocessing_no_nulls(tmp_path: Path):
    data_file = tmp_path / "transactions.csv"
    _make_dataset(data_file)

    rows = load_data(str(data_file), target_column="Class")
    processed = preprocess_data(rows, target_column="Class")

    for row in processed.X_train + processed.X_test:
        for val in row:
            assert val is not None


def test_model_training_and_prediction_shape(tmp_path: Path):
    data_file = tmp_path / "transactions.csv"
    _make_dataset(data_file)

    rows = load_data(str(data_file), target_column="Class")
    processed = preprocess_data(rows, target_column="Class")

    baseline = train_baseline(processed.X_train, processed.y_train)
    advanced = train_advanced_model(processed.X_train, processed.y_train)

    assert baseline.is_fitted is True
    assert advanced.is_fitted is True

    baseline_probs = baseline.predict_proba(processed.X_test)
    advanced_probs = advanced.predict_proba(processed.X_test)
    assert len(baseline_probs) == len(processed.X_test)
    assert len(advanced_probs) == len(processed.X_test)


def test_recall_calculation_correct():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0]

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    assert cm == [[2, 1], [1, 2]]
    assert abs(precision - (2 / 3)) < 1e-9
    assert abs(recall - (2 / 3)) < 1e-9
    assert abs(f1 - (2 / 3)) < 1e-9
    assert abs(fraud_detection_rate(y_true, y_pred) - (2 / 3)) < 1e-9


def test_improved_recall_ge_baseline(tmp_path: Path):
    data_file = tmp_path / "transactions.csv"
    _make_dataset(data_file, rows=240)

    rows = load_data(str(data_file), target_column="Class")
    processed = preprocess_data(rows, target_column="Class")

    baseline = train_baseline(processed.X_train, processed.y_train)
    advanced = train_advanced_model(processed.X_train, processed.y_train)

    base_probs = baseline.predict_proba(processed.X_test)
    adv_probs = advanced.predict_proba(processed.X_test)

    base_t, base_recall, _, _ = optimize_threshold(processed.y_test, base_probs)
    adv_t, adv_recall, _, _ = optimize_threshold(processed.y_test, adv_probs)

    assert 0.1 <= base_t <= 0.9
    assert 0.1 <= adv_t <= 0.9
    assert adv_recall >= base_recall
