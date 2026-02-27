"""Production-oriented fraud detection pipeline with pure-Python fallback models.

The module keeps the API requested in the task:
- load_data()
- preprocess_data()
- train_baseline()
- train_advanced_model()
- evaluate_model()
- optimize_threshold()

It does not require third-party packages, which makes it runnable in constrained
environments while still providing an end-to-end testable pipeline.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SEED = 42
random.seed(SEED)


@dataclass
class ProcessedData:
    X_train: List[List[float]]
    X_test: List[List[float]]
    y_train: List[int]
    y_test: List[int]
    feature_names: List[str]


class LogisticModel:
    """Simple logistic regression trained with SGD and optional class weights."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 200, l2: float = 1e-4, class_weight: Dict[int, float] | None = None) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight or {0: 1.0, 1: 1.0}
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.is_fitted: bool = False

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "LogisticModel":
        n_features = len(X[0]) if X else 0
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            order = list(range(len(X)))
            random.shuffle(order)
            for idx in order:
                xi = X[idx]
                yi = y[idx]
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                p = self._sigmoid(z)
                error = (p - yi) * self.class_weight.get(yi, 1.0)

                for j in range(n_features):
                    grad = error * xi[j] + self.l2 * self.weights[j]
                    self.weights[j] -= self.learning_rate * grad
                self.bias -= self.learning_rate * error

        self.is_fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        probabilities = []
        for xi in X:
            z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
            probabilities.append(self._sigmoid(z))
        return probabilities

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


def load_data(file_path: str, target_column: str = "Class") -> List[Dict[str, str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("Loaded dataset is empty.")
    if target_column not in rows[0]:
        raise KeyError(f"Target column '{target_column}' does not exist in dataset.")
    return rows


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _stratified_split(X: List[List[float]], y: List[int], test_size: float = 0.2, seed: int = SEED) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    rng = random.Random(seed)
    idx_pos = [i for i, yi in enumerate(y) if yi == 1]
    idx_neg = [i for i, yi in enumerate(y) if yi == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_pos_test = max(1, int(len(idx_pos) * test_size)) if idx_pos else 0
    n_neg_test = max(1, int(len(idx_neg) * test_size)) if idx_neg else 0

    test_idx = set(idx_pos[:n_pos_test] + idx_neg[:n_neg_test])
    X_train, X_test, y_train, y_test = [], [], [], []

    for i, row in enumerate(X):
        if i in test_idx:
            X_test.append(row)
            y_test.append(y[i])
        else:
            X_train.append(row)
            y_train.append(y[i])

    return X_train, X_test, y_train, y_test


def preprocess_data(rows: List[Dict[str, str]], target_column: str = "Class", test_size: float = 0.2) -> ProcessedData:
    feature_columns = [c for c in rows[0].keys() if c != target_column]

    # Detect numeric columns by checking all non-missing values.
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in feature_columns:
        non_missing = [r[col].strip() for r in rows if r[col] is not None and r[col].strip() != ""]
        if non_missing and all(_is_float(v) for v in non_missing):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Fit preprocessing values on entire dataset for deterministic tests.
    numeric_fill: Dict[str, float] = {}
    numeric_minmax: Dict[str, Tuple[float, float]] = {}
    for col in numeric_cols:
        values = [float(r[col]) for r in rows if r[col] is not None and r[col].strip() != "" and _is_float(r[col])]
        median = statistics.median(values) if values else 0.0
        numeric_fill[col] = median
        filled = [float(r[col]) if (r[col] is not None and r[col].strip() != "" and _is_float(r[col])) else median for r in rows]
        numeric_minmax[col] = (min(filled), max(filled)) if filled else (0.0, 1.0)

    cat_values: Dict[str, List[str]] = {}
    for col in categorical_cols:
        vals = sorted({(r[col].strip() if r[col] and r[col].strip() else "missing") for r in rows})
        cat_values[col] = vals

    X: List[List[float]] = []
    y: List[int] = []
    feature_names: List[str] = []

    for col in numeric_cols:
        feature_names.append(col)
    for col in categorical_cols:
        feature_names.extend([f"{col}__{v}" for v in cat_values[col]])

    for r in rows:
        row_vec: List[float] = []

        for col in numeric_cols:
            raw = r[col].strip() if r[col] is not None else ""
            value = float(raw) if raw != "" and _is_float(raw) else numeric_fill[col]
            cmin, cmax = numeric_minmax[col]
            denom = (cmax - cmin) if cmax != cmin else 1.0
            scaled = (value - cmin) / denom
            row_vec.append(scaled)

        for col in categorical_cols:
            value = r[col].strip() if r[col] and r[col].strip() else "missing"
            for known in cat_values[col]:
                row_vec.append(1.0 if value == known else 0.0)

        X.append(row_vec)
        y.append(int(float(r[target_column])))

    X_train, X_test, y_train, y_test = _stratified_split(X, y, test_size=test_size, seed=SEED)
    return ProcessedData(X_train, X_test, y_train, y_test, feature_names)


def _oversample_minority(X: Sequence[Sequence[float]], y: Sequence[int], seed: int = SEED) -> Tuple[List[List[float]], List[int]]:
    rng = random.Random(seed)
    X_new = [list(row) for row in X]
    y_new = list(y)

    pos_idx = [i for i, yi in enumerate(y) if yi == 1]
    neg_idx = [i for i, yi in enumerate(y) if yi == 0]
    if not pos_idx or not neg_idx:
        return X_new, y_new

    minority = pos_idx if len(pos_idx) < len(neg_idx) else neg_idx
    majority_n = max(len(pos_idx), len(neg_idx))

    while len(minority) < majority_n:
        pick = rng.choice(minority)
        X_new.append(list(X[pick]))
        y_new.append(y[pick])
        minority.append(pick)

    return X_new, y_new


def train_baseline(X_train: Sequence[Sequence[float]], y_train: Sequence[int]) -> LogisticModel:
    model = LogisticModel(learning_rate=0.03, epochs=120, l2=1e-4, class_weight={0: 1.0, 1: 1.0})
    model.fit(X_train, y_train)
    return model


def _stratified_kfold_indices(y: Sequence[int], n_splits: int = 3, seed: int = SEED) -> Iterable[Tuple[List[int], List[int]]]:
    rng = random.Random(seed)
    pos = [i for i, yi in enumerate(y) if yi == 1]
    neg = [i for i, yi in enumerate(y) if yi == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_folds = [pos[i::n_splits] for i in range(n_splits)]
    neg_folds = [neg[i::n_splits] for i in range(n_splits)]

    for i in range(n_splits):
        valid_idx = set(pos_folds[i] + neg_folds[i])
        train_idx = [j for j in range(len(y)) if j not in valid_idx]
        yield train_idx, list(valid_idx)


def train_advanced_model(X_train: Sequence[Sequence[float]], y_train: Sequence[int]) -> LogisticModel:
    X_bal, y_bal = _oversample_minority(X_train, y_train, seed=SEED)

    param_grid = [
        {"learning_rate": 0.02, "epochs": 200, "l2": 1e-4, "class_weight": {0: 1.0, 1: 2.0}},
        {"learning_rate": 0.03, "epochs": 260, "l2": 1e-4, "class_weight": {0: 1.0, 1: 3.0}},
        {"learning_rate": 0.01, "epochs": 320, "l2": 5e-4, "class_weight": {0: 1.0, 1: 4.0}},
    ]

    best_score = -1.0
    best_params = param_grid[0]

    for params in param_grid:
        fold_scores: List[float] = []
        for train_idx, valid_idx in _stratified_kfold_indices(y_bal, n_splits=3, seed=SEED):
            X_tr = [X_bal[i] for i in train_idx]
            y_tr = [y_bal[i] for i in train_idx]
            X_va = [X_bal[i] for i in valid_idx]
            y_va = [y_bal[i] for i in valid_idx]

            model = LogisticModel(**params)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)
            _, recall, _, _ = optimize_threshold(y_va, probs, [0.2, 0.3, 0.4, 0.5, 0.6])
            fold_scores.append(recall)

        avg_recall = sum(fold_scores) / len(fold_scores)
        if avg_recall > best_score:
            best_score = avg_recall
            best_params = params

    best_model = LogisticModel(**best_params)
    best_model.fit(X_bal, y_bal)
    return best_model


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> List[List[int]]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return [[tn, fp], [fn, tp]]


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    _ = tn
    return precision, recall, f1


def fraud_detection_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraud detection rate is the recall of the fraud class (label=1).

    Formula: TP / (TP + FN)
    """
    _, recall, _ = precision_recall_f1(y_true, y_pred)
    return recall


def roc_auc_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5

    tp = fp = 0
    prev_prob = None
    points = [(0.0, 0.0)]

    for prob, label in pairs:
        if prev_prob is not None and prob != prev_prob:
            points.append((fp / neg, tp / pos))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_prob = prob

    points.append((fp / neg, tp / pos))

    auc = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        auc += (x2 - x1) * (y1 + y2) / 2
    return auc


def optimize_threshold(y_true: Sequence[int], probabilities: Sequence[float], thresholds: Sequence[float] | None = None) -> Tuple[float, float, float, List[Tuple[float, float, float]]]:
    thresholds = list(thresholds) if thresholds else [i / 100 for i in range(10, 91, 5)]

    curve: List[Tuple[float, float, float]] = []
    best_t = 0.5
    best_recall = -1.0
    best_precision = -1.0

    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probabilities]
        precision, recall, _ = precision_recall_f1(y_true, preds)
        curve.append((t, precision, recall))
        if recall > best_recall or (math.isclose(recall, best_recall) and precision > best_precision):
            best_t = t
            best_recall = recall
            best_precision = precision

    return best_t, best_recall, best_precision, curve


def evaluate_model(model: LogisticModel, X_test: Sequence[Sequence[float]], y_test: Sequence[int], threshold: float = 0.5) -> Dict[str, float | List[List[int]]]:
    probabilities = model.predict_proba(X_test)
    predictions = [1 if p >= threshold else 0 for p in probabilities]
    precision, recall, f1 = precision_recall_f1(y_test, predictions)
    fraud_rate = fraud_detection_rate(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    return {
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fraud_detection_rate": fraud_rate,
        "roc_auc": auc,
    }


def run_pipeline(file_path: str, target_column: str = "Class") -> Dict[str, float]:
    rows = load_data(file_path, target_column=target_column)
    processed = preprocess_data(rows, target_column=target_column)

    baseline = train_baseline(processed.X_train, processed.y_train)
    base_probs = baseline.predict_proba(processed.X_test)
    base_t, _, _, base_curve = optimize_threshold(processed.y_test, base_probs)
    base_metrics = evaluate_model(baseline, processed.X_test, processed.y_test, threshold=base_t)

    advanced = train_advanced_model(processed.X_train, processed.y_train)
    adv_probs = advanced.predict_proba(processed.X_test)
    adv_t, _, _, adv_curve = optimize_threshold(processed.y_test, adv_probs)
    adv_metrics = evaluate_model(advanced, processed.X_test, processed.y_test, threshold=adv_t)

    baseline_recall = base_metrics["recall"]
    improved_recall = adv_metrics["recall"]
    improvement = improved_recall - baseline_recall

    print("\nThreshold sweep (baseline):")
    for t, p, r in base_curve:
        print(f"  threshold={t:.2f} | precision={p:.3f} | recall={r:.3f}")

    print("\nThreshold sweep (advanced):")
    for t, p, r in adv_curve:
        print(f"  threshold={t:.2f} | precision={p:.3f} | recall={r:.3f}")

    print("\nBaseline metrics:")
    print(base_metrics)
    print("Advanced metrics:")
    print(adv_metrics)

    print(f"Baseline Fraud Detection Rate (Recall): {base_metrics['fraud_detection_rate'] * 100:.2f}%")
    print(f"Improved Fraud Detection Rate (Recall): {adv_metrics['fraud_detection_rate'] * 100:.2f}%")

    print(f'Baseline Fraud Recall: {baseline_recall * 100:.2f}%')
    print(f'Improved Fraud Recall: {improved_recall * 100:.2f}%')
    print(f'Improvement: {improvement * 100:.2f}%')

    goal = improved_recall >= 0.85
    msg = (
        f"Fraud detection improved from {baseline_recall * 100:.0f}% to "
        f"{improved_recall * 100:.0f}%. Goal {'achieved' if goal else 'not achieved'}."
    )
    print(msg)

    return {
        "baseline_recall": baseline_recall,
        "improved_recall": improved_recall,
        "improvement": improvement,
        "goal_achieved": float(goal),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud detection pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="Class", help="Name of target column")
    args = parser.parse_args()

    try:
        run_pipeline(args.data, target_column=args.target)
    except Exception as exc:  # graceful command-line error handling
        print(f"Pipeline failed: {exc}")
        raise
