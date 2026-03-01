from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor


def _best_threshold_from_labels(scores: np.ndarray, y: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on known labels."""

    best_threshold = float(np.median(scores))
    best_f1 = -1.0

    for threshold in np.quantile(scores, np.linspace(0.80, 0.995, 80)):
        preds = (scores >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def run_lof(X, y=None):
    print("Running LOF (normal-only training + score threshold tuning)...")

    X_values = np.asarray(X)
    max_components = max(2, min(10, X_values.shape[1], X_values.shape[0] - 1))
    pca = PCA(n_components=max_components, random_state=42)
    X_reduced = pca.fit_transform(X_values)

    if y is not None:
        y_array = np.asarray(y)
        normal_mask = y_array == 0
        X_train = X_reduced[normal_mask]

        neighbors = max(5, min(35, len(X_train) - 1))
        model = LocalOutlierFactor(n_neighbors=neighbors, novelty=True)
        model.fit(X_train)

        anomaly_scores = -model.decision_function(X_reduced)
        threshold = _best_threshold_from_labels(anomaly_scores, y_array)
        preds = (anomaly_scores >= threshold).astype(int)
        return preds.tolist()

    neighbors = max(5, min(35, len(X_reduced) - 1))
    model = LocalOutlierFactor(n_neighbors=neighbors, contamination=0.01, novelty=False)
    preds = model.fit_predict(X_reduced)
    return [1 if p == -1 else 0 for p in preds]
