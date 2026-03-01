"""
Local Outlier Factor (LOF) model for anomaly/fraud detection.

Supports:
- Unsupervised mode: fit and detect on all data
- Supervised mode: train on normal-only data, tune threshold on separate validation set
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor


class LOFResult(NamedTuple):
    """Result from LOF model: predictions and anomaly scores."""

    predictions: np.ndarray
    anomaly_scores: np.ndarray


def _to_numpy(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert input to numpy array, handling pandas DataFrames."""
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _reduce_dimensionality(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Apply PCA if needed to reduce to reasonable dimensionality."""
    if X.shape[1] <= 10:
        return X

    max_components = max(2, min(10, X.shape[1], X.shape[0] - 1))
    pca = PCA(n_components=max_components, random_state=random_state)
    return pca.fit_transform(X)


def _select_n_neighbors(n_samples: int) -> int:
    """Select appropriate n_neighbors based on sample size."""
    return max(5, min(35, n_samples - 1))


def _optimize_threshold_on_validation(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    precision_floor: float = 0.10,
) -> float:
    """
    Tune threshold on validation set only (no leakage).

    Maximize recall subject to precision >= precision_floor.
    Falls back to maximizing F1 if precision_floor cannot be met.

    Args:
        val_scores: Anomaly scores on validation set.
        val_labels: Ground truth labels on validation set (0=normal, 1=fraud).
        precision_floor: Minimum acceptable precision.

    Returns:
        Optimal threshold for classification.
    """
    best_threshold = float(np.median(val_scores))
    best_recall = 0.0
    best_f1 = -1.0
    threshold_with_best_f1 = best_threshold

    # Try thresholds across the score distribution
    for threshold in np.quantile(val_scores, np.linspace(0.80, 0.995, 80)):
        preds = (val_scores >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (val_labels == 1)))
        fp = int(np.sum((preds == 1) & (val_labels == 0)))
        fn = int(np.sum((preds == 0) & (val_labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # Track best F1 as fallback
        if f1 > best_f1:
            best_f1 = f1
            threshold_with_best_f1 = float(threshold)

        # If precision meets floor, prioritize recall
        if precision >= precision_floor and recall > best_recall:
            best_recall = recall
            best_threshold = float(threshold)

    # If no threshold met precision floor, use best F1
    if best_recall == 0.0:
        best_threshold = threshold_with_best_f1

    return best_threshold


def fit_and_predict_lof(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | None = None,
    val_X: np.ndarray | pd.DataFrame | None = None,
    val_y: np.ndarray | None = None,
    precision_floor: float = 0.10,
    random_state: int = 42,
) -> LOFResult:
    """
    Train LOF and predict anomalies with optional threshold tuning on validation set.

    **Supervised mode (y provided):**
    - Trains on normal samples only from training set.
    - If validation data provided: tunes threshold on validation set only (no leakage).
    - If no validation data: returns anomaly scores (caller should tune threshold elsewhere).

    **Unsupervised mode (y is None):**
    - Standard LOF fit_predict on all data with fixed contamination.

    Args:
        X: Training data (n_samples, n_features). Can be numpy array or pandas DataFrame.
        y: Training labels (0=normal, 1=fraud). If None, uses unsupervised mode.
        val_X: Validation data for threshold tuning. Required for supervised threshold tuning.
        val_y: Validation labels for threshold tuning.
        precision_floor: Minimum precision to maintain during threshold optimization.
        random_state: Random seed for reproducibility.

    Returns:
        LOFResult with binary predictions (0/1) and anomaly scores.
    """
    X_array = _to_numpy(X)
    X_reduced = _reduce_dimensionality(X_array, random_state=random_state)

    if y is None:
        # Unsupervised mode: standard LOF
        n_samples = X_reduced.shape[0]
        n_neighbors = _select_n_neighbors(n_samples)
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=0.01,
            novelty=False,
        )
        raw_preds = model.fit_predict(X_reduced)
        anomaly_scores = -model.negative_outlier_factor_
        predictions = np.where(raw_preds == -1, 1, 0).astype(int)
        return LOFResult(predictions=predictions, anomaly_scores=anomaly_scores)

    # Supervised mode: train on normal-only data with novelty=True
    y_array = np.asarray(y)
    normal_mask = y_array == 0
    X_train = X_reduced[normal_mask]

    n_neighbors = _select_n_neighbors(len(X_train))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    model.fit(X_train)

    # Score all training data
    anomaly_scores = -model.decision_function(X_reduced)

    # Threshold tuning on validation set (no leakage)
    if val_X is not None and val_y is not None:
        val_X_array = _to_numpy(val_X)
        val_X_reduced = _reduce_dimensionality(val_X_array, random_state=random_state)
        val_scores = -model.decision_function(val_X_reduced)
        val_y_array = np.asarray(val_y)

        threshold = _optimize_threshold_on_validation(
            val_scores,
            val_y_array,
            precision_floor=precision_floor,
        )
    else:
        # No validation set: use median as default threshold
        threshold = float(np.median(anomaly_scores))

    predictions = (anomaly_scores >= threshold).astype(int)
    return LOFResult(predictions=predictions, anomaly_scores=anomaly_scores)


def run_lof(X: np.ndarray | pd.DataFrame, y: np.ndarray | None = None) -> list:
    """
    Legacy interface for backward compatibility.

    Args:
        X: Feature data.
        y: Optional labels for supervised mode.

    Returns:
        Binary predictions as a list.
    """
    result = fit_and_predict_lof(X, y)
    return result.predictions.tolist()
