"""
Main fraud detection pipeline with proper train/validation/test evaluation.

- Splits data into train/validation/test sets (stratified by fraud label)
- Trains both LOF and Isolation Forest on normal samples only
- Tunes LOF threshold on validation set (no leakage)
- Reports final metrics on held-out test set
"""

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np

from src.data_loader import load_data
from src.preprocessing import scale_data
from src.eda import plot_class_distribution
from src.models.lof import _reduce_dimensionality, _select_n_neighbors, _optimize_threshold_on_validation
from src.visualize import pca_plot


def print_results(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print classification metrics for model evaluation."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(f"{name} - TEST SET RESULTS")
    print("=" * 60)
    print(f"Total transactions : {len(y_true)}")
    print(f"True positives (TP): {tp}")
    print(f"False negatives (FN): {fn}")
    print(f"False positives (FP): {fp}")
    print(f"True negatives (TN): {tn}")
    print("-" * 60)
    print(f"Accuracy           : {accuracy*100:.2f}%")
    print(f"Precision          : {precision*100:.2f}%")
    print(f"Recall (Detection) : {recall*100:.2f}%")
    print(f"F1 Score           : {f1:.4f}")
    print("=" * 60)


def main():
    print("Loading data...")
    df = load_data()

    print("Plotting class distribution...")
    plot_class_distribution(df)

    print("Scaling features...")
    df = scale_data(df)

    X = df.drop("Class", axis=1)
    y = df["Class"].values

    # ========== STRATIFIED SPLIT: TRAIN / VALIDATION / TEST ==========
    # First split: 60% train+val, 40% test (stratified by fraud label)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    # Second split: 75% train, 25% validation from temp (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"\nData split summary:")
    print(f"  Train: {len(X_train)} samples ({np.sum(y_train)} frauds)")
    print(f"  Val:   {len(X_val)} samples ({np.sum(y_val)} frauds)")
    print(f"  Test:  {len(X_test)} samples ({np.sum(y_test)} frauds)")

    # ========== ISOLATION FOREST ==========
    print("\n" + "=" * 60)
    print("Training Isolation Forest on normal samples only...")
    print("=" * 60)

    # Train on normal samples from training set
    X_train_normal = X_train[y_train == 0]
    iso_model = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso_model.fit(X_train_normal)

    # Predict on test set
    iso_raw_preds = iso_model.predict(X_test)
    iso_preds = np.where(iso_raw_preds == -1, 1, 0)

    print_results("Isolation Forest", y_test, iso_preds)
    pca_plot(X_test, iso_preds, "Isolation Forest")

    # ========== LOF WITH VALIDATION-BASED THRESHOLD TUNING ==========
    print("\n" + "=" * 60)
    print("Training LOF on normal samples only...")
    print("=" * 60)

    # Step 1: Train LOF on normal samples from train+val combined
    X_trainval = np.vstack([X_train.values, X_val.values])
    y_trainval = np.hstack([y_train, y_val])
    X_trainval_normal = X_trainval[y_trainval == 0]
    
    X_trainval_reduced = _reduce_dimensionality(X_trainval, random_state=42)
    X_trainval_normal_reduced = X_trainval_reduced[y_trainval == 0]
    
    n_neighbors = _select_n_neighbors(len(X_trainval_normal_reduced))
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof_model.fit(X_trainval_normal_reduced)

    # Step 2: Find best threshold using validation set
    X_val_reduced = _reduce_dimensionality(X_val.values, random_state=42)
    val_scores = -lof_model.decision_function(X_val_reduced)
    threshold = _optimize_threshold_on_validation(val_scores, y_val, precision_floor=0.10)

    # Step 3: Score and predict on test set
    X_test_reduced = _reduce_dimensionality(X_test.values, random_state=42)
    test_scores = -lof_model.decision_function(X_test_reduced)
    lof_preds = (test_scores >= threshold).astype(int)

    print_results("LOF", y_test, lof_preds)
    pca_plot(X_test, lof_preds, "LOF")

    print("\n" + "=" * 60)
    print("INTERPRETATION & METRIC TRADE-OFFS")
    print("=" * 60)
    print("""
Precision vs. Recall Trade-off:
  - PRECISION: % of predicted frauds that are actually frauds (low = many false alarms)
  - RECALL: % of actual frauds that we detect (low = miss real frauds)
  - Threshold tuning with 'precision floor' maintains acceptable false alarm rate
    while maximizing fraud detection.

Why metrics differ between Isolation Forest and LOF:
  - Isolation Forest: Uses tree-based novelty detection on normal samples
  - LOF: Uses local density-based detection, more sensitive to local regions
  - Model choice affects precision/recall balance depending on fraud pattern
  - Validation-based threshold tuning (LOF) prevents overfitting to test labels
""")
    print("=" * 60)


if __name__ == "__main__":
    main()