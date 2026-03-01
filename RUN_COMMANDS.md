===============================================================================
FRAUD DETECTION PIPELINE - REFACTORING COMPLETE
===============================================================================

Data Leakage Fixed ✓
Proper Train/Val/Test Evaluation Implemented ✓
Comprehensive Test Suite Added ✓
All Tests Passing (17/17) ✓

===============================================================================
EXACT COMMANDS TO RUN
===============================================================================

1. Run Full Test Suite
─────────────────────────────────────────────────────────────────────────────

Command:
  C:/Users/ragha/fraud-detection/.venv/Scripts/python.exe -m pytest -q

Expected Output:
  17 passed in ~4s

What it tests:
  ✓ Old test: test_fraud_pipeline.py (5 tests from fraud_pipeline.py module)
  ✓ New tests: test_lof_model.py (12 tests covering all refactored LOF functionality)
  
Tests verify:
  • No data leakage in threshold tuning
  • Unsupervised and supervised LOF modes work correctly
  • Precision floor constraint is enforced
  • Data type handling (numpy arrays, pandas DataFrames)
  • Threshold optimization logic


2. Run Main Pipeline
─────────────────────────────────────────────────────────────────────────────

Command:
  C:/Users/ragha/fraud-detection/.venv/Scripts/python.exe main.py

Expected Output:
  Loading data...
  Plotting class distribution...
  Scaling features...

  Data split summary:
    Train: XXX samples (YYY frauds)
    Val:   ZZZ samples (WWW frauds)
    Test:  AAA samples (BBB frauds)

  ============================================================
  Training Isolation Forest on normal samples only...
  ============================================================

  ============================================================
  Isolation Forest - TEST SET RESULTS
  ============================================================
  Total transactions : AAA
  True positives (TP): ???
  False negatives (FN): ???
  False positives (FP): ???
  True negatives (TN): ???
  Accuracy           : X.XX%
  Precision          : X.XX%
  Recall (Detection) : X.XX%
  F1 Score           : X.XXXX
  ============================================================

  ============================================================
  Training LOF on normal samples only...
  ============================================================

  ============================================================
  LOF - TEST SET RESULTS
  ============================================================
  Total transactions : AAA
  True positives (TP): ???
  False negatives (FN): ???
  False positives (FP): ???
  True negatives (TN): ???
  Accuracy           : X.XX%
  Precision          : X.XX%
  Recall (Detection) : X.XX%
  F1 Score           : X.XXXX
  ============================================================

  ============================================================
  INTERPRETATION & METRIC TRADE-OFFS
  ============================================================
  
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


Notes:
  - Requires CSV data at: data/creditcard.csv
  - Requires features: Time, Amount, V1-V28, Class
  - Generates plots in: outputs/plots/ (if outputs dir exists)


===============================================================================
KEY IMPROVEMENTS SUMMARY
===============================================================================

BEFORE (Data Leakage):
  ❌ Threshold tuned on test labels directly
  ❌ No separation between model development and evaluation
  ❌ LOF performance artificially inflated
  ❌ Mixed concerns in run_lof() function
  ❌ Unclear evaluation methodology

AFTER (No Leakage):
  ✓ Stratified train/validation/test split
  ✓ Threshold tuned on validation set ONLY
  ✓ Test set held-out for final evaluation
  ✓ Clean separation of concerns
  ✓ Both models evaluated on same held-out test set
  ✓ Precision floor constraint maintains practical false alarm rates
  ✓ Comprehensive test coverage (12 new tests)
  ✓ Type hints and detailed docstrings throughout
  ✓ Backward-compatible API (run_lof() still works)
  ✓ Robust input handling (numpy, pandas)


===============================================================================
METRIC TRADE-OFFS EXPLAINED
===============================================================================

Precision vs. Recall

PRECISION = TP / (TP + FP)
  "Of frauds we predicted, how many were real?"
  High precision = Few false alarms
  Low precision = Many false alarms

RECALL = TP / (TP + FN)
  "Of actual frauds, how many did we detect?"
  High recall = Catch more fraud
  Low recall = Miss fraud

Threshold's Effect:
  HIGHER THRESHOLD → Higher Precision, Lower Recall
  LOWER THRESHOLD → Lower Precision, Higher Recall

LOF's Precision Floor Strategy:
  Goal: Maximize recall while keeping precision >= 10%
  
  This means:
  - Accept up to 10 false alarms per true fraud (in expectation)
  - Conservative but practical approach
  - Adapts automatically based on validation data
  
  Example: If validation set has 100 frauds and 10,000 normals
    Best threshold might flag 1500 samples as fraud
    If 150 are real frauds (10% precision), recall = 150/100 = 1.5 (impossible, cap at actual)
    
  Better example: 150 flagged, 110 real frauds = 73% precision, 110%+ recall (capped at actuals)


Why Models Differ:

Isolation Forest:
  - Creates random decision trees on subsets of data
  - Isolates anomalies in trees (takes fewer splits for anomalies)
  - Gives consistent normality scores across regions
  - May miss local density anomalies

LOF (Local Outlier Factor):
  - Compares local density of a point vs its neighbors
  - Excellent for density-based anomalies
  - May vary significantly across regions
  - With validation tuning: Better recall on test data


Expected Results With Refactored Code:

If LOF recall >> Isolation Forest recall:
  → LOF threshold tuned for high recall on validation set
  → Works well if frauds have density-based patterns
  → False alarm rate controllable via precision_floor parameter

If Isolation Forest recall >> LOF recall:
  → Frauds form distinct groups in feature space
  → Isolation-based detection more reliable
  → More stable across different data regions

If metrics similar:
  → Both models learned similar patterns
  → Good sign of robust fraud signal
  → Can use either, or ensemble both


===============================================================================
VERIFYING NO DATA LEAKAGE
===============================================================================

Leakage-Free Pipeline Verified:

Stage 1: DATA SPLITTING
  Input: Full Dataset (Class = 0 or 1)
  Process: Stratified split to maintain fraud ratio
    60% → Train validation pool (300 samples if N=500 total)
    40% → Test set (200 samples) ← LOCKED AWAY
  Output: X_train, y_train | X_val, y_val | X_test, y_test

Stage 2: TRAINING
  Input: X_train, y_train
  - Extract: X_train_normal = X_train[y_train==0]
  - Train isolation_forest on X_train_normal
  - Train LOF on X_train_normal
  Output: Two trained models
  Key: y_train labels used only for extracting normal samples, NOT for tuning thresholds

Stage 3: VALIDATION-BASED THRESHOLD TUNING (LOF only)
  Input: X_val, y_val (validation set)
  - Get anomaly scores: model.decision_function(X_val)
  - Tune threshold to maximize recall with precision >= 0.10
  - Track best threshold
  Output: Tuned threshold
  Key: VALIDATION labels used, test labels NOT TOUCHED

Stage 4: EVALUATION (Final metrics)
  Input: X_test, y_test (held-out test set)
  Process:
    - iso_forest.predict(X_test) → predictions
    - lof_model with tuned threshold on X_test → predictions
    - Compare predictions vs y_test (ground truth)
    - Calculate metrics
  Output: Precision, Recall, F1, Confusion Matrix
  Key: Test labels used ONLY for final metric calculation, not model tuning

No Leakage Checklist:
  ✓ Test labels not used during training
  ✓ Test labels not used during threshold tuning
  ✓ Test labels used only for final evaluation
  ✓ Threshold optimized on VALIDATION set (separate from test)
  ✓ Both models evaluated on same held-out test set
  ✓ Stratified splits maintain fraud distribution across all sets


===============================================================================
FILE CHANGES AT A GLANCE
===============================================================================

1. src/models/lof.py
   Lines before: ~49
   Lines after:  ~219
   Major changes:
   - New: LOFResult, _to_numpy, _reduce_dimensionality, _select_n_neighbors
   - Replaced: _best_threshold_from_labels → _optimize_threshold_on_validation
   - New: fit_and_predict_lof(X, y, val_X, val_y, precision_floor, random_state)
   - Enhanced: run_lof() now just calls fit_and_predict_lof

2. main.py
   Lines before: ~33
   Lines after:  ~119
   Major changes:
   - New: Stratified data splitting (train 45%, val 15%, test 40%)
   - New: Separate Isolation Forest training block
   - New: Separate LOF training block with validation tuning
   - Enhanced: print_results() with confusion matrix and F1
   - New: Metric interpretation guide printed to console

3. tests/test_lof_model.py
   Lines before: ~40
   Lines after:  ~227
   Major changes:
   - New: TestDataConversion (2 tests)
   - New: TestUnsupervisedMode (2 tests)
   - New: TestSupervisedMode (2 tests)
   - New: TestValidationBasedThresholdTuning (2 tests) ← LEAKAGE PREVENTION ✓
   - New: TestThresholdOptimization (2 tests)
   - New: TestDataTypes (2 tests)
   - Old tests removed (replaced with comprehensive suite)


===============================================================================
BACKWARD COMPATIBILITY
===============================================================================

Old Code:
  from src.models.lof import run_lof
  preds = run_lof(X, y)  # Returns list of predictions

New Code Still Works:
  from src.models.lof import run_lof
  preds = run_lof(X, y)  # Still returns list of predictions
  
  The run_lof() function is a backward-compatible wrapper around
  the new fit_and_predict_lof() API.

New Code (Advanced):
  from src.models.lof import fit_and_predict_lof
  result = fit_and_predict_lof(
    X_train, y_train,
    val_X=X_val, val_y=y_val,
    precision_floor=0.10
  )
  predictions = result.predictions  # np.ndarray
  scores = result.anomaly_scores    # np.ndarray


===============================================================================
DEPENDENCIES
===============================================================================

Required Python Packages:
  - numpy
  - pandas
  - scikit-learn (sklearn)
  - matplotlib
  - seaborn
  - pytest (for testing)

Versions (tested on):
  - Python 3.13.2
  - All packages installed via pip, latest stable versions

Installation:
  C:/Users/ragha/fraud-detection/.venv/Scripts/python.exe -m pip install -r requirements.txt

Or manually:
  pip install numpy pandas scikit-learn matplotlib seaborn pytest


===============================================================================
TROUBLESHOOTING
===============================================================================

Issue: ModuleNotFoundError: No module named 'X'

Solution:
  C:/Users/ragha/fraud-detection/.venv/Scripts/python.exe -m pip install X

Specific packages:
  - matplotlib: for visualization
  - seaborn: for enhanced plots  
  - pytest: for running tests


Issue: Data file not found (data/creditcard.csv)

Solution:
  - Place creditcard.csv in data/ directory (or adjust DATA_PATH in config.py)
  - File should have columns: Time, Amount, V1-V28, Class
  - Class column: 0 = Normal, 1 = Fraud


Issue: Tests pass but main.py fails on data loading

Solution:
  - Verify data file exists and is readable
  - Check that all feature columns (V1-V28, Time, Amount, Class) are present
  - Run: python -c "import pandas as pd; df = pd.read_csv('data/creditcard.csv'); print(df.shape, df.columns)"


Issue: Different metrics than expected

Possible Reasons:
  - Different random seed in train/val/test split (change random_state=42)
  - Different data file version/subset  
  - Different sklearn version (may affect LOF/IsolationForest implementations)
  - Different data preprocessing (scaling is applied in pipeline)


===============================================================================
NEXT STEPS (OPTIONAL)
===============================================================================

To Further Improve:

1. Hyperparameter Tuning:
   - Adjust contamination parameter in Isolation Forest
   - Adjust precision_floor parameter in LOF
   - Adjust train/val/test split ratios

2. Feature Engineering:
   - Add interaction features
   - Add domain-specific features (time of day, merchant category, etc.)
   - Apply feature selection

3. Model Ensemble:
   - Create voting ensemble of Isolation Forest + LOF
   - Use soft predictions (probabilities) instead of hard predictions
   - Calibrate probability outputs

4. Cross-Validation:
   - Implement k-fold cross-validation on training set
   - Better estimate of model uncertainty
   - More stable hyperparameter estimates

5. Monitoring:
   - Track model performance over time
   - Detect data drift
   - Retrain periodically with new data

