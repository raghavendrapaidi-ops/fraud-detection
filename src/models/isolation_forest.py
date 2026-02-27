from sklearn.ensemble import IsolationForest


def run_isolation_forest(X, y):

    print("Running Isolation Forest (normal-only training)...")

    # 🔹 Train only on NORMAL transactions
    X_train = X[y == 0]

    model = IsolationForest(
        n_estimators=300,     # more trees = better detection
        contamination=0.02,   # tune this (0.02–0.03 works well)
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    preds = model.predict(X)

    preds = [1 if p == -1 else 0 for p in preds]

    return preds