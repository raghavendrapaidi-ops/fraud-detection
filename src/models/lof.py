from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


def run_lof(X):

    print("Running LOF (with PCA + tuning)...")

    # 🔹 Step 1: Reduce dimensions (VERY important for LOF)
    pca = PCA(n_components=10, random_state=42)
    X_reduced = pca.fit_transform(X)

    # 🔹 Step 2: Tuned LOF
    model = LocalOutlierFactor(
        n_neighbors=50,        # bigger neighborhood = better density estimate
        contamination=0.02,    # slightly higher fraud expectation
        novelty=False
    )

    preds = model.fit_predict(X_reduced)

    # convert (-1 anomaly, 1 normal) → (1 fraud, 0 normal)
    preds = [1 if p == -1 else 0 for p in preds]

    return preds