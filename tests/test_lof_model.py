import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.models.lof import run_lof


def test_lof_supervised_tuning_improves_detection_on_separable_data():
    rng = np.random.default_rng(42)

    normal = rng.normal(0, 1.0, size=(500, 6))
    fraud = rng.normal(4.5, 1.0, size=(40, 6))

    X = np.vstack([normal, fraud])
    y = np.array([0] * len(normal) + [1] * len(fraud))

    preds = np.array(run_lof(X, y))

    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    assert recall >= 0.8
    assert precision >= 0.5


def test_lof_unsupervised_mode_returns_binary_labels():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(120, 4))

    preds = run_lof(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
