from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.config import PLOT_PATH


def pca_plot(X, preds, name):
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    plt.scatter(comp[:, 0], comp[:, 1], c=preds, s=4)
    plt.title(name + " PCA")
    plt.savefig(PLOT_PATH + f"{name}_pca.png")
    plt.close()
