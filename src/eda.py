import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOT_PATH


def plot_class_distribution(df):
    sns.countplot(x="Class", data=df)
    plt.title("Class Imbalance")
    plt.savefig(PLOT_PATH + "class_distribution.png")
    plt.close()
