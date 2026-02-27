from src.data_loader import load_data
from src.preprocessing import scale_data
from src.eda import plot_class_distribution
from src.models.isolation_forest import run_isolation_forest
from src.models.lof import run_lof
from src.visualize import pca_plot

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


def print_results(name, y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print("\n" + "=" * 45)
    print(f"{name} RESULTS")
    print("=" * 45)

    print(f"Total transactions : {len(y_true)}")
    print(f"Detected frauds    : {tp}")
    print(f"Missed frauds      : {fn}")
    print(f"False alarms       : {fp}")
    print(f"Accuracy           : {accuracy*100:.2f}%")
    print(f"Precision          : {precision*100:.2f}%")
    print(f"Fraud detection rate (Recall): {recall*100:.2f}%")


def main():

    print("Loading data...")
    df = load_data()

    print("Plotting class distribution...")
    plot_class_distribution(df)

    print("Scaling features...")
    df = scale_data(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("Running Isolation Forest...")
    iso_preds = run_isolation_forest(X, y)
    print_results("Isolation Forest", y, iso_preds)
    pca_plot(X, iso_preds, "Isolation Forest")

    print("Running LOF...")
    lof_preds = run_lof(X)
    print_results("LOF", y, lof_preds)
    pca_plot(X, lof_preds, "LOF")


if __name__ == "__main__":
    main()