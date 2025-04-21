import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# read the predictions from CSV files
nltk_df = pd.read_csv("nltk_predictions_4000.csv")
sklearn_df = pd.read_csv("sklearn_predictions_4000.csv")

# calculate accuracy for both models
nltk_accuracy = (nltk_df["true_label"] == nltk_df["predicted_label"]).mean()
sklearn_accuracy = (
    sklearn_df["true_label"] == sklearn_df["predicted_label"]
).mean()

print(f"NLTK Model Accuracy: {nltk_accuracy:.4f}")
print(f"Scikit-Learn TF-IDF Model Accuracy: {sklearn_accuracy:.4f}")


# plot and save confusion matrix with accuracy in title
def plot_confusion_matrix(
    true_labels,
    pred_labels,
    model_name,
    accuracy,
    cmap_style="Blues",
    save_path=None,
):
    labels = ["positive", "neutral", "negative"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_percent = (
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    )  # convert to percentages

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap=cmap_style,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title(
        f"Confusion Matrix - {model_name} (Accuracy: {accuracy*100:.1f}%)\n(Percentage %)"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


# plot and save NLTK model confusion matrix
plot_confusion_matrix(
    nltk_df["true_label"],
    nltk_df["predicted_label"],
    "NLTK (CountVectorizer)",
    nltk_accuracy,
    cmap_style="Blues",
    save_path="nltk.png",
)

# plot and save Scikit-Learn model confusion matrix
plot_confusion_matrix(
    sklearn_df["true_label"],
    sklearn_df["predicted_label"],
    "Scikit-Learn (TF-IDF)",
    sklearn_accuracy,
    cmap_style="Greens",
    save_path="sklearn.png",
)
