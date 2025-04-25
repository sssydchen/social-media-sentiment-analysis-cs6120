import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# ===============================
# Function: plot_confusion_matrix
# What it does:
#   - Plot a confusion matrix heatmap (in percentage) given true and predicted labels.
#
# Inputs:
#   - true_labels: Series or list of ground truth labels
#   - pred_labels: Series or list of predicted labels
#   - model_name: String, name of the model for title display
#   - accuracy: Float, accuracy score for the model (between 0 and 1)
#   - cmap_style: String, matplotlib colormap style
#   - save_path: String, file path to save the figure
#
# Output:
#   - Displays the confusion matrix heatmap, optionally saves the figure to a file
# ===============================
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
