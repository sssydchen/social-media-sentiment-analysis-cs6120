import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# read the predictions from CSV files
nltk_df = pd.read_csv("nltk_predictions_5000.csv")
sklearn_df = pd.read_csv("sklearn_predictions_5000.csv")

# calculate accuracy for both models
nltk_accuracy = (
    nltk_df["airline_sentiment"] == nltk_df["predicted_sentiment"]
).mean()
sklearn_accuracy = (
    sklearn_df["airline_sentiment"] == sklearn_df["predicted_sentiment"]
).mean()

print(f"NLTK Model Accuracy: {nltk_accuracy:.4f}")
print(f"Scikit-Learn TF-IDF Model Accuracy: {sklearn_accuracy:.4f}")


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


# plot and save NLTK model confusion matrix
plot_confusion_matrix(
    nltk_df["airline_sentiment"],
    nltk_df["predicted_sentiment"],
    "NLTK (CountVectorizer)",
    nltk_accuracy,
    cmap_style="Blues",
    save_path="nltk.png",
)

# plot and save Scikit-Learn model confusion matrix
plot_confusion_matrix(
    sklearn_df["airline_sentiment"],
    sklearn_df["predicted_sentiment"],
    "Scikit-Learn (TF-IDF)",
    sklearn_accuracy,
    cmap_style="Greens",
    save_path="sklearn.png",
)


# ===============================
# Unit Test for plot_confusion_matrix function
# ===============================
def test_plot_confusion_matrix():
    # Small dummy data
    true = ["positive", "neutral", "negative", "positive", "neutral"]
    pred = ["positive", "positive", "negative", "neutral", "neutral"]

    try:
        plot_confusion_matrix(
            true,
            pred,
            model_name="Dummy Model",
            accuracy=(np.array(true) == np.array(pred)).mean(),
            cmap_style="Oranges",
            save_path=None,  # Do not save during test
        )
        # If no exception occurs, the test passes
        print("test_plot_confusion_matrix passed.")
    except Exception as e:
        print(f"test_plot_confusion_matrix failed: {e}")


# Run unit test
test_plot_confusion_matrix()

# Additional basic accuracy check
assert 0 <= nltk_accuracy <= 1, "nltk_accuracy is out of bounds (0-1)"
assert 0 <= sklearn_accuracy <= 1, "sklearn_accuracy is out of bounds (0-1)"
