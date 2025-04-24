import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# read the predictions from CSV files
nltk_df = pd.read_csv("nltk_predictions.csv")
logreg_df = pd.read_csv("logreg_predictions.csv")
tf_df = pd.read_csv("tf_predictions.csv")
bert_df = pd.read_csv("bert_predictions.csv")

# calculate accuracy for both models
nltk_accuracy = (
    nltk_df["airline_sentiment"] == nltk_df["predicted_sentiment"]
).mean()
logreg_accuracy = (
    logreg_df["airline_sentiment"] == logreg_df["predicted_sentiment"]
).mean()
tf_accuracy = (
    tf_df["airline_sentiment"] == tf_df["predicted_sentiment"]
).mean()
bert_accuracy = (
    bert_df["airline_sentiment"] == bert_df["predicted_sentiment"]
).mean()

# print the accuracy results
print(f"NLTK Model Accuracy: {nltk_accuracy:.4f}")
print(f"Logistic Regression Model Accuracy: {logreg_accuracy:.4f}")
print(f"Tf Model Accuracy: {tf_accuracy:.4f}")
print(f"Bert Model Accuracy: {bert_accuracy:.4f}")


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
    save_path="nltk_matrix.png",
)

# plot and save Scikit-Learn model confusion matrix
plot_confusion_matrix(
    logreg_df["airline_sentiment"],
    logreg_df["predicted_sentiment"],
    "Logistic Regression (TF-IDF)",
    logreg_accuracy,
    cmap_style="Greens",
    save_path="logreg_matrix.png",
)

# plot and save Tf model confusion matrices
plot_confusion_matrix(
    tf_df["airline_sentiment"],
    tf_df["predicted_sentiment"],
    "Tf Model",
    tf_accuracy,
    cmap_style="Oranges",
    save_path="tf_matrix.png",
)

# plot and save Bert model confusion matrices
plot_confusion_matrix(
    bert_df["airline_sentiment"],
    bert_df["predicted_sentiment"],
    "Bert Model",
    bert_accuracy,
    cmap_style="Purples",
    save_path="bert_matrix.png",
)


# ===============================
# Unit Test for plot_confusion_matrix function
# ===============================
matplotlib.use("Agg")  # Use non-interactive backend for testing


def test_plot_confusion_matrix():
    # Dummy true and predicted labels
    true = ["positive", "neutral", "negative", "positive", "neutral"]
    pred = ["positive", "positive", "negative", "neutral", "neutral"]

    # Calculate dummy accuracy
    accuracy = (np.array(true) == np.array(pred)).mean()

    try:
        plot_confusion_matrix(
            true_labels=true,
            pred_labels=pred,
            model_name="Dummy Model",
            accuracy=accuracy,
            cmap_style="Oranges",
            save_path=None,  # Not saving the plot during test
        )
        print("✅ test_plot_confusion_matrix passed.")
    except Exception as e:
        print(f"❌ test_plot_confusion_matrix failed: {e}")


# Run unit test
test_plot_confusion_matrix()

# Ensure all accuracy values are within bounds
for model_name, acc in [
    ("nltk_accuracy", nltk_accuracy),
    ("logreg_accuracy", logreg_accuracy),
    ("tf_accuracy", tf_accuracy),
    ("bert_accuracy", bert_accuracy),
]:
    assert 0 <= acc <= 1, f"{model_name} is out of bounds (0-1)"
