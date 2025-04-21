import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# read the predictions from CSV files
nltk_df = pd.read_csv("nltk_predictions_1000.csv")
sklearn_df = pd.read_csv("sklearn_predictions_1000.csv")

# calculate accuracy for both models
nltk_accuracy = (nltk_df["true_label"] == nltk_df["predicted_label"]).mean()
sklearn_accuracy = (
    sklearn_df["true_label"] == sklearn_df["predicted_label"]
).mean()

print(f"NLTK Model Accuracy: {nltk_accuracy:.4f}")
print(f"Scikit-Learn TF-IDF Model Accuracy: {sklearn_accuracy:.4f}")


# plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, model_name):
    cm = confusion_matrix(
        true_labels, pred_labels, labels=["positive", "neutral", "negative"]
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["positive", "neutral", "negative"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


# plot NLTK model confusion matrix
plot_confusion_matrix(
    nltk_df["true_label"], nltk_df["predicted_label"], "NLTK (CountVectorizer)"
)

# plot Scikit-Learn model confusion matrix
plot_confusion_matrix(
    sklearn_df["true_label"],
    sklearn_df["predicted_label"],
    "Scikit-Learn (TF-IDF)",
)
