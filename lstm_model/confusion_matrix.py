# ───────────────────────────────────────────────────────────────
# Compare three sentiment models with confusion‑matrix heat‑maps
# ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------
# 1. Load prediction CSVs
#    Each file must contain two columns:
#      • true labels
#      • predicted labels
# ----------------------------------------------------------------
FILES = {
    # "NLTK (CountVectorizer)" : "nltk_predictions.csv",
    # "TF‑IDF (MultinomNB)"    : "sklearn_predictions.csv",
    "TF Bi-LSTM"             : "tf_predictions.csv",
}

# Mapping from possible column names to standard names
STD_COLS = {
    "airline_sentiment":   "true",
    "true_label":          "true",
    "true_label_name":     "true",
    "predicted_sentiment": "pred",
    "predicted_label":     "pred",
}

models  = {}
for model_name, path in FILES.items():
    df = pd.read_csv(path)
    df = df.rename(columns={c: STD_COLS.get(c, c) for c in df.columns})
    if {"true", "pred"} <= set(df.columns):
        models[model_name] = df[["true", "pred"]]
    else:
        raise ValueError(f"{path} must contain ground‑truth and prediction columns")

# ----------------------------------------------------------------
# 2. Global label order (modify if needed)
# ----------------------------------------------------------------
LABELS = ["negative", "neutral", "positive"]   # or list(label_encoder.classes_)

# ----------------------------------------------------------------
# 3. Helper: plot + save confusion matrix
# ----------------------------------------------------------------
def plot_confusion_matrix(true_labels,
                          pred_labels,
                          model_name,
                          accuracy,
                          cmap="Blues",
                          save_path=None):
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm_pct,
                annot=True, fmt=".1f", cmap=cmap,
                xticklabels=LABELS, yticklabels=LABELS,
                cbar=True, vmin=0, vmax=100)
    plt.title(f"{model_name}\nConfusion Matrix (Acc {accuracy*100:.1f} %)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

# ----------------------------------------------------------------
# 4. Loop over models and draw/save
# ----------------------------------------------------------------
CMAP_PICK = {"NLTK (CountVectorizer)": "Blues",
             "TF‑IDF (MultinomNB)"   : "Greens",
             "TF Bi‑LSTM"            : "Purples"}

for name, df in models.items():
    acc = (df["true"] == df["pred"]).mean()
    plot_confusion_matrix(df["true"], df["pred"],
                          model_name=name,
                          accuracy=acc,
                          cmap=CMAP_PICK.get(name, "Blues"),
                          save_path=f"{name.lower().replace(' ', '_')}.png")

# ----------------------------------------------------------------
# 5. Quick sanity checks
# ----------------------------------------------------------------
for name, df in models.items():
    acc = (df["true"] == df["pred"]).mean()
    assert 0 <= acc <= 1, f"{name} accuracy out of bounds"
print("All accuracies within [0, 1]; confusion matrices generated.")
