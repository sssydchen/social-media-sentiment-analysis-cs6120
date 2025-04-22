# ───────────────────────────────────────────────────────────────
# Compare sentiment models with confusion‑matrix heat‑maps
# ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------
# 1. Load prediction CSVs
#    Each file must contain two columns:
#      • airline_sentiment (ground truth)
#      • bert_pred (model prediction)
# ----------------------------------------------------------------
FILES = {
    "BERT-1000data": "bert_1000data.csv",
}

# ----------------------------------------------------------------
# 2. Read each CSV and standardize column names
# ----------------------------------------------------------------
models = {}
for model_name, path in FILES.items():
    df = pd.read_csv(path)
    # Rename the columns from your CSV to the script's internal names
    df = df.rename(columns={
        "airline_sentiment":   "true",
        "bert_pred": "pred"
    })
    # Check presence of required columns
    if {"true", "pred"} <= set(df.columns):
        models[model_name] = df[["true", "pred"]]
    else:
        raise ValueError(
            f"{path} must contain 'airline_sentiment' and 'bert_pred' columns"
        )

# ----------------------------------------------------------------
# 3. Define label order (adjust if needed)
# ----------------------------------------------------------------
LABELS = ["negative", "neutral", "positive"]

# ----------------------------------------------------------------
# 4. Plotting function
# ----------------------------------------------------------------
def plot_confusion_matrix(true_labels,
                          pred_labels,
                          model_name,
                          accuracy,
                          cmap="Blues",
                          save_path=None):
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm_pct,
        annot=True, fmt=".1f",
        cmap=cmap,
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=True,
        vmin=0, vmax=100
    )
    plt.title(f"{model_name}\nConfusion Matrix (Acc {accuracy*100:.1f} %) )")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

# ----------------------------------------------------------------
# 5. Generate and save the plots
# ----------------------------------------------------------------
for name, df in models.items():
    acc = (df["true"] == df["pred"]).mean()
    fname = f"{name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(
        df["true"], df["pred"],
        model_name=name,
        accuracy=acc,
        save_path=fname
    )

# ----------------------------------------------------------------
# 6. Sanity check
# ----------------------------------------------------------------
for name, df in models.items():
    acc = (df["true"] == df["pred"]).mean()
    assert 0 <= acc <= 1, f"{name} accuracy out of bounds"
print("All accuracies within [0, 1]; confusion matrices generated.")
