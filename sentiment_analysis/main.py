from preprocessing import clean_text
from dataset import load_and_split
from evaluate import print_report, results
from save_predictions import save_predictions
from confusion_matrix import plot_confusion_matrix
from models.nb_model import NBModel
from models.lr_model import LRModel
from models.lstm_model import LSTMModel
from models.bert_model import BERTModel
import pandas as pd

# 1. Load and split
X_train, X_test, y_train, y_test = load_and_split("Tweets.csv")


# 2. Clean text
X_train = [clean_text(x) for x in X_train]
X_test = [clean_text(x) for x in X_test]


# 3. Train & evaluate each model
# ====================================
# Naive Bayes
# ====================================
nb = NBModel()
nb.fit(X_train, y_train)
y_nb = nb.predict(X_test)
print_report("Naive Bayes", y_test, y_nb)
save_predictions("nltk", X_test, y_test, y_nb)

# ====================================
# Logistic Regression
# ====================================
lr = LRModel()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)
print_report("Logistic Regression", y_test, y_lr)
save_predictions("logreg", X_test, y_test, y_lr)

# ====================================
# Bi-LSTM
# ====================================
lstm = LSTMModel()
lstm.fit(X_train, y_train)
y_lstm = lstm.predict(X_test)
print_report("Bi-LSTM", y_test, y_lstm)
save_predictions("Bi-LSTM", X_test, y_test, y_lstm)

# ====================================
# BERT
# ====================================
bert = BERTModel()
bert.fit(X_train, y_train)
y_bert = bert.predict(X_test)
print_report("BERT", y_test, y_bert)
save_predictions("BERT", X_test, y_test, y_bert)


# 4. Generate confusion matrices
# read the predictions from CSV files
nb_df = pd.read_csv("nltk_predictions.csv")
lr_df = pd.read_csv("logreg_predictions.csv")
lstm_df = pd.read_csv("tf_predictions.csv")
bert_df = pd.read_csv("bert_predictions.csv")

# plot and save NLTK model confusion matrix
plot_confusion_matrix(
    nb_df["true_label"],
    nb_df["predicted_label"],
    "NLTK (CountVectorizer)",
    accuracy=results["Naive Bayes"],
    cmap_style="Blues",
    save_path="nltk_matrix.png",
)

# plot and save Scikit-Learn model confusion matrix
plot_confusion_matrix(
    lr_df["true_label"],
    lr_df["predicted_label"],
    "Logistic Regression (TF-IDF)",
    accuracy=results["Logistic Regression"],
    cmap_style="Greens",
    save_path="logreg_matrix.png",
)

# plot and save Tf model confusion matrices
plot_confusion_matrix(
    lstm_df["true_label"],
    lstm_df["predicted_label"],
    "Tf Model",
    accuracy=results["Bi-LSTM"],
    cmap_style="Oranges",
    save_path="tf_matrix.png",
)

# plot and save Bert model confusion matrices
plot_confusion_matrix(
    bert_df["true_label"],
    bert_df["predicted_label"],
    "Bert Model",
    accuracy=results["BERT"],
    cmap_style="Purples",
    save_path="bert_matrix.png",
)
