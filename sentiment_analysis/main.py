from preprocessing import clean_text
from dataset import load_and_split
from evaluate import evaluate, results
from save_predictions import save_predictions
from models.nb_model import NBModel
from models.lr_model import LRModel
from models.lstm_model import LSTMModel
from models.bert_model import BERTModel

# 1. Load and split
X_train, X_test, y_train, y_test = load_and_split('Tweets.csv')

# 2. Clean text
X_train = [clean_text(x) for x in X_train]
X_test  = [clean_text(x) for x in X_test]

# 3. Train & evaluate each model
# Naive Bayes
nb = NBModel(); nb.fit(X_train, y_train)
y_nb = nb.predict(X_test); evaluate('Naive Bayes', y_test, y_nb)
save_predictions('nltk', X_test, y_test, y_nb)

# Logistic Regression
lr = LRModel(); lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

evaluate('Logistic Regression', y_test, y_lr)

# 1) Save out the CSV for your plotting script
save_predictions('logreg', X_test, y_test, y_lr)


# Bi-LSTM
lstm = LSTMModel(); lstm.fit(X_train, y_train)
y_lstm = lstm.predict(X_test); evaluate('Bi-LSTM', y_test, y_lstm)

# print the raw confusion matrix to console
cm = confusion_matrix(y_test, y_lstm, labels=lstm.label_encoder.classes_)
print('Bi-LSTM Confusion Matrix:\n', cm)

# BERT
bert = BERTModel(); bert.fit(X_train, y_train)
y_bert = bert.predict(X_test); evaluate('BERT', y_test, y_bert)

# 4. Print final results
print("Final accuracies:")
for name, acc in results.items():
    print(f"{name}: {acc:.3f}")