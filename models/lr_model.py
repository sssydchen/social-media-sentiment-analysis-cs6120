from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ===============================
# Class: LRModel
# What it does:
#   - A Logistic Regression text classification model using TfidfVectorizer.
#   - Provides fit and predict methods for text input.
# ===============================
class LRModel:
    # ===============================
    # Function: __init__
    # What it does:
    #   - Initialize the TfidfVectorizer and LogisticRegression classifier.
    #
    # Inputs:
    #   - None
    #
    # Output:
    #   - Initialized LRModel instance with vectorizer and classifier
    # ===============================
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(max_iter=200)

    # ===============================
    # Function: fit
    # What it does:
    #   - Train the model using training data and labels.
    #
    # Inputs:
    #   - X_train: list of text samples for training
    #   - y_train: list or array of labels corresponding to X_train
    #
    # Output:
    #   - None (updates the model's internal state)
    # ===============================
    def fit(self, X_train, y_train):
        Xv = self.vectorizer.fit_transform(X_train)
        self.clf.fit(Xv, y_train)

    # ===============================
    # Function: predict
    # What it does:
    #   - Predict labels for new text inputs.
    #
    # Inputs:
    #   - X: list of text samples to predict
    #
    # Output:
    #   - Array of predicted labels
    # ===============================
    def predict(self, X):
        Xv = self.vectorizer.transform(X)
        return self.clf.predict(Xv)
