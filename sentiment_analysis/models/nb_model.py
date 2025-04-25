from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# ===============================
# Class: NBModel
# What it does:
#   - A Naive Bayes text classification model using CountVectorizer.
#   - Provides fit and predict methods for text input.
# ===============================
class NBModel:
    # ===============================
    # Function: __init__
    # What it does:
    #   - Initialize the CountVectorizer and MultinomialNB classifier.
    #
    # Inputs:
    #   - None
    #
    # Output:
    #   - Initialized NBModel instance with vectorizer and classifier
    # ===============================
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()

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
