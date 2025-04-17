import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download("punkt")
nltk.download("stopwords")

# Load the dataset
csv_file = "Tweets.csv"
df = pd.read_csv(csv_file).head(1000)

# Select relevant columns
df = df[
    [
        "tweet_id",
        "airline_sentiment",
        "airline_sentiment_confidence",
        "negativereason",
        "negativereason_confidence",
        "airline",
        "airline_sentiment_gold",
        "name",
        "negativereason_gold",
        "retweet_count",
        "text",
        "tweet_coord",
        "tweet_created",
        "tweet_location",
        "user_timezone",
    ]
]


# Data Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\@\\w+|\\#", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


# Apply preprocessing
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Split data into training and testing sets
X = df["cleaned_text"]
y = df["airline_sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### 1. NLTK-based Model (Naive Bayes) ###
vectorizer_nltk = CountVectorizer()
X_train_nltk = vectorizer_nltk.fit_transform(X_train)
X_test_nltk = vectorizer_nltk.transform(X_test)

model_nltk = MultinomialNB()
model_nltk.fit(X_train_nltk, y_train)

y_pred_nltk = model_nltk.predict(X_test_nltk)
print("\nNLTK Model Evaluation:")
print(classification_report(y_test, y_pred_nltk))

# Save NLTK model predictions
nltk_output_df = pd.DataFrame(
    {
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred_nltk,
    }
)
nltk_output_df.to_csv("nltk_predictions.csv", index=False)
print("Saved NLTK predictions to 'nltk_predictions.csv'")

### 2. Scikit-Learn Model (TF-IDF + Naive Bayes) ###
vectorizer_sklearn = TfidfVectorizer()
X_train_sklearn = vectorizer_sklearn.fit_transform(X_train)
X_test_sklearn = vectorizer_sklearn.transform(X_test)

model_sklearn = MultinomialNB()
model_sklearn.fit(X_train_sklearn, y_train)

y_pred_sklearn = model_sklearn.predict(X_test_sklearn)
print("\nScikit-Learn Model Evaluation:")
print(classification_report(y_test, y_pred_sklearn))

# Save TF-IDF model predictions
sklearn_output_df = pd.DataFrame(
    {
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred_sklearn,
    }
)
sklearn_output_df.to_csv("sklearn_predictions.csv", index=False)
print("Saved Scikit-Learn predictions to 'sklearn_predictions.csv'")
