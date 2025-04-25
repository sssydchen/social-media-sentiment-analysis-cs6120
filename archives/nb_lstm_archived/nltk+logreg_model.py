import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the dataset
csv_file = "Tweets.csv"
df = pd.read_csv(csv_file)

# Select necessary columns
df = df[["tweet_id", "airline_sentiment", "text"]]


# Preprocessing function
STOP_WORDS = set(stopwords.words("english"))
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "cannot",
    "can't",
    "won't",
    "n't",
    "never",
}
CUSTOM_STOPWORDS = STOP_WORDS - NEGATION_WORDS


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    tokens = word_tokenize(text.strip())
    tokens = [word for word in tokens if word not in CUSTOM_STOPWORDS]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)


# Apply preprocessing
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Train/test split
X = df["cleaned_text"]
y = df["airline_sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### 1. NLTK Model (CountVectorizer + Naive Bayes) ###
vectorizer_nltk = CountVectorizer()
X_train_nltk = vectorizer_nltk.fit_transform(X_train)
X_test_nltk = vectorizer_nltk.transform(X_test)

model_nltk = MultinomialNB()
model_nltk.fit(X_train_nltk, y_train)
y_pred_nltk = model_nltk.predict(X_test_nltk)

print("\nNLTK Model Evaluation (CountVectorizer + NB):")
print(classification_report(y_test, y_pred_nltk))

nltk_output_df = pd.DataFrame(
    {
        "text": X_test.values,
        "airline_sentiment": y_test.values,
        "predicted_sentiment": y_pred_nltk,
    }
)
nltk_output_df.to_csv("nltk_predictions.csv", index=False)
print("Saved NLTK predictions to 'nltk_predictions.csv'")


### 2. Logistic Regression Model (TF-IDF + N-gram) ###
vectorizer_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

model_logreg = LogisticRegression(max_iter=1000)
model_logreg.fit(X_train_tfidf, y_train)
y_pred_logreg = model_logreg.predict(X_test_tfidf)

print("\nLogistic Regression Model Evaluation (TF-IDF + N-gram):")
print(classification_report(y_test, y_pred_logreg))

logreg_output_df = pd.DataFrame(
    {
        "text": X_test.values,
        "airline_sentiment": y_test.values,
        "predicted_sentiment": y_pred_logreg,
    }
)
logreg_output_df.to_csv("logreg_predictions.csv", index=False)
print("Saved Logistic Regression predictions to 'logreg_predictions.csv'")


# ===============================
# Unit Test for updated preprocess_text function
# ===============================


def test_preprocess_text():
    # Case 1: Normal sentence
    assert (
        preprocess_text("I love flying with United Airlines!")
        == "love flying united airline"
    )

    # Case 2: Text with URL (URL removed, https kept due to apostrophe-safe regex)
    assert (
        preprocess_text("Check out our flights at https://united.com!")
        == "check flight"
    )

    # Case 3: Text with mention and hashtag
    assert (
        preprocess_text("@UnitedAirlines #badservice delayed again")
        == "badservice delayed"
    )

    # Case 4: Empty string
    assert preprocess_text("") == ""

    # Case 5: Only non-alphabetic characters
    assert preprocess_text("1234567890 !!! ???") == ""

    # Case 6: Sentence with negation
    assert (
        preprocess_text("I don't like delayed flights")
        == "n't like delayed flight"
    )

    print("All updated preprocess_text tests passed.")


# Run the tests
test_preprocess_text()
