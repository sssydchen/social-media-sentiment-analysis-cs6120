# import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertForSequenceClassification

# For compatibility with Transformers + Keras3, use tf-keras optimizers
try:
    from tf_keras.optimizers import Adam as TFAdam
except ImportError:
    from tensorflow.keras.optimizers import Adam as TFAdam

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load & preprocess
CSV_PATH = 'Tweets.csv'  # or 'movie.csv' for movie reviews
DF = pd.read_csv(CSV_PATH)

# simple cleaning + NLTK tokenization + stopword removal (keep negations)
STOP = set(stopwords.words('english'))
NEG = {'no','not','never',"n't"}
KEEP = STOP - NEG

def clean(text):
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"@\w+|#", "", t)
    t = re.sub(r"[^a-zA-Z]", " ", t)
    toks = word_tokenize(t)
    return ' '.join([w for w in toks if w not in KEEP])

DF['clean'] = DF['text'].astype(str).apply(clean)

# features & labels
X = DF['clean'].values
# adjust column name for label if using movie.csv
label_col = 'airline_sentiment' if 'airline_sentiment' in DF.columns else 'label'
y = DF[label_col].values

# Split once for all models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Utility to collect and print reports
results = {}

def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    results[name] = acc

# NLTK + Naive Bayes
vect_nb = CountVectorizer()
X_tr_nb = vect_nb.fit_transform(X_train)
X_te_nb = vect_nb.transform(X_test)
nb = MultinomialNB().fit(X_tr_nb, y_train)
y_pred_nb = nb.predict(X_te_nb)
evaluate('Naive Bayes', y_test, y_pred_nb)

# TF-IDF + Logistic Regression
vect_lr = TfidfVectorizer()
X_tr_lr = vect_lr.fit_transform(X_train)
X_te_lr = vect_lr.transform(X_test)
lr = LogisticRegression(max_iter=200).fit(X_tr_lr, y_train)
y_pred_lr = lr.predict(X_te_lr)
evaluate('Logistic Regression', y_test, y_pred_lr)

# TensorFlow Bi-LSTM
# encode labels
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_train)
y_te_enc = le.transform(y_test)
num_classes = len(le.classes_)
y_tr_cat = to_categorical(y_tr_enc, num_classes)
y_te_cat = to_categorical(y_te_enc, num_classes)

# Tokenize + pad
MAX_VOCAB=10000; MAX_LEN=100; EMBED=64
tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tok.fit_on_texts(X_train)
X_tr_seq = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_te_seq = pad_sequences(tok.texts_to_sequences(X_test),  maxlen=MAX_LEN)

# Build model
tf.random.set_seed(42); np.random.seed(42)
model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=EMBED, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# class weights
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_tr_enc), y=y_tr_enc)
class_weights = dict(enumerate(cw))

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_tr_seq, y_tr_cat, epochs=15, batch_size=32,
          validation_split=0.2, class_weight=class_weights, callbacks=[es], verbose=2)
y_pred_lstm = model.predict(X_te_seq).argmax(axis=1)
evaluate('Bi-LSTM', y_te_enc, y_pred_lstm)

# BERT fine-tuning
bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
train_enc = bert_tok(list(X_train), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_enc  = bert_tok(list(X_test),  truncation=True, padding=True, max_length=128, return_tensors='tf')

train_ds = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_tr_enc)).batch(16)
test_ds  = tf.data.Dataset.from_tensor_slices((dict(test_enc),  y_te_enc )).batch(16)

# Load model & compile using tf-keras optimizer for compatibility
bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
bert.compile(
    optimizer=TFAdam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
bert.fit(train_ds, epochs=3, verbose=2)
y_pred_bert = np.argmax(bert.predict(test_ds).logits, axis=1)
evaluate('BERT', y_te_enc, y_pred_bert)

# Summary
print("\nFinal Accuracies:")
for m, a in results.items(): print(f"{m}: {a:.3f}")

