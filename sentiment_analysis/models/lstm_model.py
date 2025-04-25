import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


tf.random.set_seed(42)
np.random.seed(42)


class LSTMModel:
    def __init__(self, max_vocab=10000, max_len=100, embed_dim=64):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X_train, y_train):
        # Encode labels to integers
        y_enc = self.label_encoder.fit_transform(y_train)
        classes = self.label_encoder.classes_
        num_classes = len(classes)
        y_cat = to_categorical(y_enc, num_classes)

        # Tokenize & pad sequences
        self.tokenizer.fit_on_texts(X_train)
        X_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X_train),
            maxlen=self.max_len
        )

        # Compute class weights
        cw = compute_class_weight(
            'balanced', classes=np.unique(y_enc), y=y_enc
        )
        class_weights = dict(enumerate(cw))

        # Build and compile model
        model = Sequential([
            Embedding(
                input_dim=self.max_vocab,
                output_dim=self.embed_dim,
                input_length=self.max_len
            ),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Train with early stopping
        model.fit(
            X_seq, y_cat,
            epochs=15,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )],
            verbose=2
        )
        self.model = model

    def predict(self, X):
        # Convert texts to padded sequences
        X_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X),
            maxlen=self.max_len
        )
        # Predict and decode labels
        y_prob = self.model.predict(X_seq)
        y_idx = y_prob.argmax(axis=1)
        return self.label_encoder.inverse_transform(y_idx)