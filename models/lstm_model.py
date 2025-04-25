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
    """
    Bi-directional LSTM sentiment classifier.

    This class encapsulates a complete pipeline for training and
    evaluating a bidirectional LSTM on text data. It supports
    fitting on (X_train, y_train) and predicting labels for new texts.

    Attributes
    ----------
    max_vocab : int
        Maximum size of the vocabulary (most frequent tokens). Tokens outside
        this range are mapped to an out-of-vocabulary token.
    max_len : int
        All sequences are padded or truncated to this fixed length.
    embed_dim : int
        Dimensionality of the word embedding vectors.
    tokenizer : Tokenizer
        Keras tokenizer that handles text-to-integer mapping.
    label_encoder : LabelEncoder
        Scikit-learn encoder mapping string labels to integers and back.
    model : Sequential
        The compiled Keras model after training; None before fit().
    """
    def __init__(self, max_vocab=10000, max_len=100, embed_dim=64):
        """
        Initialize the LSTMModel with hyperparameters.

        Parameters
        ----------
        max_vocab : int, default=10000
            The maximum number of words to keep in the vocabulary.
        max_len : int, default=100
            The length to which all sequences will be padded or truncated.
        embed_dim : int, default=64
            The size of the embedding vectors.
        """
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X_train, y_train):
        """
        Train the bi-directional LSTM model on the provided data.

        This method:
          1. Encodes string labels to integers.
          2. One-hot encodes the integer labels.
          3. Tokenizes and pads the training texts.
          4. Computes class weights to handle imbalance.
          5. Builds, compiles, and fits the Keras model with early stopping.

        Parameters
        ----------
        X_train : list or array-like of str
            Cleaned text samples for training.
        y_train : list or array-like of str
            Corresponding sentiment labels (strings) for training.

        Returns
        -------
        None
            After fitting, the trained model is stored in self.model.
        """
        # Encode labels to integers
        y_enc = self.label_encoder.fit_transform(y_train)
        num_classes = len(self.label_encoder.classes_)
        y_cat = to_categorical(y_enc, num_classes)

        # Tokenize & pad sequences to uniform length
        self.tokenizer.fit_on_texts(X_train)
        X_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X_train),
            maxlen=self.max_len
        )

        # Compute balanced class weights
        cw = compute_class_weight(
            'balanced', classes=np.unique(y_enc), y=y_enc
        )
        class_weights = dict(enumerate(cw))

        # Build LSTM model architecture
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

        # Train with early stopping on validation loss
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
        """
        Generate sentiment predictions for new text samples.

        This method:
          1. Tokenizes and pads the input texts using the trained tokenizer.
          2. Uses the trained model to predict class probabilities.
          3. Converts predicted probability indices back to original labels.

        Parameters
        ----------
        X : list or array-like of str
            Cleaned text samples to classify.

        Returns
        -------
        array-like of str
            Predicted sentiment labels corresponding to each input sample.
        """
        # Convert texts to padded integer sequences
        X_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X),
            maxlen=self.max_len
        )
        # Predict class probabilities
        y_prob = self.model.predict(X_seq)
        # Pick index of highest probability, then map back to label
        y_idx = y_prob.argmax(axis=1)
        return self.label_encoder.inverse_transform(y_idx)