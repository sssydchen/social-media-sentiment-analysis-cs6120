import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification

# Use tf-keras or fallback to tf.keras
try:
    from tf_keras.optimizers import Adam as TFAdam
except ImportError:
    from tensorflow.keras.optimizers import Adam as TFAdam

class BERTModel:
    def __init__(self,
                 model_name='bert-base-uncased',
                 max_len=128,
                 learning_rate=2e-5):
        """
        Initialize tokenizer and config placeholders for BERT model.
        Actual model weights are loaded in `fit` when num_labels is known.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X_train, y_train, epochs=3, batch_size=16):
        """
        Encode labels, load BERT with correct num_labels, tokenize inputs,
        and fine-tune the model on the training data.
        """
        # Encode labels
        y_enc = self.label_encoder.fit_transform(y_train)
        num_labels = len(self.label_encoder.classes_)

        # Load BERT model with correct output size
        self.model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )

        # Tokenize and create tf.data.Dataset
        encodings = self.tokenizer(
            list(X_train), truncation=True,
            padding=True, max_length=self.max_len,
            return_tensors='tf'
        )
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings), y_enc
        )).shuffle(1000).batch(batch_size)

        # Compile with legacy Adam for speed on M1/M2
        self.model.compile(
            optimizer=TFAdam(self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train
        self.model.fit(dataset, epochs=epochs, verbose=2)

    def predict(self, X, batch_size=16):
        """
        Tokenize inputs and return array of predicted labels.
        """
        encodings = self.tokenizer(
            list(X), truncation=True,
            padding=True, max_length=self.max_len,
            return_tensors='tf'
        )
        dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(batch_size)
        logits = self.model.predict(dataset).logits
        preds = np.argmax(logits, axis=1)
        return self.label_encoder.inverse_transform(preds)



