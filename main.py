import os
import pandas as pd
from utils.preprocessing import clean_text
from utils.dataset import load_and_split
from utils.evaluate import print_report, results
from utils.save_predictions import save_predictions
from utils.confusion_matrix import plot_confusion_matrix
from models.nb_model import NBModel
from models.lr_model import LRModel
from models.lstm_model import LSTMModel
from models.bert_model import BERTModel

# Get the configurations of each model
def get_model_configs():
    return [
        {'key':'nltk', 'name':'Naive Bayes', 'model':NBModel(), 'cmap':'Blues'},
        {'key':'logreg', 'name':'Logistic Regression','model':LRModel(), 'cmap':'Greens'},
        {'key':'bi-lstm', 'name':'Bi-LSTM', 'model':LSTMModel(),  'cmap':'Oranges'},
        {'key':'bert', 'name':'BERT','model':BERTModel(),  'cmap':'Purples'},
    ]

def main():
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('matrices', exist_ok=True)

    # load & preprocess
    X_train, X_test, y_train, y_test = load_and_split('Tweets.csv', nrows=2000) # change the value of nrows to adjust the data size, set to None for all data
    X_train = [clean_text(x) for x in X_train]
    X_test  = [clean_text(x) for x in X_test]

    # train, evaluate, and save
    for cfg in get_model_configs():
        print(f"\n--- Processing {cfg['name']} ---")
        cfg['model'].fit(X_train, y_train)
        y_pred = cfg['model'].predict(X_test)

        print_report(cfg['name'], y_test, y_pred)
        save_predictions(cfg['key'], X_test, y_test, y_pred)

    # plot confusion matrices
    for cfg in get_model_configs():
        df = pd.read_csv(f"predictions/{cfg['key']}_predictions.csv")
        plot_confusion_matrix(
            df['true_label'],
            df['predicted_label'],
            cfg['name'],
            accuracy=results.get(cfg['name'], 0),
            cmap_style=cfg['cmap'],
            save_path=f"matrices/{cfg['key']}_matrix.png"
        )

if __name__ == "__main__":
    main()
