# Multi-Model Sentiment Analysis Pipeline for Social Media


## 📖 Overview

This project implements and compares four approaches for sentiment analysis on social media comments:

1. **NLTK Baseline**: Bag-of-Words + Multinomial Naive Bayes  
2. **Scikit-Learn**: TF-IDF + Logistic Regression  
3. **TensorFlow**: Bidirectional LSTM  
4. **HuggingFace BERT**: Fine-tuned Transformer  

Each model is evaluated on accuracy, precision, recall, and F1 score, and the  and results are visualized with confusion-matrix heatmaps.

## 🛠️   Features

- **Standardized preprocessing**: cleaning, tokenization, negation-aware stopword filtering & lemmatization  
- **Unified train/test split** (80 / 20), with stratification to preserve class balance  
- **Modular codebase** with reusable utils, model classes, and evaluation functions  
- **Automatic CSV export** of per‐model predictions  
- **Confusion-matrix heatmaps** saved for each model  


## 🚀 Quickstart

### 1. Clone & install

```bash
git clone <this-repo-url>
cd sentiment_analysis
```
### 2. Install dependencies

Install the following dependencies using pip:
```
pip install pandas scikit-learn matplotlib seaborn nltk tensorflow torch transformers
```
### 3. Prepare dataset
Download the dataset from Kaggle or use your own dataset: [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

After downloading, place the Tweets.csv file under the following directory structure: 
```
sentiment_analysis/
├── Tweets.csv
├── main.py
├── utils/
└── models/
└── ...
```

The sentiment_analysis folder should be at the same level as main.py.

### 4. Run the script
```bash
python main.py
```

* Default: trains on the first 2,000 tweets (nrows=2000)
* Full dataset: edit ```main.py``` →

```diff
- load_and_split('Tweets.csv', nrows=2000)
+ load_and_split('Tweets.csv', nrows=None)
```
* Smoke-test (eg, first 1,000 rows): set `nrows=1000`

* **Note**:  BERT on the full tweets dataset may take 40 min – 1 hr. 

The script will automatically:
* Load and split the dataset into training and testing sets.
* Preprocess the text data.
* Train four models: Naive Bayes, Logistic Regression, Bi-LSTM, and BERT.
* Evaluate the models and print performance metrics.
* Save the model predictions to CSV files.
* Generate and save confusion matrix plots for each model.

## 🔍 Expected Outputs
After running `main.py`, you’ll find:

**Prediction CSVs** in `predictions/`:
- `nltk_predictions.csv`
- `logreg_predictions.csv`
- `bi-lstm_predictions.csv`
- `bert_predictions.csv`

**Confusion-matrix plots** in `matrices/`:
- `nltk_matrix.png`
- `logreg_matrix.png`
- `bi-lstm_matrix.png`
- `bert_matrix.png`

## 🛠️  Troubleshooting Tips
* `Module Not Found Errors`:   
Ensure all libraries are installed. For the Keras-3 + Transformers issue, run:
```bash
pip install tf-keras
export TF_USE_LEGACY_KERAS=True
```
* **Dataset Errors**:  
Verify that `Tweets.csv` is in the project root, not in a subfolder.

* **Long Training Time**:  
– Use GPU acceleration if available.  
– Smoke-test on a subset via nrows.


