import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Build custom stopword set (retain negations)
STOP = set(stopwords.words('english'))
NEGATIONS = {"no","not","never","n't","can't","won't"}
CUSTOM_STOPWORDS = STOP - NEGATIONS
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw text:
      - lowercase
      - remove URLs, mentions, hashtags
      - keep alphabetic chars only
      - tokenize, remove custom stopwords
      - lemmatize tokens
    Returns cleaned string.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text.strip())
    tokens = [tok for tok in tokens if tok not in CUSTOM_STOPWORDS]
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)
