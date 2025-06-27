import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))



def load_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')  # handles special characters
    df = df[['v1', 'v2']]  # keep only label and message columns
    df.columns = ['label', 'message']  # rename for consistency
    return df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text  # ← Do NOT remove stopwords now


def preprocess_data(df):
    df = df.dropna(subset=['label', 'message']).copy()
    df['clean_message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df[df['label'].notna()]
    df = df[df['clean_message'].notna()]
    df = df[df['clean_message'].str.strip() != ""]
    return df.reset_index(drop=True)

def get_train_test(df, test_size=0.2, random_state=42):
    X = df['clean_message']
    y = df['label'].map({'ham': 0, 'spam': 1})
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer 