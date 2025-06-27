from data_loader import load_data, preprocess_data, get_train_test, vectorize_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import string

MODEL_PATH = 'spam_classifier.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'

STOPWORDS = set(["the", "is", "in", "and", "to", "a", "of"])  # Expand as needed

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

def preprocess_data(df):
    df = df.dropna(subset=['label', 'message'])
    df['clean_message'] = df['message'].apply(clean_text)
    return df

if __name__ == '__main__':
    df = load_data('spam.csv')
    print("Initial DataFrame preview:")
    print(df.head())
    print("Shape:", df.shape)
    print("Label value counts:\n", df['label'].value_counts())

    df = preprocess_data(df)
    print("Check for NaN labels:", df['label'].isnull().sum())
    print("Check for NaN messages:", df['clean_message'].isnull().sum())
    print("Data preview after cleaning:")
    print(df[['label', 'clean_message']].head())

    X_train, X_test, y_train, y_test = get_train_test(df)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f'Model saved to {MODEL_PATH}')
    print(f'Vectorizer saved to {VECTORIZER_PATH}')
