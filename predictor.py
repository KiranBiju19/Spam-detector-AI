import joblib
import os
from data_loader import clean_text

MODEL_PATH = 'spam_classifier.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'
LOG_PATH = 'suspicious_log.txt'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_message(message, log_suspicious=True):
    clean_msg = clean_text(message)
    X = vectorizer.transform([clean_msg])
    pred = model.predict(X)[0]
    if pred == 1 and log_suspicious:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    return 'spam' if pred == 1 else 'ham' 