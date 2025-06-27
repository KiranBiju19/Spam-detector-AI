## Spam/Phishing Detection with AI & Cybersecurity Concepts

This project uses machine learning to detect phishing or spam messages using the SMS Spam Collection Dataset. It demonstrates AI and cybersecurity concepts for text-based threat detection.

## Features
- Loads and preprocesses SMS/email messages
- Trains a text classification model (scikit-learn or transformers)
- Simple web GUI for real-time message prediction
- Optionally logs suspicious messages

## Project Structure
- `data_loader.py`: Loads and preprocesses the dataset
- `model_trainer.py`: Trains and saves the model
- `predictor.py`: Loads the model and predicts new messages
- `app.py`: Web app (Flask)
- `spam.csv`: Dataset (from Kaggle)
- `suspicious_log.txt`: (Optional) Log of suspicious messages

## Setup
1. Clone this repository and navigate to the project folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Ensure `spam.csv` is in the project directory.

## Training the Model
Run the following to train and save the model:
```
python model_trainer.py
```

## Running the Web App
Start the web app with:
```
python app.py
```
Then open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Customization
- To use a transformer model (DistilBERT), uncomment the relevant lines in `requirements.txt` and update the code in `model_trainer.py` and `predictor.py`.
- Suspicious messages are logged in `suspicious_log.txt`.

## Accuray and output
```
Accuracy: 0.9614349775784753
              precision    recall  f1-score   support

           0       0.96      1.00      0.98       965
           1       0.98      0.73      0.84       150

    accuracy                           0.96      1115
   macro avg       0.97      0.86      0.91      1115
weighted avg       0.96      0.96      0.96      1115

Model saved to spam_classifier.joblib
Vectorizer saved to vectorizer.joblib
```
## License
This project is licensed under the MIT License.

**Dataset Attribution**:  
The dataset used in this project is from the [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and is provided for research purposes by its original authors.
