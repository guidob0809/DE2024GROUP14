import json
import logging
import os
import pickle
from io import StringIO

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class SMSPredictor:
    def __init__(self):
        self.model = None
        self.tfidf = TfidfVectorizer()  # Initialize TfidfVectorizer directly

    def load_model(self, model_file_path):
        """
        Load the machine learning model from a pickle file.
        """
        self.model = pickle.load(open(model_file_path, 'rb'))

    def fit_tfidf(self, messages):
        """
        Fit the TF-IDF vectorizer on the training data.
        """
        self.tfidf.fit(messages)

    def predict_classification(self, prediction_input):
        """
        Predict whether the given SMS message is spam or not.
        """
        logging.debug(prediction_input)
        if self.model is None:
            try:
                model_repo = os.environ['MODEL_REPO']
                model_file_path = os.path.join(model_repo, "model.pkl")
                self.model = pickle.load(open(model_file_path, 'rb'))
            except KeyError:
                print("MODEL_REPO is undefined")
                self.model = pickle.load(open('model.pkl', 'rb'))

        # Convert the input into a DataFrame for consistency
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        messages = df['message'].values

        # Transform the input message using the loaded TF-IDF vectorizer
        message_tfidf = self.tfidf.transform(messages)

        # Make predictions
        predictions = self.model.predict(message_tfidf)
        result = ["Spam" if pred == 1 else "Not Spam" for pred in predictions]

        # Return predictions in a structured JSON response
        return {"predictions": result}
