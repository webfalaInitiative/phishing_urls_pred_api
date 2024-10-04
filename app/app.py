# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model and label encoder
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
classifier = joblib.load('./model/phishing_classifier_20240927_013119.pkl')
label_encoder = joblib.load('./model/label_encoder_20240927_013119.pkl')

# Define input and output schemas
class URLInput(BaseModel):
    url: str

class PredictionOutput(BaseModel):
    predicted_label: str
    confidence_score: float

# Preprocessing function (if applicable)
def preprocess_url(url):
    # Example preprocessing - customize this as needed
    return url.strip().lower()

# Prediction logic with thresholds
def predict_url_with_threshold(url, model, classifier, label_encoder):
    # Preprocess the URL
    cleaned_url = preprocess_url(url)

    # Convert the input URL to embeddings
    embedding = model.encode([cleaned_url])

    # Get the prediction (0 or 1)
    prediction = classifier.predict(embedding)[0]

    # Get the prediction probabilities (confidence scores)
    prediction_prob = classifier.predict_proba(embedding)[0]

    # Confidence score of the predicted class
    confidence_score = np.max(prediction_prob) * 100  # in percentage

    # Convert prediction (0 or 1) to the corresponding label ('good' or 'bad')
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Define your class-specific confidence thresholds
    bad_link_threshold = 95.0
    good_link_threshold = 70.0

    # Apply the thresholds
    if predicted_label == 'bad' and confidence_score >= bad_link_threshold:
        return predicted_label, confidence_score
    elif predicted_label == 'good' and confidence_score >= good_link_threshold:
        return predicted_label, confidence_score
    else:
        return "uncertain", confidence_score

# Define the API route for URL prediction
@app.post("/predict", response_model=PredictionOutput)
def predict_url(input_data: URLInput):
    predicted_label, confidence_score = predict_url_with_threshold(
        input_data.url, model, classifier, label_encoder
    )

    return {"predicted_label": predicted_label, "confidence_score": confidence_score}
