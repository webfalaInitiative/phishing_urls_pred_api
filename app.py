# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import preprocess_url
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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
    bad_link_threshold = 80.0
    good_link_threshold = 50.0

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
