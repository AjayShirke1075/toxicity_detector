# src/predict.py
import os
import joblib
from typing import Dict
from .preprocess import clean_text

# Define paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "toxic_model.pkl")  # trained model from train_on_hf.py

# Load trained model
try:
    MODEL = joblib.load(MODEL_PATH)
    print(f"✅ Loaded trained model from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not load model from {MODEL_PATH}. Error: {e}")
    MODEL = None

def predict_text(text: str) -> Dict:
    """
    Predict if a given text is toxic or not.
    
    Returns:
        Dict: {
            "label": "toxic" or "not_toxic",
            "score": float probability of toxic,
            "cleaned": preprocessed text
        }
    """
    if MODEL is None:
        return {"label": "not_toxic", "score": 0.0, "cleaned": text}

    # Clean text
    cleaned = clean_text(text)

    # Transform and predict
    X = [cleaned]  # single-sample list
    probs = MODEL.predict_proba(X)[0]  # [not_toxic_prob, toxic_prob]
    toxic_prob = float(probs[1])
    label = "toxic" if toxic_prob >= 0.5 else "not_toxic"

    return {"label": label, "score": toxic_prob, "cleaned": cleaned}
