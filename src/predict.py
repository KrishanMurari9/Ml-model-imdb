"""Prediction utilities used by the Streamlit application."""

from pathlib import Path
import joblib

from src.data_preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Please run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(review_text: str):
    """Return label and an approximate confidence score."""
    model = load_model()
    cleaned = clean_text(review_text)

    if not cleaned:
        return {
            "sentiment": "Unknown",
            "confidence": 0.0,
            "score": 0.0,
            "word_count": 0,
        }

    prediction = int(model.predict([cleaned])[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        score = float(probabilities[prediction])
    elif hasattr(model, "decision_function"):
        raw = float(model.decision_function([cleaned])[0])
        # Convert the margin to a bounded, probability-like confidence.
        import math
        score = 1.0 / (1.0 + math.exp(-abs(raw)))
    else:
        score = 1.0

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": score,
        "score": float(model.decision_function([cleaned])[0])
        if hasattr(model, "decision_function")
        else 0.0,
        "word_count": len(cleaned.split()),
    }
