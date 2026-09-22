"""Streamlit UI for the IMDB sentiment analysis project."""

from pathlib import Path
import json

import streamlit as st

from src.predict import predict_sentiment

ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    layout="centered",
)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


st.title("IMDB Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review and the trained NLP model will predict "
    "whether the sentiment is positive or negative."
)

metadata = load_metadata()

if metadata:
    st.caption(
        f"Model: {metadata['selected_vectorizer']} + "
        f"{metadata['selected_model']}"
    )

review = st.text_area(
    "Enter your movie review here:",
    height=220,
    placeholder="Example: The acting was excellent and the story was very engaging...",
)

analyze = st.button("Analyze Sentiment", use_container_width=True)
clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if analyze:
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        try:
            result = predict_sentiment(review)

            st.divider()
            st.subheader("Prediction")

            if result["sentiment"] == "Positive":
                st.success("POSITIVE")
            else:
                st.error("NEGATIVE")

            confidence_percent = result["confidence"] * 100

            st.metric("Confidence", f"{confidence_percent:.2f}%")
            st.metric("Review word count", result["word_count"])

            st.info(
                "Confidence is model-specific. For Linear SVM, the displayed "
                "value is a bounded score derived from the decision margin, "
                "not a calibrated probability."
            )

        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Run `python src/train.py` before launching the app.")
