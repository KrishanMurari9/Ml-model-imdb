"""Text cleaning utilities for the IMDB sentiment project."""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep sentiment-important negation words.
NEGATION_WORDS = {"no", "not", "nor", "never", "neither", "nothing", "nowhere", "hardly"}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS


def clean_text(text: str) -> str:
    """Clean one review while retaining words useful for sentiment analysis."""
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()

    # Keep alphabetic tokens and simple apostrophes.
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]

    return " ".join(tokens)


def clean_texts(texts):
    """Clean an iterable/Series of reviews."""
    return [clean_text(t) for t in texts]
