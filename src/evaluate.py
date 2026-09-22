"""Evaluate the saved sentiment model and perform simple error analysis."""

from pathlib import Path
import json

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report

from src.data_preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train.py"
        )

    dataset = load_dataset("stanfordnlp/imdb")
    test_df = pd.DataFrame(dataset["test"])

    model = joblib.load(MODEL_PATH)

    cleaned = test_df["text"].map(clean_text).tolist()
    predictions = model.predict(cleaned)

    print(classification_report(
        test_df["label"],
        predictions,
        target_names=["Negative", "Positive"],
        digits=4,
    ))

    errors = test_df.copy()
    errors["predicted"] = predictions
    errors["error_type"] = "Correct"

    errors.loc[
        (errors["label"] == 0) & (errors["predicted"] == 1),
        "error_type"
    ] = "False Positive"

    errors.loc[
        (errors["label"] == 1) & (errors["predicted"] == 0),
        "error_type"
    ] = "False Negative"

    errors[errors["error_type"] != "Correct"][
        ["text", "label", "predicted", "error_type"]
    ].to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)

    print("Saved:", OUTPUT_DIR / "error_analysis.csv")
    print("\nSample False Positives:")
    print(
        errors[errors["error_type"] == "False Positive"]["text"]
        .head(5)
        .to_string(index=False)
    )

    print("\nSample False Negatives:")
    print(
        errors[errors["error_type"] == "False Negative"]["text"]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
