"""Train and compare classical NLP sentiment models on IMDB."""

from pathlib import Path
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_preprocessing import clean_text

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MAX_FEATURES = 50000


def load_imdb():
    """Download/load the official IMDB dataset via Hugging Face Datasets."""
    dataset = load_dataset("stanfordnlp/imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    return train_df, test_df


def save_eda(train_df):
    """Save basic EDA plots and summary information."""
    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": 25000,
        "missing_reviews": int(train_df["text"].isna().sum()),
        "duplicate_reviews": int(train_df["text"].duplicated().sum()),
        "positive_train": int((train_df["label"] == 1).sum()),
        "negative_train": int((train_df["label"] == 0).sum()),
    }

    with open(OUTPUT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=train_df, x="label")
    plt.title("IMDB Training Class Distribution")
    plt.xlabel("Label (0 = Negative, 1 = Positive)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
    plt.close()

    lengths = train_df["text"].fillna("").str.split().str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=60)
    plt.title("Review Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "review_length_distribution.png", dpi=160)
    plt.close()


def build_vectorizer(kind):
    if kind == "count":
        return CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=MAX_FEATURES,
        )
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "predictions": predictions,
    }


def main():
    print("Loading IMDB dataset...")
    train_df, test_df = load_imdb()

    save_eda(train_df)

    print("Cleaning reviews. This may take a little time...")
    x_train = train_df["text"].map(clean_text).tolist()
    x_test = test_df["text"].map(clean_text).tolist()
    y_train = train_df["label"]
    y_test = test_df["label"]

    results = []
    best = None

    model_configs = [
        ("CountVectorizer", "count"),
        ("TF-IDF", "tfidf"),
    ]

    for vectorizer_name, vectorizer_kind in model_configs:
        for model_name, classifier in build_models().items():
            print(f"\nTraining: {vectorizer_name} + {model_name}")

            pipeline = Pipeline(
                [
                    ("vectorizer", build_vectorizer(vectorizer_kind)),
                    ("classifier", classifier),
                ]
            )

            start = time.time()
            pipeline.fit(x_train, y_train)
            elapsed = time.time() - start

            metrics = evaluate_model(pipeline, x_test, y_test)

            row = {
                "vectorizer": vectorizer_name,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "training_seconds": round(elapsed, 2),
            }
            results.append(row)

            print(classification_report(
                y_test,
                metrics["predictions"],
                target_names=["Negative", "Positive"],
                digits=4,
            ))

            # F1 is used only as the technical selection metric.
            if best is None or row["f1"] > best["f1"]:
                best = {
                    "pipeline": pipeline,
                    "f1": row["f1"],
                    "row": row,
                    "predictions": metrics["predictions"],
                }

    results_df = pd.DataFrame(results).sort_values(
        ["f1", "accuracy"], ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(10, 6))
    plot_df = results_df.copy()
    plot_df["configuration"] = (
        plot_df["vectorizer"] + " + " + plot_df["model"]
    )
    sns.barplot(data=plot_df, x="f1", y="configuration")
    plt.title("Model Comparison by F1 Score")
    plt.xlabel("F1 Score")
    plt.ylabel("")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_f1_comparison.png", dpi=160)
    plt.close()

    # Save confusion matrix for the selected model.
    cm = confusion_matrix(y_test, best["predictions"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(
        f"Confusion Matrix: {best['row']['vectorizer']} + "
        f"{best['row']['model']}"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(best["pipeline"], MODEL_DIR / "sentiment_model.joblib")

    metadata = {
        "selected_vectorizer": best["row"]["vectorizer"],
        "selected_model": best["row"]["model"],
        "selection_metric": "F1 score",
        "metrics": best["row"],
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\nSaved model:", MODEL_DIR / "sentiment_model.joblib")
    print("Best technical configuration:", metadata["selected_model"])
    print("Best vectorizer:", metadata["selected_vectorizer"])
    print("F1:", round(metadata["metrics"]["f1"], 4))


if __name__ == "__main__":
    main()
