# 🎬 IMDB Movie Review Sentiment Analysis

A college-level NLP + Machine Learning project that predicts whether an IMDB movie review is **Positive** or **Negative**.

## 1. Objective

Build a binary sentiment classifier using the IMDB 50K Movie Reviews dataset and compare classical NLP/ML approaches.

Models:
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

Feature representations:
- CountVectorizer
- TF-IDF

The project also includes:
- Exploratory data analysis
- Precision, recall, accuracy and F1
- Confusion matrix
- Error analysis
- Streamlit web application
- Experimental aspect-level analysis

## 2. Dataset

The project loads the IMDB dataset using Hugging Face Datasets:

```python
load_dataset("imdb")
```

Dataset size:
- 25,000 training reviews
- 25,000 test reviews
- Binary labels:
  - 0 = Negative
  - 1 = Positive

## 3. Project Architecture

```text
IMDB Reviews
     |
     v
Text Cleaning
     |
     v
CountVectorizer / TF-IDF
     |
     v
Logistic Regression / Naive Bayes / Linear SVM
     |
     v
Evaluation
     |
     v
Saved Best Model
     |
     v
Streamlit App
     |
     v
Positive / Negative + Confidence
```

## 4. Folder Structure

```text
imdb-sentiment-analysis/
│
├── data/
│   └── README.md
├── models/
├── outputs/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Installation

Recommended Python version: 3.10 or newer.

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Train the Models

From the project root:

```bash
python src/train.py
```

The script:
1. Downloads/loads IMDB.
2. Performs basic EDA.
3. Cleans reviews.
4. Trains six configurations:
   - CountVectorizer + Logistic Regression
   - CountVectorizer + Multinomial Naive Bayes
   - CountVectorizer + Linear SVM
   - TF-IDF + Logistic Regression
   - TF-IDF + Multinomial Naive Bayes
   - TF-IDF + Linear SVM
5. Calculates accuracy, precision, recall and F1.
6. Saves comparison graphs.
7. Selects the highest-F1 configuration.
8. Saves the trained pipeline.

## 7. Evaluate Error Analysis

After training:

```bash
python src/evaluate.py
```

This creates:

```text
outputs/error_analysis.csv
```

The file contains false-positive and false-negative examples.

## 8. Run the Streamlit Application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## 9. Evaluation Metrics

### Accuracy

Percentage of all predictions that are correct.

### Precision

Among reviews predicted as positive, how many are actually positive.

### Recall

Among truly positive reviews, how many were correctly identified.

### F1 Score

Harmonic mean of precision and recall.

### Confusion Matrix

Shows:
- True Negative
- False Positive
- False Negative
- True Positive

## 10. Why TF-IDF?

TF-IDF gives more importance to terms that are useful in a document but not equally common across all documents.

For sentiment analysis, words and phrases such as "excellent", "terrible", "waste", "highly recommended", and useful bigrams can help distinguish sentiment.

## 11. Why Linear SVM / Logistic Regression?

These models are strong classical baselines for high-dimensional sparse text features. They are also easier to train and explain than a deep neural network.

## 12. Error Analysis

Some reviews are difficult because of:
- sarcasm
- negation
- mixed sentiment
- context
- unusual wording
- very short reviews

Example:

> This movie was so bad that it was actually hilarious.

A simple bag-of-words system may see strong negative words without fully understanding the sentence's context.

## 13. Innovative Feature

The Streamlit app includes an experimental **aspect analysis** for:
- Acting
- Story
- Direction
- Music
- Visuals
- Cinematography
- Characters

This component is deliberately labeled as a heuristic. It should not be described as a fully trained aspect-based sentiment model during a presentation.

## 14. Limitations

- Classical bag-of-words methods have limited understanding of context.
- Sarcasm can be difficult.
- The confidence score for Linear SVM is not a calibrated probability.
- Aspect analysis is heuristic rather than a dedicated ABSA model.

## 15. Future Scope

Possible future improvements:
- LSTM/GRU
- BERT/DistilBERT
- Calibrated probabilities
- True aspect-based sentiment analysis
- Multilingual sentiment analysis
- Explainable AI methods such as SHAP/LIME
- Deployment to a cloud platform

## 16. Viva Questions

### Q1. What is sentiment analysis?
It is an NLP task that identifies the emotional polarity of text, such as positive or negative.

### Q2. Why is TF-IDF used?
It converts text into numerical features while reducing the importance of terms that occur in many documents.

### Q3. Why do we need vectorization?
Machine learning algorithms require numerical input, so text must be converted into numerical feature vectors.

### Q4. What is a confusion matrix?
It summarizes classification results using true positives, true negatives, false positives and false negatives.

### Q5. Why do we use F1?
F1 combines precision and recall into a single metric and is useful when both types of classification errors matter.

### Q6. What is data leakage?
Data leakage occurs when information from the test set improperly influences training.

### Q7. Why use a Pipeline?
A Pipeline keeps vectorization and classification together and helps ensure transformations are learned only from training data.

### Q8. Why preserve words like "not"?
Negation can completely change sentiment. For example, "good" and "not good" have different meanings.

## 17. Conclusion

This project demonstrates a complete classical NLP workflow: data understanding, text preprocessing, feature engineering, machine learning, evaluation, error analysis and deployment through Streamlit.

## 18. Changelog / Recent Commits

- **Simplified UI**: Stripped out emojis and experimental features from `app.py` for a cleaner interface.
- **Dataset Loading Fix**: Updated HF dataset URI from `imdb` to `stanfordnlp/imdb` to fix `HfUriError`.
- **Bug Fixes**: Removed invalid `sublinear_tf` argument from `CountVectorizer`.
- **Import Fixes**: Updated absolute imports in `predict.py`, `train.py`, and `evaluate.py` to correctly resolve `src.data_preprocessing`.
- **Deployment**: Added `Dockerfile` and `.dockerignore` for easy containerized deployment.
