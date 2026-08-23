# app/predictor.py

import os
import pickle
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure project root path for modular imports across execution contexts
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils import calibrate_probability, clean_text, derive_sentiment_label

# Directory constants and sequence length boundaries
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
DISTILBERT_DIR: str = os.path.join(MODEL_DIR, "distilbert_model")
SARCASM_DIR: str = os.path.join(MODEL_DIR, "roberta_sarcasm")

MAX_SEQUENCE_LENGTH: int = 200
BERT_MAX_LENGTH: int = 128

# Confidence threshold required to flag input text as sarcastic
SARCASM_THRESHOLD: float = 0.85


def _load_pickle(filename: str) -> Any:
    """Helper function to load serialized Python artifacts from the models directory."""
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "rb") as file:
        return pickle.load(file)


# Load Core Machine Learning Artifacts
tfidf = _load_pickle("tfidf_vectorizer.pkl")
naive_bayes = _load_pickle("naive_bayes_model.pkl")
logistic_model = _load_pickle("logistic_regression_model.pkl")

# Load Deep Learning Artifacts
lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
lstm_tokenizer = _load_pickle("tokenizer.pkl")

# Load Transformer Sentiment Artifacts (Step 4)
bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
bert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
bert_model.eval()

# Load Transformer Sarcasm Artifacts (Step 5)
roberta_tokenizer = AutoTokenizer.from_pretrained(SARCASM_DIR)
roberta_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_DIR)
roberta_model.eval()


def _extract_positive_index(model_classes: np.ndarray) -> int:
    """Locates the column index corresponding to the positive class in Scikit-Learn classifiers."""
    classes_list = list(model_classes)
    for index, label in enumerate(classes_list):
        if str(label).lower() == "positive":
            return index
    return 1


def _extract_bert_positive_index() -> int:
    """Locates the positive class logit index from DistilBERT model configuration metadata."""
    if hasattr(bert_model.config, "id2label"):
        for index, label in bert_model.config.id2label.items():
            if "positive" in str(label).lower():
                return int(index)
    return 1


def _extract_sarcasm_index() -> int:
    """Maps the target sarcasm label index dynamically from the fine-tuned RoBERTa config."""
    if hasattr(roberta_model.config, "id2label"):
        for index, label in roberta_model.config.id2label.items():
            label_text = str(label).lower()
            if any(key in label_text for key in ["sarcastic", "sarcasm", "pos", "1"]):
                return int(index)
    return 1


def predict_sarcasm(review: str) -> Tuple[bool, float]:
    """Evaluates input text for sarcastic tone using the fine-tuned RoBERTa model."""
    inputs = roberta_tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    sarcasm_idx = _extract_sarcasm_index()
    sarcasm_prob = float(probabilities[sarcasm_idx].item())

    # Evaluate sarcasm confidence score against pre-configured decision threshold
    is_sarcastic = sarcasm_prob >= SARCASM_THRESHOLD
    return is_sarcastic, round(sarcasm_prob, 4)


def _apply_sarcasm_adjustment(raw_prob: float, is_sarcastic: bool, sarcasm_prob: float) -> float:
    """Adjusts raw sentiment probability by scaling score inversion proportionally to sarcasm confidence."""
    if is_sarcastic:
        weight = (sarcasm_prob - SARCASM_THRESHOLD) / (1.0 - SARCASM_THRESHOLD)
        inverted = 1.0 - raw_prob
        return (raw_prob * (1 - weight)) + (inverted * weight)
    return raw_prob


def derive_7class_sentiment(prob: float) -> str:
    """Maps a continuous positivity probability score [0.0, 1.0] to a 7-class sentiment scale."""
    if prob >= 0.90:
        return "Overwhelmingly Positive"
    elif prob >= 0.75:
        return "Very Positive"
    elif prob >= 0.55:
        return "Positive"
    elif prob >= 0.45:
        return "Mixed"
    elif prob >= 0.25:
        return "Negative"
    elif prob >= 0.10:
        return "Very Negative"
    else:
        return "Overwhelmingly Negative"


def _build_result(review: str, raw_prob: float) -> Dict[str, Any]:
    """Integrates sentiment calibration, sarcasm correction, and 7-class label assignment into output structure."""
    raw_prob = max(0.0, min(1.0, raw_prob))

    is_sarcastic, sarcasm_prob = predict_sarcasm(review)
    adjusted_prob = _apply_sarcasm_adjustment(raw_prob, is_sarcastic, sarcasm_prob)

    calibrated_prob = calibrate_probability(review, adjusted_prob)
    calibrated_prob = max(0.0, min(1.0, float(calibrated_prob)))

    return {
        "sentiment": derive_7class_sentiment(calibrated_prob),
        "positive_prob": round(calibrated_prob, 4),
        "is_sarcastic": is_sarcastic,
        "sarcasm_prob": sarcasm_prob,
    }


def predict_naive_bayes(review: str) -> Dict[str, Any]:
    """Generates sentiment prediction using TF-IDF feature extraction and Naive Bayes classification."""
    text = clean_text(review) or review
    vector = tfidf.transform([text])
    probabilities = naive_bayes.predict_proba(vector)[0]
    pos_idx = _extract_positive_index(naive_bayes.classes_)
    return _build_result(review, float(probabilities[pos_idx]))


def predict_logistic_regression(review: str) -> Dict[str, Any]:
    """Generates sentiment prediction using TF-IDF feature extraction and Logistic Regression classification."""
    text = clean_text(review) or review
    vector = tfidf.transform([text])
    probabilities = logistic_model.predict_proba(vector)[0]
    pos_idx = _extract_positive_index(logistic_model.classes_)
    return _build_result(review, float(probabilities[pos_idx]))


def predict_lstm(review: str) -> Dict[str, Any]:
    """Generates sentiment prediction using tokenized padding sequences and deep LSTM network inference."""
    text = clean_text(review) or review
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="pre"
    )
    raw_prob = float(lstm_model.predict(padded_sequence, verbose=0)[0][0])
    return _build_result(review, raw_prob)


def predict_distilbert(review: str) -> Dict[str, Any]:
    """Generates sentiment prediction using DistilBERT transformer sequence classification."""
    inputs = bert_tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    pos_idx = _extract_bert_positive_index()
    return _build_result(review, float(probabilities[pos_idx].item()))
