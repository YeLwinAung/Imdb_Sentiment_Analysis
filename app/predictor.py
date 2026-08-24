import os
import pickle
import sys
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

# Configure project root path for modular imports
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils import calibrate_probability, clean_text

# Directory constants
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
DISTILBERT_DIR: str = os.path.join(MODEL_DIR, "distilbert_model")
SARCASM_DIR: str = os.path.join(MODEL_DIR, "roberta_sarcasm")

MAX_SEQUENCE_LENGTH: int = 200
BERT_MAX_LENGTH: int = 128

# Confidence threshold required to flag sarcasm
SARCASM_THRESHOLD: float = 0.85


# ============================================================
# PICKLE MODEL LOADING
# ============================================================

@st.cache_resource
def _load_pickle(filename: str) -> Any:
    """
    Load and cache a pickle model only when it is first needed.
    """
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file not found: {filepath}"
        )

    with open(filepath, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def get_tfidf():
    return _load_pickle("tfidf_vectorizer.pkl")


@st.cache_resource
def get_naive_bayes():
    return _load_pickle("naive_bayes_model.pkl")


@st.cache_resource
def get_logistic_regression():
    return _load_pickle("logistic_regression_model.pkl")


@st.cache_resource
def get_lstm_tokenizer():
    return _load_pickle("tokenizer.pkl")


# ============================================================
# LSTM LOADING
# ============================================================

@st.cache_resource
def get_lstm_model():
    """
    Import TensorFlow only when the LSTM model is used.
    This prevents TensorFlow from loading during normal app startup.
    """
    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LSTM model not found: {model_path}"
        )

    return load_model(model_path)


# ============================================================
# DISTILBERT LOADING
# ============================================================

@st.cache_resource
def get_distilbert():
    """
    Load DistilBERT only when the user requests a prediction.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if not os.path.exists(DISTILBERT_DIR):
        raise FileNotFoundError(
            f"DistilBERT directory not found: {DISTILBERT_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_DIR,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_DIR,
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model


# ============================================================
# SARCASM MODEL LOADING
# ============================================================

@st.cache_resource
def get_sarcasm_model():
    """
    Load RoBERTa sarcasm model only when sarcasm detection is needed.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if not os.path.exists(SARCASM_DIR):
        raise FileNotFoundError(
            f"Sarcasm model directory not found: {SARCASM_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        SARCASM_DIR,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SARCASM_DIR,
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _extract_positive_index(model_classes: np.ndarray) -> int:
    """
    Find the probability index corresponding to the positive class.
    """
    classes_list = list(model_classes)

    for index, label in enumerate(classes_list):
        if str(label).lower() == "positive":
            return index

    return 1


def _extract_bert_positive_index(bert_model) -> int:
    """
    Find the positive class index from the DistilBERT model.
    """
    if hasattr(bert_model.config, "id2label"):
        for index, label in bert_model.config.id2label.items():
            if "positive" in str(label).lower():
                return int(index)

    return 1


def _extract_sarcasm_index(roberta_model) -> int:
    """
    Find the sarcasm class index from the RoBERTa model.
    """
    if hasattr(roberta_model.config, "id2label"):
        for index, label in roberta_model.config.id2label.items():
            label_text = str(label).lower()

            if any(
                key in label_text
                for key in ["sarcastic", "sarcasm", "pos", "1"]
            ):
                return int(index)

    return 1


# ============================================================
# SARCASM PREDICTION
# ============================================================

def predict_sarcasm(review: str) -> Tuple[bool, float]:
    """
    Evaluate input text for sarcastic tone.
    """

    import torch

    roberta_tokenizer, roberta_model = get_sarcasm_model()

    inputs = roberta_tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    sarcasm_idx = _extract_sarcasm_index(roberta_model)

    sarcasm_prob = float(
        probabilities[sarcasm_idx].item()
    )

    is_sarcastic = sarcasm_prob >= SARCASM_THRESHOLD

    return is_sarcastic, round(sarcasm_prob, 4)


# ============================================================
# SENTIMENT ADJUSTMENT
# ============================================================

def _apply_sarcasm_adjustment(
    raw_prob: float,
    is_sarcastic: bool,
    sarcasm_prob: float,
) -> float:
    """
    Adjust sentiment probability when sarcasm is detected.
    """

    if is_sarcastic:
        weight = (
            (sarcasm_prob - SARCASM_THRESHOLD)
            / (1.0 - SARCASM_THRESHOLD)
        )

        weight = max(0.0, min(1.0, weight))

        inverted = 1.0 - raw_prob

        return (
            raw_prob * (1 - weight)
            + inverted * weight
        )

    return raw_prob


def derive_7class_sentiment(prob: float) -> str:
    """
    Convert positive probability to a 7-class sentiment label.
    """

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


def _build_result(
    review: str,
    raw_prob: float,
) -> Dict[str, Any]:
    """
    Build the final prediction result.
    """

    raw_prob = max(
        0.0,
        min(1.0, raw_prob)
    )

    # Sarcasm model loads only when a prediction is requested
    is_sarcastic, sarcasm_prob = predict_sarcasm(review)

    adjusted_prob = _apply_sarcasm_adjustment(
        raw_prob,
        is_sarcastic,
        sarcasm_prob,
    )

    calibrated_prob = calibrate_probability(
        review,
        adjusted_prob,
    )

    calibrated_prob = max(
        0.0,
        min(1.0, float(calibrated_prob))
    )

    return {
        "sentiment": derive_7class_sentiment(
            calibrated_prob
        ),
        "positive_prob": round(
            calibrated_prob,
            4
        ),
        "is_sarcastic": is_sarcastic,
        "sarcasm_prob": sarcasm_prob,
    }


# ============================================================
# NAIVE BAYES
# ============================================================

def predict_naive_bayes(
    review: str,
) -> Dict[str, Any]:

    tfidf = get_tfidf()
    naive_bayes = get_naive_bayes()

    text = clean_text(review) or review

    vector = tfidf.transform([text])

    probabilities = naive_bayes.predict_proba(
        vector
    )[0]

    pos_idx = _extract_positive_index(
        naive_bayes.classes_
    )

    return _build_result(
        review,
        float(probabilities[pos_idx]),
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def predict_logistic_regression(
    review: str,
) -> Dict[str, Any]:

    tfidf = get_tfidf()
    logistic_model = get_logistic_regression()

    text = clean_text(review) or review

    vector = tfidf.transform([text])

    probabilities = logistic_model.predict_proba(
        vector
    )[0]

    pos_idx = _extract_positive_index(
        logistic_model.classes_
    )

    return _build_result(
        review,
        float(probabilities[pos_idx]),
    )


# ============================================================
# LSTM
# ============================================================

def predict_lstm(
    review: str,
) -> Dict[str, Any]:

    from tensorflow.keras.preprocessing.sequence import (
        pad_sequences
    )

    lstm_model = get_lstm_model()
    lstm_tokenizer = get_lstm_tokenizer()

    text = clean_text(review) or review

    sequence = lstm_tokenizer.texts_to_sequences(
        [text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="pre",
        truncating="pre",
    )

    raw_prob = float(
        lstm_model.predict(
            padded_sequence,
            verbose=0,
        )[0][0]
    )

    return _build_result(
        review,
        raw_prob,
    )


# ============================================================
# DISTILBERT
# ============================================================

def predict_distilbert(
    review: str,
) -> Dict[str, Any]:

    import torch

    bert_tokenizer, bert_model = get_distilbert()

    inputs = bert_tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )[0]

    pos_idx = _extract_bert_positive_index(
        bert_model
    )

    raw_prob = float(
        probabilities[pos_idx].item()
    )

    return _build_result(
        review,
        raw_prob,
    )
