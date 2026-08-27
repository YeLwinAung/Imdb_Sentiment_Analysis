import os
import sys
import re
import pickle
from typing import Any, Dict, Callable

# 1. Environment Setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Setup project  imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")

for path in [BASE_DIR, SRC_DIR, APP_DIR]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy

# Central utilities import
from utils import ID2LABEL, process_sentiment_output

# Initialize SpaCy model for syntactic parsing
nlp = spacy.load("en_core_web_sm")

MODELS_DIR = os.path.join(BASE_DIR, "models")
_model_cache: Dict[str, Any] = {}

# 2. Text Normalization
def normalize_typos_and_elongations(text: str) -> str:

    text = re.sub(r'(.)\1{2,}', r'\1', text)
    slang_map = {
        "luvd": "loved", 
        "luvs": "loves", 
        "awsum": "awesome", 
        "freaking": "really"
    }
    tokens = text.split()
    return " ".join([slang_map.get(t.lower(), t) for t in tokens])


def enhanced_clean_raw(text: str) -> str:

    return normalize_typos_and_elongations(text.strip())


def enhanced_clean_text(text: str) -> str:

    return normalize_typos_and_elongations(text.lower().strip())


def detect_idiomatic_adverbs(text: str) -> bool:
    doc = nlp(text.lower())

    positive_verbs = {"like", "love", "enjoy", "want", "miss", "dig"}
    intensifiers = {"bad", "hard", "crazy", "much"}

    # Handle common idiomatic expressions such as:
    # "I like this movie so bad"
    # "I love this movie so much"
    idiomatic_pattern = re.compile(
        r"\b(?:like|love|enjoy|want|miss|dig)"
        r"\b.*\bso\s+(?:bad|hard|much|crazy)\b"
    )

    if idiomatic_pattern.search(text.lower()):
        return True

    # SpaCy dependency-based detection
    for token in doc:
        if token.lemma_ in intensifiers:
            if token.dep_ in {"advmod", "amod"}:
                head = token.head

                if (
                    head.lemma_ in positive_verbs
                    or head.head.lemma_ in positive_verbs
                ):
                    return True

    return False


def adjust_for_structural_patterns(text: str, score: float) -> float:

    lower_text = text.lower()

    #Boost score
    if detect_idiomatic_adverbs(text):
        score = max(score, 0.90)

    #double negations
    double_neg_pattern = r"\b(can'?t|cannot)\s+say\s+(?:that\s+)?i\s+didn'?t\b|\bdidn'?t\s+dislike\b"
    if re.search(double_neg_pattern, lower_text):
        if score < 0.50:
            # push score
            score = 0.65 + (0.50 - score) * 0.40

    # 3. Soft contrastive scaling
    contrast_words = [" but ", " however ", " although ", " yet "]
    if any(cw in lower_text for cw in contrast_words):
        score = 0.50 + (score - 0.50) * 0.35

    return float(np.clip(score, 0.0, 1.0))

#Lazy Loading Model s
def get_sarcasm():
    if "sarcasm" not in _model_cache:
        path = os.path.join(MODELS_DIR, "roberta_sarcasm")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        _model_cache["sarcasm"] = (tokenizer, model)
    return _model_cache["sarcasm"]


def get_distilbert():

    if "distilbert" not in _model_cache:
        path = os.path.join(MODELS_DIR, "distilbert_model")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        _model_cache["distilbert"] = (tokenizer, model)
    return _model_cache["distilbert"]


def get_naive_bayes():
    if "naive_bayes" not in _model_cache:
        nb_path = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        with open(nb_path, "rb") as f:
            nb_model = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
        _model_cache["naive_bayes"] = (nb_model, tfidf)
    return _model_cache["naive_bayes"]


def get_logistic_regression():

    if "logistic" not in _model_cache:
        lr_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
        with open(lr_path, "rb") as f:
            lr_model = pickle.load(f)
        _, tfidf = get_naive_bayes()
        _model_cache["logistic"] = (lr_model, tfidf)
    return _model_cache["logistic"]


def get_bilstm():
    
    if "bilstm" not in _model_cache:
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
        tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
        lstm_model = tf.keras.models.load_model(lstm_path)
        with open(tok_path, "rb") as f:
            tok = pickle.load(f)
        _model_cache["bilstm"] = (lstm_model, tok)
    return _model_cache["bilstm"]

# Core execution
def _run_model_pipeline(model_name: str, text: str, prob_extractor: Callable[[str], np.ndarray]) -> Dict[str, Any]:

    try:
        probs = prob_extractor(text)
        
        # Continuous score
        raw_score = float(np.sum(probs * np.array([0.0, 0.25, 0.50, 0.75, 1.0])))
        adjusted_score = adjust_for_structural_patterns(text, raw_score)
        processed = process_sentiment_output(text, adjusted_score)

        return {
            "model": model_name,
            "class_id": processed["predicted_id"],
            "sentiment": processed["predicted_label"],
            "confidence": round(float(np.max(probs)), 4),
            "positive_prob": processed["calibrated_score"],
            "probabilities": [round(float(p), 4) for p in probs],
        }
    except Exception as e:
        return {
            "model": model_name,
            "class_id": 2,
            "sentiment": "Neutral",
            "confidence": 0.20,
            "positive_prob": 0.50,
            "probabilities": [0.20] * 5,
        }

# Predictor Functions
def predict_sarcasm(text: str) -> Dict[str, Any]:
    try:
        tokenizer, model = get_sarcasm()
        raw_text = enhanced_clean_raw(text)
        inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

        sarcasm_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        
        # Suppress false positives on ultra-short standard texts
        if len(raw_text.strip().split()) <= 4:
            sarcasm_prob = min(sarcasm_prob, 0.20)

        return {
            "is_sarcastic": sarcasm_prob >= 0.85, 
            "sarcasm_prob": round(sarcasm_prob, 4)
        }
    except Exception:
        return {"is_sarcastic": False, "sarcasm_prob": 0.0}


def predict_distilbert(text: str) -> Dict[str, Any]:
    try:
        tokenizer, model = get_distilbert()
        raw_text = enhanced_clean_raw(text)
        inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

        pred_class_id = int(np.argmax(probs))
        raw_score = float(np.sum(probs * np.array([0.0, 0.25, 0.50, 0.75, 1.0])))

        # Sarcasm check & score inversion adjustment
        sarcasm_res = predict_sarcasm(text)
        if pred_class_id in [3, 4] and sarcasm_res["is_sarcastic"]:
            raw_score *= (1.0 - sarcasm_res["sarcasm_prob"])

        adjusted_score = adjust_for_structural_patterns(text, raw_score)
        processed = process_sentiment_output(text, adjusted_score)

        return {
            "model": "DistilBERT",
            "class_id": processed["predicted_id"],
            "sentiment": processed["predicted_label"],
            "confidence": round(float(np.max(probs)), 4),
            "positive_prob": processed["calibrated_score"],
            "probabilities": [round(float(p), 4) for p in probs],
            "is_sarcastic": sarcasm_res["is_sarcastic"],
            "sarcasm_prob": sarcasm_res["sarcasm_prob"],
        }
    except Exception:
        return {
            "model": "DistilBERT", 
            "class_id": 2, 
            "sentiment": "Neutral", 
            "confidence": 0.20, 
            "positive_prob": 0.50, 
            "probabilities": [0.20] * 5
        }


def predict_naive_bayes(text: str) -> Dict[str, Any]:
    def extract(t: str) -> np.ndarray:
        model, tfidf = get_naive_bayes()
        return model.predict_proba(tfidf.transform([enhanced_clean_text(t)]))[0]
    
    return _run_model_pipeline("Naive Bayes", text, extract)


def predict_logistic_regression(text: str) -> Dict[str, Any]:
    def extract(t: str) -> np.ndarray:
        model, tfidf = get_logistic_regression()
        return model.predict_proba(tfidf.transform([enhanced_clean_text(t)]))[0]
    
    return _run_model_pipeline("Logistic Regression", text, extract)


def predict_bilstm(text: str) -> Dict[str, Any]:
    def extract(t: str) -> np.ndarray:
        model, tokenizer = get_bilstm()
        seqs = tokenizer.texts_to_sequences([enhanced_clean_text(t)])
        
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seqs, maxlen=100, padding="post", truncating="post"
        )
        return model.predict(padded, verbose=0)[0]
    
    return _run_model_pipeline("BiLSTM", text, extract)

# Alias for backwards compatibility
predict_lstm = predict_bilstm


def predict_all_models(text: str) -> Dict[str, Dict[str, Any]]:
    return {
        "distilbert": predict_distilbert(text),
        "logistic_regression": predict_logistic_regression(text),
        "naive_bayes": predict_naive_bayes(text),
        "bilstm": predict_bilstm(text),
    }
