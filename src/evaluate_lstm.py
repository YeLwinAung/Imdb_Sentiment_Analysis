import os
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration Paths and Settings
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_DIR: str = "models"
MAX_LENGTH: int = 200


# Helper utility to safely load pickle artifacts
def _load_pickle(filename: str) -> Any:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "rb") as file:
        return pickle.load(file)


# Main evaluation pipeline function for BiLSTM
def evaluate_lstm(input_csv: str, model_dir: str) -> Dict[str, Any]:
    print("Loading saved LSTM artifacts and model...")

    # Load artifacts
    tokenizer = _load_pickle("tokenizer.pkl")
    label_encoder = _load_pickle("label_encoder.pkl")
    model_path = os.path.join(model_dir, "lstm_model.keras")
    model = load_model(model_path)

    # Check for pre-saved test split artifact for speed and consistency
    test_artifact_path = os.path.join(model_dir, "lstm_test_data.pkl")
    if os.path.exists(test_artifact_path):
        print("Loading pre-processed test split artifact...")
        X_test_pad, y_test = _load_pickle("lstm_test_data.pkl")
    else:
        print("Re-creating test dataset split...")
        df = pd.read_csv(input_csv)
        df["clean_review"] = df["clean_review"].fillna("")

        X = df["clean_review"]
        y = label_encoder.transform(df["sentiment"])

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_test_sequences = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(
            X_test_sequences,
            maxlen=MAX_LENGTH,
            padding="pre",
            truncating="pre",
        )

    # Prediction
    print("Generating predictions...")
    predictions = model.predict(X_test_pad, verbose=0)
    y_pred = (predictions > 0.5).astype(int).flatten()

    # Calculate Evaluation Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Print Formatted Evaluation Report
    header_divider = "=" * 40
    print(f"\n{header_divider}")
    print(" BiLSTM Model Evaluation Results")
    print(f"{header_divider}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return {
        "Model": "BiLSTM",
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
    }


if __name__ == "__main__":
    evaluate_lstm(INPUT_PATH, MODEL_DIR)