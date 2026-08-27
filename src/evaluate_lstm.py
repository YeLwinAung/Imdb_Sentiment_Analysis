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
from tensorflow.keras.utils import to_categorical

# Configuration Paths and Settings
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_DIR: str = "models"
MAX_LENGTH: int = 64
NUM_CLASSES: int = 5


# load pickle
def _load_pickle(filename: str) -> Any:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "rb") as file:
        return pickle.load(file)


# evulate BiLSTM
def evaluate_lstm(input_csv: str, model_dir: str) -> Dict[str, Any]:
    print("Loading saved BiLSTM artifacts and model...")

    # Load artifacts
    tokenizer = _load_pickle("tokenizer.pkl")
    label_encoder = _load_pickle("label_encoder.pkl")
    model_path = os.path.join(model_dir, "lstm_model.keras")
    model = load_model(model_path)

    # Check for pre-saved test split
    test_artifact_path = os.path.join(model_dir, "lstm_test_data.pkl")
    if os.path.exists(test_artifact_path):
        print("Loading pre-processed test split artifact...")
        X_test_pad, y_test_raw = _load_pickle("lstm_test_data.pkl")
    else:
        print("Re-creating test dataset split...")
        df = pd.read_csv(input_csv)
        df["clean_review"] = df["clean_review"].fillna("")

        X = df["clean_review"]
        y_encoded = label_encoder.transform(df["sentiment"])
        y_one_hot = to_categorical(y_encoded, num_classes=NUM_CLASSES)

        _, X_test, _, y_test_raw = train_test_split(
            X, y_one_hot, test_size=0.20, random_state=42, stratify=y_encoded
        )

        X_test_sequences = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(
            X_test_sequences,
            maxlen=MAX_LENGTH,
            padding="post",
            truncating="post",
        )

    if len(y_test_raw.shape) > 1 and y_test_raw.shape[1] > 1:
        y_true = np.argmax(y_test_raw, axis=1)
    else:
        y_true = y_test_raw

    # Multi-class predictions
    print("Generating predictions...")
    predictions = model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Calculate Evaluation Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted"))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    target_names = [str(c) for c in label_encoder.classes_]

    #Evaluation Report
    header_divider = "=" * 45
    print(f"\n{header_divider}")
    print(" 5-Class BiLSTM Model Evaluation Results")
    print(f"{header_divider}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision (W)      : {precision_weighted:.4f}")
    print(f"Recall (W)         : {recall_weighted:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro)   : {f1_macro:.4f}")

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true, y_pred, target_names=target_names, digits=4
        )
    )

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    return {
        "Model": "BiLSTM",
        "Accuracy": round(accuracy, 4),
        "Precision (W)": round(precision_weighted, 4),
        "Recall (W)": round(recall_weighted, 4),
        "F1 (Weighted)": round(f1_weighted, 4),
        "F1 (Macro)": round(f1_macro, 4),
    }


if __name__ == "__main__":
    evaluate_lstm(INPUT_PATH, MODEL_DIR)