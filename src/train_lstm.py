import os
import pickle
from typing import Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Configuration Paths and Hyperparameters
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_DIR: str = "models"
MAX_WORDS: int = 10000
MAX_LENGTH: int = 200
EMBEDDING_DIM: int = 128
BATCH_SIZE: int = 64
EPOCHS: int = 5


# Helper utility to save pickle artifacts
def _save_pickle(obj: Any, filename: str) -> None:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


# Main training function for BiLSTM
def train_lstm_model(
    input_csv: str, model_output_dir: str
) -> Tuple[float, float]:
    os.makedirs(model_output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading processed dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    df["clean_review"] = df["clean_review"].fillna("")

    X = df["clean_review"]
    y = df["sentiment"]

    # Encode sentiment labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # Tokenizer setup
    print("Fitting Keras Tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    # Convert text to padded sequences
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_sequences, maxlen=MAX_LENGTH, padding="pre", truncating="pre"
    )
    X_test_pad = pad_sequences(
        X_test_sequences, maxlen=MAX_LENGTH, padding="pre", truncating="pre"
    )

    # Construct Bidirectional LSTM Architecture
    print("Building BiLSTM model...")
    model = Sequential(
        [
            Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, dropout=0.2)),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    # Fit Model
    print("\nStarting LSTM Training...")
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_pad, y_test),
        callbacks=[early_stopping],
    )

    # Save artifacts
    print(f"\nSaving model artifacts to: {model_output_dir}")
    _save_pickle(tokenizer, "tokenizer.pkl")
    _save_pickle(label_encoder, "label_encoder.pkl")
    _save_pickle((X_test_pad, y_test), "lstm_test_data.pkl")

    model_save_path = os.path.join(model_output_dir, "lstm_model.keras")
    model.save(model_save_path)

    # Summary
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    print("\n===================================")
    print(" LSTM Training Completed Successfully")
    print("===================================")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
    print(f"Vocabulary Size  : {len(tokenizer.word_index)}")
    print(f"Validation Acc   : {final_val_acc:.4f}")

    return final_train_acc, final_val_acc


if __name__ == "__main__":
    train_lstm_model(INPUT_PATH, MODEL_DIR)