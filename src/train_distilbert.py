import os
from typing import Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Configuration and Paths
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_NAME: str = "distilbert-base-uncased"
MODEL_DIR: str = "models/distilbert_model"
OUTPUT_DIR: str = "distilbert_output"
MAX_LENGTH: int = 256

# Explicit Label Mappings
LABEL_MAPPING: Dict[str, int] = {"negative": 0, "positive": 1}
ID2LABEL: Dict[int, str] = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID: Dict[str, int] = {"NEGATIVE": 0, "POSITIVE": 1}

# Load Hugging Face evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Computes weighted evaluation metrics during training validation rounds."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["recall"]

    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
    }


def train_distilbert_model() -> None:
    """Executes the DistilBERT fine-tuning pipeline using Train/Validation sets."""
    print("==========================================")
    print("       DistilBERT Sentiment Training")
    print("==========================================")

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and validate dataset
    print(f"\nLoading dataset from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Original record count: {len(df)}")

    required_columns = ["raw_review", "sentiment"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' missing from dataset input."
            )

    # Map sentiment labels to integer IDs
    df["label"] = df["sentiment"].astype(str).str.lower().map(LABEL_MAPPING)

    # Clean missing or invalid rows
    df = df.dropna(subset=["raw_review", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"Valid record count: {len(df)}")

    # 80/20 Train and Validation Split (Test set removed for training run)
    train_df, validation_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )

    print("\nDataset split allocation:")
    print("------------------------------------------")
    print(f"Training samples   : {len(train_df)}")
    print(f"Validation samples : {len(validation_df)}")

    # Convert to Hugging Face Datasets using un-modified raw text
    train_dataset = Dataset.from_pandas(
        train_df[["raw_review", "label"]], preserve_index=False
    )
    validation_dataset = Dataset.from_pandas(
        validation_df[["raw_review", "label"]], preserve_index=False
    )

    # Load Tokenizer
    print(f"\nLoading pretrained tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(
            batch["raw_review"], truncation=True, max_length=MAX_LENGTH
        )

    # Tokenize datasets
    print("Tokenizing training and validation splits...")
    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["raw_review"]
    )
    validation_dataset = validation_dataset.map(
        tokenize_function, batched=True, remove_columns=["raw_review"]
    )

    # Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Model with label metadata
    print("\nInitializing DistilBERT sequence classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Hardware detection
    use_fp16 = torch.cuda.is_available()
    print(
        f"\nHardware Setup: {'GPU (FP16 Enabled)' if use_fp16 else 'CPU Acceleration'}"
    )

    # Fine-tuning parameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=use_fp16,
        report_to="none",
    )

    # Hugging Face Trainer setup using processing_class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train Transformer Model
    print("\n==========================================")
    print("Starting DistilBERT fine-tuning execution...")
    print("==========================================")
    trainer.train()

    # Validation evaluation
    print("\n==========================================")
    print("Validation Set Metrics")
    print("==========================================")
    validation_results = trainer.evaluate(eval_dataset=validation_dataset)
    for key, val in validation_results.items():
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "").capitalize()
            print(f"{metric_name:12s}: {val}")

    # Export final model artifacts
    print("\n==========================================")
    print(f"Saving fine-tuned model artifacts to: {MODEL_DIR}")
    print("==========================================")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("\nDistilBERT training completed successfully!")


if __name__ == "__main__":
    train_distilbert_model()