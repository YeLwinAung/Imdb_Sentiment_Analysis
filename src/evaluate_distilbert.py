import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# Configuration and Paths
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_PATH: str = "models/distilbert_model"
MAX_LENGTH: int = 256
BATCH_SIZE: int = 32 

LABEL_MAPPING: Dict[str, int] = {"negative": 0, "positive": 1}
TARGET_NAMES = ["negative", "positive"]


def evaluate_distilbert(input_csv: str, model_dir: str) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        raise FileNotFoundError(
            f"Could not find model.safetensors in {model_dir}. Please check your path."
        )

    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Standardize and map labels
    df["label"] = df["sentiment"].astype(str).str.lower().map(LABEL_MAPPING)

    # Clean missing or invalid rows
    df = df.dropna(subset=["raw_review", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Recreate identical 20% validation split
    _, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    val_df = val_df.reset_index(drop=True)
    print(f"Evaluating model on {len(val_df)} validation samples...")

    # Load local tokenizer and model with local_files_only=True
    print(f"Loading local model weights and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )

    # Set up hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Running inference on device: {device}")

    # Convert to tokenize
    val_dataset = Dataset.from_pandas(
        val_df[["raw_review", "label"]], preserve_index=False
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["raw_review"], truncation=True, max_length=MAX_LENGTH
        )

    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["raw_review"]
    )

    # Create PyTorch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator
    )

    # Fast PyTorch evaluation loop
    all_predictions = []
    print("Executing evaluation loop...")

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(preds)

    y_pred = np.array(all_predictions)
    y_true = val_df["label"].values

    # Compute classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted"))
    rec = float(recall_score(y_true, y_pred, average="weighted"))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # Print results
    header_divider = "=" * 40
    print(f"\n{header_divider}")
    print(" DistilBERT Validation Evaluation Results")
    print(f"{header_divider}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    return {
        "Model": "DistilBERT",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    }


if __name__ == "__main__":
    evaluate_distilbert(INPUT_PATH, MODEL_PATH)