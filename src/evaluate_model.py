import os
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Paths and File Configuration
MODEL_DIR: str = "models"


# Helper utility to safely load pickle artifacts
def _load_pickle(filename: str) -> Any:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "rb") as file:
        return pickle.load(file)


# Evaluates a single model against test dataset and prints evaluation metrics
def evaluate_model(
    model_name: str, model: Any, X_test: Any, y_test: Any
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    # Calculate weighted classification metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Print formatted model evaluation output
    header_divider = "=" * 40
    print(f"\n{header_divider}")
    print(f" Model Evaluation: {model_name}")
    print(f"{header_divider}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
    }


# Main evaluation pipeline function
def run_evaluation_pipeline() -> pd.DataFrame:
    print("Loading test artifacts and trained models...")

    # Load artifacts
    naive_bayes_model = _load_pickle("naive_bayes_model.pkl")
    logistic_regression_model = _load_pickle("logistic_regression_model.pkl")
    X_test_tfidf, y_test = _load_pickle("test_data.pkl")

    # Evaluate models
    nb_results = evaluate_model(
        "Naive Bayes", naive_bayes_model, X_test_tfidf, y_test
    )
    lr_results = evaluate_model(
        "Logistic Regression", logistic_regression_model, X_test_tfidf, y_test
    )

    # Generate model comparison summary dataframe
    results_list: List[Dict[str, Any]] = [nb_results, lr_results]
    comparison_df = pd.DataFrame(results_list)

    summary_divider = "=" * 40
    print(f"\n{summary_divider}")
    print(" Overall Model Comparison Summary")
    print(f"{summary_divider}\n")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    run_evaluation_pipeline()