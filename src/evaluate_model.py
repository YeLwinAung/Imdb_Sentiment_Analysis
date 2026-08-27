import os
import pickle
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Paths
MODEL_DIR: str = "models"
TARGET_NAMES: List[str] = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
]


# load pickle
def _load_pickle(filename: str) -> Any:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "rb") as file:
        return pickle.load(file)


# Evaluates mdoel
def evaluate_model(
    model_name: str, model: Any, X_test: Any, y_test: Any
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    # Calculate classification metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted"))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    # Print evaluation output
    header_divider = "=" * 45
    print(f"\n{header_divider}")
    print(f" 5-Class Evaluation: {model_name}")
    print(f"{header_divider}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision (W)     : {precision_weighted:.4f}")
    print(f"Recall (W)        : {recall_weighted:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro)   : {f1_macro:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=TARGET_NAMES, digits=4
        )
    )

    print("Confusion Matrix (0=Very Neg -> 4=Very Pos):")
    print(confusion_matrix(y_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision (W)": round(precision_weighted, 4),
        "Recall (W)": round(recall_weighted, 4),
        "F1 (Weighted)": round(f1_weighted, 4),
        "F1 (Macro)": round(f1_macro, 4),
    }


# evaluation
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

    # model comparison summary
    results_list: List[Dict[str, Any]] = [nb_results, lr_results]
    comparison_df = pd.DataFrame(results_list)

    summary_divider = "=" * 45
    print(f"\n{summary_divider}")
    print(" Overall 5-Class Model Comparison Summary")
    print(f"{summary_divider}\n")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    run_evaluation_pipeline()