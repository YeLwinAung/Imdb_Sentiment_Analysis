import os
import pickle
from typing import Any, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

# Paths and Directory Setup
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_DIR: str = "models"


# Helper utility to save pickle artifacts
def _save_pickle(obj: Any, filename: str) -> None:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


# Main training pipeline function
def train_traditional_models(
    input_csv: str, model_output_dir: str
) -> Tuple[float, float]:
    os.makedirs(model_output_dir, exist_ok=True)

    # Load processed dataset
    print(f"Loading processed dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    df["clean_review"] = df["clean_review"].fillna("")

    X = df["clean_review"]
    y = df["sentiment"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Enhanced TF-IDF Vectorizer (ngram_range=(1, 3) captures n-grams and negations)
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=40000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes GridSearch
    print("\nTraining Naive Bayes...")
    nb_grid = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid={"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    nb_grid.fit(X_train_tfidf, y_train)
    naive_bayes_model = nb_grid.best_estimator_
    print(
        f"Best NB Params: {nb_grid.best_params_} | "
        f"Best NB CV Accuracy: {nb_grid.best_score_:.4f}"
    )

    # Train Logistic Regression GridSearch
    print("\nTraining Logistic Regression...")
    lr_grid = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        param_grid={
            "C": [0.1, 1, 5, 10],
            "solver": ["liblinear", "lbfgs"],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    lr_grid.fit(X_train_tfidf, y_train)
    logistic_model = lr_grid.best_estimator_
    print(
        f"Best LR Params: {lr_grid.best_params_} | "
        f"Best LR CV Accuracy: {lr_grid.best_score_:.4f}"
    )

    # Save artifacts
    print(f"\nSaving model artifacts to: {model_output_dir}")
    _save_pickle(vectorizer, "tfidf_vectorizer.pkl")
    _save_pickle(naive_bayes_model, "naive_bayes_model.pkl")
    _save_pickle(logistic_model, "logistic_regression_model.pkl")
    _save_pickle((X_test_tfidf, y_test), "test_data.pkl")

    # Final summary report
    print("\n===================================")
    print("Training Completed Successfully")
    print("===================================")
    print(
        f"Training Samples: {len(X_train)} | "
        f"Testing Samples: {len(X_test)} | "
        f"TF-IDF Vocabulary Features: {X_train_tfidf.shape[1]}"
    )

    return float(nb_grid.best_score_), float(lr_grid.best_score_)


if __name__ == "__main__":
    train_traditional_models(INPUT_PATH, MODEL_DIR)