import os
import pickle
from typing import Any, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

# Paths and Directory Setup
INPUT_PATH: str = "data/processed/processed_data.csv"
MODEL_DIR: str = "models"


#utility to save pickle artifacts
def _save_pickle(obj: Any, filename: str) -> None:
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


# training models
def train_traditional_models(
    input_csv: str, model_output_dir: str
) -> Tuple[float, float]:
    os.makedirs(model_output_dir, exist_ok=True)

    # Load processed dataset
    print(f"Loading processed dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    df["clean_review"] = df["clean_review"].fillna("")

    X = df["clean_review"]
    y = df["sentiment"].astype(int)

    #train-test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Enhanced TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    nb_grid = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid={"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    nb_grid.fit(X_train_tfidf, y_train)
    naive_bayes_model = nb_grid.best_estimator_

    # Evaluate NB on test set
    nb_preds = naive_bayes_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_preds)
    nb_f1 = f1_score(y_test, nb_preds, average="macro")
    print(
        f"Best NB Params: {nb_grid.best_params_} | "
        f"Test Acc: {nb_acc:.4f} | Test F1 (Macro): {nb_f1:.4f}"
    )

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_grid = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        param_grid={
            "C": [0.1, 1.0, 3.0, 5.0],
            "solver": ["lbfgs", "saga"],
        },
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    lr_grid.fit(X_train_tfidf, y_train)
    logistic_model = lr_grid.best_estimator_

    # Evaluate LR on test set
    lr_preds = logistic_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro")
    print(
        f"Best LR Params: {lr_grid.best_params_} | "
        f"Test Acc: {lr_acc:.4f} | Test F1 (Macro): {lr_f1:.4f}"
    )

    # Detailed Classification Reports
    target_names = [
        "Very Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very Positive",
    ]
    print("\n--- Logistic Regression Test Report ---")
    print(classification_report(y_test, lr_preds, target_names=target_names))

    # Save artifacts
    print(f"Saving model artifacts to: {model_output_dir}")
    _save_pickle(vectorizer, "tfidf_vectorizer.pkl")
    _save_pickle(naive_bayes_model, "naive_bayes_model.pkl")
    _save_pickle(logistic_model, "logistic_regression_model.pkl")
    _save_pickle((X_test_tfidf, y_test), "test_data.pkl")

    # summary report
    print("\n===================================")
    print("5-Class Traditional Training Completed")
    print("===================================")
    print(
        f"Training Samples: {len(X_train)} | "
        f"Testing Samples: {len(X_test)} | "
        f"TF-IDF Features: {X_train_tfidf.shape[1]}"
    )

    return float(nb_acc), float(lr_acc)


if __name__ == "__main__":
    train_traditional_models(INPUT_PATH, MODEL_DIR)