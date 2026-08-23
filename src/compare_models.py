import pandas as pd

results = [
    {
        "Model": "Naive Bayes",
        "Category": "Machine Learning",
        "Accuracy": 0.8545,
        "Precision": 0.8546,
        "Recall": 0.8545,
        "F1 Score": 0.8545
    },
    {
        "Model": "Logistic Regression",
        "Category": "Machine Learning",
        "Accuracy": 0.8910,
        "Precision": 0.8911,
        "Recall": 0.8910,
        "F1 Score": 0.8910
    },
    {
    "Model": "LSTM",
    "Category": "Artificial Neural Network",
    "Accuracy": 0.6254,
    "Precision": 0.7451,
    "Recall": 0.6254,
    "F1 Score": 0.5733
    },
    {
        "Model": "DistilBERT",
        "Category": "Transformer",
        "Accuracy": 0.8932,
        "Precision": 0.8947,
        "Recall": 0.8932,
        "F1 Score": 0.8931
    }
]

comparison = pd.DataFrame(results)

print("\n========== Model Comparison ==========\n")
print(comparison)

comparison.to_csv(
"model_comparison.csv",
    index=False
)

print("\nComparison table saved as model_comparison.csv")