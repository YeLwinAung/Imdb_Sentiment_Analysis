import os
import pandas as pd

#Evaluation Data
results = [
    {
        "Model": "Naive Bayes",
        "Category": "Machine Learning",
        "Accuracy": 0.3842,
        "Precision (W)": 0.3810,
        "Recall (W)": 0.3842,
        "F1 (Weighted)": 0.3785,
        "F1 (Macro)": 0.3621,
    },
    {
        "Model": "Logistic Regression",
        "Category": "Machine Learning",
        "Accuracy": 0.4215,
        "Precision (W)": 0.4180,
        "Recall (W)": 0.4215,
        "F1 (Weighted)": 0.4150,
        "F1 (Macro)": 0.3980,
    },
    {
        "Model": "BiLSTM",
        "Category": "Deep Learning (RNN)",
        "Accuracy": 0.4050,
        "Precision (W)": 0.4012,
        "Recall (W)": 0.4050,
        "F1 (Weighted)": 0.3995,
        "F1 (Macro)": 0.3810,
    },
    {
        "Model": "DistilBERT",
        "Category": "Transformer",
        "Accuracy": 0.5131,
        "Precision (W)": 0.5075,
        "Recall (W)": 0.5131,
        "F1 (Weighted)": 0.5075,
        "F1 (Macro)": 0.4991,
    },
]


def generate_comparison_report():
    # Ensure sdirectory exists
    os.makedirs("results", exist_ok=True)

    comparison = pd.DataFrame(results)

    # Sort models
    comparison = comparison.sort_values(
        by="F1 (Macro)", ascending=False
    ).reset_index(drop=True)

    # Print  Report
    divider = "=" * 80
    print(f"\n{divider}")
    print("                      SST-5 Model Benchmark Comparison")
    print(f"{divider}\n")
    print(comparison.to_string(index=False))
    print(f"\n{divider}")

    # Export to CSV 
    csv_path = "results/model_comparison.csv"
    md_path = "results/model_comparison.md"

    comparison.to_csv(csv_path, index=False)
    comparison.to_markdown(md_path, index=False)

    print(f"\nSaved CSV comparison artifact to      : {csv_path}")
    print(f"Saved Markdown summary table to       : {md_path}\n")


if __name__ == "__main__":
    generate_comparison_report()