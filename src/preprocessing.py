import os
import re
from typing import List, Set

import contractions
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Paths and File Configuration
INPUT_PATH: str = "data/raw/IMDb_Dataset.csv"
OUTPUT_PATH: str = "data/processed/processed_data.csv"


# Ensure NLTK resources are available silently
def _ensure_nltk_resource(resource_name: str, path_check: str) -> None:
    try:
        nltk.data.find(path_check)
    except LookupError:
        nltk.download(resource_name, quiet=True)


_ensure_nltk_resource("punkt", "tokenizers/punkt")
_ensure_nltk_resource("stopwords", "corpora/stopwords")
_ensure_nltk_resource("wordnet", "corpora/wordnet")


# Stopword and Negation Definitions
NEGATION_WORDS: Set[str] = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "cannot",
    "without",
    "hardly",
    "barely",
    "rarely",
}

STOP_WORDS: Set[str] = set(stopwords.words("english")) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()


# Apply NEG_ prefix to token scope window following negation words
def mark_negation(tokens: List[str]) -> List[str]:
    result: List[str] = []
    negation_active: bool = False
    negation_count: int = 0

    for token in tokens:
        if token in NEGATION_WORDS:
            result.append(token)
            negation_active = True
            negation_count = 0
            continue

        if negation_active:
            if token not in STOP_WORDS and token.isalpha():
                result.append(f"NEG_{token}")
                negation_count += 1
                if negation_count >= 3:
                    negation_active = False
            else:
                result.append(token)
        else:
            result.append(token)

    return result


# Full cleaning pipeline for traditional ML vectorizers
def clean_text(text: str) -> str:
    # Expand contractions and lowercase
    text = contractions.fix(str(text)).lower()

    # Clean HTML, URLs, numbers, and non-alphabetic characters
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and filter stopwords
    tokens = [
        word
        for word in word_tokenize(text)
        if word.isalpha() and (word not in STOP_WORDS or word in NEGATION_WORDS)
    ]

    # Lemmatize and apply negation scope tagging
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    negated_tokens = mark_negation(lemmatized_tokens)

    return " ".join(negated_tokens)


# Light cleaner preserving raw text structure for Transformer models (DistilBERT)
def clean_raw_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Main execution pipeline
def preprocess_dataset(input_file: str, output_file: str) -> None:
    print(f"Loading raw dataset from: {input_file}")
    df = pd.read_csv(input_file)

    # Apply preprocessing transformations
    df["clean_review"] = df["review"].apply(clean_text)
    df["raw_review"] = df["review"].apply(clean_raw_text)

    # Filter empty reviews
    df = df[df["clean_review"].str.strip() != ""].reset_index(drop=True)

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Data successfully processed and saved to: {output_file}")
    print(f"Total valid reviews: {len(df)}")


if __name__ == "__main__":
    preprocess_dataset(INPUT_PATH, OUTPUT_PATH)