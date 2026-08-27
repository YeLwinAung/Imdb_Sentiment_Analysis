import re
from typing import Dict, Union

#Mapping
ID2LABEL: Dict[int, str] = {
    0: "Very Negative",
    1: "Negative",
    2: "Mixed",
    3: "Positive",
    4: "Very Positive",
}

def correct_typos(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def calibrate_probability(text: str, raw_prob: float) -> float:
    text_lower = text.lower().strip()

    negation_match = re.search(
        r"\b(didn't|did not|not|isn't|wasn't|wouldn't|haven't|never)\s+([a-z]+)\b",
        text_lower,
    )

    if negation_match:
        word_after_neg = negation_match.group(2)
        negative_words = {
            "hate", "bad", "terrible", "awful", "horrible",
            "boring", "disappointing", "worst", "poor",
        }
        if word_after_neg in negative_words and raw_prob < 0.40:
            return 0.52

    return max(0.0, min(1.0, float(raw_prob)))

def derive_sentiment_id(prob: float) -> int:
    """
    Single source of truth for granular 5-class SST-5 mapping:
    [0.00 - 0.18) -> 0 (Very Negative)
    [0.18 - 0.40) -> 1 (Negative)
    [0.40 - 0.60) -> 2 (Neutral)
    [0.60 - 0.82) -> 3 (Positive)
    [0.82 - 1.00] -> 4 (Very Positive)
    """
    if prob >= 0.82:
        return 4
    elif prob >= 0.60:
        return 3
    elif prob >= 0.40:
        return 2
    elif prob >= 0.18:
        return 1
    return 0

def derive_sentiment_label(prob: float) -> str:
    return ID2LABEL[derive_sentiment_id(prob)]

def process_sentiment_output(text: str, raw_prob: float) -> Dict[str, Union[str, int, float]]:

    cleaned_text = correct_typos(text)
    calibrated_score = calibrate_probability(cleaned_text, raw_prob)
    class_id = derive_sentiment_id(calibrated_score)

    return {
        "cleaned_text": cleaned_text,
        "calibrated_score": round(calibrated_score, 4),
        "predicted_id": class_id,
        "predicted_label": ID2LABEL[class_id],
    }