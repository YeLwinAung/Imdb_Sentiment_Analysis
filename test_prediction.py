import os
import sys

# Ensure root directory is added to sys.path for relative imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.predictor import (
    predict_distilbert,
    predict_logistic_regression,
    predict_lstm,
    predict_naive_bayes,
)


# STEAM RATING MAPPER
def map_to_steam_rating(prob):
    if prob >= 0.95:
        return "Overwhelmingly Positive"
    elif prob >= 0.80:
        return "Very Positive"
    elif prob >= 0.70:
        return "Positive"
    elif prob >= 0.40:
        return "Mixed"
    elif prob >= 0.20:
        return "Negative"
    elif prob >= 0.05:
        return "Very Negative"
    else:
        return "Overwhelmingly Negative"


# GET PROBABILITY SAFELY
def get_positive_prob(result):
    if isinstance(result, dict):
        return result.get("positive_prob", 0.5)
    return 0.5


# RUN ALL MODELS
def get_model_predictions(review):
    return {
        "Naive Bayes": predict_naive_bayes(review),
        "Logistic Regression": predict_logistic_regression(review),
        "LSTM": predict_lstm(review),
        "DistilBERT": predict_distilbert(review),
    }


# ANALYZE REVIEW
def analyze_review(review):
    predictions = get_model_predictions(review)

    results = {}
    total_prob = 0.0

    for model_name, output in predictions.items():
        prob = get_positive_prob(output)
        total_prob += prob

        results[model_name] = {
            "sentiment": output.get("sentiment", "Unknown") if isinstance(output, dict) else output,
            "probability": prob,
            "steam_rating": map_to_steam_rating(prob),
            "is_sarcastic": output.get("is_sarcastic", False) if isinstance(output, dict) else False,
            "sarcasm_prob": output.get("sarcasm_prob", 0.0) if isinstance(output, dict) else 0.0,
        }

    avg_prob = total_prob / len(predictions)

    results["AVERAGE"] = {
        "probability": avg_prob,
        "steam_rating": map_to_steam_rating(avg_prob),
    }

    return results


# PRINT RESULTS
def print_results(review, results):
    print("=" * 80)
    print("REVIEW:", review)
    print("=" * 80)

    for model_name, data in results.items():
        if model_name != "AVERAGE":
            print(f"\n[{model_name}]")
            print(f"  Sentiment:      {data['sentiment']}")
            print(f"  Is Sarcastic:   {data['is_sarcastic']} (Score: {data['sarcasm_prob']:.2%})")
            print(f"  Probability:    {data['probability']:.2%}")
            print(f"  Steam Rating:   {data['steam_rating']}")

    avg_data = results["AVERAGE"]
    print(f"\n---> [AVERAGE ENSEMBLE RESULT]")
    print(f"     Probability:  {avg_data['probability']:.2%}")
    print(f"     Steam Rating: {avg_data['steam_rating']}\n")


# TEST RUNNER WITH 10 SARCASTIC & 3 SINCERE SAMPLES
if __name__ == "__main__":

    sarcastic_reviews = [
        "Wow, what an amazing movie. I absolutely loved falling asleep halfway through it.",
        "Great graphics, if you enjoy looking at blurry powerpoint presentations.",
        "10/10 masterpiece, can't wait to never play this unoptimized mess again.",
        "I love paying full price for a game that crashes every five minutes. Best purchase ever!",
        "Truly a revolutionary experience. My computer turned into an oven in two minutes.",
        "Thanks developers for deleting my save file, I really wanted to restart from scratch!",
        "Fantastic voice acting, it sounds like everyone was reading off a napkin at gunpoint.",
        "So glad I spent 60 dollars to stare at a loading screen all evening.",
        "Amazing story line, I especially loved how none of the character choices mattered at all.",
        "Super fun game if your idea of fun is standing in a virtual line for three hours."
    ]

    sincere_reviews = [
        "The story is amazing and the graphics are incredible.",
        "Boring gameplay and terrible graphics, waste of money.",
        "It was decent, nothing special but enjoyable."
    ]

    print("\n" + "#" * 80)
    print("1. RUNNING SARCASTIC REVIEWS TEST SUITE (10 SAMPLES)")
    print("#" * 80)

    sarcasm_detection_count = 0

    for idx, review in enumerate(sarcastic_reviews, 1):
        print(f"\n--- Sarcastic Test Case #{idx} ---")
        result = analyze_review(review)
        print_results(review, result)
        
        # Check if DistilBERT detected sarcasm
        if result["DistilBERT"]["is_sarcastic"]:
            sarcasm_detection_count += 1

    print("\n" + "#" * 80)
    print("2. RUNNING SINCERE REVIEWS TEST SUITE (3 BASELINE SAMPLES)")
    print("#" * 80)

    for idx, review in enumerate(sincere_reviews, 1):
        print(f"\n--- Sincere Test Case #{idx} ---")
        result = analyze_review(review)
        print_results(review, result)

    print("=" * 80)
    print(f"SUMMARY: DistilBERT detected sarcasm in {sarcasm_detection_count}/10 test cases.")
    print("=" * 80)