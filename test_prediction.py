import os
import sys
import time

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.predictor import predict_distilbert

REVIEW_TYPES = [
    "Extreme Positive","Strong Positive","Moderate Positive","Mild Positive",
    "Neutral","Mixed","Mild Negative","Strong Negative","Extreme Negative",
    "Sarcasm","Negation","Double Negation","Positive + Negative",
    "Negative + Positive","Short Positive","Short Negative",
    "Intensified Positive","Intensified Negative","Slang","Typos"
]

EXPECTED_TARGETS = [
    "Very Positive","Very Positive","Positive","Positive",
    "Mixed","Mixed","Negative","Very Negative","Very Negative",
    ("Negative","Very Negative"),"Positive","Positive","Mixed",
    "Mixed","Very Positive","Very Negative","Very Positive",
    "Very Negative","Very Positive","Very Positive"
]

REVIEWS_STRESS_TEST = [
    "A stunning achievement that completely captured my attention. The direction was superb and every performance felt convincing.",
    "This is a highly enjoyable film with excellent acting, an engaging plot, and several genuinely memorable moments.",
    "A good movie overall. The characters were interesting and the story was entertaining enough to keep me watching.",
    "I had a decent time with this one. It has some enjoyable scenes, even though it isn't particularly impressive.",
    "Neither good nor bad. The movie has a few strong moments, but several parts felt ordinary and forgettable.",
    "The actors were fantastic and the visuals looked incredible, but the weak storyline and slow middle section prevented it from being great.",
    "The movie wasn't completely awful, but the boring dialogue and predictable events made it a disappointing experience.",
    "I struggled to finish this film. The story was confusing, the acting felt forced, and almost nothing kept me interested.",
    "An utterly miserable experience. The script was awful, the performances were embarrassing, and the entire movie felt like a waste of time.",
    "Absolutely brilliant! I especially enjoyed the thrilling experience of watching nothing happen for nearly two hours.",
    "I didn't dislike the movie as much as I expected. There were actually several scenes that I thought were enjoyable.",
    "I can't really say that I didn't have a good time watching this film.",
    "The lead performances were excellent and the music was beautiful, but the repetitive story and weak ending were major problems.",
    "The movie started terribly and I nearly gave up, but the second half became engaging and the final scenes were surprisingly good.",
    "Superb!",
    "Horrendous!",
    "Everything about this film was exceptional. The writing was clever, the cast was phenomenal, and the emotional payoff was perfect.",
    "A painfully bad film with awful dialogue, irritating characters, ridiculous decisions, and an ending that made absolutely no sense.",
    "This film was insanely fun! The jokes landed, the action was wild, and I had a great time from start to finish.",
    "Luv this film! The charcters were awsum, the plot was amazng, and I wud totaly watch it agin!"
]
def run_stress_test():
    total = len(REVIEWS_STRESS_TEST)
    passed = 0
    failed_cases = []

    print("=" * 85)
    print(f"RUNNING MODEL STRESS TEST & FAILURE ANALYSIS ({total} SAMPLES)")
    print("=" * 85 + "\n")

    for idx, (rtype, review, expected) in enumerate(zip(REVIEW_TYPES, REVIEWS_STRESS_TEST, EXPECTED_TARGETS), 1):
        result = predict_distilbert(review)
        pred_sentiment = result["sentiment"]
        prob = result["positive_prob"]

        if isinstance(expected, tuple):
            is_pass = pred_sentiment in expected
            exp_str = " / ".join(expected)
        else:
            is_pass = pred_sentiment == expected
            exp_str = expected

        if is_pass:
            passed += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"
            failed_cases.append({
                "id": idx,
                "type": rtype,
                "review": review,
                "predicted": pred_sentiment,
                "expected": exp_str,
                "prob": prob
            })

        print(f"{status} | #{idx:02d} | {rtype:<20} | Score: {prob:.4f} | Pred: {pred_sentiment:<13} | Expected: {exp_str}")

    print("\n" + "=" * 85)
    accuracy = (passed / total) * 100
    print(f"SUMMARY: {passed}/{total} Passed | Accuracy: {accuracy:.1f}%")
    print("=" * 85)

    # DETAILED FAILURE BREAKDOWN REPORT
    if failed_cases:
        print("\n" + "!" * 85)
        print(f"FAILED ANALYSIS REPORT ({len(failed_cases)} REVIEWS FAILED)")
        print("!" * 85)
        for fail in failed_cases:
            print(f"\n[FAIL ITEM #{fail['id']:02d}] {fail['type']}")
            print(f"  Review Text : \"{fail['review']}\"")
            print(f"  Predicted   : {fail['predicted']} (Positive Probability: {fail['prob']:.4f})")
            print(f"  Expected    : {fail['expected']}")
        print("!" * 85)
    else:
        print("\n🎉 Perfect Run! All reviews passed analysis.")

if __name__ == "__main__":
    run_stress_test()