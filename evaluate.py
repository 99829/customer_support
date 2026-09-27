"""
STEP 6: Evaluation
====================
Runs the classifier against the labeled eval dataset and computes
objective accuracy metrics.

Concepts covered:
- Accuracy: what fraction of predictions exactly matched ground truth
- Per-field breakdown: category accuracy vs urgency accuracy separately
  (a system might be great at categories but weak at urgency - you only
  find this by measuring them separately, not as one combined score)
- Error analysis: printing out WHICH tickets were misclassified, not just
  the final number - this is what actually tells you how to improve the
  system, a bare accuracy percentage doesn't.
"""

import os
from dotenv import load_dotenv

from classifier import classify_ticket
from eval_dataset import EVAL_DATASET


def run_evaluation(api_key: str):
    total = len(EVAL_DATASET)
    category_correct = 0
    urgency_correct = 0
    both_correct = 0
    mismatches = []

    for item in EVAL_DATASET:
        ticket = item["ticket"]
        expected_category = item["expected_category"]
        expected_urgency = item["expected_urgency"]

        prediction = classify_ticket(ticket, api_key)

        cat_match = prediction.category == expected_category
        urg_match = prediction.urgency == expected_urgency

        if cat_match:
            category_correct += 1
        if urg_match:
            urgency_correct += 1
        if cat_match and urg_match:
            both_correct += 1
        else:
            mismatches.append({
                "ticket": ticket,
                "expected_category": expected_category.value,
                "predicted_category": prediction.category.value,
                "expected_urgency": expected_urgency.value,
                "predicted_urgency": prediction.urgency.value,
                "cat_match": cat_match,
                "urg_match": urg_match,
            })

    # --- Report ---
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total tickets evaluated: {total}")
    print(f"Category accuracy:       {category_correct}/{total} ({100*category_correct/total:.1f}%)")
    print(f"Urgency accuracy:        {urgency_correct}/{total} ({100*urgency_correct/total:.1f}%)")
    print(f"Both fully correct:      {both_correct}/{total} ({100*both_correct/total:.1f}%)")

    if mismatches:
        print(f"\n{'-'*70}")
        print(f"MISCLASSIFIED TICKETS ({len(mismatches)}):")
        print('-'*70)
        for m in mismatches:
            print(f"\nTicket: {m['ticket']}")
            if not m["cat_match"]:
                print(f"  Category  -> expected: {m['expected_category']:20} got: {m['predicted_category']}")
            if not m["urg_match"]:
                print(f"  Urgency   -> expected: {m['expected_urgency']:20} got: {m['predicted_urgency']}")
    else:
        print("\nNo misclassifications - perfect score on this eval set.")

    return {
        "total": total,
        "category_accuracy": category_correct / total,
        "urgency_accuracy": urgency_correct / total,
        "full_accuracy": both_correct / total,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in a .env file first.")
        exit(1)

    run_evaluation(api_key)
    