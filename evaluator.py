"""Evaluation engine for Cleaner Choice Challenge.

Provides mathematically rigorous metrics:
- Multi-class Log Loss (Cross-Entropy)
- Multi-class Brier Score
- Top-1 Accuracy
- Entrant Calibration Gap (Observed vs Expected share)
- 5-Fold GroupKFold partitioning by household_id (strictly zero household leakage)
Uses only Python standard library.
"""

import math
from collections import Counter, defaultdict
from typing import Any, Callable

PRODUCTS = ["entrant", "standard", "budget", "scented", "none"]
EPSILON = 1e-15


def multi_class_log_loss(y_true: list[str], y_prob_dicts: list[dict[str, float]]) -> float:
    """Compute average negative log-likelihood (multi-class cross-entropy)."""
    if not y_true:
        return 0.0
    total_loss = 0.0
    for true_label, prob_dict in zip(y_true, y_prob_dicts):
        prob = max(min(prob_dict.get(true_label, 0.0), 1.0 - EPSILON), EPSILON)
        total_loss += -math.log(prob)
    return total_loss / len(y_true)


def multi_class_brier_score(y_true: list[str], y_prob_dicts: list[dict[str, float]]) -> float:
    """Compute multi-class Brier score (mean squared error of probability vectors)."""
    if not y_true:
        return 0.0
    total_brier = 0.0
    for true_label, prob_dict in zip(y_true, y_prob_dicts):
        sample_brier = 0.0
        for p in PRODUCTS:
            target = 1.0 if p == true_label else 0.0
            prob = prob_dict.get(p, 0.0)
            sample_brier += (prob - target) ** 2
        total_brier += sample_brier
    return total_brier / len(y_true)


def top1_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Fraction of exact top-choice matches."""
    if not y_true:
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def offers_for_visit(visit: dict[str, Any]) -> list[dict[str, Any]]:
    """Standard offer list extractor matching starter kit."""
    return [
        {"product_id": "entrant", "price": float(visit["entrant_price"]), "message": visit.get("entrant_message", "")},
        {"product_id": "standard", "price": float(visit["standard_price"])},
        {"product_id": "budget", "price": float(visit["budget_price"])},
        {"product_id": "scented", "price": float(visit["scented_price"])},
    ]


def group_k_fold_households(household_ids: list[str], n_splits: int = 5, seed: int = 42) -> list[tuple[set[str], set[str]]]:
    """Partition household IDs into n deterministic folds.
    
    Returns list of (train_hids, val_hids).
    Ensures zero household overlap between train and validation.
    """
    sorted_ids = sorted(household_ids)
    import random
    rng = random.Random(seed)
    shuffled = list(sorted_ids)
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // n_splits
    folds = []
    for i in range(n_splits):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else len(shuffled)
        val_ids = set(shuffled[val_start:val_end])
        train_ids = set(shuffled) - val_ids
        folds.append((train_ids, val_ids))
    return folds


def evaluate_predictions(y_true: list[str], y_prob_dicts: list[dict[str, float]]) -> dict[str, Any]:
    """Compute complete metric suite on predictions."""
    n = len(y_true)
    if n == 0:
        return {}

    y_pred = [max(p, key=p.get) for p in y_prob_dicts]
    acc = top1_accuracy(y_true, y_pred)
    loss = multi_class_log_loss(y_true, y_prob_dicts)
    brier = multi_class_brier_score(y_true, y_prob_dicts)

    entrant_observed = sum(1 for y in y_true if y == "entrant") / n
    entrant_expected = sum(p.get("entrant", 0.0) for p in y_prob_dicts) / n
    entrant_cal_gap = abs(entrant_expected - entrant_observed)

    return {
        "total_samples": n,
        "accuracy": acc,
        "log_loss": loss,
        "brier_score": brier,
        "entrant_observed": entrant_observed,
        "entrant_expected": entrant_expected,
        "entrant_calibration_gap": entrant_cal_gap,
        "choice_observed": dict(Counter(y_true)),
        "choice_predicted": dict(Counter(y_pred)),
    }


def run_sequential_evaluation(
    predict_fn: Callable,
    visits: list[dict[str, Any]],
    households: dict[str, dict[str, Any]],
    allowed_household_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Simulate sequential shopping trips per household in time order.
    
    Carries history forward causally (only preceding visits are available).
    Returns (metrics_dict, detailed_results_list).
    """
    if allowed_household_ids is not None:
        filtered_visits = [v for v in visits if v["household_id"] in allowed_household_ids]
    else:
        filtered_visits = list(visits)

    # Sort visits by household and visit_number to guarantee strict causality
    sorted_visits = sorted(filtered_visits, key=lambda v: (v["household_id"], int(v["visit_number"])))

    history_by_hh: dict[str, list[dict[str, Any]]] = defaultdict(list)
    y_true: list[str] = []
    y_probs: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []

    for visit in sorted_visits:
        hh_id = visit["household_id"]
        hh = households[hh_id]
        offers = offers_for_visit(visit)
        current_history = history_by_hh[hh_id]

        pred_result = predict_fn(hh, offers, visit, current_history)
        probs = pred_result["probabilities"]
        reason = pred_result.get("reason", "")

        y_true.append(visit["choice"])
        y_probs.append(probs)

        details.append({
            "visit_id": visit.get("visit_id", ""),
            "household_id": hh_id,
            "visit_number": int(visit["visit_number"]),
            "actual_choice": visit["choice"],
            "predicted_choice": max(probs, key=probs.get),
            "probabilities": probs,
            "reason": reason,
        })

        # Append to history *after* prediction
        history_by_hh[hh_id].append(visit)

    metrics = evaluate_predictions(y_true, y_probs)
    return metrics, details
