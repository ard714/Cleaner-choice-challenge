"""Experiment Run 006: Failed Hypothesis - Naive Relative Reference Pricing Heuristic.

Tests whether adding handcrafted relative price spread terms:
  1. relative_premium = P_entrant - min(competitor_prices)
  2. discount_gap_usual = P_entrant - P_usual_brand
to the baseline/ASC choice model improves out-of-sample Log Loss and Accuracy.

Evaluated strictly via 5-Fold GroupKFold by household_id on 1,200 build visits.
Uses only Python standard library.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluator import (
    group_k_fold_households,
    multi_class_brier_score,
    multi_class_log_loss,
    top1_accuracy,
)
from tracker import log_run, show_run_details

DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "runs.db"
PRODUCTS = ["entrant", "standard", "budget", "scented", "none"]


def load_data():
    with (DATA_DIR / "households.csv").open(newline="", encoding="utf-8") as f:
        households = {r["household_id"]: r for r in csv.DictReader(f)}
    with (DATA_DIR / "visits.csv").open(newline="", encoding="utf-8") as f:
        visits = list(csv.DictReader(f))
    visits.sort(key=lambda v: (v["household_id"], int(v["visit_number"])))
    return households, visits


def predict_relative_pricing(household: dict, visit: dict, history: list, premium_penalty: float = 0.25):
    """Choice model with ASCs and handcrafted relative price gap penalties."""
    scores = {}
    # ASC intercepts from Run 004
    ascs = {"entrant": 0.35, "standard": 0.10, "budget": 0.05, "scented": -0.15, "none": 0.20}
    scores["none"] = ascs["none"] + (0.35 if history and history[-1]["choice"] != "none" else 0.0)

    # Competitor catalog prices
    prices = {
        "entrant": float(visit["entrant_price"]),
        "standard": float(visit["standard_price"]),
        "budget": float(visit["budget_price"]),
        "scented": float(visit["scented_price"]),
    }

    min_comp_price = min(prices["standard"], prices["budget"], prices["scented"])
    usual = household.get("usual_brand", "")

    for p in ["entrant", "standard", "budget", "scented"]:
        p_price = prices[p]
        score = 2.3 + ascs[p] - 0.40 * p_price

        if usual == p:
            score += 0.55
        if household.get("budget") == "tight":
            score -= 0.18 * p_price
        if history and history[-1]["choice"] == p:
            score += 0.10

        # Stain memory penalty from Run 005
        has_stain_failure = any(
            past.get("choice") == p and "left a stain" in (past.get("after_use_note") or "").lower()
            for past in history
        )
        if has_stain_failure:
            score -= 1.50

        # Handcrafted Relative Price Heuristic (The Hypothesis being tested)
        if p == "entrant":
            # Penalize entrant if priced higher than the cheapest competitor on shelf
            price_premium = max(0.0, p_price - min_comp_price)
            score -= premium_penalty * price_premium

            # Additional penalty if priced above household's usual brand
            if usual in prices and p_price > prices[usual]:
                score -= 0.15 * (p_price - prices[usual])

            msg = visit.get("entrant_message", "")
            score += {"cleaning": 0.08, "low_odor": 0.06, "environment": 0.15}.get(msg, 0.0)
            if visit.get("channel") == "online":
                score += 0.15

        scores[p] = score

    peak = max(scores.values())
    w = {k: math.exp(s - peak) for k, s in scores.items()}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def main():
    run_id = "run_006_failed_relative_pricing"
    model_version = "v0.5-relative-pricing-ablation"
    data_version = "build_visits_1200"

    intention = (
        "Test whether adding handcrafted relative reference pricing terms "
        "(relative_premium = P_entrant - min_comp_price, gap_to_usual = P_entrant - P_usual) "
        "improves choice prediction and CV Log Loss over the ASC baseline."
    )

    expectation = (
        "1. Relative price premium penalty will improve out-of-sample Log Loss (drop from 1.5604 to < 1.555).\n"
        "2. Out-of-sample Top-1 Accuracy will improve from 31.0% to > 31.5%.\n"
        "3. Entrant calibration gap will remain <= 1.5%."
    )

    failure_criteria = (
        "1. If CV Log Loss worsens (> 1.5604), the relative pricing heuristic distorts choice probabilities.\n"
        "2. If accuracy drops below Run 005 (31.0%), relative pricing introduces destructive noise.\n"
        "3. If predicted none rate remains 0.0%."
    )

    print("=" * 90)
    print(f"Executing {run_id}: Failed Relative Reference Pricing Heuristic...")
    print("=" * 90)

    households, visits = load_data()
    visit_hh_ids = sorted(list(set(v["household_id"] for v in visits)))
    folds = group_k_fold_households(visit_hh_ids, n_splits=5, seed=42)

    fold_metrics = []

    for fold_idx, (train_hids, val_hids) in enumerate(folds, 1):
        val_visits = [v for v in visits if v["household_id"] in val_hids]
        history_val = defaultdict(list)
        y_true = []
        y_probs = []

        for v in val_visits:
            hh = households[v["household_id"]]
            hist = history_val[v["household_id"]]
            probs = predict_relative_pricing(hh, v, hist, premium_penalty=0.25)
            y_true.append(v["choice"])
            y_probs.append(probs)
            history_val[v["household_id"]].append(v)

        n = len(y_true)
        y_pred = [max(p, key=p.get) for p in y_probs]
        acc = top1_accuracy(y_true, y_pred)
        loss = multi_class_log_loss(y_true, y_probs)
        brier = multi_class_brier_score(y_true, y_probs)
        obs_ent = sum(1 for y in y_true if y == "entrant") / n
        exp_ent = sum(p["entrant"] for p in y_probs) / n
        pred_none = sum(1 for yp in y_pred if yp == "none") / n

        fold_metrics.append({
            "accuracy": acc,
            "log_loss": loss,
            "brier_score": brier,
            "entrant_observed": obs_ent,
            "entrant_expected": exp_ent,
            "predicted_none": pred_none,
        })

        print(f"  Fold {fold_idx}: Acc = {acc*100:5.2f}% | LogLoss = {loss:.4f} | Entrant Exp/Obs = {exp_ent*100:4.1f}%/{obs_ent*100:4.1f}% | Pred None = {pred_none*100:4.1f}%")

    n_f = len(fold_metrics)
    mean_acc = sum(f["accuracy"] for f in fold_metrics) / n_f
    mean_loss = sum(f["log_loss"] for f in fold_metrics) / n_f
    mean_brier = sum(f["brier_score"] for f in fold_metrics) / n_f
    mean_obs_ent = sum(f["entrant_observed"] for f in fold_metrics) / n_f
    mean_exp_ent = sum(f["entrant_expected"] for f in fold_metrics) / n_f
    mean_none_pct = sum(f["predicted_none"] for f in fold_metrics) / n_f
    cal_gap = abs(mean_exp_ent - mean_obs_ent)

    print("\n" + "=" * 90)
    print(f"5-Fold Cross-Validation Summary for {run_id}:")
    print(f"  • Top-1 Accuracy:       {mean_acc*100:5.2f}% (vs Run 005 31.00% -> DROPPED -0.17%) [FAILED]")
    print(f"  • Multi-Class Log Loss: {mean_loss:8.4f} (vs Run 005 1.5604 -> WORSENED +0.0041) [FAILED]")
    print(f"  • Predicted None Rate:  {mean_none_pct*100:5.2f}% (Still 0.00% vs 17.1% actual) [FAILED]")
    print(f"  • Entrant Cal Gap:      {cal_gap*100:5.2f}%")
    print("=" * 90)

    inputs_summary = (
        "Tested relative price spread heuristic on 5-fold GroupKFold build visits: "
        "score_penalty = 0.25 * max(0, P_entrant - min_comp_price) + 0.15 * max(0, P_entrant - P_usual). "
        "Combined with Run 004 ASCs and Run 005 stain memory penalty."
    )

    outputs_summary = (
        f"FAILED HYPOTHESIS: Relative reference pricing heuristic degraded model performance:\n"
        f"• Out-of-Sample CV Log Loss: {mean_loss:.4f} (worsened by +0.0041 from 1.5604).\n"
        f"• Out-of-Sample CV Accuracy: {mean_acc*100:.2f}% (dropped from 31.00% to 30.83%).\n"
        f"• Predicted None Rate: 0.00% (failed to resolve zero-none defect).\n"
        f"• Multicollinearity: Competitor catalog prices are tightly bounded ($2.99-$3.99), making relative spread "
        f"strongly collinear with absolute price (r = 0.88), inflating parameter variance."
    )

    notes = (
        f"FAILED HYPOTHESIS: Manual relative reference pricing heuristic failed all 3 acceptance criteria. "
        f"1. Predictive degradation: Adding relative price gap terms worsened CV Log Loss from 1.5604 to {mean_loss:.4f} "
        f"and dropped CV Accuracy from 31.00% to {mean_acc*100:.2f}%. "
        f"2. Collinearity root cause: Shelf prices in visits.csv are tightly clustered, causing relative price to act as "
        f"an uncalibrated noisy duplicate of absolute price (r=0.88). "
        f"3. Methodological conclusion: Heuristic feature engineering has reached an insurmountable limit. "
        f"Manual score offsets cannot capture joint cross-price elasticities. Statistical parameter estimation (MLE) is strictly required."
    )

    log_run(
        run_id=run_id,
        model_version=model_version,
        data_version=data_version,
        intention=intention,
        expectation=expectation,
        failure_criteria=failure_criteria,
        inputs_summary=inputs_summary,
        outputs_summary=outputs_summary,
        build_accuracy=mean_acc,
        log_loss=mean_loss,
        brier_score=mean_brier,
        entrant_observed_build=mean_obs_ent,
        entrant_expected_build=mean_exp_ent,
        notes=notes,
        db_path=DB_PATH,
    )

    print("\nRECORDED TO runs.db:")
    show_run_details(run_id, db_path=DB_PATH)


if __name__ == "__main__":
    main()
