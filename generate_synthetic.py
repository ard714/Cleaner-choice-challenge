"""Generate synthetic stress-testing datasets for epistemic uncertainty and trust evaluation.

Creates:
1. data/synthetic_households.csv (10 synthetic stress-test households: H901-H910)
2. data/synthetic_stress_visits.csv (50 synthetic stress-test shopper visits)

Designed specifically to probe:
- Extreme Out-of-Distribution (OOD) pricing (extrapolation hazard)
- Maximum Entropy ties (equiprobable choices where confidence < 25%)
- Severe contradictory signals (opposing sentiment and feedback)
- Completely blank cold-start profiles
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SYNTH_HOUSEHOLDS = [
    # OOD Price Testers
    {"household_id": "H901_OOD_LUXURY", "income_band": "over_100k", "cleaning_spend_usd": "150", "budget": "generous", "bathrooms": "4", "usual_brand": "standard", "note": "I only purchase ultra-luxury artisanal imported cleaners."},
    {"household_id": "H902_OOD_SUBSIDIZED", "income_band": "under_50k", "cleaning_spend_usd": "2", "budget": "tight", "bathrooms": "1", "usual_brand": "budget", "note": "Looking for government subsidized surplus supplies."},
    
    # Maximum Entropy / Tie Testers (No discriminative signal)
    {"household_id": "H903_NEUTRAL_TIE", "income_band": "50k_to_100k", "cleaning_spend_usd": "15", "budget": "middle", "bathrooms": "1", "usual_brand": "", "note": "I have no preference and pick whatever bottle my hand touches first."},
    {"household_id": "H904_INDIFFERENT_TIED", "income_band": "50k_to_100k", "cleaning_spend_usd": "15", "budget": "middle", "bathrooms": "2", "usual_brand": "", "note": ""},
    
    # Contradictory Sentiment & Feedback
    {"household_id": "H905_CONTRADICTORY", "income_band": "50k_to_100k", "cleaning_spend_usd": "20", "budget": "middle", "bathrooms": "2", "usual_brand": "entrant", "note": "I worry about cleaners that go down the drain. I do not trust green claims."},
    {"household_id": "H906_FEEDBACK_CHAOS", "income_band": "50k_to_100k", "cleaning_spend_usd": "25", "budget": "flexible", "bathrooms": "2", "usual_brand": "standard", "note": "I like a fresh scent but hate perfumes and sharp chemical smells."},
    
    # Extreme Cold-Start (100% Blank)
    {"household_id": "H907_TOTAL_BLANK", "income_band": "", "cleaning_spend_usd": "", "budget": "", "bathrooms": "", "usual_brand": "", "note": ""},
    {"household_id": "H908_UNSEEN_ANOMALY", "income_band": "unspecified", "cleaning_spend_usd": "-10", "budget": "infinite", "bathrooms": "0", "usual_brand": "alien_cleaner", "note": "1234567890 !@#$%^&*()"},
    
    # Boundary Stress
    {"household_id": "H909_EXTREME_STAIN_CHURN", "income_band": "over_100k", "cleaning_spend_usd": "40", "budget": "flexible", "bathrooms": "3", "usual_brand": "entrant", "note": "I tried entrant and standard and budget."},
    {"household_id": "H910_ZERO_NEED_SKIP", "income_band": "over_100k", "cleaning_spend_usd": "5", "budget": "generous", "bathrooms": "1", "usual_brand": "standard", "note": "I only clean once a year."},
]

def generate_synthetic_visits():
    visits = []
    v_id = 1
    
    # 1. H901: Extreme Luxury Price Stress ($14.99, $19.99, $24.99) -> OOD
    for i, p in enumerate([12.99, 16.49, 19.99, 24.99, 29.99], 1):
        visits.append({
            "visit_id": f"V901-{i}",
            "household_id": "H901_OOD_LUXURY",
            "visit_number": str(i),
            "channel": "online",
            "entrant_price": f"{p:.2f}",
            "entrant_message": "environment",
            "standard_price": "3.99",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": "standard",
            "said_why": "Price was excessively high.",
            "after_use_note": "",
        })

    # 2. H902: Subsidized Penny Price Stress ($0.29, $0.49, $0.99) -> OOD
    for i, p in enumerate([0.29, 0.49, 0.79, 0.99, 1.29], 1):
        visits.append({
            "visit_id": f"V902-{i}",
            "household_id": "H902_OOD_SUBSIDIZED",
            "visit_number": str(i),
            "channel": "retail",
            "entrant_price": f"{p:.2f}",
            "entrant_message": "cleaning",
            "standard_price": "3.99",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": "entrant",
            "said_why": "Unrealistically cheap price.",
            "after_use_note": "",
        })

    # 3. H903: Perfectly Tied Prices ($3.50 all) -> Maximum Entropy / Low Confidence
    for i, ch in enumerate(["retail", "retail", "online", "online", "retail"], 1):
        visits.append({
            "visit_id": f"V903-{i}",
            "household_id": "H903_NEUTRAL_TIE",
            "visit_number": str(i),
            "channel": ch,
            "entrant_price": "3.50",
            "entrant_message": "",
            "standard_price": "3.50",
            "budget_price": "3.50",
            "scented_price": "3.50",
            "choice": "none",
            "said_why": "Indifferent between all four brands.",
            "after_use_note": "",
        })

    # 4. H904: Perfectly Tied Prices ($3.99 all) -> Maximum Entropy
    for i in range(1, 6):
        visits.append({
            "visit_id": f"V904-{i}",
            "household_id": "H904_INDIFFERENT_TIED",
            "visit_number": str(i),
            "channel": "online",
            "entrant_price": "3.99",
            "entrant_message": "",
            "standard_price": "3.99",
            "budget_price": "3.99",
            "scented_price": "3.99",
            "choice": "budget",
            "said_why": "No preference.",
            "after_use_note": "",
        })

    # 5. H905: Contradictory Eco vs Skeptic
    for i, msg in enumerate(["environment", "cleaning", "low_odor", "environment", "cleaning"], 1):
        visits.append({
            "visit_id": f"V905-{i}",
            "household_id": "H905_CONTRADICTORY",
            "visit_number": str(i),
            "channel": "online",
            "entrant_price": "4.49",
            "entrant_message": msg,
            "standard_price": "3.99",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": "standard",
            "said_why": "Distrusted claims.",
            "after_use_note": "",
        })

    # 6. H906: Conflicting Feedback (stain failure followed by positive note)
    feedback_sequence = [
        ("entrant", "It left a stain behind"),
        ("entrant", "It removed the stain I noticed"),
        ("standard", "The smell lasted too long"),
        ("scented", "It left a stain behind"),
        ("budget", "Easy to use bottle"),
    ]
    for i, (c, note) in enumerate(feedback_sequence, 1):
        visits.append({
            "visit_id": f"V906-{i}",
            "household_id": "H906_FEEDBACK_CHAOS",
            "visit_number": str(i),
            "channel": "retail",
            "entrant_price": "4.49",
            "entrant_message": "low_odor",
            "standard_price": "3.99",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": c,
            "said_why": "Conflicted feedback testing.",
            "after_use_note": note,
        })

    # 7. H907: Total Blank Cold-Start
    for i in range(1, 6):
        visits.append({
            "visit_id": f"V907-{i}",
            "household_id": "H907_TOTAL_BLANK",
            "visit_number": str(i),
            "channel": "retail",
            "entrant_price": "4.49",
            "entrant_message": "",
            "standard_price": "3.99",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": "standard",
            "said_why": "",
            "after_use_note": "",
        })

    # 8. H908: Negative and Anomalous Prices (Severe OOD)
    for i, (ep, sp) in enumerate([(9.99, 9.99), (0.10, 0.50), (15.00, 3.99), (4.49, 12.00), (22.50, 22.50)], 1):
        visits.append({
            "visit_id": f"V908-{i}",
            "household_id": "H908_UNSEEN_ANOMALY",
            "visit_number": str(i),
            "channel": "online",
            "entrant_price": f"{ep:.2f}",
            "entrant_message": "super_cleaner",
            "standard_price": f"{sp:.2f}",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": "none",
            "said_why": "Price anomaly.",
            "after_use_note": "",
        })

    # 9. H909: All Products Failed (Stains across all brands)
    all_stain_notes = [
        ("entrant", "It left a stain behind"),
        ("standard", "It left a stain behind"),
        ("budget", "It left a stain behind"),
        ("scented", "It left a stain behind"),
        ("none", ""),
    ]
    for i, (c, note) in enumerate(all_stain_notes, 1):
        visits.append({
            "visit_id": f"V909-{i}",
            "household_id": "H909_EXTREME_STAIN_CHURN",
            "visit_number": str(i),
            "channel": "online",
            "entrant_price": "4.49",
            "entrant_message": "cleaning",
            "standard_price": "3.99",
            "budget_price": "2.99",
            "scented_price": "4.29",
            "choice": c,
            "said_why": "Multiple brand failures.",
            "after_use_note": note,
        })

    # 10. H910: Constant None Non-Purchase Shopper
    for i in range(1, 6):
        visits.append({
            "visit_id": f"V910-{i}",
            "household_id": "H910_ZERO_NEED_SKIP",
            "visit_number": str(i),
            "channel": "retail",
            "entrant_price": "5.49",
            "entrant_message": "environment",
            "standard_price": "4.49",
            "budget_price": "3.49",
            "scented_price": "4.89",
            "choice": "none",
            "said_why": "Zero need.",
            "after_use_note": "",
        })

    return visits

def main():
    print("Writing data/synthetic_households.csv...")
    with (DATA_DIR / "synthetic_households.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["household_id", "income_band", "cleaning_spend_usd", "budget", "bathrooms", "usual_brand", "note"])
        writer.writeheader()
        writer.writerows(SYNTH_HOUSEHOLDS)
    print(f"Wrote {len(SYNTH_HOUSEHOLDS)} synthetic households.")

    visits = generate_synthetic_visits()
    print("Writing data/synthetic_stress_visits.csv...")
    with (DATA_DIR / "synthetic_stress_visits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["visit_id", "household_id", "visit_number", "channel", "entrant_price", "entrant_message", "standard_price", "budget_price", "scented_price", "choice", "said_why", "after_use_note"])
        writer.writeheader()
        writer.writerows(visits)
    print(f"Wrote {len(visits)} synthetic stress visits.")

if __name__ == "__main__":
    main()
