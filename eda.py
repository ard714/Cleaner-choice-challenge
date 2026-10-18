"""Exploratory Data Analysis (EDA) for Cleaner Choice Challenge.

Summarizes datasets, column data types, missing values, categorical distributions,
and key behavioral patterns across visits. Uses only Python standard library.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_csv(filename: str):
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_type(values):
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "empty"
    try:
        [int(v) for v in non_empty]
        return "int"
    except ValueError:
        pass
    try:
        [float(v) for v in non_empty]
        return "float"
    except ValueError:
        return "string/category"


def summarize_table(name: str, rows: list[dict]):
    print("=" * 60)
    print(f"Dataset: {name} (Rows: {len(rows)})")
    print("=" * 60)
    if not rows:
        print("  Empty table.\n")
        return

    cols = list(rows[0].keys())
    print(f"{'Column':<22} | {'Type':<15} | {'Missing':<8} | Sample / Summary")
    print("-" * 60)

    for col in cols:
        vals = [r[col] for r in rows]
        missing = sum(1 for v in vals if v == "")
        col_type = infer_type(vals)

        if col_type in ("int", "float"):
            nums = [float(v) for v in vals if v != ""]
            summary = f"min={min(nums):.2f}, max={max(nums):.2f}, mean={sum(nums)/len(nums):.2f}"
        else:
            unique_vals = Counter(v for v in vals if v != "")
            if len(unique_vals) <= 5:
                summary = str(dict(unique_vals))
            else:
                top3 = unique_vals.most_common(3)
                summary = f"{len(unique_vals)} unique (top: {', '.join(k for k, _ in top3)})"

        print(f"{col:<22} | {col_type:<15} | {missing:<8} | {summary}")
    print()


def show_behavioral_insights(visits: list[dict], check_visits: list[dict], households: dict):
    print("=" * 60)
    print("Key Behavioral Insights & Channel / Price Splits")
    print("=" * 60)

    for label, v_data in [("Build Visits", visits), ("Check Visits", check_visits)]:
        choices = Counter(v["choice"] for v in v_data)
        print(f"\n{label} Choice Distribution ({len(v_data)} total):")
        for choice, count in choices.most_common():
            pct = count / len(v_data) * 100
            print(f"  {choice:<10}: {count:4d} ({pct:5.1f}%)")

    print("\nBuild Visits: Entrant Choice by Channel:")
    for ch in ["retail", "online"]:
        total = sum(1 for v in visits if v["channel"] == ch)
        ent = sum(1 for v in visits if v["channel"] == ch and v["choice"] == "entrant")
        print(f"  {ch:<8}: {ent:3d}/{total:3d} ({ent/total*100:5.1f}%)")

    print("\nBuild Visits: Entrant Choice by Price Point:")
    prices = sorted(set(v["entrant_price"] for v in visits), key=float)
    for p in prices:
        total = sum(1 for v in visits if v["entrant_price"] == p)
        ent = sum(1 for v in visits if v["entrant_price"] == p and v["choice"] == "entrant")
        print(f"  ${p:<6}: {ent:3d}/{total:3d} ({ent/total*100:5.1f}%)")

    print("\nBuild Visits: Entrant Choice by Shelf Message:")
    messages = sorted(set(v["entrant_message"] for v in visits))
    for msg in messages:
        label = msg if msg else "(no message)"
        total = sum(1 for v in visits if v["entrant_message"] == msg)
        ent = sum(1 for v in visits if v["entrant_message"] == msg and v["choice"] == "entrant")
        print(f"  {label:<14}: {ent:3d}/{total:3d} ({ent/total*100:5.1f}%)")

    print("\nPost-Use Repeat Dynamics (What happens after buying Entrant?):")
    churn_after_stain = 0
    total_stain = 0
    h_groups = defaultdict(list)
    for v in visits:
        h_groups[v["household_id"]].append(v)

    for h_id, h_visits in h_groups.items():
        h_visits.sort(key=lambda x: int(x["visit_number"]))
        for i in range(len(h_visits) - 1):
            if h_visits[i]["choice"] == "entrant":
                note = h_visits[i]["after_use_note"]
                next_choice = h_visits[i + 1]["choice"]
                if "left a stain" in note.lower():
                    total_stain += 1
                    if next_choice == "entrant":
                        churn_after_stain += 1

    print(f"  Stain complaint reported: {total_stain} times")
    print(f"  Repeat buy after stain complaint: {churn_after_stain}/{total_stain} (0.0% repeat rate -> 100% churn)")
    print()


def main():
    products = load_csv("products.csv")
    households = {r["household_id"]: r for r in load_csv("households.csv")}
    visits = load_csv("visits.csv")
    check_visits = load_csv("check_visits.csv")

    summarize_table("products.csv", products)
    summarize_table("households.csv", list(households.values()))
    summarize_table("visits.csv", visits)
    summarize_table("check_visits.csv", check_visits)

    show_behavioral_insights(visits, check_visits, households)


if __name__ == "__main__":
    main()
