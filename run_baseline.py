"""Read the example data and print a small baseline summary."""

import csv
from pathlib import Path

from baseline import predict


DATA = Path(__file__).parent / "data"


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def offers_for(visit):
    return [
        {"product_id": "entrant", "price": float(visit["entrant_price"]), "message": visit["entrant_message"]},
        {"product_id": "standard", "price": float(visit["standard_price"])},
        {"product_id": "budget", "price": float(visit["budget_price"])},
        {"product_id": "scented", "price": float(visit["scented_price"])},
    ]


def show_summary(label, visits, households):
    history = {key: [] for key in households}
    correct = 0
    total = 0
    observed_entrant = 0
    expected_entrant = 0.0

    for visit in visits:
        household_id = visit["household_id"]
        result = predict(households[household_id], offers_for(visit), visit, history[household_id])
        probabilities = result["probabilities"]
        predicted = max(probabilities, key=probabilities.get)
        correct += predicted == visit["choice"]
        observed_entrant += visit["choice"] == "entrant"
        expected_entrant += probabilities["entrant"]
        total += 1
        history[household_id].append(visit)

    print(f"{label} visits: {total}")
    print(f"Top choice matched: {correct / total:.1%}")
    print(f"Entrant chosen: {observed_entrant / total:.1%}")
    print(f"Entrant expected: {expected_entrant / total:.1%}")


def main():
    households = {row["household_id"]: row for row in read_rows("households.csv")}
    show_summary("Build", read_rows("visits.csv"), households)
    show_summary("Check", read_rows("check_visits.csv"), households)


if __name__ == "__main__":
    main()
