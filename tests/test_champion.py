"""Unit tests for the Champion Choice Model (v1.1-mle-refined) and Trust Engine."""

import csv
import unittest
from pathlib import Path

import app

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ChampionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.init_app()

    def test_probabilities_sum_to_one(self):
        res = app.run_prediction({"household_id": "H004", "entrant_price": 4.49, "channel": "retail"})
        probs = res["probabilities"]
        self.assertEqual(set(probs.keys()), {"entrant", "standard", "budget", "scented", "none"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

    def test_law_of_demand_price_sensitivity(self):
        # Higher entrant price must strictly reduce entrant choice probability
        res_low = app.run_prediction({"household_id": "H004", "entrant_price": 3.99, "channel": "retail"})
        res_high = app.run_prediction({"household_id": "H004", "entrant_price": 6.49, "channel": "retail"})
        self.assertLess(res_high["probabilities"]["entrant"], res_low["probabilities"]["entrant"])

    def test_post_use_stain_penalty(self):
        # A prior stain failure must drastically penalize entrant utility
        res_clean = app.run_prediction({"household_id": "H004", "had_stain_failure": False})
        res_stain = app.run_prediction({"household_id": "H004", "had_stain_failure": True, "stain_product": "entrant"})
        self.assertLess(res_stain["probabilities"]["entrant"], res_clean["probabilities"]["entrant"])
        self.assertTrue(any("stain" in r.lower() for r in res_stain["grounded_reasons"]))

    def test_trust_abstention_on_extrapolation(self):
        # Out-of-distribution price $19.99 must trigger "I Don't Know" abstention
        res = app.run_prediction({"household_id": "H004", "entrant_price": 19.99})
        self.assertTrue(res["should_abstain"])
        self.assertIn("I DON'T KNOW", res["trust_status"])
        self.assertIn("extrapolate", res["trust_warning"].lower())

    def test_trust_abstention_on_synthetic_stress_household(self):
        # H901_OOD_LUXURY has extreme income and extreme spend
        res = app.run_prediction({"household_id": "H901_OOD_LUXURY", "entrant_price": 14.99})
        self.assertTrue(res["should_abstain"])

    def test_check_visits_dataset_integrity(self):
        # Strict validation of frozen check dataset
        check_path = DATA_DIR / "check_visits.csv"
        self.assertTrue(check_path.exists())
        with check_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 300)
        households = {r["household_id"] for r in rows}
        self.assertEqual(len(households), 50)


if __name__ == "__main__":
    unittest.main()
