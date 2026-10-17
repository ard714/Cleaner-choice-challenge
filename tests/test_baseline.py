import unittest
from contextlib import redirect_stdout
from io import StringIO

from baseline import predict
from run_baseline import main


HOUSEHOLD = {"household_id": "H001", "budget": "middle", "usual_brand": "standard"}
VISIT = {"visit_number": 2, "channel": "retail"}
OFFERS = [
    {"product_id": "entrant", "price": 4.49, "message": "cleaning"},
    {"product_id": "standard", "price": 3.99, "message": ""},
    {"product_id": "budget", "price": 2.99, "message": ""},
]


class BaselineTests(unittest.TestCase):
    def test_example_summary_shows_both_public_sets(self):
        output = StringIO()
        with redirect_stdout(output):
            main()
        self.assertIn("Build visits: 1200", output.getvalue())
        self.assertIn("Check visits: 300", output.getvalue())

    def test_probabilities_cover_each_offer_and_none(self):
        result = predict(HOUSEHOLD, OFFERS, VISIT, [])
        self.assertEqual(set(result["probabilities"]), {"entrant", "standard", "budget", "none"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_higher_price_reduces_entrant_probability(self):
        first = predict(HOUSEHOLD, OFFERS, VISIT, [])["probabilities"]["entrant"]
        higher = [dict(offer) for offer in OFFERS]
        higher[0]["price"] = 5.49
        second = predict(HOUSEHOLD, higher, VISIT, [])["probabilities"]["entrant"]
        self.assertLess(second, first)

    def test_duplicate_product_is_rejected(self):
        with self.assertRaises(ValueError):
            predict(HOUSEHOLD, OFFERS + [OFFERS[0]], VISIT, [])

    def test_recent_purchase_changes_none_probability(self):
        first = predict(HOUSEHOLD, OFFERS, VISIT, [])["probabilities"]["none"]
        history = [{"choice": "standard"}]
        second = predict(HOUSEHOLD, OFFERS, VISIT, history)["probabilities"]["none"]
        self.assertGreater(second, first)

    def test_usual_brand_can_be_top_choice(self):
        result = predict(HOUSEHOLD, OFFERS, VISIT, [])
        self.assertEqual(max(result["probabilities"], key=result["probabilities"].get), "standard")

    def test_channel_changes_entrant_probability(self):
        retail = predict(HOUSEHOLD, OFFERS, VISIT, [])["probabilities"]["entrant"]
        online = predict(HOUSEHOLD, OFFERS, {**VISIT, "channel": "online"}, [])["probabilities"]["entrant"]
        self.assertGreater(online, retail)

    def test_missing_survey_and_message_fields_do_not_stop_prediction(self):
        household = {**HOUSEHOLD, "usual_brand": ""}
        offers = [dict(offer) for offer in OFFERS]
        offers[0]["message"] = ""
        result = predict(household, offers, VISIT, [])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
