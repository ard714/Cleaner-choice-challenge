"""A small choice model. It does not learn from the visit data."""

from math import exp


def predict(household, offers, visit, history):
    """Return one probability for each offer and for no purchase."""
    scores = {"none": 0.2 + (0.35 if history and history[-1]["choice"] != "none" else 0)}
    reasons = {"none": "The model expects no purchase on this visit."}

    for offer in offers:
        product = offer["product_id"]
        price = float(offer["price"])
        if product == "none" or product in scores or price <= 0:
            raise ValueError("Each offer needs a unique product and a positive price.")

        score = 2.3 - 0.4 * price
        reason = "The model gives this cleaner a low price penalty."
        if household["usual_brand"] == product:
            score += 0.55
            reason = "The model favors the household's usual brand."
        if household["budget"] == "tight":
            score -= 0.18 * price
            reason = "The model gives this household a larger price penalty."
        if history and history[-1]["choice"] == product:
            score += 0.1
            reason = "The model favors the most recent choice."
        if product == "entrant":
            score += {"cleaning": 0.08, "low_odor": 0.06, "environment": 0.15}.get(offer.get("message"), 0)
            if visit["channel"] == "online":
                score += 0.15

        scores[product] = score
        reasons[product] = reason

    peak = max(scores.values())
    weights = {name: exp(score - peak) for name, score in scores.items()}
    total = sum(weights.values())
    probabilities = {name: weight / total for name, weight in weights.items()}
    top = max(probabilities, key=probabilities.get)
    return {"probabilities": probabilities, "reason": reasons[top]}
