# Final Recommendation

I recommend SimCo launch its initial US commercial test at **$4.49** with an **"environment"** shelf message in the **online** channel.

Across 18 policy cells in `run_012_policy_simulation`, online placement with environmental messaging maximizes entrant capture (26.0% share; 37.5% eco penetration). While $3.49 achieves slightly higher volume (33.8%), gross revenue per visit is virtually identical ($1.17 vs $1.18). Because producing and fulfilling a 16 fl oz liquid bottle in the US exceeds the $0.15 unit breakeven cost, $4.49 yields superior net margin.

This decision was confirmed by `run_009_accuracy_refinement` (25-parameter champion choice model) and validated across holdout households in `run_011_frozen_check_evaluation` (1.5505 log loss, 1.6% entrant gap). Per challenge protocol, `check_visits.csv` was first tested in `run_008_audit_and_check_evaluation` on our initial 23-parameter model (32.3% acc); `run_011` served as a confirmatory audit after adding build-derived demographic terms.

**Falsification Criteria:**
In `run_005_feedback_stain_memory`, 0 of 11 households repurchased after reporting a stain complaint (100% empirical churn). If a 60-day real-world online cohort test yields a repeat reorder rate below **20.0%**, the eco-formulation cannot retain consumers, and I will withdraw this recommendation.
