# Final Recommendation

I recommend SimCo launch its initial US commercial test at **$4.49** with an **"environment"** shelf message exclusively in the **online** channel.

Across 18 simulated policy cells in `run_012_policy_simulation`, online placement with environmental messaging maximizes entrant capture (26.0% market share; 37.5% eco-conscious household penetration). While testing at $3.49 achieves slightly higher volume (33.8%), gross revenue per visit is virtually identical ($1.17 at $4.49 vs. $1.18 at $3.49). Because producing and fulfilling a 16 fl oz liquid cleaner in the US exceeds the $0.15 unit breakeven cost, $4.49 delivers significantly superior net margin.

This decision was confirmed by `run_009_accuracy_refinement` (champion 25-parameter choice model) and validated out-of-sample on 50 unseen households in `run_011_frozen_check_evaluation` (1.5505 log loss, 1.6% entrant calibration gap). Relaxing purchase hierarchy via `run_010_nested_logit` degraded performance, confirming shoppers jointly evaluate brand alternatives.

**Falsification Criteria:**
In `run_005_feedback_stain_memory` and exploratory data analysis, exactly 0 of 11 households repurchased after reporting a stain complaint (100% empirical churn). If a 60-day real-world online cohort test yields a repeat reorder rate below **20.0%**, the eco-formulation cannot retain consumers, and I will withdraw this recommendation.
