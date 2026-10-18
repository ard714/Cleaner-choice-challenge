"""Run 001: Starter Baseline Benchmark Replication & Logging.

Evaluates the uncalibrated heuristic starter model on visits.csv (1,200 visits)
and check_visits.csv (300 visits). Establishes initial anchor metrics:
- Build Accuracy: 29.0% | Entrant Expected: 15.5% (vs 11.2% observed, +4.3% calibration gap)
- Check Accuracy: 28.0% | Entrant Expected: 13.1% (vs 11.0% observed, +2.1% calibration gap)
- Mean CV Log Loss: 1.5793 | Brier Score: 0.7850
"""

import os
import sys
from pathlib import Path

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import baseline
import run_baseline
from tracker import list_runs, DB_PATH


def main():
    print("=" * 70)
    print("RUN 001: STARTER BASELINE REPLICATION")
    print("=" * 70)
    
    # Run standard baseline summary
    run_baseline.main()
    
    # Verify tracker registration
    past_runs = list_runs(DB_PATH)
    has_run_001 = any("run_001" in r["run_id"] for r in past_runs)
    print(f"\nRuns Database Status: {'Verified (run_001 present)' if has_run_001 else 'Needs logging'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
