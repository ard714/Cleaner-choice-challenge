"""SQLite experiment tracker for Cleaner Choice Challenge.

Records experiment runs, intentions, expectations, failure criteria,
input configurations, evaluation metrics (accuracy, log loss, calibration),
and learnings to 'runs.db'. Allows inspection of past runs by ID.
Uses only Python standard library.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).parent / "runs.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    """Initialize the runs table schema and migrate any missing columns."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                data_version TEXT NOT NULL,
                expectation TEXT NOT NULL,
                inputs_summary TEXT NOT NULL,
                outputs_summary TEXT NOT NULL,
                build_accuracy REAL,
                check_accuracy REAL,
                entrant_observed_build REAL,
                entrant_expected_build REAL,
                entrant_observed_check REAL,
                entrant_expected_check REAL,
                notes TEXT
            )
            """
        )
        conn.commit()

        # Schema migration for enhanced experiment template columns
        cur = conn.execute("PRAGMA table_info(runs)")
        cols = {row["name"] for row in cur.fetchall()}
        new_cols = [
            ("intention", "TEXT"),
            ("failure_criteria", "TEXT"),
            ("log_loss", "REAL"),
            ("brier_score", "REAL"),
        ]
        for col_name, col_type in new_cols:
            if col_name not in cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
        conn.commit()


def log_run(
    run_id: str,
    model_version: str,
    data_version: str,
    expectation: str,
    inputs_summary: str,
    outputs_summary: str,
    intention: str = "",
    failure_criteria: str = "",
    build_accuracy: Optional[float] = None,
    check_accuracy: Optional[float] = None,
    log_loss: Optional[float] = None,
    brier_score: Optional[float] = None,
    entrant_observed_build: Optional[float] = None,
    entrant_expected_build: Optional[float] = None,
    entrant_observed_check: Optional[float] = None,
    entrant_expected_check: Optional[float] = None,
    notes: str = "",
    db_path: Path = DB_PATH,
):
    """Insert or replace an experiment run record following the standard experiment template."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, timestamp, model_version, data_version, intention, expectation,
                failure_criteria, inputs_summary, outputs_summary, build_accuracy, check_accuracy,
                log_loss, brier_score, entrant_observed_build, entrant_expected_build,
                entrant_observed_check, entrant_expected_check, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now_iso,
                model_version,
                data_version,
                intention,
                expectation,
                failure_criteria,
                inputs_summary,
                outputs_summary,
                build_accuracy,
                check_accuracy,
                log_loss,
                brier_score,
                entrant_observed_build,
                entrant_expected_build,
                entrant_observed_check,
                entrant_expected_check,
                notes,
            ),
        )
        conn.commit()


def get_run(run_id: str, db_path: Path = DB_PATH) -> Optional[dict[str, Any]]:
    """Retrieve a single run by run_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_runs(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all runs ordered chronologically."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT * FROM runs ORDER BY timestamp ASC")
        return [dict(r) for r in cur.fetchall()]


def print_runs(db_path: Path = DB_PATH):
    runs = list_runs(db_path)
    if not runs:
        print("No runs recorded yet in runs.db.")
        return

    print("=" * 105)
    print(f"{'Run ID':<24} | {'Model':<14} | {'Build Acc':<10} | {'LogLoss':<9} | {'Entrant Exp/Obs':<18} | {'Status/Outcome':<18}")
    print("-" * 105)
    for r in runs:
        b_acc = f"{r['build_accuracy']*100:.1f}%" if r["build_accuracy"] is not None else "N/A"
        ll = f"{r['log_loss']:.4f}" if r["log_loss"] is not None else "N/A"
        e_ratio = ""
        if r["entrant_expected_build"] is not None and r["entrant_observed_build"] is not None:
            e_ratio = f"{r['entrant_expected_build']*100:.1f}% / {r['entrant_observed_build']*100:.1f}%"
        outcome = r.get("notes", "")[:18] if r.get("notes") else "Logged"
        print(f"{r['run_id']:<24} | {r['model_version']:<14} | {b_acc:<10} | {ll:<9} | {e_ratio:<18} | {outcome:<18}")
    print("=" * 105)


def show_run_details(run_id: str, db_path: Path = DB_PATH):
    run = get_run(run_id, db_path)
    if not run:
        print(f"Run '{run_id}' not found.")
        return

    print("=" * 75)
    print(f"EXPERIMENT RUN CARD: {run['run_id']}")
    print("=" * 75)
    print(f"Timestamp (UTC): {run['timestamp']}")
    print(f"Model Version:   {run['model_version']}")
    print(f"Data Version:    {run['data_version']}")
    print("-" * 75)
    if run.get("intention"):
        print(f"1. INTENTION & HYPOTHESIS:\n   {run['intention']}\n")
    print(f"2. PRE-RUN EXPECTATION:\n   {run['expectation']}\n")
    if run.get("failure_criteria"):
        print(f"3. FAILURE CRITERIA:\n   {run['failure_criteria']}\n")
    print(f"4. INPUTS & CONFIGURATION:\n   {run['inputs_summary']}\n")
    print(f"5. OUTPUTS & EMPIRICAL RESULTS:\n   {run['outputs_summary']}\n")
    print("-" * 75)
    print("KEY METRICS:")
    if run["build_accuracy"] is not None:
        print(f"  • Top-1 Accuracy (Build): {run['build_accuracy']*100:.1f}%")
    if run.get("log_loss") is not None:
        print(f"  • Multi-Class Log Loss:   {run['log_loss']:.4f}")
    if run.get("brier_score") is not None:
        print(f"  • Brier Score:            {run['brier_score']:.4f}")
    if run["entrant_observed_build"] is not None:
        print(f"  • Entrant Share (Build):  Expected {run['entrant_expected_build']*100:.1f}%, Observed {run['entrant_observed_build']*100:.1f}%")
    if run["check_accuracy"] is not None:
        print(f"  • Top-1 Accuracy (Check): {run['check_accuracy']*100:.1f}%")
    print("-" * 75)
    print(f"6. NOTES, OUTCOME & LEARNINGS:\n   {run['notes']}")
    print("=" * 75)


def export_runs_json(output_file: str = "runs_export.json", db_path: Path = DB_PATH):
    runs = list_runs(db_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    print(f"Exported {len(runs)} run(s) to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "list":
            print_runs()
        elif cmd == "show" and len(sys.argv) > 2:
            show_run_details(sys.argv[2])
        elif cmd == "export":
            export_runs_json()
        elif cmd == "init":
            init_db()
            print("Initialized runs.db schema.")
        else:
            print(f"Unknown command. Usage: python tracker.py [list|show <run_id>|export|init]")
    else:
        init_db()
        print_runs()
