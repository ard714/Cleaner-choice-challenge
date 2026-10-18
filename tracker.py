"""SQLite experiment tracker for Cleaner Choice Challenge.

Records experiment runs, expectations, model versions, input configurations,
and evaluation metrics to 'runs.db'. Allows inspection of past runs by ID.
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
    """Initialize the runs table schema."""
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


def log_run(
    run_id: str,
    model_version: str,
    data_version: str,
    expectation: str,
    inputs_summary: str,
    outputs_summary: str,
    build_accuracy: Optional[float] = None,
    check_accuracy: Optional[float] = None,
    entrant_observed_build: Optional[float] = None,
    entrant_expected_build: Optional[float] = None,
    entrant_observed_check: Optional[float] = None,
    entrant_expected_check: Optional[float] = None,
    notes: str = "",
    db_path: Path = DB_PATH,
):
    """Insert or replace an experiment run record."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, timestamp, model_version, data_version, expectation,
                inputs_summary, outputs_summary, build_accuracy, check_accuracy,
                entrant_observed_build, entrant_expected_build,
                entrant_observed_check, entrant_expected_check, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now_iso,
                model_version,
                data_version,
                expectation,
                inputs_summary,
                outputs_summary,
                build_accuracy,
                check_accuracy,
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

    print("=" * 95)
    print(f"{'Run ID':<22} | {'Model':<14} | {'Build Acc':<10} | {'Check Acc':<10} | {'Entrant Exp/Obs (Build)':<24}")
    print("-" * 95)
    for r in runs:
        b_acc = f"{r['build_accuracy']*100:.1f}%" if r["build_accuracy"] is not None else "N/A"
        c_acc = f"{r['check_accuracy']*100:.1f}%" if r["check_accuracy"] is not None else "N/A"
        e_ratio = ""
        if r["entrant_expected_build"] is not None and r["entrant_observed_build"] is not None:
            e_ratio = f"{r['entrant_expected_build']*100:.1f}% / {r['entrant_observed_build']*100:.1f}%"
        print(f"{r['run_id']:<22} | {r['model_version']:<14} | {b_acc:<10} | {c_acc:<10} | {e_ratio:<24}")
    print("=" * 95)


def show_run_details(run_id: str, db_path: Path = DB_PATH):
    run = get_run(run_id, db_path)
    if not run:
        print(f"Run '{run_id}' not found.")
        return

    print("=" * 70)
    print(f"Run ID: {run['run_id']}")
    print(f"Timestamp (UTC): {run['timestamp']}")
    print(f"Model Version:   {run['model_version']}")
    print(f"Data Version:    {run['data_version']}")
    print("-" * 70)
    print(f"Expectation:\n  {run['expectation']}\n")
    print(f"Inputs Summary:\n  {run['inputs_summary']}\n")
    print(f"Outputs Summary:\n  {run['outputs_summary']}\n")
    print(f"Build Accuracy:  {run['build_accuracy']*100:.1f}%" if run["build_accuracy"] is not None else "Build Acc: N/A")
    print(f"Check Accuracy:  {run['check_accuracy']*100:.1f}%" if run["check_accuracy"] is not None else "Check Acc: N/A")
    if run["entrant_observed_build"] is not None:
        print(f"Build Entrant:   Expected {run['entrant_expected_build']*100:.1f}%, Observed {run['entrant_observed_build']*100:.1f}%")
    if run["entrant_observed_check"] is not None:
        print(f"Check Entrant:   Expected {run['entrant_expected_check']*100:.1f}%, Observed {run['entrant_observed_check']*100:.1f}%")
    print("-" * 70)
    print(f"Notes & Learnings:\n  {run['notes']}")
    print("=" * 70)


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
