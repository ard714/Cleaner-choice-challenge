"""SimCo Cleaner Choice Platform - Single Command Unified Local Runner.

Usage:
    python run.py [port]

Features:
1. Validates local Python runtime and directory environment (zero external dependencies).
2. Audits dataset files (visits.csv, households.csv, check_visits.csv, synthetic stress sets).
3. Executes quick integrity unit tests.
4. Boots the decision intelligence web server on http://localhost:8000.
5. Automatically opens the user's default web browser.
"""

import os
import sys
import time
import socket
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def check_port_available(port: int) -> bool:
    """Check if a port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def verify_deliverables():
    """Verify all required files exist before launch."""
    required_files = [
        ROOT_DIR / "FIRST.md",
        ROOT_DIR / "FINAL.md",
        ROOT_DIR / "README.md",
        ROOT_DIR / "app.py",
        ROOT_DIR / "runs.db",
        ROOT_DIR / "data" / "households.csv",
        ROOT_DIR / "data" / "visits.csv",
        ROOT_DIR / "data" / "check_visits.csv",
    ]
    missing = [str(f.relative_to(ROOT_DIR)) for f in required_files if not f.exists()]
    if missing:
        print(f"Error: Missing required project files: {', '.join(missing)}")
        sys.exit(1)


def run_quick_tests():
    """Run core mathematical unit tests in-memory to confirm model integrity."""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT_DIR / "tests"))
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("Warning: Quick unit tests reported failures. Proceeding with caution.")
    else:
        print("  [PASS] Mathematical and choice probability unit tests verified.")


def main():
    print("=" * 75)
    print("  SIMCO CLEANER CHOICE CHALLENGE - DECISION INTELLIGENCE PLATFORM")
    print("=" * 75)
    print("Auditing environment and deliverables...")

    verify_deliverables()
    print("  [PASS] Required datasets, FIRST.md, FINAL.md, and runs.db present.")

    run_quick_tests()

    # Determine port
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    elif not check_port_available(port):
        # If 8000 is occupied by our background daemon or another process, find next free
        for p in range(8001, 8010):
            if check_port_available(p):
                port = p
                break

    url = f"http://localhost:{port}/"
    print(f"\nLaunching SimCo Interactive Platform at: {url}")
    print("Press Ctrl+C to stop the local server at any time.\n")

    # Try to open the default web browser after 1 second
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    from app import run_server
    try:
        run_server(port=port)
    except KeyboardInterrupt:
        print("\nServer gracefully shut down. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
