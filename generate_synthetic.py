"""Shim redirecting to scripts/generate_synthetic.py for backward compatibility."""

from scripts.generate_synthetic import (
    SYNTH_HOUSEHOLDS,
    generate_synthetic_visits,
    main,
)

if __name__ == "__main__":
    main()
