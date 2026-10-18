# Utility & Analysis Scripts

This directory contains standalone tools and data preparation utilities:

- **`eda.py`**: Exploratory data analysis summarizing datasets, discrete phrase taxonomies, categorical splits, and feature token discovery.
  ```bash
  python scripts/eda.py
  ```
- **`generate_synthetic.py`**: Generates synthetic out-of-distribution stress datasets (`data/synthetic_households.csv` and `data/synthetic_stress_visits.csv`) used to audit the "I Don't Know" trust engine in Run 013.
  ```bash
  python scripts/generate_synthetic.py
  ```

*(Note: Root shims `eda.py` and `generate_synthetic.py` are also maintained for backward compatibility).*
