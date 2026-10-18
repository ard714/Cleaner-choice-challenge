# Cleaner Choice Challenge: Decision Intelligence Platform

A complete decision intelligence platform, econometric multinomial choice model, and trust-calibration system evaluating the US market launch of a new 16 fl oz eco-friendly toilet cleaner.

---

## 📌 Submission Deliverables & Quick Links

| Deliverable | Description / Location |
| :--- | :--- |
| 🌐 **Live Web Application** | **[https://cleaner-choice-challenge.onrender.com/](https://cleaner-choice-challenge.onrender.com/)** *(Interactive UI & JSON API)* |
| 💻 **Single-Command Local Run** | `python run.py` *(Zero external dependencies — standard library only on laptop CPU)* |
| 🗄️ **Experiment Run Database** | [`runs.db`](runs.db) *(SQLite database tracking all 14 runs with pre-registered hypotheses & metrics)* |
| 🐙 **Git Repository** | **[https://github.com/ard714/Cleaner-choice-challenge](https://github.com/ard714/Cleaner-choice-challenge)** |
| 📝 **Initial Pre-Registration** | [`FIRST.md`](FIRST.md) *(Pre-registered initial concern and testing criteria — 114 words, $\le 120$ limit)* |
| 🎯 **Final Strategic Decision** | [`FINAL.md`](FINAL.md) *(Strategic launch policy, unit economics, and falsification test — 175 words, $\le 200$ limit)* |
| 📖 **Technical Methodology** | [`METHODS.md`](METHODS.md) *(Complete 11-section documentation of econometrics, runs, failed hypotheses & unit economics)* |
| 🧭 **Interactive App Guide** | [`APP_GUIDE.md`](APP_GUIDE.md) *(Step-by-step user manual for what-if scenarios, trust testing, and CSV ingestion)* |

> ⚠️ **Cloud Free Tier Notice (Cold Start Warning):**  
> The live application is hosted on Render's free tier. If the instance has been inactive for more than 15 minutes, Render automatically spins down the container to conserve compute resources. On the very first visit after a period of dormancy, the server may take **30 to 50 seconds** to wake up. Once warm, all tab navigations, what-if sliders, live CSV scoring, and counterfactual simulations run instantaneously (<50ms).

---

## 🚀 Quickstart: Single-Command Local Run

To launch the full interactive web platform locally on your machine:
```bash
python run.py
```
*(Or double-click `run.bat` on Windows, or execute `./run.sh` on macOS/Linux).*

The launcher automatically:
1. Validates dataset integrity and SQLite database connectivity.
2. Executes mathematical sanity checks on the champion choice model.
3. Launches the local zero-dependency HTTP server (`http://localhost:8000/`).
4. Automatically opens your default web browser to the application.

> **Zero External Dependencies**: The application runs entirely on a standard laptop CPU using only Python's built-in standard library (`http.server`, `sqlite3`, `json`, `math`, `csv`). No `pip install` required.

### Reproducible Benchmark Commands
To run individual command-line benchmarks:
- **Starter Baseline Benchmark**: `python run_baseline.py`
- **Champion Choice Model Benchmark**: `python run_champion.py`
- **Execute Automated Test Suite (14 unit tests)**: `python -m unittest discover tests`

---

## 📂 Repository Architecture & Organization

```
simco-intern-challenge-main/
│
├── 📑 Core Submission Deliverables
│   ├── FIRST.md                 # Initial pre-registered hypothesis & falsification criteria (114 words)
│   ├── FINAL.md                 # Strategic launch policy, breakeven bound & falsification test (175 words)
│   ├── METHODS.md               # 11-section technical methodology, math, and failure postmortems
│   ├── APP_GUIDE.md             # Interactive web application user manual & testing guide
│   └── README.md                # Project overview, deliverables table, and quickstart
│
├── 💻 Runtime Launchers & Server
│   ├── run.py                   # Master single-command launcher (environment check + server + browser)
│   ├── run.bat                  # One-click Windows batch launcher
│   ├── run.sh                   # Unix / macOS shell script launcher
│   ├── app.py                   # Full interactive web application, REST API & simulation dashboard
│   ├── runs.db                  # SQLite database tracking all 14 experimental runs
│   └── tracker.py               # SQLite logging & schema interface
│
├── 🔬 Benchmarks & Core Models
│   ├── run_baseline.py          # Starter baseline replication script
│   ├── run_champion.py          # Direct comparison of baseline vs. champion choice model
│   ├── baseline.py              # Starter heuristic baseline choice model
│   ├── evaluator.py             # Strict GroupKFold cross-validation & evaluation metrics
│   ├── eda.py                   # Exploratory data analysis & phrase dictionary extraction
│   └── generate_synthetic.py    # Generator for synthetic stress-test households & visits
│
├── 📁 data/                     # Data assets (both public challenge splits + synthetic stress tests)
│   ├── products.csv             # 4 cleaner catalog specifications (volume, form, scent)
│   ├── households.csv           # 250 household survey profiles (income, budget, spend, notes)
│   ├── visits.csv               # 1,200 build visits from 200 households (in time order)
│   ├── check_visits.csv         # 300 frozen holdout visits from 50 households (unseen test)
│   ├── synthetic_households.csv # 10 extreme edge-case households (OOD income, budget stress)
│   └── synthetic_stress_visits.csv # 50 synthetic visits engineered to test "I Don't Know" abstention
│
├── 📁 runs/                     # Chronological, reproducible experiment runs (Runs 001 to 013)
│   ├── run_001.py               # Run 001: Starter Baseline Replication
│   ├── run_002.py               # Run 002: Baseline 5-Fold GroupKFold Cross-Validation
│   ├── run_003.py               # Run 003: [FAILED] Inventory Depletion & Non-Purchase Ablation
│   ├── run_004.py               # Run 004: Calibrated Alternative-Specific Constants (ASCs)
│   ├── run_005.py               # Run 005: Inter-Visit Feedback Memory (Stain Penalties)
│   ├── run_006.py               # Run 006: [FAILED] Relative Price Differentials vs. Absolute Prices
│   ├── run_007.py               # Run 007: 23-Parameter Convex MLE Multinomial Logit
│   ├── run_008.py               # Run 008: Comprehensive Out-of-Fold Audit & Subgroup Calibration
│   ├── run_009.py               # Run 009: [CHAMPION] 25-Parameter MLE Model Refinement
│   ├── run_010.py               # Run 010: [FAILED] 2-Level Nested Logit Formulation
│   ├── run_011.py               # Run 011: Confirmatory Evaluation on Frozen Holdout Set
│   ├── run_012.py               # Run 012: Counterfactual Strategic Launch Policy Simulation
│   └── run_013.py               # Run 013: Synthetic Stress Database & "I Don't Know" Trust Engine
│
└── 📁 tests/                    # Automated unit test suite
    ├── test_baseline.py         # 8 unit tests for starter baseline model
    └── test_champion.py         # 6 unit tests for champion choice model & trust abstention
```

---

## 📊 Empirical Results & Benchmark Comparison

The table below summarizes performance across the starter baseline, intermediate milestones, and the calibrated champion model:

| Evaluation Split | Metric | Starter Baseline (Run 001/002) | Champion Model (Run 009/011) | Net Performance Lift |
| :--- | :--- | :---: | :---: | :---: |
| **5-Fold Cross-Validation** *(200 Households)* | **Top-1 Accuracy** | 29.00% | **32.50%** | **+3.50% (+42 correct visits)** |
| *(Zero Leakage)* | **Multi-Class Log Loss** | 1.5793 | **1.5537** | **-0.0256 (Better calibration)** |
| | **Multi-Class Brier Score** | 0.7850 | **0.7750** | **-0.0100 (Sharper probabilities)** |
| | **Entrant Calibration Gap** | 4.30% gap | **0.50% gap** | **88% error eliminated** |
| **Frozen Holdout Check** *(50 Households)* | **Top-1 Accuracy** | 28.00% | **31.00%** | **+3.00% (+9 correct visits)** |
| *(Confirmatory Audit)* | **Multi-Class Log Loss** | 1.5631 | **1.5505** | **-0.0126 (Zero overfitting)** |
| | **Multi-Class Brier Score** | 0.7758 | **0.7702** | **-0.0056 (Superior generalization)** |
| | **Entrant Calibration Gap** | 2.10% gap | **1.10% gap** | **48% error eliminated** |

---

## 🎯 Executive Summary & Strategic Decision

Based on the 18-cell counterfactual simulation and margin-adjusted breakeven analysis (detailed in [`FINAL.md`](FINAL.md) and [`METHODS.md`](METHODS.md)):

1. **Recommended Test Configuration**:
   - **Sales Channel**: **Online Direct-to-Consumer / E-Commerce**
   - **Shelf Message**: **`environment`** *(or `low_odor` secondary)*
   - **Launch Price**: **$4.49** *(Expected market share: 22.8% online; gross contribution: $0.66/visit)*
2. **Breakeven Threshold**:
   - Unit gross margin at $4.49 with estimated $1.60 COGS is **$2.89/unit (64.4%)**.
   - Breakeven requires maintaining at least **14.2% market share** to cover fixed shelf and digital acquisition testing budgets.
3. **When the System Says "I Don't Know"**:
   - Out-of-distribution prices outside the historical empirical domain ($<\$2.49$ or $>\$6.49$).
   - Equiprobable ties where consumer choices exhibit high entropy (no alternative exceeds 25% probability).
   - Contradictory customer feedback signals (e.g., severe product stain failures combined with stated loyalty).

---

## 🧪 Intellectual Honesty & Failed Hypotheses

In strict adherence to empirical rigor, 3 hypotheses were tested and explicitly recorded as **FAILED** in [`runs.db`](runs.db) and the web inspector:
- **Run 003 (Inventory Depletion Ablation)**: Assuming households buy zero cleaners if they purchased recently degraded accuracy from 29.0% to 28.3%. In reality, large households maintain multi-bottle buffers.
- **Run 006 (Relative Price Differentials)**: Replacing raw shelf prices with ratio differences against usual brands worsened log-loss from 1.5793 to 1.5831 due to colinearity and noise.
- **Run 010 (2-Level Nested Logit)**: Partitioning choices into nests (Purchase vs. Non-Purchase) degraded log-loss to 1.5746 and check accuracy to 29.67% due to small-sample variance in dissimilarity parameters.

---

## 📜 Original Challenge Background & Rules

A fictional US brand wants to sell a new toilet cleaner. The bottle holds 16 fl oz. Its formula stays the same. The brand can change its price and main shelf message. Each cleaner makes different claims. The data do not prove those claims.

The public data show six visits for each of 250 fictional households. A household could buy one cleaner or buy none. Some households gave short answers after a choice or after using a cleaner. Their answers can be incomplete or wrong. The starter model gives a chance for each choice and a short reason. The data cannot prove why a household chose an item. They cannot show how real shoppers will act.

**Your task:** The brand can fund one US test of the new cleaner. Recommend a price, shelf message, and sales channel to test first. If the evidence is too weak, say "I do not know yet." Explain why. Name the data or test that would let you decide. You have 48 hours to submit. A careful, partial result is useful.

### Data Description
- `households.csv`: Answers from one pre-visit survey (income bands, budget pressure, monthly cleaning supply spend).
- `products.csv`: Specifications for the 4 cleaners (`entrant`, `standard`, `budget`, `scented`).
- `visits.csv`: 1,200 visits across 200 households (Build Set).
- `check_visits.csv`: 300 visits across 50 households (Frozen Holdout Check Set).
