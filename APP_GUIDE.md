# Interactive Web Application Guide: SimCo Cleaner Intelligence

A quick reference guide for evaluating, demonstrating, and testing the SimCo Cleaner Choice Decision Intelligence web application.

---

## 1. Quick Launch Options

### Option A: Cloud Live Demo (Instant Access)
Visit the deployed live application directly in your browser:
🔗 **[https://cleaner-choice-challenge.onrender.com/](https://cleaner-choice-challenge.onrender.com/)**

> ⚠️ **Cloud Free Tier Notice (Cold Start Warning):** The live application is hosted on Render's free tier. If the instance has been inactive for more than 15 minutes, Render automatically spins down the container to conserve compute resources. On the very first visit after a period of dormancy, the page may take **30 to 50 seconds** to wake up. Once warm, all tab navigations, what-if sliders, live CSV scoring, and counterfactual simulations run instantaneously (<50ms).

---

### Option B: Local Single-Command Run
From the project root directory, run:
```bash
python run.py
```
*(Or double-click `run.bat` on Windows / execute `./run.sh` on macOS/Linux).*

The launcher automatically audits datasets, executes mathematical sanity tests, starts the local HTTP server, and opens your default browser at:
```
http://localhost:8000/
```
> **Zero External Dependencies**: The application runs entirely on a laptop CPU using Python's standard library (`http.server`, `json`, `sqlite3`, `math`). No `pip install` required.

---

## 2. Tab-by-Tab Walkthrough

```
[ What-If Simulator ]  [ SQLite Runs Inspector ]  [ Holdout Benchmark ]  [ Policy Matrix (Final Run) ]  [ Live Ingestion & Scoring ]
```

### Tab 1: What-If Simulator (`#tab-predict`)
*Interactive scenario planning and household choice prediction.*

1. **Select a Shopper Household**:
   - Choose from pre-configured archetypes in the dropdown (e.g., `H004` Affluent Eco-Conscious, `H013` Tight Budget, `H201` Holdout Shopper, or `H901_OOD_LUXURY` Synthetic Stress).
2. **Adjust Launch Levers**:
   - Slide the **Entrant Launch Price** slider ($\$2.49\text{--}\$6.49$).
   - Toggle **Sales Channel** (`Online` vs. `Retail`) and **Shelf Message** (`environment`, `cleaning`, `low_odor`).
   - Toggle prior purchase history and post-use notes (e.g., check `Prior Stain Failure` to watch utility collapse).
3. **Test Hypothetical Demographics**:
   - Click **`⚙️ Custom Demographic Overrides`** to test custom income bands, bathroom counts, or free-text survey notes live.
4. **Observe Outputs**:
   - **Simulated Choice & Confidence**: Real-time Top-1 choice with confidence percentage.
   - **Market Probabilities**: Dynamic animated bar charts across all 5 alternatives.
   - **Grounded Factual Explanations**: Bullet points citing exact household survey sentiments and price terms.
   - **Trust Warning Banner**: If you set an extreme price (e.g. $\$19.99$) or create an equiprobable tie, an amber banner appears: **`⚠️ TRUST ABSTENTION: I DON'T KNOW`**, explaining why the data cannot support a confident choice.

---

### Tab 2: SQLite Runs Inspector (`#tab-runs`)
*Full experiment audit trail querying `runs.db`.*

1. **Select Any Experiment Run**:
   - Open the **SQLite Run Selector** dropdown containing all 14 saved runs (`run_001` to `run_013`).
   - Notice the visual indicators:
     - `❌ [FAILED]`: Clearly marks negative results (Runs 003, 006, 010).
     - `⭐ [CHAMPION]`: Marks the champion 25-parameter choice model (Run 009).
2. **Inspect Experiment Run Cards**:
   - View the exact pre-run expectations, failure criteria, model version, dataset version, and empirical metrics stored in the SQLite database.
   - Read the critical outcome notes explaining the econometric lessons learned from each experiment.

---

### Tab 3: Holdout Benchmark (`check_visits.csv`) (`#tab-check`)
*Audited holdout generalization on 50 previously unseen households.*

1. **README Protocol Transparency Banner**:
   - Prominently displays the README holdout disclosure, explaining the two-stage audit sequence (Run 008 Initial Check $\to$ Run 009 Build Refinement $\to$ Run 011 Confirmatory Audit).
2. **Side-by-Side KPI Grid**:
   - Check Top-1 Accuracy: **31.0%** (vs. 28.0% starter baseline | $+3.0\%$ lift).
   - Multi-Class Log Loss: **1.5505** (vs. 1.5793 baseline | $-0.0288$).
   - Multi-Class Brier Score: **0.7725** (vs. 0.7850 baseline | $-0.0125$).
   - Entrant Calibration Gap: **1.6%** (gap closed from $5.2\%$).
3. **Interactive 300-Row Scored Table**:
   - Scroll through all 300 holdout visits (`H201` to `H250`) showing actual shopper choice vs. predicted choice, confidence, match pills, and grounded reasons.
4. **Re-Run Live**:
   - Click **`⚡ Re-Run Holdout Benchmark Live`** to re-score the entire holdout dataset in-memory in under 50ms.

---

### Tab 4: Policy Matrix (Final Run) (`#tab-policy`)
*Strategic launch optimization, unit economics, and falsification criteria.*

1. **18-Cell Simulation Ranking**:
   - Ranks all 18 launch policies (2 Channels $\times$ 3 Prices $\times$ 3 Messages) across 1,500 shopper visits.
   - Highlights the optimal launch recommendation: **$4.49 | Online | Environment** (26.0% market share, 37.5% eco penetration, $1.17 gross revenue/visit).
2. **Interactive Unit Breakeven Slider**:
   - Drag the **Estimated Unit Cost Slider** ($\$0.00\text{--}\$3.00$).
   - Dynamically verifies the analytical breakeven bound: for any unit cost $c > \$0.15/\text{bottle}$, $\$4.49$ delivers strictly superior net profit compared to $\$3.49$.
3. **Stain Complaint Churn Hazard & Falsification Test**:
   - Displays empirical churn metrics ($0$ of $11$ households repurchased after a stain failure).
   - Documents the pre-committed **$< 20.0\%$ 60-day repeat reorder rate** falsification threshold.

---

### Tab 5: Live Ingestion & Scoring Sandbox (`#tab-ingest`)
*Reviewer sandbox for live data ingestion and arbitrary batch testing.*

#### Section 1: Ingest Single Household
- Enter custom attributes for a brand-new household (`household_id`, spend, income, budget, bathrooms, usual brand, and survey note).
- Click **`⚡ Ingest & Predict Choice`**: The household is ingested into memory, scored immediately, and added to the Tab 1 dropdown for further what-if exploration.

#### Section 2: Ingest & Score Batch CSV Dataset
- **Load 50 Holdout Visits**: Click **`Load 50 Holdout Visits`** $\to$ click **`🚀 Process & Benchmark Batch Dataset`** to benchmark `data/check_visits.csv` live.
- **Load Synthetic Stress Dataset**: Click **`🧪 Load Synthetic Stress Dataset (50 Visits - Trust Test)`** $\to$ click **`🚀 Process & Benchmark Batch Dataset`** to watch the model emit **`⚠️ I DON'T KNOW`** warning badges on out-of-distribution prices and equiprobable ties.
- **Upload Any External File**: Click **`📁 Upload CSV File`** to upload any external `.csv` from your local machine.

---

## 3. Keyboard Shortcuts & Tips

- **Switching Tabs**: Click the tab headers or press Tab to navigate.
- **Resetting Simulator**: Refresh the browser page or re-select `H004` to restore default launch values.
- **Inspecting JSON API**: You can test API endpoints directly in your browser:
  - `http://localhost:8000/api/runs` (all 14 runs JSON)
  - `http://localhost:8000/api/benchmark/check_visits` (300 scored visits JSON)
  - `http://localhost:8000/health` (health check)
