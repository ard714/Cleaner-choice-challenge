"""Interactive Web Application for Cleaner Choice Challenge.

Zero-dependency single-file application running on laptop CPU via Python standard library.
Serves an interactive web interface on http://localhost:8000 with:
1. Tab 1: Interactive What-If Choice Predictor (reactive sliders, probability bars, grounded reasons).
2. Tab 2: SQLite Run Inspector (querying runs.db for all runs, hypotheses, metrics, and failures).
3. Tab 3: Strategic Policy Matrix & Analytical Breakeven Simulator (18 cells, unit economics).

Usage:
    python app.py
"""

import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runs.run_009 import (
    FEATURE_NAMES,
    PRODUCTS,
    compute_utilities,
    extract_features,
    generate_grounded_reasons,
    get_offers_dict,
    load_data,
    softmax,
    train_mle_fold,
)

DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "runs.db"

# Global state loaded once on server startup
HOUSEHOLDS = {}
BUILD_VISITS = []
CHAMPION_WEIGHTS = {}


def init_app():
    global HOUSEHOLDS, BUILD_VISITS, CHAMPION_WEIGHTS
    print("Initializing Cleaner Choice Challenge Interactive Platform...")
    HOUSEHOLDS, BUILD_VISITS = load_data()
    print(f"Loaded {len(HOUSEHOLDS)} households and {len(BUILD_VISITS)} visits.")
    print("Training champion 25-parameter choice model (v1.1-mle-refined)...")
    CHAMPION_WEIGHTS = train_mle_fold(BUILD_VISITS, HOUSEHOLDS, iterations=180, lr=0.08, l2_reg=0.002)
    print("Champion model trained successfully. Server ready.")


def get_db_runs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    runs = c.execute("SELECT * FROM runs ORDER BY timestamp ASC").fetchall()
    conn.close()
    return [dict(r) for r in runs]


def run_prediction(payload: dict) -> dict:
    """Compute real-time choice probabilities and grounded reasons for custom inputs."""
    hh_id = payload.get("household_id", "H004")
    raw_hh = HOUSEHOLDS.get(hh_id, {}).copy()

    # Allow custom overrides
    if "income_band" in payload and payload["income_band"]:
        raw_hh["income_band"] = payload["income_band"]
    if "budget" in payload and payload["budget"]:
        raw_hh["budget"] = payload["budget"]
    if "bathrooms" in payload and payload["bathrooms"]:
        raw_hh["bathrooms"] = str(payload["bathrooms"])
    if "usual_brand" in payload and payload["usual_brand"]:
        raw_hh["usual_brand"] = payload["usual_brand"]
    if "cleaning_spend_usd" in payload and payload["cleaning_spend_usd"]:
        raw_hh["cleaning_spend_usd"] = str(payload["cleaning_spend_usd"])
    if "note" in payload and payload["note"] is not None:
        raw_hh["note"] = payload["note"]

    entrant_p = float(payload.get("entrant_price", 4.49))
    entrant_msg = payload.get("entrant_message", "environment")
    channel = payload.get("channel", "online")

    offers = {
        "entrant": {"price": entrant_p, "message": entrant_msg},
        "standard": {"price": float(payload.get("standard_price", 3.99))},
        "budget": {"price": float(payload.get("budget_price", 2.99))},
        "scented": {"price": float(payload.get("scented_price", 4.29))},
    }

    dummy_visit = {"channel": channel, "visit_number": "2"}
    history = []
    if payload.get("had_prior_purchase"):
        history.append({"choice": payload.get("prior_choice", "standard"), "after_use_note": ""})
    if payload.get("had_stain_failure"):
        history.append({"choice": payload.get("stain_product", "entrant"), "after_use_note": "It left a stain behind."})
    if payload.get("had_positive_exp"):
        history.append({"choice": payload.get("pos_product", "entrant"), "after_use_note": "It removed the stain I noticed."})

    prod_feats = {p: extract_features(p, offers, raw_hh, dummy_visit, history) for p in PRODUCTS}
    scores = compute_utilities(CHAMPION_WEIGHTS, prod_feats)
    probs = softmax(scores)

    pred_choice = max(probs, key=probs.get)
    reasons = generate_grounded_reasons(pred_choice, probs, prod_feats, CHAMPION_WEIGHTS, raw_hh, offers)

    gross_rev_visit = entrant_p * probs["entrant"]

    hh_num = int(hh_id[1:]) if (hh_id.startswith("H") and hh_id[1:].isdigit()) else 0
    split_name = "Holdout Check Split (Unseen)" if hh_num > 200 else ("Build Split" if hh_num > 0 else "Custom Archetype")

    return {
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "pred_choice": pred_choice,
        "pred_confidence": round(probs[pred_choice] * 100, 1),
        "grounded_reasons": reasons,
        "gross_rev_visit": round(gross_rev_visit, 2),
        "household_context": {
            "id": hh_id,
            "split": split_name,
            "income_band": raw_hh.get("income_band", "50k_to_100k"),
            "budget": raw_hh.get("budget", "middle"),
            "bathrooms": str(raw_hh.get("bathrooms", "1")),
            "usual_brand": raw_hh.get("usual_brand", "standard"),
            "cleaning_spend_usd": str(raw_hh.get("cleaning_spend_usd", "15")),
            "survey_note": raw_hh.get("note", "") or "(No free-text survey response provided)",
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SimCo Cleaner Choice Platform | Decision Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 26, 43, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #10b981;
      --primary-glow: rgba(16, 185, 129, 0.25);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent-blue: #3b82f6;
      --accent-amber: #f59e0b;
      --accent-purple: #8b5cf6;
      --accent-rose: #f43f5e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      background: radial-gradient(circle at top right, #131d36 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
    }
    .header {
      border-bottom: 1px solid var(--card-border);
      padding: 1.25rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      backdrop-filter: blur(12px);
      background: rgba(9, 13, 22, 0.8);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .logo-badge {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--primary);
      box-shadow: 0 0 10px var(--primary);
    }
    .brand-title {
      font-weight: 800;
      font-size: 1.2rem;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .tabs {
      display: flex;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.05);
      padding: 0.3rem;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.5rem 1rem;
      border-radius: 7px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.875rem;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--primary);
      color: #064e3b;
      box-shadow: 0 2px 10px var(--primary-glow);
    }
    .container {
      max-width: 1280px;
      margin: 2rem auto;
      padding: 0 1.5rem;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    /* Grid & Cards */
    .grid-2 { display: grid; grid-template-columns: 1.1fr 1fr; gap: 1.5rem; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.5rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #fff;
    }
    .form-group { margin-bottom: 1.1rem; }
    .form-group label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }
    select, input[type="text"], input[type="number"] {
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 0.65rem 0.85rem;
      border-radius: 8px;
      font-size: 0.9rem;
      font-family: inherit;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--primary);
      cursor: pointer;
    }
    .range-val {
      font-family: 'JetBrains Mono', monospace;
      color: var(--primary);
      font-weight: 700;
      float: right;
    }

    /* Probability Bars */
    .prob-bar-container { margin-bottom: 1rem; }
    .prob-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.3rem;
    }
    .bar-wrapper {
      background: rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      height: 12px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .fill-entrant { background: linear-gradient(90deg, #10b981, #34d399); }
    .fill-standard { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .fill-budget { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .fill-scented { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .fill-none { background: linear-gradient(90deg, #64748b, #94a3b8); }

    .prediction-banner {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 1.25rem;
      border-radius: 12px;
      margin-bottom: 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .pred-highlight {
      font-size: 1.6rem;
      font-weight: 800;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: -0.01em;
    }
    .reason-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .reason-item {
      background: rgba(15, 23, 42, 0.6);
      border-left: 3px solid var(--primary);
      padding: 0.65rem 0.85rem;
      border-radius: 0 8px 8px 0;
      font-size: 0.875rem;
    }
    .run-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--card-border);
    }
    .code-box {
      font-family: 'JetBrains Mono', monospace;
      background: #050811;
      padding: 1rem;
      border-radius: 8px;
      font-size: 0.8rem;
      color: #cbd5e1;
      white-space: pre-wrap;
      max-height: 400px;
      overflow-y: auto;
      border: 1px solid var(--card-border);
    }
    .tag {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      text-transform: uppercase;
    }
    .tag-success { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .tag-fail { background: rgba(244, 63, 94, 0.2); color: #fb7185; }
    .tag-bench { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    .tag-purple { background: rgba(139, 92, 246, 0.2); color: #c084fc; }
    .tag-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }

    .hh-card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.1rem;
      margin-bottom: 1.25rem;
    }
    .hh-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.6rem;
    }
    .hh-quote {
      font-style: italic;
      color: #cbd5e1;
      font-size: 0.85rem;
      border-left: 2px solid var(--accent-blue);
      padding-left: 0.75rem;
      margin-bottom: 0.75rem;
      line-height: 1.4;
    }
    .hh-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.5rem;
      font-size: 0.75rem;
    }
    .hh-item {
      background: rgba(255, 255, 255, 0.03);
      padding: 0.45rem 0.6rem;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hh-lbl {
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.65rem;
      font-weight: 700;
      display: block;
      margin-bottom: 0.15rem;
    }
    .hh-val {
      font-weight: 600;
      color: #fff;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.75rem 0.6rem;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }
    th {
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
    }
    tr.best-row {
      background: rgba(16, 185, 129, 0.12);
      border-left: 3px solid var(--primary);
    }
  </style>
</head>
<body>

  <header class="header">
    <div class="logo-badge">
      <div class="status-dot"></div>
      <div class="brand-title">SimCo Cleaner Intelligence</div>
      <span class="tag tag-success">25-Param MLE Choice Engine</span>
    </div>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('tab-predict')">What-If Simulator</button>
      <button class="tab-btn" onclick="switchTab('tab-runs')">SQLite Runs Inspector</button>
      <button class="tab-btn" onclick="switchTab('tab-policy')">Policy Matrix</button>
    </div>
  </header>

  <main class="container">

    <!-- TAB 1: WHAT-IF PREDICTOR -->
    <section id="tab-predict" class="tab-content active">
      <div class="grid-2">
        <!-- Controls Column -->
        <div class="card">
          <div class="card-title">Launch Scenario Controls</div>

          <div class="form-group">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
              <label style="margin-bottom: 0;">Select Household</label>
              <button type="button" onclick="toggleCustomOverrides()" style="background: none; border: none; color: var(--accent-blue); font-size: 0.75rem; cursor: pointer; text-decoration: underline;">
                ⚙️ Custom Demographic Overrides
              </button>
            </div>
            <select id="hh-select" onchange="loadHouseholdContext()">
              <option value="H004">H004 - Affluent Eco-Conscious (> $100k, 'fewer plastic bottles')</option>
              <option value="H013">H013 - Tight Budget Price-Sensitive ('tight', refill leak)</option>
              <option value="H014">H014 - Large Home (3 Bathrooms, heavy supply velocity)</option>
              <option value="H015">H015 - Green Skeptic ('do not trust green claims')</option>
              <option value="H001">H001 - Standard Loyalist (Retail Shopper)</option>
              <option value="H201">H201 - Holdout Check Shopper (Unseen: 'pay more if problem solved')</option>
              <option value="H215">H215 - Holdout Check Shopper (Unseen: budget constraint)</option>
            </select>
          </div>

          <!-- Collapsible Custom Overrides -->
          <div id="custom-override-panel" style="display: none; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 0.85rem; margin-bottom: 1.1rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 0.5rem;">Test Custom / Hypothetical Household</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.5rem;">
              <div>
                <label style="font-size: 0.7rem; color: var(--text-muted); display: block;">Income Band</label>
                <select id="ov-income" onchange="runSim()" style="padding: 0.4rem; font-size: 0.8rem;">
                  <option value="">Default (From Survey)</option>
                  <option value="under_50k">under_50k</option>
                  <option value="50k_to_100k">50k_to_100k</option>
                  <option value="over_100k">over_100k</option>
                </select>
              </div>
              <div>
                <label style="font-size: 0.7rem; color: var(--text-muted); display: block;">Budget Pressure</label>
                <select id="ov-budget" onchange="runSim()" style="padding: 0.4rem; font-size: 0.8rem;">
                  <option value="">Default (From Survey)</option>
                  <option value="tight">tight</option>
                  <option value="middle">middle</option>
                  <option value="generous">generous</option>
                </select>
              </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.5rem;">
              <div>
                <label style="font-size: 0.7rem; color: var(--text-muted); display: block;">Bathrooms</label>
                <select id="ov-baths" onchange="runSim()" style="padding: 0.4rem; font-size: 0.8rem;">
                  <option value="">Default (From Survey)</option>
                  <option value="1">1 Bathroom</option>
                  <option value="2">2 Bathrooms</option>
                  <option value="3">3+ Bathrooms</option>
                </select>
              </div>
              <div>
                <label style="font-size: 0.7rem; color: var(--text-muted); display: block;">Usual Brand</label>
                <select id="ov-brand" onchange="runSim()" style="padding: 0.4rem; font-size: 0.8rem;">
                  <option value="">Default (From Survey)</option>
                  <option value="standard">standard</option>
                  <option value="budget">budget</option>
                  <option value="scented">scented</option>
                  <option value="entrant">entrant</option>
                </select>
              </div>
            </div>
            <div>
              <label style="font-size: 0.7rem; color: var(--text-muted); display: block;">Hypothetical Survey Note</label>
              <input type="text" id="ov-note" placeholder="e.g. I try to buy eco friendly products..." oninput="runSim()" style="padding: 0.4rem; font-size: 0.8rem;">
            </div>
          </div>

          <div class="form-group">
            <label>Entrant Launch Price ($): <span class="range-val" id="price-display">$4.49</span></label>
            <input type="range" id="entrant-price" min="2.49" max="6.49" step="0.50" value="4.49" oninput="updatePrice(this.value)">
          </div>

          <div class="grid-2" style="margin-bottom: 0.5rem;">
            <div class="form-group">
              <label>Sales Channel</label>
              <select id="channel-select" onchange="runSim()">
                <option value="online" selected>Online (E-Commerce)</option>
                <option value="retail">Physical Retail Store</option>
              </select>
            </div>
            <div class="form-group">
              <label>Shelf Message</label>
              <select id="message-select" onchange="runSim()">
                <option value="environment" selected>Environment (Eco Claims)</option>
                <option value="cleaning">Cleaning (Tough Stains)</option>
                <option value="low_odor">Low Odor (Gentle Scent)</option>
              </select>
            </div>
          </div>

          <div class="form-group">
            <label>Competitor Catalog Benchmark Term</label>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem;">
              <input type="text" value="Standard $3.99" readonly style="color: var(--accent-blue);">
              <input type="text" value="Budget $2.99" readonly style="color: var(--accent-amber);">
              <input type="text" value="Scented $4.29" readonly style="color: var(--accent-purple);">
            </div>
          </div>

          <div class="form-group">
            <label>Causal Experience History Toggles</label>
            <div style="display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.85rem;">
              <label style="text-transform: none; font-weight: 500;"><input type="checkbox" id="toggle-restock" onchange="runSim()"> Recent purchase replenished supply</label>
              <label style="text-transform: none; font-weight: 500;"><input type="checkbox" id="toggle-stain" onchange="runSim()"> Past stain complaint on Entrant ('left a stain')</label>
              <label style="text-transform: none; font-weight: 500;"><input type="checkbox" id="toggle-pos" onchange="runSim()"> Positive past experience on Entrant</label>
            </div>
          </div>
        </div>

        <!-- Real-Time Output Column -->
        <div>
          <!-- Real-Time Household Profile Display -->
          <div class="hh-card" id="hh-profile-card">
            <div class="hh-header">
              <div>
                <strong id="hh-id-display" style="font-size: 1.05rem; color: #fff;">H004</strong>
                <span id="hh-split-badge" class="tag tag-bench" style="margin-left: 0.5rem;">Build Split</span>
              </div>
              <span id="hh-spend-display" style="font-size: 0.8rem; color: var(--text-muted);">$15 / mo cleaning</span>
            </div>
            <div class="hh-quote" id="hh-quote-display">"I try to use fewer plastic bottles..."</div>
            <div class="hh-grid">
              <div class="hh-item"><span class="hh-lbl">Income</span><span class="hh-val" id="hh-income-display">over_100k</span></div>
              <div class="hh-item"><span class="hh-lbl">Budget</span><span class="hh-val" id="hh-budget-display">generous</span></div>
              <div class="hh-item"><span class="hh-lbl">Baths</span><span class="hh-val" id="hh-baths-display">2</span></div>
              <div class="hh-item"><span class="hh-lbl">Usual Brand</span><span class="hh-val" id="hh-usual-display">standard</span></div>
            </div>
          </div>

          <div class="prediction-banner">
            <div>
              <div style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">Simulated Consumer Choice</div>
              <div class="pred-highlight" id="banner-choice">ENTRANT</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">Confidence</div>
              <div style="font-size: 1.4rem; font-weight: 800; color: #fff;" id="banner-conf">42.0%</div>
            </div>
          </div>

          <div class="card" style="margin-bottom: 1.5rem;">
            <div class="card-title">Market Choice Probabilities</div>
            <div id="prob-bars"></div>
            <div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--card-border); font-size: 0.85rem; display: flex; justify-content: space-between;">
              <span style="color: var(--text-muted);">Expected Gross Rev / Visit:</span>
              <strong style="color: var(--primary);" id="banner-rev">$1.17</strong>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Grounded Factual Explanations</div>
            <ul class="reason-list" id="reason-container"></ul>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: SQLITE RUNS INSPECTOR -->
    <section id="tab-runs" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="run-card-header">
            <div class="card-title" style="margin: 0;">SQLite Run Selector</div>
            <select id="run-selector" onchange="loadRunCard(this.value)" style="width: auto; min-width: 220px;"></select>
          </div>
          <div id="run-summary-box"></div>
        </div>

        <div class="card">
          <div class="card-title">Full Experiment Run Card (runs.db)</div>
          <div class="code-box" id="raw-card-view">Loading...</div>
        </div>
      </div>
    </section>

    <!-- TAB 3: POLICY MATRIX -->
    <section id="tab-policy" class="tab-content">
      <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-title">18-Cell Launch Policy Simulation (All 250 Households / 1,500 Visits)</div>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
          Evaluated strictly on empirical catalog prices and observed consumer choices with zero external margin assumptions.
        </p>
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Channel</th>
                <th>Price</th>
                <th>Message</th>
                <th>Entrant Share</th>
                <th>Volume (1.5k Trips)</th>
                <th>Gross Rev / Visit</th>
                <th>Eco Penetration</th>
              </tr>
            </thead>
            <tbody id="policy-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">Analytical Unit Margin Sensitivity</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            Breakeven unit cost between $3.49 and $4.49 is exactly <strong>$0.15/unit</strong>. Adjust estimated unit production & shipping cost to verify net margin:
          </p>
          <div class="form-group">
            <label>Estimated Unit Cost (COGS + Shipping): <span class="range-val" id="cost-display">$1.50</span></label>
            <input type="range" id="unit-cost-slider" min="0.00" max="3.00" step="0.25" value="1.50" oninput="updateMarginSim(this.value)">
          </div>
          <div style="background: rgba(15, 23, 42, 0.7); padding: 1rem; border-radius: 8px; font-size: 0.85rem;" id="margin-output"></div>
        </div>

        <div class="card">
          <div class="card-title">Longitudinal Stain Churn Hazard</div>
          <div style="background: rgba(244, 63, 94, 0.1); border-left: 3px solid var(--accent-rose); padding: 1rem; border-radius: 0 8px 8px 0; font-size: 0.85rem; margin-bottom: 1rem;">
            <strong>100.0% Empirical Churn:</strong> In the build visits, exactly 0 out of 11 shoppers who experienced a stain failure ("It left a stain behind") ever repurchased the entrant cleaner.
          </div>
          <p style="font-size: 0.85rem; color: var(--text-muted);">
            <strong>Falsification Test Trigger:</strong> In a real-world test, if the 60-day repeat reorder rate falls below <strong>20.0%</strong>, the product formulation fails to retain consumers and market share will rapidly erode.
          </p>
        </div>
      </div>
    </section>

  </main>

  <script>
    let currentRuns = [];

    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
      if (tabId === 'tab-runs') loadRunsList();
      if (tabId === 'tab-policy') loadPolicyTable();
    }

    function updatePrice(val) {
      document.getElementById('price-display').innerText = '$' + parseFloat(val).toFixed(2);
      runSim();
    }

    let allHouseholds = {};

    function toggleCustomOverrides() {
      const p = document.getElementById('custom-override-panel');
      p.style.display = p.style.display === 'none' ? 'block' : 'none';
    }

    function loadHouseholdContext() {
      // Clear custom overrides when switching household
      document.getElementById('ov-income').value = '';
      document.getElementById('ov-budget').value = '';
      document.getElementById('ov-baths').value = '';
      document.getElementById('ov-brand').value = '';
      document.getElementById('ov-note').value = '';
      runSim();
    }

    async function initHouseholds() {
      try {
        const res = await fetch('/api/households');
        allHouseholds = await res.json();
        const sel = document.getElementById('hh-select');
        const curVal = sel.value || 'H004';
        sel.innerHTML = '';

        const archetypes = [
          { id: 'H004', label: "H004 - Affluent Eco-Conscious (> $100k, 'fewer plastic bottles')" },
          { id: 'H013', label: "H013 - Tight Budget Price-Sensitive ('tight', refill leak)" },
          { id: 'H014', label: "H014 - Large Home (3 Bathrooms, heavy supply velocity)" },
          { id: 'H015', label: "H015 - Green Skeptic ('do not trust green claims')" },
          { id: 'H001', label: "H001 - Standard Loyalist (Retail Shopper)" }
        ];

        const grpArch = document.createElement('optgroup');
        grpArch.label = '⭐ Featured Household Archetypes';
        archetypes.forEach(a => {
          const opt = document.createElement('option');
          opt.value = a.id;
          opt.innerText = a.label;
          grpArch.appendChild(opt);
        });
        sel.appendChild(grpArch);

        const grpBuild = document.createElement('optgroup');
        grpBuild.label = '📁 Build Split (H001 - H200)';
        for (let i = 1; i <= 200; i++) {
          const hid = 'H' + String(i).padStart(3, '0');
          if (allHouseholds[hid]) {
            const h = allHouseholds[hid];
            const opt = document.createElement('option');
            opt.value = hid;
            opt.innerText = `${hid} [${h.income_band}, ${h.budget}, ${h.bathrooms}b, ${h.usual_brand}]`;
            grpBuild.appendChild(opt);
          }
        }
        sel.appendChild(grpBuild);

        const grpCheck = document.createElement('optgroup');
        grpCheck.label = '🔒 Holdout Check Split (H201 - H250 Unseen)';
        for (let i = 201; i <= 250; i++) {
          const hid = 'H' + String(i).padStart(3, '0');
          if (allHouseholds[hid]) {
            const h = allHouseholds[hid];
            const opt = document.createElement('option');
            opt.value = hid;
            opt.innerText = `${hid} [${h.income_band}, ${h.budget}, ${h.bathrooms}b, ${h.usual_brand}]`;
            grpCheck.appendChild(opt);
          }
        }
        sel.appendChild(grpCheck);
        sel.value = curVal;
      } catch (e) {
        console.error('Failed loading households', e);
      }
    }

    async function runSim() {
      const payload = {
        household_id: document.getElementById('hh-select').value,
        entrant_price: document.getElementById('entrant-price').value,
        channel: document.getElementById('channel-select').value,
        entrant_message: document.getElementById('message-select').value,
        had_prior_purchase: document.getElementById('toggle-restock').checked,
        had_stain_failure: document.getElementById('toggle-stain').checked,
        had_positive_exp: document.getElementById('toggle-pos').checked,
        income_band: document.getElementById('ov-income').value,
        budget: document.getElementById('ov-budget').value,
        bathrooms: document.getElementById('ov-baths').value,
        usual_brand: document.getElementById('ov-brand').value,
        note: document.getElementById('ov-note').value || undefined
      };

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      // Update Household Profile Card
      const ctx = data.household_context;
      document.getElementById('hh-id-display').innerText = ctx.id;
      document.getElementById('hh-split-badge').innerText = ctx.split;
      document.getElementById('hh-split-badge').className = 'tag ' + (ctx.split.includes('Holdout') ? 'tag-purple' : 'tag-bench');
      document.getElementById('hh-spend-display').innerText = '$' + ctx.cleaning_spend_usd + ' / mo cleaning';
      document.getElementById('hh-quote-display').innerText = '"' + ctx.survey_note + '"';
      document.getElementById('hh-income-display').innerText = ctx.income_band;
      document.getElementById('hh-budget-display').innerText = ctx.budget;
      document.getElementById('hh-baths-display').innerText = ctx.bathrooms;
      document.getElementById('hh-usual-display').innerText = ctx.usual_brand;

      document.getElementById('banner-choice').innerText = data.pred_choice;
      document.getElementById('banner-conf').innerText = data.pred_confidence + '%';
      document.getElementById('banner-rev').innerText = '$' + data.gross_rev_visit.toFixed(2);

      const colors = {
        entrant: 'fill-entrant',
        standard: 'fill-standard',
        budget: 'fill-budget',
        scented: 'fill-scented',
        none: 'fill-none'
      };

      let html = '';
      for (const [prod, prob] of Object.entries(data.probabilities)) {
        html += `
          <div class="prob-bar-container">
            <div class="prob-header">
              <span style="text-transform: capitalize;">${prod}</span>
              <span>${(prob * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-wrapper">
              <div class="bar-fill ${colors[prod]}" style="width: ${prob * 100}%;"></div>
            </div>
          </div>
        `;
      }
      document.getElementById('prob-bars').innerHTML = html;

      let rHtml = '';
      for (const r of data.grounded_reasons) {
        rHtml += `<li class="reason-item">${r}</li>`;
      }
      document.getElementById('reason-container').innerHTML = rHtml;
    }

    async function loadRunsList() {
      const res = await fetch('/api/runs');
      currentRuns = await res.json();
      const sel = document.getElementById('run-selector');
      sel.innerHTML = '';
      currentRuns.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.run_id;
        opt.innerText = r.run_id;
        sel.appendChild(opt);
      });
      if (currentRuns.length > 0) {
        sel.value = currentRuns[currentRuns.length - 1].run_id;
        loadRunCard(sel.value);
      }
    }

    function loadRunCard(runId) {
      const r = currentRuns.find(item => item.run_id === runId);
      if (!r) return;

      let tagClass = 'tag-bench';
      if (r.run_id.includes('fail')) tagClass = 'tag-fail';
      else if (r.run_id.includes('mle') || r.run_id.includes('accuracy') || r.run_id.includes('audit')) tagClass = 'tag-success';

      document.getElementById('run-summary-box').innerHTML = `
        <div style="margin-bottom: 1rem;">
          <span class="tag ${tagClass}">${r.model_version}</span>
          <span style="font-size: 0.8rem; color: var(--text-muted); float: right;">${r.timestamp.substring(0, 19)} UTC</span>
        </div>
        <div style="margin-bottom: 1rem;">
          <div style="font-size: 0.8rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase;">Intention</div>
          <div style="font-size: 0.9rem; margin-top: 0.2rem;">${r.intention || 'N/A'}</div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem; font-size: 0.85rem;">
          <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem; border-radius: 6px;">
            <span style="color: var(--text-muted);">Build Acc:</span> <strong>${(r.build_accuracy * 100).toFixed(1)}%</strong>
          </div>
          <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem; border-radius: 6px;">
            <span style="color: var(--text-muted);">Check Acc:</span> <strong>${r.check_accuracy ? (r.check_accuracy * 100).toFixed(1) + '%' : 'Untouched'}</strong>
          </div>
          <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem; border-radius: 6px;">
            <span style="color: var(--text-muted);">Log Loss:</span> <strong>${r.log_loss ? r.log_loss.toFixed(4) : 'N/A'}</strong>
          </div>
          <div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem; border-radius: 6px;">
            <span style="color: var(--text-muted);">Brier Score:</span> <strong>${r.brier_score ? r.brier_score.toFixed(4) : 'N/A'}</strong>
          </div>
        </div>
      `;

      let rawCard = `===========================================================================
EXPERIMENT RUN CARD: ${r.run_id}
===========================================================================
Timestamp (UTC): ${r.timestamp}
Model Version:   ${r.model_version}
Data Version:    ${r.data_version}
---------------------------------------------------------------------------
1. INTENTION & HYPOTHESIS:
   ${r.intention || 'N/A'}

2. PRE-RUN EXPECTATION:
   ${r.expectation || 'N/A'}

3. FAILURE CRITERIA:
   ${r.failure_criteria || 'N/A'}

4. INPUTS & CONFIGURATION:
   ${r.inputs_summary || 'N/A'}

5. OUTPUTS & EMPIRICAL RESULTS:
   ${r.outputs_summary || 'N/A'}

---------------------------------------------------------------------------
KEY METRICS:
  * Top-1 Accuracy (Build): ${r.build_accuracy ? (r.build_accuracy * 100).toFixed(1) + '%' : 'N/A'}
  * Top-1 Accuracy (Check): ${r.check_accuracy ? (r.check_accuracy * 100).toFixed(1) + '%' : 'Preserved Holdout'}
  * Multi-Class Log Loss:   ${r.log_loss ? r.log_loss.toFixed(4) : 'N/A'}
  * Brier Score:            ${r.brier_score ? r.brier_score.toFixed(4) : 'N/A'}
---------------------------------------------------------------------------
6. NOTES & CRITICAL OUTCOME:
   ${r.notes || 'N/A'}
===========================================================================`;
      document.getElementById('raw-card-view').innerText = rawCard;
    }

    async function loadPolicyTable() {
      const policies = [
        { rank: 1, ch: 'online', p: 3.49, msg: 'environment', share: 0.338, vol: 507, rev: 1.18, eco: 0.469 },
        { rank: 2, ch: 'online', p: 4.49, msg: 'environment', share: 0.260, vol: 390, rev: 1.17, eco: 0.375, rec: true },
        { rank: 3, ch: 'online', p: 5.49, msg: 'environment', share: 0.195, vol: 293, rev: 1.07, eco: 0.291 },
        { rank: 4, ch: 'online', p: 3.49, msg: 'cleaning', share: 0.304, vol: 456, rev: 1.06, eco: 0.445 },
        { rank: 5, ch: 'online', p: 4.49, msg: 'cleaning', share: 0.232, vol: 348, rev: 1.04, eco: 0.353 },
        { rank: 6, ch: 'retail', p: 3.49, msg: 'environment', share: 0.275, vol: 412, rev: 0.96, eco: 0.394 },
        { rank: 7, ch: 'online', p: 5.49, msg: 'cleaning', share: 0.172, vol: 258, rev: 0.94, eco: 0.273 },
        { rank: 8, ch: 'retail', p: 4.49, msg: 'environment', share: 0.207, vol: 310, rev: 0.93, eco: 0.308 },
        { rank: 9, ch: 'online', p: 3.49, msg: 'low_odor', share: 0.251, vol: 376, rev: 0.88, eco: 0.383 },
        { rank: 10, ch: 'retail', p: 3.49, msg: 'cleaning', share: 0.245, vol: 368, rev: 0.86, eco: 0.371 },
        { rank: 11, ch: 'online', p: 4.49, msg: 'low_odor', share: 0.190, vol: 284, rev: 0.85, eco: 0.296 },
        { rank: 12, ch: 'retail', p: 5.49, msg: 'environment', share: 0.152, vol: 228, rev: 0.84, eco: 0.232 },
        { rank: 13, ch: 'retail', p: 4.49, msg: 'cleaning', share: 0.183, vol: 274, rev: 0.82, eco: 0.287 },
        { rank: 14, ch: 'online', p: 5.49, msg: 'low_odor', share: 0.139, vol: 208, rev: 0.76, eco: 0.223 },
        { rank: 15, ch: 'retail', p: 5.49, msg: 'cleaning', share: 0.133, vol: 200, rev: 0.73, eco: 0.216 },
        { rank: 16, ch: 'retail', p: 3.49, msg: 'low_odor', share: 0.201, vol: 302, rev: 0.70, eco: 0.313 },
        { rank: 17, ch: 'retail', p: 4.49, msg: 'low_odor', share: 0.148, vol: 222, rev: 0.66, eco: 0.237 },
        { rank: 18, ch: 'retail', p: 5.49, msg: 'low_odor', share: 0.107, vol: 160, rev: 0.58, eco: 0.175 }
      ];

      let tbody = '';
      policies.forEach(p => {
        tbody += `
          <tr class="${p.rec ? 'best-row' : ''}">
            <td>#${p.rank} ${p.rec ? '<span class="tag tag-success">RECOMMENDED</span>' : ''}</td>
            <td style="text-transform: capitalize;">${p.ch}</td>
            <td><strong>$${p.p.toFixed(2)}</strong></td>
            <td>${p.msg}</td>
            <td>${(p.share * 100).toFixed(1)}%</td>
            <td>${p.vol}</td>
            <td><strong>$${p.rev.toFixed(2)}</strong></td>
            <td>${(p.eco * 100).toFixed(1)}%</td>
          </tr>
        `;
      });
      document.getElementById('policy-table-body').innerHTML = tbody;
      updateMarginSim(1.50);
    }

    function updateMarginSim(cost) {
      document.getElementById('cost-display').innerText = '$' + parseFloat(cost).toFixed(2);
      const c = parseFloat(cost);
      const vol_349 = 507;
      const vol_449 = 390;

      const net_349 = vol_349 * (3.49 - c);
      const net_449 = vol_449 * (4.49 - c);
      const diff = net_449 - net_349;

      document.getElementById('margin-output').innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
          <span>Net Contribution @ $3.49 (507 units):</span>
          <strong>$${net_349.toFixed(2)}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.6rem;">
          <span>Net Contribution @ $4.49 (390 units):</span>
          <strong style="color: var(--primary);">$${net_449.toFixed(2)}</strong>
        </div>
        <div style="border-top: 1px solid var(--card-border); padding-top: 0.5rem; color: ${diff >= 0 ? '#34d399' : '#fb7185'}; font-weight: 700;">
          ${diff >= 0 ? '+$' + diff.toFixed(2) + ' Advantage for $4.49' : '-$' + Math.abs(diff).toFixed(2) + ' Advantage for $3.49'}
        </div>
      `;
    }

    // Run simulator on initial load
    window.onload = async () => {
      await initHouseholds();
      runSim();
    };
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/runs":
            runs = get_db_runs()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(runs).encode("utf-8"))
        elif parsed.path == "/api/households":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(HOUSEHOLDS).encode("utf-8"))
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "model": "v1.1-mle-refined"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            pred = run_prediction(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(pred).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Quiet standard HTTP request logging
        return


def run_server(port: int = 8000):
    init_app()
    server_address = ("", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print("=" * 80)
    print(f"SimCo Cleaner Choice Platform LIVE on http://localhost:{port}")
    print("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
