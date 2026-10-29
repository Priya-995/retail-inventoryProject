# Optimized Retail Inventory Management System

A four-week educational prototype for inventory analytics, demand-forecasting baselines,
ML demand forecasting, and reorder recommendations.

**HCLTech Career Shaper Programme — AIML Domain (Individual, 4-Week Project)**

---

## Important data note

`data/dataset_original.csv` is the supplied source file and is **never modified**.
`generate_dataset.py` retains those 500 rows (after deduplication: 488) and creates
4,512 synthetic records, then saves `data/dataset.csv` (5,000 rows total).

> **Synthetic data disclaimer:** All generated demand values are illustrative.
> Seasonal multipliers (summer +20%, Diwali window +40%, weekends +8%) are
> assumptions, not measured real-world effects. No metric or chart in this project
> should be represented as actual retailer data.

---

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Running the pipeline

```powershell
# 1. Generate dataset (500 original + 4,512 synthetic = 5,000 rows)
python generate_dataset.py

# 2. Exploratory data analysis + three output charts
python -m src.eda

# 3. Aggregate daily-sales mean baseline (chronological 80/20 split)
python -m src.forecast_baseline

# 4. ML demand forecasting (Ridge, RandomForest, GradientBoosting)
#    Saves best model to models/best_model.joblib
python -m src.forecast_model

# 5. Streamlit dashboard (four tabs: Overview, Inventory, Reorder, Suppliers)
#    IMPORTANT: use the venv's streamlit, not the system/Anaconda one
streamlit run app.py          # works if venv is activated (step above)
# Or without activating:
# .\.venv\Scripts\streamlit.exe run app.py
```

---

## Project structure

```
retail_inventory_project/
├── generate_dataset.py          # Dataset generation (SEED=42, do not modify)
├── app.py                       # Streamlit dashboard (4 tabs)
├── requirements.txt             # Pinned dependencies
├── README.md                    # This file
│
├── data/
│   ├── dataset_original.csv     # Original 500 rows (NEVER modified)
│   ├── dataset.csv              # Generated working dataset (5,000 rows)
│   ├── dataset_validation.json  # Auditable row/column counts
│   ├── data_dictionary.csv      # Full column descriptions (this project)
│   └── data_dictionary_original.csv  # Supplied template
│
├── src/
│   ├── __init__.py              # Package marker for python -m src.*
│   ├── eda.py                   # EDA audit + 3 charts
│   ├── forecast_baseline.py     # Aggregate mean baseline
│   └── forecast_model.py        # ML pipeline (Ridge / RF / GBT)
│
├── outputs/
│   ├── eda_data_audit.csv
│   ├── 01_sales_by_category.png
│   ├── 02_stock_vs_reorder.png
│   ├── 03_daily_sales_trend.png
│   ├── baseline_metrics.csv
│   ├── baseline_forecast_test.csv
│   ├── model_metrics.csv        # Baseline + ML model comparison
│   ├── model_predictions_test.csv
│   ├── 04_actual_vs_predicted.png
│   ├── 05_feature_importance.png
│   ├── run_log.md               # Phase 1 verification log
│   ├── data_cleaning_log.md     # Cleaning/normalisation audit
│   └── ai_disclosure.md         # AI tool usage appendix
│
└── models/
    └── best_model.joblib        # Saved best ML model
```

---

## What is included

| Script | Purpose |
|---|---|
| `generate_dataset.py` | Loads original 500 rows, normalises product metadata, generates 4,512 synthetic rows, saves to `data/dataset.csv` |
| `src/eda.py` | Schema audit, missing-value check, duplicate check, three output charts |
| `src/forecast_baseline.py` | Chronological 80/20 split on aggregate daily sales; training-period mean baseline with MAE, RMSE, WAPE |
| `src/forecast_model.py` | Per-record ML forecasting with Ridge, RandomForest, GradientBoosting; chronological split; no data leakage |
| `app.py` | Streamlit dashboard with Overview, Inventory, Reorder, and Suppliers tabs; anonymise-pricing toggle |

---

## Limitations to disclose

1. **Sparsity (~10 rows per product):** With 5,000 rows spread across 500 products
   and 366 dates, each product appears on roughly 10 distinct dates on average.
   This severely limits per-product time-series forecasting; the ML model uses a
   product-level mean feature computed from the training split as a proxy.

2. **"Latest record = current stock" caveat:** The dashboard treats each product's
   most recent record as its current inventory snapshot. In a real system, this
   field would come from a live WMS feed. If the data is weeks old, recommendations
   may be materially out of date.

3. **No real-world validation:** The ML model largely learns the synthetic
   generator's seasonal rules (Diwali window, summer boost, weekends), not genuine
   shopper behaviour. R² on the test set is approximately 0.23 — a weak signal,
   as expected on noisy synthetic data.

4. **Aggregate baseline only:** `src/forecast_baseline.py` evaluates at the
   store-wide daily aggregate level, not per-SKU. Direct comparison with the ML
   model (which is per-record) should be done with care.

5. **No open purchase orders, pack sizes, or MOQs:** Reorder quantities are
   purely arithmetic (target stock − current stock) with no supplier constraints.

6. **Pricing currency assumed INR:** `unit_cost` values are assumed to be in
   Indian Rupees based on the range and context; this is not confirmed by the
   original source.

---

## Data schema

`date, product_id, category, season, stock_qty, reorder_level, daily_sales, unit_cost, supplier_id, lead_days`

See `data/data_dictionary.csv` for full column descriptions, data types, and sample values.
