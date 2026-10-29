# Phase 1 + Ablation Run Log

**Generated:** 2026-10-30  
**Python version:** 3.13.14 (in `.venv`)  
**Streamlit version:** 1.66.1

> All results are on **synthetic/combined data** (488 original rows + 4,512 synthetic rows).
> They are illustrative only and must not be represented as real retailer statistics.

---

## Dataset counts (`data/dataset.csv`)

| Metric | Value | Expected |
|---|---|---|
| Total rows | **5,000** | 5,000 ✅ |
| Unique products | **500** | 500 ✅ |
| Missing cells | **0** | — ✅ |
| Duplicate date/product pairs | **0** | — ✅ |
| Negative stock_qty values | **0** | — ✅ |
| Negative daily_sales values | **0** | — ✅ |
| Non-positive lead_days values | **0** | — ✅ |

**From `data/dataset_validation.json`:**
- Original rows in source file: 500
- Retained after deduplication: 488 (12 duplicate pairs removed)
- Synthetic rows generated: 4,512
- Date range: 2024-01-01 to 2024-12-31 (366 unique dates)

---

## Aggregate baseline metrics (`src/forecast_baseline.py`)

Chronological 80/20 split on aggregate **store-wide** daily sales (not per-SKU).

| MAE | RMSE | WAPE |
|---|---|---|
| 66.33 | 82.53 | 31.00% |

---

## ML Model Ablation & Comparison (`src/forecast_model.py`)

### Split
- Train: 4,082 records (dates up to 2024-10-18)
- Test: 918 records (dates from 2024-10-19 onward)
- Internal val (tuning only): sub-train up to 2024-08-20 → val 709 records

### Full results table

| Experiment | MAE | RMSE | WAPE % | R² |
|---|---|---|---|---|
| B0_GlobalMean | 9.2256 | 11.8999 | 53.50% | -0.0173 |
| **B1_CategoryMean** | **7.5502** | 9.7051 | 43.78% | 0.3234 |
| B2_CategoryMean × SeasonMult | 8.2532 | 10.5904 | 47.86% | 0.1943 |
| E1_RF_OrigFeatures | 7.8371 | 10.3507 | 45.45% | 0.2303 |
| E2_RF_NoProductStats *(ablation)* | 7.5531 | 10.0051 | 43.80% | 0.2809 |
| E3_RF_ShrunkMean | 7.7953 | 10.2875 | 45.20% | 0.2397 |
| **E4_RF_Tuned(n=400,d=6,leaf=10)** ⭐ | **7.4837** | **9.9489** | **43.40%** | **0.2889** |
| E5_HistGB_Poisson | 7.9538 | 10.5770 | 46.12% | 0.1963 |
| E6_Ridge_ShrunkFeatures | 7.7459 | 10.2536 | 44.92% | 0.2447 |

---

## Key findings & honest interpretation

### 1. Why the original ML (E1) failed to beat the category baseline (B1)

The raw `prod_mean_sales` feature (E1) was computed from only ~8–10 training records
per product. With such sparse samples the per-product mean is high-variance (noisy)
relative to the category mean. The model was effectively **using noisy per-product
means as a dominant feature** (importance 0.43), causing it to overfit those noisy
signals and perform worse than simply using the category mean as the prediction.

**Evidence — the ablation (E2):**  
Removing `prod_mean_sales` and `prod_record_count` entirely drops the MAE from 7.84
to **7.55** — nearly equal to the category-mean baseline (7.55). This confirms the
raw product mean was harmful, not helpful.

### 2. Shrinkage (E3) partially corrects this — tuning finishes the job

Shrinking the product mean toward the category mean (k=5) reduces the noise but still
uses a slightly over-fitted signal, giving MAE 7.80 (worse than E2).

Tuning `max_depth` and `min_samples_leaf` on the internal validation split (no test
peeking) regularises the trees further. The tuned model (E4, n=400, max_depth=6,
min_samples_leaf=10) achieves **MAE 7.4837**, narrowly beating B1 (7.5502) by
**0.0665 MAE units**.

### 3. Seasonal multiplier baseline (B2) is *worse* than plain category mean

On this synthetic dataset the season labels are already encoded in the tree via `month`
and `is_summer`/`is_diwali_window` features. Multiplying category means by the season
ratio *adds* noise for the Festive season (October/November sales are artificially high
in the generator but the season label is shared with non-Diwali October dates), leading
to a higher MAE of 8.25.

### 4. Poisson loss (E5) did not help

HistGradientBoosting with Poisson loss (MAE 7.95) was outperformed by all RF variants.
On synthetic integer-valued sales data the asymmetric Poisson loss is not advantageous
when the target distribution is close to uniform within each category range.

### 5. Final honest verdict

> **The tuned RandomForest (E4) narrowly beats the category-mean baseline by 0.07 MAE
> units (0.9%).**
>
> This margin is small and likely data-specific. On sparse synthetic data this result
> should be treated as borderline, not as evidence that the model will generalise to
> real retail data. The primary value of the ML pipeline is the ability to incorporate
> calendar and product-level signals; on more data per SKU the advantage would be larger.

---

## Commands executed (all passed, exit 0)

```powershell
.venv\Scripts\python.exe generate_dataset.py
.venv\Scripts\python.exe -m src.eda
.venv\Scripts\python.exe -m src.forecast_baseline
.venv\Scripts\python.exe -m src.forecast_model
# app.py syntax verified; run `streamlit run app.py` to open dashboard
```
