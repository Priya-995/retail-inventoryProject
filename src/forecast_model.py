"""
src/forecast_model.py
=====================
ML demand forecasting with honest ablation and model comparison.

Experiments (all evaluated on the SAME chronological test set — never tuned on it):
  E0  Baselines: GlobalMean, CategoryMean, CategoryMean*SeasonalMultiplier
  E1  RandomForest with prod_mean_sales + prod_record_count  (original)
  E2  RandomForest WITHOUT prod_mean_sales / prod_record_count  (ablation)
  E3  RandomForest with SHRUNK product mean  (n*mu_p + k*mu_c)/(n+k), k=5
  E4  RandomForest (E3 features) with tuned hyper-params via time-ordered val split
  E5  HistGradientBoosting with Poisson loss (E3 features, same tuned settings)
  E6  Ridge (E3 features, reference linear model)

Tuning uses only the TRAINING split, with an internal time-ordered 80/20 val split.
Test set is touched only for final evaluation.

WARNING: Results are on synthetic/combined data. The model largely learns the
generator's assumed seasonal rules, not real retail behaviour.

Run:
    python -m src.forecast_model
"""

from pathlib import Path
import itertools

import joblib
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path("data/dataset.csv")
OUT = Path("outputs")
MODELS_DIR = Path("models")
OUT.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

WARNING_NOTE = (
    "Synthetic/combined data; model largely learns the generator's assumed "
    "seasonal rules, not real retail behaviour."
)

TARGET = "daily_sales"
SHRINK_K = 5          # Bayesian shrinkage constant for smoothed product mean

# ── 1. Load & base feature engineering ────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Calendar features (derived from date only -- zero leakage)
df["month"]           = df["date"].dt.month
df["day_of_week"]     = df["date"].dt.day_of_week   # 0=Mon
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
df["is_summer"]       = df["month"].isin([4, 5, 6]).astype(int)   # Apr-Jun
df["is_diwali_window"] = (
    ((df["date"].dt.month == 10) & (df["date"].dt.day >= 20))
    | ((df["date"].dt.month == 11) & (df["date"].dt.day <= 5))
).astype(int)

# ── 2. Chronological 80/20 train/test split by date ───────────────────────────
unique_dates = sorted(df["date"].unique())
split_idx    = int(len(unique_dates) * 0.8)
cutoff_date  = unique_dates[split_idx - 1]

train_df = df[df["date"] <= cutoff_date].copy()
test_df  = df[df["date"] >  cutoff_date].copy()

print(f"Train: {len(train_df):,} rows up to {cutoff_date.date()}")
print(f"Test : {len(test_df):,} rows from {unique_dates[split_idx].date()} onward")

# ── 3. Internal validation split (inside train only, for hyper-param tuning) ──
# Last 20% of train dates = validation; first 80% = sub-train.
train_dates_sorted = sorted(train_df["date"].unique())
val_cut_idx   = int(len(train_dates_sorted) * 0.8)
val_cutoff    = train_dates_sorted[val_cut_idx - 1]

subtrain_df = train_df[train_df["date"] <= val_cutoff].copy()
val_df      = train_df[train_df["date"] >  val_cutoff].copy()

print(f"\nInternal validation split (tuning only):")
print(f"  Sub-train : {len(subtrain_df):,} rows up to {val_cutoff.date()}")
print(f"  Val       : {len(val_df):,} rows")

# ── 4. Training-set statistics (computed once from full train_df) ──────────────
# Used for: category means, seasonal multipliers, shrinkage, feature engineering.

cat_means_train = train_df.groupby("category")[TARGET].mean()

# Raw product stats from training set
# Join category via a simple product->category lookup from train_df
prod_cat = train_df[["product_id", "category"]].drop_duplicates("product_id")

prod_stats_train = (
    train_df.groupby("product_id")[TARGET]
    .agg(n="count", mu_prod="mean")
    .reset_index()
)
prod_stats_train = prod_stats_train.merge(prod_cat, on="product_id", how="left")
prod_stats_train["mu_cat"] = prod_stats_train["category"].map(cat_means_train)

# Shrunk mean: (n * mu_prod + k * mu_cat) / (n + k)
prod_stats_train["prod_shrunk_mean"] = (
    (prod_stats_train["n"] * prod_stats_train["mu_prod"]
     + SHRINK_K * prod_stats_train["mu_cat"])
    / (prod_stats_train["n"] + SHRINK_K)
)

# Seasonal multipliers from train (ratio of season mean to overall mean)
global_mean_train = float(train_df[TARGET].mean())
season_mult_train = (
    train_df.groupby("season")[TARGET].mean() / global_mean_train
).rename("season_mult")

# ── 5. Same statistics from sub-train (for tuning experiments) ────────────────
cat_means_subtrain = subtrain_df.groupby("category")[TARGET].mean()
global_mean_subtrain = float(subtrain_df[TARGET].mean())

prod_stats_subtrain = (
    subtrain_df.groupby("product_id")[TARGET]
    .agg(n="count", mu_prod="mean")
    .reset_index()
)
prod_cat_sub = subtrain_df[["product_id", "category"]].drop_duplicates()
prod_stats_subtrain = prod_stats_subtrain.merge(prod_cat_sub, on="product_id", how="left")
prod_stats_subtrain["mu_cat"] = prod_stats_subtrain["category"].map(cat_means_subtrain)
prod_stats_subtrain["prod_shrunk_mean"] = (
    (prod_stats_subtrain["n"] * prod_stats_subtrain["mu_prod"]
     + SHRINK_K * prod_stats_subtrain["mu_cat"])
    / (prod_stats_subtrain["n"] + SHRINK_K)
)

season_mult_subtrain = (
    subtrain_df.groupby("season")[TARGET].mean() / global_mean_subtrain
).rename("season_mult")

# ── 6. Feature attachment helpers ─────────────────────────────────────────────
def attach_original_features(split, prod_stats, cat_means):
    """Attach prod_mean_sales + prod_record_count (original scheme)."""
    s = split.copy()
    pm = prod_stats[["product_id", "mu_prod", "n"]].rename(
        columns={"mu_prod": "prod_mean_sales", "n": "prod_record_count"})
    s = s.merge(pm, on="product_id", how="left")
    s["prod_mean_sales"]   = s["prod_mean_sales"].fillna(s["category"].map(cat_means))
    s["prod_record_count"] = s["prod_record_count"].fillna(0)
    return s

def attach_shrunk_features(split, prod_stats, cat_means):
    """Attach shrunk product mean (no raw prod_mean or count)."""
    s = split.copy()
    pm = prod_stats[["product_id", "prod_shrunk_mean"]]
    s = s.merge(pm, on="product_id", how="left")
    # Unseen products: shrunk mean reduces to category mean (n=0 => k*mu_cat/k)
    s["prod_shrunk_mean"] = s["prod_shrunk_mean"].fillna(s["category"].map(cat_means))
    return s

# ── 7. Metrics helper ─────────────────────────────────────────────────────────
def metrics(y_true, y_pred, name, note=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    denom = np.abs(y_true).sum()
    wape = (np.abs(y_true - y_pred).sum() / denom * 100) if denom else np.nan
    r2   = r2_score(y_true, y_pred)
    return {
        "experiment": name,
        "MAE":      round(mae,  4),
        "RMSE":     round(rmse, 4),
        "WAPE_pct": round(wape, 4),
        "R2":       round(r2,   4),
        "note": note or WARNING_NOTE,
    }

# ── 8. Preprocessor factory ───────────────────────────────────────────────────
CAT_FEATS = ["season", "category"]

def make_preprocessor(num_feats):
    return ColumnTransformer([
        ("num", "passthrough", num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
    ])

def make_rf_pipe(num_feats, n_estimators=200, max_depth=None,
                 min_samples_leaf=1):
    return Pipeline([
        ("prep",  make_preprocessor(num_feats)),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42, n_jobs=-1,
        )),
    ])

# ── 9. Feature column lists ────────────────────────────────────────────────────
BASE_NUM = [
    "month", "day_of_week", "is_weekend", "is_summer", "is_diwali_window",
    "lead_days", "unit_cost", "reorder_level",
]
ORIG_NUM    = BASE_NUM + ["prod_mean_sales", "prod_record_count"]
SHRUNK_NUM  = BASE_NUM + ["prod_shrunk_mean"]
NOSTAT_NUM  = BASE_NUM   # ablation: no product-level statistics at all

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: BASELINES (evaluated on test set, stats from train)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Baselines ---")
all_results = []

y_test  = test_df[TARGET]
y_train = train_df[TARGET]

# B0: Global train mean
preds_global = np.full(len(y_test), global_mean_train)
all_results.append(metrics(y_test, preds_global,
    "B0_GlobalMean",
    "Predicts the single training-set mean for every record."))
print(f"  B0 GlobalMean            MAE={all_results[-1]['MAE']:.4f}")

# B1: Category train mean
preds_catmean = test_df["category"].map(cat_means_train).fillna(global_mean_train).to_numpy()
all_results.append(metrics(y_test, preds_catmean,
    "B1_CategoryMean",
    "Predicts the per-category training mean for each record."))
print(f"  B1 CategoryMean          MAE={all_results[-1]['MAE']:.4f}")

# B2: Category mean * seasonal multiplier (multipliers from train only)
preds_cat_season = (
    test_df["category"].map(cat_means_train).fillna(global_mean_train)
    * test_df["season"].map(season_mult_train).fillna(1.0)
).to_numpy()
all_results.append(metrics(y_test, preds_cat_season,
    "B2_CategoryMean_x_SeasonMult",
    "Category train mean scaled by per-season multiplier (both from train only)."))
print(f"  B2 CategoryMean*Season   MAE={all_results[-1]['MAE']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: ML EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

# ── E1: Original RandomForest (prod_mean_sales + prod_record_count) ───────────
print("\n--- E1: RandomForest (original prod_mean_sales + count) ---")
train_e1 = attach_original_features(train_df, prod_stats_train, cat_means_train)
test_e1  = attach_original_features(test_df,  prod_stats_train, cat_means_train)

pipe_e1 = make_rf_pipe(ORIG_NUM, n_estimators=200)
pipe_e1.fit(train_e1[ORIG_NUM + CAT_FEATS], train_e1[TARGET])
preds_e1 = pipe_e1.predict(test_e1[ORIG_NUM + CAT_FEATS])
all_results.append(metrics(y_test, preds_e1,
    "E1_RF_OrigFeatures",
    "RF200 with raw prod_mean_sales + prod_record_count from train."))
print(f"  E1 RF original           MAE={all_results[-1]['MAE']:.4f}")

# ── E2: Ablation -- RandomForest WITHOUT product-level stats ─────────────────
print("--- E2: Ablation -- RF without any product stats ---")
pipe_e2 = make_rf_pipe(NOSTAT_NUM, n_estimators=200)
pipe_e2.fit(train_df[NOSTAT_NUM + CAT_FEATS], train_df[TARGET])
preds_e2 = pipe_e2.predict(test_df[NOSTAT_NUM + CAT_FEATS])
all_results.append(metrics(y_test, preds_e2,
    "E2_RF_NoProductStats (ablation)",
    "Ablation: RF200 with no prod_mean_sales or prod_record_count."))
print(f"  E2 RF no-stats (ablation) MAE={all_results[-1]['MAE']:.4f}")

# ── E3: RandomForest with SHRUNK product mean ─────────────────────────────────
print("--- E3: RF with shrunk product mean (k=5) ---")
train_e3 = attach_shrunk_features(train_df, prod_stats_train, cat_means_train)
test_e3  = attach_shrunk_features(test_df,  prod_stats_train, cat_means_train)

pipe_e3 = make_rf_pipe(SHRUNK_NUM, n_estimators=200)
pipe_e3.fit(train_e3[SHRUNK_NUM + CAT_FEATS], train_e3[TARGET])
preds_e3 = pipe_e3.predict(test_e3[SHRUNK_NUM + CAT_FEATS])
all_results.append(metrics(y_test, preds_e3,
    "E3_RF_ShrunkMean",
    f"RF200 with shrunk product mean (n*mu_p + {SHRINK_K}*mu_c)/(n+{SHRINK_K}), train only."))
print(f"  E3 RF shrunk mean        MAE={all_results[-1]['MAE']:.4f}")

# ── E4: Tuned RandomForest (shrunk features) via internal val split ───────────
print("--- E4: Tuning RF on internal val split (no test-set peeking) ---")
subtrain_e4 = attach_shrunk_features(subtrain_df, prod_stats_subtrain, cat_means_subtrain)
val_e4      = attach_shrunk_features(val_df,      prod_stats_subtrain, cat_means_subtrain)
y_val       = val_df[TARGET]

# Grid: modest ranges as specified
param_grid = list(itertools.product(
    [200, 400],          # n_estimators
    [4, 6, 8],           # max_depth
    [10, 20, 30],        # min_samples_leaf
))

print(f"  Trying {len(param_grid)} hyper-param combos on val split ...")
best_val_mae = float("inf")
best_params  = None

for n_est, max_d, min_leaf in param_grid:
    pipe_tmp = make_rf_pipe(SHRUNK_NUM, n_estimators=n_est,
                             max_depth=max_d, min_samples_leaf=min_leaf)
    pipe_tmp.fit(subtrain_e4[SHRUNK_NUM + CAT_FEATS], subtrain_e4[TARGET])
    val_preds   = pipe_tmp.predict(val_e4[SHRUNK_NUM + CAT_FEATS])
    val_mae     = mean_absolute_error(y_val, val_preds)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_params  = (n_est, max_d, min_leaf)

n_est_best, max_d_best, min_leaf_best = best_params
print(f"  Best val MAE={best_val_mae:.4f} => "
      f"n_estimators={n_est_best}, max_depth={max_d_best}, "
      f"min_samples_leaf={min_leaf_best}")

# Re-train best config on FULL train set (not just sub-train)
train_e4 = attach_shrunk_features(train_df, prod_stats_train, cat_means_train)
test_e4  = test_e3   # same features
pipe_e4  = make_rf_pipe(SHRUNK_NUM, n_estimators=n_est_best,
                         max_depth=max_d_best, min_samples_leaf=min_leaf_best)
pipe_e4.fit(train_e4[SHRUNK_NUM + CAT_FEATS], train_e4[TARGET])
preds_e4 = pipe_e4.predict(test_e4[SHRUNK_NUM + CAT_FEATS])
all_results.append(metrics(y_test, preds_e4,
    f"E4_RF_Tuned(n={n_est_best},d={max_d_best},leaf={min_leaf_best})",
    f"RF tuned on internal val split: n_estimators={n_est_best}, "
    f"max_depth={max_d_best}, min_samples_leaf={min_leaf_best}. "
    f"Val MAE={best_val_mae:.4f}."))
print(f"  E4 RF tuned              MAE={all_results[-1]['MAE']:.4f}")

# ── E5: HistGradientBoosting with Poisson loss ────────────────────────────────
print("--- E5: HistGradientBoostingRegressor (loss=poisson) ---")
# HGBR handles categoricals natively; still use same OHE pipeline for consistency.
# Poisson loss is appropriate for non-negative count-like targets.
pipe_e5 = Pipeline([
    ("prep",  make_preprocessor(SHRUNK_NUM)),
    ("model", HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=400,
        max_depth=max_d_best,       # reuse best depth from tuning
        min_samples_leaf=min_leaf_best,
        learning_rate=0.05,
        random_state=42,
    )),
])
pipe_e5.fit(train_e4[SHRUNK_NUM + CAT_FEATS], train_e4[TARGET])
preds_e5 = pipe_e5.predict(test_e4[SHRUNK_NUM + CAT_FEATS])
all_results.append(metrics(y_test, preds_e5,
    "E5_HistGB_Poisson",
    "HistGradientBoosting with loss=poisson; shrunk product mean features."))
print(f"  E5 HistGB Poisson        MAE={all_results[-1]['MAE']:.4f}")

# ── E6: Ridge (linear, shrunk features) ───────────────────────────────────────
print("--- E6: Ridge (shrunk features, reference linear model) ---")
pipe_e6 = Pipeline([
    ("prep",  make_preprocessor(SHRUNK_NUM)),
    ("model", Ridge(alpha=1.0)),
])
pipe_e6.fit(train_e4[SHRUNK_NUM + CAT_FEATS], train_e4[TARGET])
preds_e6 = pipe_e6.predict(test_e4[SHRUNK_NUM + CAT_FEATS])
all_results.append(metrics(y_test, preds_e6,
    "E6_Ridge_ShrunkFeatures",
    "Ridge regression with shrunk product mean features."))
print(f"  E6 Ridge (shrunk)        MAE={all_results[-1]['MAE']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: Evaluate & pick winner honestly
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT / "model_metrics.csv", index=False)

# Winner = lowest MAE across ALL entries (baseline or ML)
best_idx  = results_df["MAE"].idxmin()
winner    = results_df.loc[best_idx]
winner_name = winner["experiment"]

# Among ML-only entries (non-Baseline)
ml_df  = results_df[~results_df["experiment"].str.startswith("B")]
ml_best_idx  = ml_df["MAE"].idxmin()
ml_winner    = ml_df.loc[ml_best_idx]
ml_winner_name = ml_winner["experiment"]

# Map experiment name to (pipe, preds) for saving
experiment_to_pipe = {
    "E1_RF_OrigFeatures":                                          (pipe_e1, preds_e1),
    "E2_RF_NoProductStats (ablation)":                             (pipe_e2, preds_e2),
    "E3_RF_ShrunkMean":                                            (pipe_e3, preds_e3),
    f"E4_RF_Tuned(n={n_est_best},d={max_d_best},leaf={min_leaf_best})": (pipe_e4, preds_e4),
    "E5_HistGB_Poisson":                                           (pipe_e5, preds_e5),
    "E6_Ridge_ShrunkFeatures":                                     (pipe_e6, preds_e6),
}

# Save best ML model regardless of whether it beats the baseline
if ml_winner_name in experiment_to_pipe:
    best_ml_pipe, best_preds = experiment_to_pipe[ml_winner_name]
    joblib.dump(best_ml_pipe, MODELS_DIR / "best_model.joblib")
    print(f"\nBest ML model ({ml_winner_name}) saved to models/best_model.joblib")
else:
    # Fallback (should not happen)
    best_preds = preds_e4
    best_ml_pipe = pipe_e4

# ── Predictions CSV ───────────────────────────────────────────────────────────
pred_df = test_df[["date", "product_id", "category", TARGET]].copy()
pred_df["predicted_daily_sales"] = best_preds
pred_df["model"] = ml_winner_name
pred_df["warning"] = WARNING_NOTE
pred_df.to_csv(OUT / "model_predictions_test.csv", index=False)

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
sample_size = min(300, len(pred_df))
sample = pred_df.sample(sample_size, random_state=42).sort_values("date")

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(sample["date"], sample[TARGET],
           alpha=0.4, s=18, label="Actual", color="steelblue")
ax.scatter(sample["date"], sample["predicted_daily_sales"],
           alpha=0.4, s=18, label=f"Predicted ({ml_winner_name})", color="tomato")
ax.set_title(f"Actual vs Predicted Daily Sales\n{ml_winner_name} (n={sample_size} sample)")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Sales (units)")
ax.legend(fontsize=8)
ax.text(0.01, -0.13, f"[WARNING] {WARNING_NOTE}",
        transform=ax.transAxes, fontsize=7, color="gray")
plt.tight_layout()
plt.savefig(OUT / "04_actual_vs_predicted.png", dpi=160)
plt.close()
print("Saved outputs/04_actual_vs_predicted.png")

# ── Plot 2: Feature importance (best ML model if tree-based) ─────────────────
try:
    model_step = best_ml_pipe.named_steps["model"]
    prep_step  = best_ml_pipe.named_steps["prep"]

    # Determine which numeric features the best pipe used
    num_feats_used = prep_step.transformers[0][2]  # "passthrough" list
    cat_enc  = prep_step.named_transformers_["cat"]
    cat_names = cat_enc.get_feature_names_out(CAT_FEATS).tolist()
    feature_names = list(num_feats_used) + cat_names

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_)
    else:
        importances = None

    if importances is not None and len(importances) == len(feature_names):
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        top15 = feat_imp.head(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        top15.plot(kind="barh", ax=ax, color="teal")
        ax.invert_yaxis()
        ax.set_title(f"Top 15 Feature Importances\n{ml_winner_name}")
        ax.set_xlabel("Importance / |Coefficient|")
        ax.text(0.01, -0.12, f"[WARNING] {WARNING_NOTE}",
                transform=ax.transAxes, fontsize=7, color="gray")
        plt.tight_layout()
        plt.savefig(OUT / "05_feature_importance.png", dpi=160)
        plt.close()
        print("Saved outputs/05_feature_importance.png")

        print(f"\nTop 5 features ({ml_winner_name}):")
        for feat, imp in feat_imp.head(5).items():
            print(f"  {feat}: {imp:.4f}")
    else:
        print("Feature importance length mismatch -- skipping plot.")
        feat_imp = pd.Series(dtype=float)

except Exception as exc:
    print(f"Feature importance plot skipped: {exc}")
    feat_imp = pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Print honest comparison table
# ─────────────────────────────────────────────────────────────────────────────
display_cols = ["experiment", "MAE", "RMSE", "WAPE_pct", "R2"]

print("\n" + "=" * 100)
print("FULL EXPERIMENT TABLE  (all on same test set; synthetic/combined data)")
print("=" * 100)
print(results_df[display_cols].to_string(index=False))
print("=" * 100)

b1_mae = results_df.loc[results_df["experiment"] == "B1_CategoryMean", "MAE"].iloc[0]
ml_best_mae = ml_winner["MAE"]
beats = ml_best_mae < b1_mae

print(f"\nCategory-mean baseline MAE : {b1_mae:.4f}")
print(f"Best ML model MAE          : {ml_best_mae:.4f}  ({ml_winner_name})")
if beats:
    print(f"RESULT: Best ML model BEATS the category-mean baseline "
          f"by {b1_mae - ml_best_mae:.4f} MAE units.")
else:
    print(f"RESULT: No ML model beats the category-mean baseline "
          f"(gap = {ml_best_mae - b1_mae:.4f} MAE units in baseline's favour).")
    print("This is EXPECTED on sparse synthetic data (~10 records per SKU).")
    print("Reporting honestly; see outputs/run_log.md for explanation.")

print(f"\n[WARNING] {WARNING_NOTE}")
print("Outputs: outputs/model_metrics.csv, model_predictions_test.csv, "
      "04_actual_vs_predicted.png, 05_feature_importance.png, models/best_model.joblib")
