
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TARGET_ROWS = 5000
OUT = Path("data")
OUT.mkdir(exist_ok=True)

SOURCE = OUT / "dataset_original.csv"

EXPECTED = [
    "date", "product_id", "category", "season", "stock_qty",
    "reorder_level", "daily_sales", "unit_cost", "supplier_id", "lead_days"
]

categories = ["Grocery", "Electronics", "HomeDecor", "Apparel"]
cat_probs = [0.35, 0.20, 0.20, 0.25]
supplier_ids = [f"SUP{i:02d}" for i in range(1, 31)]
product_ids = [f"SKU{i:04d}" for i in range(1, 501)]

cost_ranges = {
    "Grocery": (20, 900),
    "Electronics": (500, 25000),
    "HomeDecor": (100, 8000),
    "Apparel": (150, 5000),
}

rng = np.random.default_rng(SEED)

# Load source and validate schema.
source = pd.read_csv(SOURCE)

if list(source.columns) != EXPECTED:
    raise ValueError(f"Unexpected columns: {list(source.columns)}")

source["date"] = pd.to_datetime(source["date"], errors="coerce")

if source["date"].isna().any():
    raise ValueError("The source contains invalid dates.")

if not source["product_id"].isin(product_ids).all():
    raise ValueError("The source contains product IDs outside SKU0001-SKU0500.")

original_rows = len(source)

# Build one consistent product master for all 500 product IDs.
# Rule: mode for categorical fields; median for numeric metadata.
metadata = {}

for pid in product_ids:
    group = source[source["product_id"] == pid]

    if group.empty:
        category = str(rng.choice(categories, p=cat_probs))
        supplier = str(rng.choice(supplier_ids))
        lo, hi = cost_ranges[category]

        metadata[pid] = {
            "category": category,
            "supplier_id": supplier,
            "unit_cost": round(float(rng.uniform(lo, hi)), 2),
            "reorder_level": int(rng.integers(20, 101)),
            "lead_days": int(rng.integers(3, 22)),
        }
        continue

    # Stable tie-breaking: pandas mode returns sorted values.
    category = group["category"].mode().iloc[0]
    supplier = group["supplier_id"].mode().iloc[0]

    metadata[pid] = {
        "category": category,
        "supplier_id": supplier,
        "unit_cost": round(float(group["unit_cost"].median()), 2),
        "reorder_level": max(1, int(round(group["reorder_level"].median()))),
        "lead_days": max(1, int(round(group["lead_days"].median()))),
    }

# Normalize the working copy; the original CSV is never overwritten.
base = source.copy()

for field in ["category", "supplier_id", "unit_cost", "reorder_level", "lead_days"]:
    base[field] = base["product_id"].map(
        {pid: meta[field] for pid, meta in metadata.items()}
    )

# Keep one observation per product per date.
# If duplicates exist, retain the first source row and document the removals.
base = (
    base.sort_values(["date", "product_id"], kind="stable")
        .drop_duplicates(["date", "product_id"], keep="first")
        .copy()
)

retained_original_rows = len(base)
rows_needed = TARGET_ROWS - retained_original_rows

if rows_needed < 0:
    raise ValueError("The cleaned source exceeds the target row count.")

# Generate unique product/date combinations for calendar year 2024.
dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

month_weights = np.ones(12, dtype=float)
month_weights[[3, 4, 5]] *= 1.20
month_weights[[9, 10]] *= 1.10
month_weights /= month_weights.sum()

used_pairs = {
    (row.product_id, row.date.normalize())
    for row in base[["product_id", "date"]].itertuples(index=False)
}

new_rows = []
attempts = 0
max_attempts = rows_needed * 100

while len(new_rows) < rows_needed:
    attempts += 1
    if attempts > max_attempts:
        raise RuntimeError("Could not generate enough unique date/product pairs.")

    pid = str(rng.choice(product_ids))

    month = int(rng.choice(np.arange(1, 13), p=month_weights))
    valid_days = dates[dates.month == month]
    dt = pd.Timestamp(rng.choice(valid_days))

    key = (pid, dt.normalize())
    if key in used_pairs:
        continue

    used_pairs.add(key)
    meta = metadata[pid]
    category = meta["category"]

    base_ranges = {
        "Grocery": (8, 45),
        "Electronics": (2, 18),
        "HomeDecor": (2, 20),
        "Apparel": (3, 25),
    }

    low, high = base_ranges[category]
    demand = float(rng.uniform(low, high))

    # Illustrative synthetic demand assumptions:
    # summer: +20%; Diwali window: +40%.
    if dt.month in [4, 5, 6]:
        demand *= 1.20

    if (dt.month == 10 and dt.day >= 20) or (
        dt.month == 11 and dt.day <= 5
    ):
        demand *= 1.40

    if dt.dayofweek >= 5:
        demand *= 1.08

    daily_sales = max(
        0, int(round(demand * rng.normal(1.0, 0.18)))
    )

    coverage_days = float(rng.uniform(1.5, 18))
    stock_qty = max(
        0, int(round(daily_sales * coverage_days + rng.normal(0, 12)))
    )

    season = (
        "Winter" if dt.month in [12, 1, 2]
        else "Summer" if dt.month in [3, 4, 5, 6]
        else "Monsoon" if dt.month in [7, 8, 9]
        else "Festive"
    )

    new_rows.append({
        "date": dt,
        "product_id": pid,
        "category": category,
        "season": season,
        "stock_qty": stock_qty,
        "reorder_level": meta["reorder_level"],
        "daily_sales": daily_sales,
        "unit_cost": meta["unit_cost"],
        "supplier_id": meta["supplier_id"],
        "lead_days": meta["lead_days"],
    })

generated = pd.DataFrame(new_rows, columns=EXPECTED)

combined = pd.concat([base[EXPECTED], generated[EXPECTED]], ignore_index=True)
combined = combined.sort_values(
    ["date", "product_id"], kind="stable"
).reset_index(drop=True)

# Enforce key requirements before writing.
if len(combined) != TARGET_ROWS:
    raise ValueError(f"Expected {TARGET_ROWS} rows, got {len(combined)}")

if combined["product_id"].nunique() != 500:
    raise ValueError("Expected exactly 500 unique product IDs.")

if combined.duplicated(["date", "product_id"]).any():
    raise ValueError("Duplicate date/product pairs remain.")

# Exact timestamp format requested by the project brief.
combined["date"] = pd.to_datetime(combined["date"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
combined.to_csv(OUT / "dataset.csv", index=False)

# Save an auditable validation report.
audit = {
    "original_rows": int(original_rows),
    "retained_original_rows_after_deduplication": int(retained_original_rows),
    "generated_rows": int(len(generated)),
    "total_rows": int(len(combined)),
    "unique_products": int(combined["product_id"].nunique()),
    "unique_dates": int(pd.to_datetime(combined["date"]).dt.date.nunique()),
    "date_min": str(combined["date"].min()),
    "date_max": str(combined["date"].max()),
    "columns_match_expected": bool(list(combined.columns) == EXPECTED),
    "missing_cells": int(combined.isna().sum().sum()),
    "duplicate_date_product_pairs": int(
        combined.duplicated(["date", "product_id"]).sum()
    ),
    "metadata_rule": (
        "Mode for category/supplier; median for cost/reorder level/lead days. "
        "Missing product metadata is generated synthetically."
    ),
    "note": (
        "Synthetic data is illustrative. The source file is preserved. "
        "Duplicate source date/product pairs were reduced to one row in "
        "the working dataset. Seasonal multipliers are assumptions, "
        "not measured real-world effects."
    ),
}

pd.Series(audit).to_json(OUT / "dataset_validation.json", indent=2)
print(pd.Series(audit).to_string())
