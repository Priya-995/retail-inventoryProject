from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/dataset.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"])
# The source is sparse by product and date, so start with aggregate daily demand.
# This evaluates the baseline at store-wide daily level, not per SKU.
daily = (
    df.groupby(df["date"].dt.floor("D"))["daily_sales"]
      .sum()
      .sort_index()
      .asfreq("D", fill_value=0)
)
if len(daily) < 30:
    raise ValueError("Not enough distinct dates for a chronological holdout.")

split = int(len(daily) * 0.8)
train, test = daily.iloc[:split], daily.iloc[split:]

# Naive baseline: predict each test day using the mean of the training period.
pred = np.full(len(test), train.mean())
mae = mean_absolute_error(test, pred)
rmse = mean_squared_error(test, pred) ** 0.5
denom = test.sum()
wape = (np.abs(test.to_numpy() - pred).sum() / denom * 100) if denom else np.nan

results = pd.DataFrame({
    "date": test.index,
    "actual_daily_sales_sum": test.to_numpy(),
    "baseline_prediction": pred,
})
results.to_csv(OUT / "baseline_forecast_test.csv", index=False)
metrics = pd.DataFrame([{
    "target": "aggregate daily recorded sales",
    "model": "training-period mean baseline",
    "train_days": len(train),
    "test_days": len(test),
    "MAE": mae,
    "RMSE": rmse,
    "WAPE_percent": wape,
    "warning": "Synthetic/combined data; aggregate-level baseline only, not per-product validation."
}])
metrics.to_csv(OUT / "baseline_metrics.csv", index=False)
print(metrics.to_string(index=False))
