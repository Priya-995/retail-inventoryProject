from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/dataset.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"])
required = [
    "date", "product_id", "category", "season", "stock_qty",
    "reorder_level", "daily_sales", "unit_cost", "supplier_id", "lead_days"
]
assert list(df.columns) == required, "Column names/order do not match project schema."

# Basic audit
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "unique_count": df.nunique()
})
audit.to_csv(OUT / "eda_data_audit.csv")
print("Shape:", df.shape)
print(audit)
print("Duplicate date-product pairs:", df.duplicated(["date", "product_id"]).sum())
print("Negative stock:", (df["stock_qty"] < 0).sum())
print("Negative sales:", (df["daily_sales"] < 0).sum())
print("Nonpositive lead days:", (df["lead_days"] <= 0).sum())

# Chart 1: recorded sales by category
df.groupby("category")["daily_sales"].sum().sort_values(ascending=False).plot(kind="bar")
plt.title("Recorded Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sum of recorded daily_sales")
plt.tight_layout()
plt.savefig(OUT / "01_sales_by_category.png", dpi=160)
plt.close()

# Chart 2: inventory stock vs reorder level for the 25 most frequently recorded SKUs
top_products = df["product_id"].value_counts().head(25).index
sample = df[df["product_id"].isin(top_products)]
summary = sample.groupby("product_id")[["stock_qty", "reorder_level"]].mean().sort_values("stock_qty")
summary.plot(kind="bar", figsize=(12, 5))
plt.title("Mean Stock vs Reorder Level (25 Most Frequently Recorded Products)")
plt.xlabel("Product ID")
plt.ylabel("Mean units")
plt.tight_layout()
plt.savefig(OUT / "02_stock_vs_reorder.png", dpi=160)
plt.close()

# Chart 3: aggregate recorded sales by date
daily = df.groupby(df["date"].dt.date)["daily_sales"].sum()
daily.plot(figsize=(12, 4))
plt.title("Aggregate Recorded Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sum of recorded daily_sales")
plt.tight_layout()
plt.savefig(OUT / "03_daily_sales_trend.png", dpi=160)
plt.close()

print("EDA outputs saved in outputs/.")
