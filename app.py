"""
app.py  –  Optimized Retail Inventory Management System
=========================================================
Streamlit dashboard with four tabs:
  Overview  |  Inventory  |  Reorder  |  Suppliers

All metrics and recommendations are on synthetic/combined data and are
illustrative only – not production purchasing advice.
"""

from pathlib import Path
import math

import pandas as pd
import streamlit as st
import plotly.express as px

# ── Optional: ML model for forecast-based demand ─────────────────────────────
MODEL_PATH = Path("models/best_model.joblib")
_model = None
if MODEL_PATH.exists():
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
    except Exception:
        _model = None

DATA = Path("data/dataset.csv")

st.set_page_config(
    page_title="Retail Inventory Optimizer",
    layout="wide"
)

st.title("Optimized Retail Inventory Management System")
st.caption(
    "Prototype using supplied and synthetic inventory records. "
    "Inventory recommendations are illustrative, not production purchasing advice."
)

if not DATA.exists():
    with st.spinner("Dataset not found. Generating data/dataset.csv..."):
        try:
            import subprocess
            import sys
            subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        except Exception as e:
            st.error(f"data/dataset.csv not found and auto-generation failed: {e}")
            st.stop()

# ── Load and validate data ────────────────────────────────────────────────────
df = pd.read_csv(DATA, parse_dates=["date"])

required_columns = [
    "date", "product_id", "category", "season", "stock_qty",
    "reorder_level", "daily_sales", "unit_cost",
    "supplier_id", "lead_days"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

df = df.dropna(subset=["date", "product_id"]).copy()
df = df.sort_values(["date", "product_id"])

# Historical record-level low-stock flag:
# A record is flagged as low-stock when the recorded stock_qty at that
# point in time was at or below the product's fixed reorder_level threshold.
# This differs from the "low on stock" flag computed below for the *latest*
# snapshot, which also considers how many more units the target-stock
# formula recommends ordering.
df["low_stock"] = df["stock_qty"] <= df["reorder_level"]

# Use the latest available record for each product as its current snapshot.
# Assumption: later records represent more recent stock observations.
latest = (
    df.sort_values("date")
      .drop_duplicates("product_id", keep="last")
      .copy()
)

# Estimate demand from all available observations for each product (mean proxy).
demand = (
    df.groupby("product_id", as_index=False)
      .agg(estimated_daily_demand=("daily_sales", "mean"))
)

# Drop inventory_value column if it exists before merge to avoid duplication;
# it will be recomputed below from the final 'latest' frame.
latest = latest.drop(columns=["inventory_value"], errors="ignore")
latest = latest.merge(demand, on="product_id", how="left")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    safety_pct = st.slider(
        "Safety stock allowance (%)",
        min_value=0,
        max_value=50,
        value=20,
        step=5,
        help="Illustrative buffer added to estimated demand during supplier lead time."
    )

    # Ethics: anonymise pricing toggle
    anonymise = st.toggle(
        "Anonymise pricing (hide unit costs & spend)",
        value=True,
        help=(
            "When ON, unit_cost values are replaced by a relative spend index "
            "(0.00–1.00) and CSV downloads of raw prices are disabled."
        )
    )

    # Demand source selector (only show ML option if model is available)
    demand_source_options = ["Mean-sales proxy (default)"]
    if _model is not None:
        demand_source_options.append("Forecast-based demand (ML model)")
    demand_source = st.radio(
        "Demand estimate for reorder",
        demand_source_options,
        index=0,
        help="Forecast-based demand uses the trained ML model (models/best_model.joblib)."
    )

# ── Forecast-based demand (optional) ─────────────────────────────────────────
use_ml_demand = (demand_source == "Forecast-based demand (ML model)") and (_model is not None)

if use_ml_demand:
    # Build feature frame for latest snapshot rows
    _feat = latest.copy()
    _feat["month"] = _feat["date"].dt.month
    _feat["day_of_week"] = _feat["date"].dt.day_of_week
    _feat["is_weekend"] = (_feat["day_of_week"] >= 5).astype(int)
    _feat["is_summer"] = _feat["month"].isin([4, 5, 6]).astype(int)
    _feat["is_diwali_window"] = (
        ((_feat["date"].dt.month == 10) & (_feat["date"].dt.day >= 20))
        | ((_feat["date"].dt.month == 11) & (_feat["date"].dt.day <= 5))
    ).astype(int)

    # Build product-level statistics from the full dataset (inference proxy;
    # the model was trained with training-split-only stats, but at serve-time
    # we use all available history as the best approximation).
    cat_mean_fb = df.groupby("category")["daily_sales"].mean()
    prod_stats = (
        df.groupby("product_id")["daily_sales"]
        .agg(n="count", mu_prod="mean")
        .reset_index()
    )
    prod_cat_map = df[["product_id", "category"]].drop_duplicates("product_id")
    prod_stats = prod_stats.merge(prod_cat_map, on="product_id", how="left")
    prod_stats["mu_cat"] = prod_stats["category"].map(cat_mean_fb)
    SHRINK_K = 5
    prod_stats["prod_shrunk_mean"] = (
        (prod_stats["n"] * prod_stats["mu_prod"] + SHRINK_K * prod_stats["mu_cat"])
        / (prod_stats["n"] + SHRINK_K)
    )
    prod_stats["prod_mean_sales"] = prod_stats["mu_prod"]
    prod_stats["prod_record_count"] = prod_stats["n"]

    _feat = _feat.merge(
        prod_stats[["product_id", "prod_shrunk_mean", "prod_mean_sales", "prod_record_count"]],
        on="product_id", how="left"
    )
    _feat["prod_shrunk_mean"]   = _feat["prod_shrunk_mean"].fillna(_feat["category"].map(cat_mean_fb))
    _feat["prod_mean_sales"]    = _feat["prod_mean_sales"].fillna(_feat["category"].map(cat_mean_fb))
    _feat["prod_record_count"]  = _feat["prod_record_count"].fillna(0)

    # Detect which feature columns the saved model's preprocessor expects
    # by inspecting the ColumnTransformer's passthrough list.
    try:
        _num_cols = list(_model.named_steps["prep"].transformers[0][2])
        _cat_cols = ["season", "category"]
        _feature_cols = _num_cols + _cat_cols
    except Exception:
        # Fallback to the tuned-model feature set
        _feature_cols = [
            "month", "day_of_week", "is_weekend", "is_summer", "is_diwali_window",
            "lead_days", "unit_cost", "reorder_level",
            "prod_shrunk_mean", "season", "category"
        ]

    try:
        ml_preds = _model.predict(_feat[_feature_cols])
        latest["estimated_daily_demand"] = ml_preds.clip(min=0)
        st.sidebar.success("Using ML forecast-based demand.")
    except Exception as e:
        st.sidebar.warning(f"ML demand failed, using mean proxy. ({e})")
        use_ml_demand = False

# ── Reorder calculations (single definition used everywhere) ─────────────────
latest["lead_time_demand"] = (
    latest["estimated_daily_demand"] * latest["lead_days"]
)
latest["safety_stock"] = (
    latest["lead_time_demand"] * safety_pct / 100
)
latest["target_stock"] = (
    latest["lead_time_demand"] + latest["safety_stock"]
).apply(math.ceil)
latest["suggested_order_qty"] = (
    latest["target_stock"] - latest["stock_qty"]
).clip(lower=0).astype(int)
latest["inventory_value"] = latest["stock_qty"] * latest["unit_cost"]
latest["reorder_spend"] = latest["suggested_order_qty"] * latest["unit_cost"]

# Low-stock flag on the latest snapshot.
# NOTE – two "low-stock" definitions are used in this app:
#   (1) Historical flag (df["low_stock"]): any record where stock_qty <= reorder_level
#       at the time of that observation.
#   (2) Current snapshot flag (latest["low_stock"]): the same condition applied to
#       each product's *latest* record only. This is what drives KPI cards and
#       reorder recommendations.
# A product can appear "not low" by definition (1) in most records yet still appear
# "low" by definition (2) if its most recent snapshot crossed the threshold.
# The target-stock recommendation also flags products that have stock above
# reorder_level but below the lead-time demand + safety buffer.
latest["low_stock"] = latest["stock_qty"] <= latest["reorder_level"]

# ── Pricing helper ────────────────────────────────────────────────────────────
def _price_col(series: pd.Series, col_name: str) -> pd.Series:
    """Return the series masked or as a relative index when anonymise is ON."""
    if anonymise:
        max_val = series.max()
        if max_val and max_val > 0:
            return (series / max_val).round(4)
        return series * 0
    return series

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_inventory, tab_reorder, tab_suppliers = st.tabs(
    ["Overview", "Inventory", "Reorder", "Suppliers"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Key Performance Indicators")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory records", f"{len(df):,}")
    c2.metric("Unique products", f"{latest['product_id'].nunique():,}")
    c3.metric("Products currently low on stock", f"{int(latest['low_stock'].sum()):,}")

    inv_val = latest["inventory_value"].sum()
    c4.metric(
        "Latest recorded inventory value",
        f"{inv_val:,.2f}" if not anonymise else "Hidden (anonymised)"
    )

    st.caption(
        "Current inventory figures use the latest available record for each product. "
        "They are only as current as the dataset's latest dates. "
        "All values are on synthetic/combined data."
    )

    # Sales analysis
    st.subheader("Sales analysis")

    left, right = st.columns(2)

    with left:
        category_sales = (
            df.groupby("category", as_index=False)
              .agg(recorded_sales=("daily_sales", "sum"))
        )
        st.plotly_chart(
            px.bar(
                category_sales,
                x="category",
                y="recorded_sales",
                title="Total Recorded Sales by Category"
            ),
            width="stretch"
        )

    with right:
        # Average sales per recorded product, not raw sum (corrects for varying coverage)
        daily = (
            df.groupby(df["date"].dt.date, as_index=False)
              .agg(
                  average_sales=("daily_sales", "mean"),
                  product_records=("product_id", "nunique")
              )
        )
        daily.columns = ["date", "average_sales", "product_records"]
        st.plotly_chart(
            px.line(
                daily,
                x="date",
                y="average_sales",
                title="Average Recorded Sales per Product by Date"
            ),
            width="stretch"
        )

    st.caption(
        "Sales charts summarise the available synthetic records. "
        "They should not be interpreted as complete store-wide demand."
    )

    with st.expander("Inspect daily data coverage"):
        st.dataframe(
            daily.sort_values("date"),
            width="stretch",
            hide_index=True
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: INVENTORY
# ════════════════════════════════════════════════════════════════════════════
with tab_inventory:
    st.subheader("Explore inventory")

    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    category = st.selectbox("Category", categories, key="inv_category")
    show_low = st.checkbox("Show only low-stock records", key="inv_low")

    view = df.copy()
    if category != "All":
        view = view[view["category"] == category]
    if show_low:
        view = view[view["low_stock"]]

    st.dataframe(
        view[
            [
                "date", "product_id", "category", "stock_qty",
                "reorder_level", "daily_sales", "supplier_id",
                "lead_days", "low_stock"
            ]
        ].sort_values("date", ascending=False),
        width="stretch",
        hide_index=True
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: REORDER
# ════════════════════════════════════════════════════════════════════════════
with tab_reorder:
    st.subheader("Reorder recommendations")

    if use_ml_demand:
        st.info(
            "Demand estimates are from the trained ML model (models/best_model.joblib). "
            "Synthetic/combined data only – not validated production forecasts."
        )
    else:
        st.warning(
            "Recommendations use each product's average recorded sales as a demand proxy. "
            "The safety allowance is illustrative. This prototype does not account for "
            "outstanding purchase orders, stockouts affecting observed sales, supplier "
            "constraints, or a validated per-product forecast."
        )

    st.caption(
        "**Two 'low stock' definitions used here:**\n"
        "- **Reorder level threshold** (`stock_qty <= reorder_level`): the product's "
        "fixed replenishment trigger point (column `low_stock` in the Inventory tab).\n"
        "- **Target-stock recommendation** (`suggested_order_qty > 0`): considers lead "
        "time, estimated daily demand, and the safety-stock allowance you set in the "
        "sidebar. A product may show a suggested order even when above its reorder level "
        "if the lead-time demand exceeds current stock."
    )

    recommendations = (
        latest[latest["suggested_order_qty"] > 0]
        .sort_values("suggested_order_qty", ascending=False)
    )

    display_cols = [
        "product_id", "category", "stock_qty",
        "reorder_level", "estimated_daily_demand",
        "lead_days", "lead_time_demand", "safety_stock",
        "target_stock", "suggested_order_qty", "supplier_id"
    ]

    reorder_display = recommendations[display_cols].head(100).copy()

    st.dataframe(reorder_display, width="stretch", hide_index=True)

    st.caption(
        "Order quantity = max(0, target stock − latest recorded stock). "
        "Target stock = estimated daily demand × lead time + safety stock."
    )

    # Download reorder list (respects anonymise toggle)
    dl_df = recommendations[display_cols].copy()
    if not anonymise:
        dl_df["unit_cost"] = recommendations["unit_cost"]
        dl_df["estimated_reorder_spend"] = recommendations["reorder_spend"]
        btn_label = "Download full reorder list (with prices)"
    else:
        btn_label = "Download full reorder list (prices hidden)"

    st.download_button(
        label=btn_label,
        data=dl_df.to_csv(index=False).encode("utf-8"),
        file_name="reorder_recommendations.csv",
        mime="text/csv",
        disabled=False
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: SUPPLIERS
# ════════════════════════════════════════════════════════════════════════════
with tab_suppliers:
    st.subheader("Supplier management")

    # ── Supplier summary table ────────────────────────────────────────────
    sup_agg = (
        latest.groupby("supplier_id")
        .agg(
            num_products=("product_id", "count"),
            avg_lead_days=("lead_days", "mean"),
            avg_daily_demand=("estimated_daily_demand", "mean"),
            low_stock_products=("low_stock", "sum"),
            total_suggested_order_qty=("suggested_order_qty", "sum"),
            total_reorder_spend=("reorder_spend", "sum"),
            latest_inventory_value=("inventory_value", "sum"),
        )
        .reset_index()
    )

    sup_display = sup_agg.copy()
    sup_display["avg_lead_days"] = sup_display["avg_lead_days"].round(1)
    sup_display["avg_daily_demand"] = sup_display["avg_daily_demand"].round(2)

    if anonymise:
        max_spend = sup_display["total_reorder_spend"].max()
        max_inv = sup_display["latest_inventory_value"].max()
        sup_display["total_reorder_spend"] = (
            (sup_display["total_reorder_spend"] / max_spend).round(4)
            if max_spend else 0
        )
        sup_display["latest_inventory_value"] = (
            (sup_display["latest_inventory_value"] / max_inv).round(4)
            if max_inv else 0
        )
        spend_label = "reorder_spend_index (0-1)"
        inv_label = "inventory_value_index (0-1)"
    else:
        spend_label = "total_reorder_spend (INR)"
        inv_label = "latest_inventory_value (INR)"

    sup_display = sup_display.rename(columns={
        "total_reorder_spend": spend_label,
        "latest_inventory_value": inv_label,
    })

    st.dataframe(sup_display.sort_values("low_stock_products", ascending=False),
                 width="stretch", hide_index=True)

    # ── Bar chart: reorder spend by supplier ─────────────────────────────
    st.subheader("Reorder spend by supplier")

    chart_agg = sup_agg.copy()
    if anonymise:
        max_spend = chart_agg["total_reorder_spend"].max()
        chart_agg["spend_metric"] = (
            chart_agg["total_reorder_spend"] / max_spend
        ).round(4) if max_spend else 0
        y_label = "Relative spend index (0-1)"
    else:
        chart_agg["spend_metric"] = chart_agg["total_reorder_spend"]
        y_label = "Estimated reorder spend (INR)"

    st.plotly_chart(
        px.bar(
            chart_agg.sort_values("spend_metric", ascending=False),
            x="supplier_id",
            y="spend_metric",
            labels={"spend_metric": y_label, "supplier_id": "Supplier"},
            title="Estimated Reorder Spend by Supplier"
        ),
        width="stretch"
    )

    # ── Scatter: avg lead time vs low-stock products ──────────────────────
    st.subheader("Average lead time vs. low-stock product count")
    st.plotly_chart(
        px.scatter(
            sup_agg,
            x="avg_lead_days",
            y="low_stock_products",
            text="supplier_id",
            size="num_products",
            color="num_products",
            labels={
                "avg_lead_days": "Avg lead time (days)",
                "low_stock_products": "Products currently low on stock",
                "num_products": "# Products"
            },
            title="Supplier Lead Time vs Low-Stock Product Count"
        ),
        width="stretch"
    )

    # ── Supplier selector ─────────────────────────────────────────────────
    st.subheader("Drill down by supplier")
    selected_supplier = st.selectbox(
        "Select supplier",
        sorted(latest["supplier_id"].unique()),
        key="sup_selector"
    )

    sup_products = latest[latest["supplier_id"] == selected_supplier].copy()

    show_cols = [
        "product_id", "category", "stock_qty", "reorder_level",
        "low_stock", "estimated_daily_demand", "lead_days",
        "suggested_order_qty"
    ]

    if not anonymise:
        show_cols += ["unit_cost", "reorder_spend"]

    st.write(f"**{selected_supplier}** — {len(sup_products)} products")
    st.dataframe(
        sup_products[show_cols].sort_values("suggested_order_qty", ascending=False),
        width="stretch",
        hide_index=True
    )

    # Download supplier-specific reorder list
    dl_sup = sup_products[show_cols].copy()
    if anonymise:
        sup_dl_label = f"Download {selected_supplier} reorder list (prices hidden)"
    else:
        sup_dl_label = f"Download {selected_supplier} reorder list (with prices)"

    st.download_button(
        label=sup_dl_label,
        data=dl_sup.to_csv(index=False).encode("utf-8"),
        file_name=f"reorder_{selected_supplier}.csv",
        mime="text/csv"
    )

    st.caption(
        "Supplier spend estimates use suggested_order_qty * unit_cost. "
        "Figures are illustrative on synthetic/combined data."
    )
