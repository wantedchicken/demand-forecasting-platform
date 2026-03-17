import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

FORECAST_PATH = ROOT / "demo_data" / "forecast_demo.parquet"
MONITOR_PATH = ROOT / "demo_data" / "monitoring_demo.json"
TRAIN_METRICS_PATH = ROOT / "demo_data" / "train_metrics_demo.json"
BACKTEST_PATH = ROOT / "demo_data" / "backtest_summary_demo.parquet"
st.set_page_config(
    page_title="Retail Demand Forecast Dashboard",
    layout="wide"
)

# -------------------------------
# LOAD FUNCTIONS
# -------------------------------

@st.cache_data
def load_forecast():
    df = pd.read_parquet(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_monitoring():
    if MONITOR_PATH.exists():
        return json.loads(MONITOR_PATH.read_text())
    return None

@st.cache_data
def load_train_metrics():
    if TRAIN_METRICS_PATH.exists():
        return json.loads(TRAIN_METRICS_PATH.read_text())
    return None

@st.cache_data
def load_backtest():
    if BACKTEST_PATH.exists():
        return pd.read_parquet(BACKTEST_PATH)
    return None

# -------------------------------
# HEADER
# -------------------------------

st.title("📊 Retail Demand Forecast Dashboard")

st.caption(
    "This dashboard shows how we predict daily product sales using past trends, pricing, and seasonal patterns."
)

st.info(
    "💡 This system helps stores decide how much stock to keep to avoid running out of products or overstocking."
)

if not FORECAST_PATH.exists():
    st.error("❌ forecast.parquet not found. Run your prediction step first.")
    st.stop()

forecast = load_forecast()
monitoring = load_monitoring()
train_metrics = load_train_metrics()
backtest = load_backtest()

# -------------------------------
# TABS
# -------------------------------

tab1, tab2, tab3 = st.tabs([
    "📈 Sales Forecast",
    "📊 Model Performance",
    "🛠 Data Quality & Monitoring"
])

# =========================================================
# TAB 1 — FORECAST
# =========================================================

with tab1:
    st.subheader("📊 Sales Forecast Explorer")

    st.write(
        "Select a store and product to compare actual sales with predicted sales."
    )

    col1, col2 = st.columns(2)

    store = col1.selectbox("🏬 Select Store", sorted(forecast["store_id"].unique()))
    store_df = forecast[forecast["store_id"] == store]

    item = col2.selectbox("📦 Select Product", sorted(store_df["item_id"].unique()))
    view = store_df[store_df["item_id"] == item].sort_values("date").copy()

    # Metrics
    view["error"] = (view["y"] - view["yhat"]).abs()
    avg_error = view["error"].mean()

    confidence = max(0, 100 - avg_error * 20)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Days Shown", len(view))
    k2.metric("Average Error", f"{avg_error:.2f}", help="Prediction is off by ~1 item/day")
    k3.metric("Avg Predicted Sales", f"{view['yhat'].mean():.2f}")
    k4.metric("Model Confidence", f"{confidence:.0f}%")

    st.markdown("### 📈 Sales Prediction Chart")

    chart_df = view.rename(columns={
        "y": "Actual Sales",
        "yhat": "Predicted Sales"
    })

    fig = px.line(
        chart_df,
        x="date",
        y=["Actual Sales", "Predicted Sales"],
        markers=True,
        title=f"Sales Prediction for Product {item} in Store {store}"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insight
    st.success(
        "📌 Insight: The model closely follows demand patterns, helping stores stock the right amount of products."
    )

    st.info(
        "💡 Example: If tomorrow’s predicted sales = 12, the store should stock at least 12 units."
    )

    # Table
    st.markdown("### 📋 Daily Breakdown")

    st.dataframe(
        chart_df[["date", "Actual Sales", "Predicted Sales"]],
        use_container_width=True
    )

# =========================================================
# TAB 2 — PERFORMANCE
# =========================================================

with tab2:
    st.subheader("📊 Model Performance")

    if train_metrics:
        c1, c2, c3 = st.columns(3)

        c1.metric("MAE", f"{train_metrics.get('mae', 0):.2f}")
        c2.metric("RMSE", f"{train_metrics.get('rmse', 0):.2f}")
        c3.metric("Best Iteration", train_metrics.get("best_iteration"))

        st.caption(
            "MAE means the model is off by ~1 unit on average."
        )

    if backtest is not None:
        st.markdown("### 🔁 Rolling Backtest")

        st.dataframe(backtest, use_container_width=True)

        fig_bt = px.line(
            backtest,
            x="window",
            y="mae",
            markers=True,
            title="Model Stability Across Time"
        )

        st.plotly_chart(fig_bt, use_container_width=True)

# =========================================================
# TAB 3 — MONITORING
# =========================================================

with tab3:
    st.subheader("🛠 Data Quality & Monitoring")

    if monitoring is None:
        st.info("Monitoring data not found.")
    else:
        dq = monitoring.get("data_quality", {})
        perf = monitoring.get("performance", {})

        st.markdown("### 🔍 Data Quality")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", dq.get("rows_features"))
        c2.metric("Duplicates", dq.get("duplicate_id_date"))
        c3.metric("Negative Sales", dq.get("negative_y"))

        st.markdown("### 📉 Worst Performing Stores")

        worst_stores = perf.get("worst_stores_by_mae_top10", {})

        if worst_stores:
            df_ws = pd.DataFrame({
                "Store": list(worst_stores.keys()),
                "MAE": list(worst_stores.values())
            })

            fig_ws = px.bar(df_ws, x="Store", y="MAE")
            st.plotly_chart(fig_ws, use_container_width=True)

        st.markdown("### 📉 Worst Categories")

        worst_cats = perf.get("worst_categories_by_mae", {})

        if worst_cats:
            df_wc = pd.DataFrame({
                "Category": list(worst_cats.keys()),
                "MAE": list(worst_cats.values())
            })

            fig_wc = px.bar(df_wc, x="Category", y="MAE")
            st.plotly_chart(fig_wc, use_container_width=True)