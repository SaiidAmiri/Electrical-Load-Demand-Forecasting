import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3, os
from pathlib import Path
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np


# --------------
# Config
# --------------

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "electrical-load-forecasting-data")
REGION = os.getenv("AWS_REGION", "eu-central-1")

s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    """
    Download files from S3 if not already cached locally
    """
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"Downloading {key} from S3")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# Path (ensure available locally by fetching from S3 if missing)

HOLDOUT_RAW_PATH = load_from_s3(
    "data/cleaned/holdout.csv",
    "data/cleaned/holdout.csv"
)

# --------------
# Data Loading
# --------------

@st.cache_data
def load_data():

    raw_df = pd.read_csv(HOLDOUT_RAW_PATH)

    disp_df = pd.DataFrame(index=raw_df.index)
    disp_df["time"] = raw_df["timestamp"]
    disp_df["time"] = pd.to_datetime(disp_df["time"])
    disp_df["year"] = disp_df["time"].dt.year
    disp_df["month"] = disp_df["time"].dt.month
    disp_df["day"] = disp_df["time"].dt.day
    if "time_idx" not in disp_df.columns:
        disp_df["time_idx"] = (
            (disp_df["time"] - disp_df["time"].min()).dt.total_seconds() // (15 * 60)
            ).astype(int)
    disp_df["actual_load(kW)"] = raw_df["load_demand_(kw)"]

    return raw_df, disp_df

raw_df, disp_df = load_data()

# --------------
# UI
# --------------

st.title("Electrical Load Demand Forecasting")

days = sorted(disp_df["day"].unique())
months = sorted(disp_df["month"].unique())
years = sorted(disp_df["year"].unique())

col1, col2, col3 = st.columns(3)

with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    day = st.selectbox("Select Day", days, index=0)

model = st.selectbox(
    "Select Model",
    ["LightGBM", "Temporal Fusion Transformer"]
)

if st.button("Show Predictions"):
    model_name = "lightgbm" if model == "LightGBM" else "tft"
    mask = (
            (disp_df["year"] == year) &
            (disp_df["month"] == month) &
            (disp_df["day"] == day)
        )
    idx = disp_df.index[mask]
    if len(idx) == 0:
        st.warning("No data found for these filters")
        st.stop()
    
    request_body = {
        "model": model_name,
    }
    if model_name == "tft":
        # TFT ONLY needs a reference timestamp (NOT idx)
        target_timestamp = disp_df.loc[idx, "time"].max()
        request_body["date"] = target_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    elif model_name == "lightgbm":
        
        payload = raw_df.loc[idx].to_dict("records")
        request_body["features"] = payload
    
    try:
        resp = requests.post(API_URL, json = request_body, timeout=60)
        print("Status:", resp.status_code)
        print("Response:", resp.text)
        resp.raise_for_status()
        output = resp.json()
        preds = output.get("predictions", [])
        actuals = output.get("actuals", None)

        view = disp_df.loc[idx, ["time", "actual_load(kW)"]].copy()
        view = view.sort_values("time")
        view["prediction"] = pd.Series(preds, index=view.index).astype(float)
        #view = view.merge(pred_df, on="time_idx", how="left")
        if actuals is not None and len(actuals) == len(view):
            view["actual_load(kW)"] = pd.Series(actuals, index=view.index).astype(float)
            
        # Metrics
        valid = view.dropna(subset=["prediction", "actual_load(kW)"])
        mae = (valid["prediction"] - valid["actual_load(kW)"]).abs().mean()
        rmse = ((valid["prediction"] - valid["actual_load(kW)"]) ** 2).mean() ** 0.5

        st.subheader("Predictions vs Actuals")
        st.dataframe(
            view[["time", "actual_load(kW)", "prediction"]].reset_index(drop=True),
            use_container_width=True
            )
        c1, c2 =st.columns(2)
        with c1:
            st.metric("MAE", f"{mae:,.2f}")
        with c2:
            st.metric("RMSE", f"{rmse:,.2f}")

        # -------------
        # Trend Charts
        # -------------

        
        fig = px.line(
            view,
            x="time",  # or "day" if aggregated
            y=["actual_load(kW)", "prediction"],
            markers=True,
            labels={"value": "load demand", "time": "time"},
            title="Daily Trend",
            color_discrete_map={
                "actual_load(kW)": "#1f77b4",   # blue (truth / observed)
                "prediction": "#ff7f0e"        # orange (forecast)
                }
            )
        
        #fig.show()
            
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.exception(e)
    
else:
    st.info("Choose filters and click **Show Predictions** to compute.")