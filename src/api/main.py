
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from pathlib import Path
import os
import torch
import pandas as pd
from src.inference.inference import gbm_predict, tft_predict
from typing import List, Dict, Any, Optional
from src.utils.logger import setup_logger
logger = setup_logger()

# ---------------
# Configuration
# ---------------

S3_BUCKET = os.getenv("S3_BUCKET", "electrical-load-forecasting-data")
REGION = os.getenv("REGION", "eu-central-1")
s3 = boto3.client("s3", region_name=REGION)

# Avoid re-downloading the model/data every time the app starts
def load_from_s3(s3_key, local_path):
    """
    Download files from S3 if not already cached locally
    """
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        logger.info(
            f"Downloading {s3_key} from S3"
        )
        s3.download_file(S3_BUCKET, s3_key, str(local_path))
    return str(local_path)

# --------------
# Paths
# --------------

GBM_MODEL_PATH = Path(load_from_s3(s3_key="models/lightgbm/lightgbm_best_model.pkl", local_path="artifacts/models/lightgbm/lightgbm_best_model.pkl"))
TFT_MODEL_PATH = Path(load_from_s3(s3_key="models/tft/best_tft.ckpt", local_path="artifacts/models/tft/best_tft.ckpt"))
TFT_TRAINING_DATASET_PATH = Path(load_from_s3(s3_key="models/tft/training_dataset.pkl", local_path="artifacts/models/tft/training_dataset.pkl"))
DEV_DATASET_PATH = Path(load_from_s3(s3_key="data/cleaned/development.csv", local_path="data/cleaned/development.csv"))
HOLDOUT_DATASET_PATH = Path(load_from_s3(s3_key="data/cleaned/holdout.csv", local_path="data/cleaned/holdout.csv"))

# -----------------
# App
# -----------------

# Instantiate the FastAPI app
app = FastAPI(title="Electrical Load Forecasting API")

class ModelName(str, Enum):
    lightgbm = "lightgbm"
    tft = "tft"

class PredictionRequest(BaseModel):
    model: ModelName
    date: Optional[str] = None   # for TFT
    features: Optional[List[dict]] = None  # for LightGBM

# Helper functions

def construct_dataset(dev_path, holdout_path):
    development_df = pd.read_csv(dev_path)
    holdout_df = pd.read_csv(holdout_path)
    full_df = pd.concat([development_df, holdout_df], ignore_index=True)
    full_df = full_df.sort_values("timestamp")
    full_df["series"] = 0
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df["time_idx"] = (
        (full_df["timestamp"] - full_df["timestamp"].min())
        .dt.total_seconds() // (15 * 60)
        ).astype(int)
    return full_df
FULL_DF = construct_dataset(DEV_DATASET_PATH, HOLDOUT_DATASET_PATH)
def build_tft_window(target_date: str,
                     df: pd.DataFrame = FULL_DF,
                     encoder_length: int = 692,
                     prediction_length: int = 96):


    target_date = pd.to_datetime(target_date)

    cutoff_idx = df.loc[df["timestamp"] <= target_date, "time_idx"].max()

    start_idx = cutoff_idx - encoder_length
    end_idx = cutoff_idx + prediction_length
    window_df = df[(df["time_idx"] >= start_idx) &
                   (df["time_idx"] <= end_idx)].copy()

    return window_df

# Simple landing endpoint to confirm API is alive
@app.get("/")
def root():
    return {"message": "Electrical Load Forecasting API is running"}

# Health: Check if model exists, return status info 
@app.get("/health")
def health():
    status: Dict[str, Any] = {
        "tft_model_path": str(TFT_MODEL_PATH),
        "gbm_model_path": str(GBM_MODEL_PATH),
        "tft_training_dataset_path": str(TFT_TRAINING_DATASET_PATH),
        "checks": {}
        }
    
    status["checks"]["tft_model_path"] = TFT_MODEL_PATH.exists()
    status["checks"]["gbm_model_path"] = GBM_MODEL_PATH.exists()
    status["checks"]["tft_training_dataset_path"] = TFT_TRAINING_DATASET_PATH.exists()

    if all(status["checks"].values()):
        status["status"] = "healthy"
    else:
        status["status"] = "unhealthy"
    
    return status

# Prediction endpoint:  Core ML Serving Endpoint
@app.post("/predict")
def predict_batch(request: PredictionRequest):

    if request.model == "lightgbm":
        df = pd.DataFrame(request.features)
        preds_df = gbm_predict(df, model_path=GBM_MODEL_PATH)
        response = {"predictions": preds_df["predicted_load_demand(kW)"].astype(float).tolist()}
        if "actual_load_demand(kW)" in preds_df.columns:
            response["actuals"] = preds_df["actual_load_demand(kW)"].astype(float).tolist()
        return response
    elif request.model == "tft":

        if request.date is None:
            return {"error": "date is required for TFT"}

        df = build_tft_window(request.date)
        if df is None:
            return {"error": "not enough data for prediction"}
        logger.info(f"Window length: {len(df)}")
        logger.info(f"time_idx min: {df["time_idx"].min()}")
        logger.info(f"time_idx max: {df["time_idx"].max()}")
        preds_df = tft_predict(
            training_dataset = torch.load(TFT_TRAINING_DATASET_PATH, weights_only=False),
            pred_df = df,
            model_path=TFT_MODEL_PATH
        )

        response = {"predictions": preds_df["predicted_load_demand(kW)"].astype(float).tolist()}
        if "actual_load_demand(kW)" in preds_df.columns:
            response["actuals"] = preds_df["actual_load_demand(kW)"].astype(float).tolist()
       
    return response


   
    

