import argparse
import pandas as pd
import joblib
from pathlib import Path
from src.etl.clean_data import clean_data
from src.etl.feature_engineering import feature_engineering
from src.utils.logger import setup_logger
logger = setup_logger()
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


# --------------------------------
# Default paths and configuration
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GBM_MODEL = PROJECT_ROOT / "artifacts" / "models" / "lightgbm" / "lightgbm_best_model.pkl"
DEFAULT_TFT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "tft" / "best_tft.ckpt"
DEFAULT_GBM_OUTPUT = PROJECT_ROOT / "gbm_predictions.csv"
DEFAULT_TFT_OUTPUT = PROJECT_ROOT / "tft_predictions.csv"
DEFAULT_TFT_TRAINING_DATASET = PROJECT_ROOT / "artifacts" / "models" / "tft" / "training_dataset.pkl"

# -------------------------
# Core inference functions
# -------------------------

def gbm_predict(
        df: pd.DataFrame,
        model_path: Path | str = DEFAULT_GBM_MODEL
        ):
    logger.info(
        f"Columns received by gbm_predict(): {df.columns.tolist()}"
    )
    # Step 1: Preprocess raw data
    df = clean_data(df)
    # Step 2: Create time index if not present
    if not "time_idx" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // (15 * 60)
                          ).astype(int)  
    # Step 3: Feature engineering
    df = feature_engineering(df)
    # Step 4: Separate actuals if present
    y_true = None
    if "load_demand_(kw)" in df.columns:
        y_true = df["load_demand_(kw)"].tolist()
        df = df.drop(columns=["load_demand_(kw)"])
        logger.info("Extracted and removed target column from features")
        
    # Step 5: Load model and predict
    model = joblib.load(model_path)
    preds = model.predict(df)

    # Step 6: Build output
    output_df = df.copy()
    output_df["predicted_load_demand(kW)"] = preds
    if y_true is not None:
        output_df["actual_load_demand(kW)"] = y_true
    
    return output_df


def tft_predict(
        training_dataset,
        pred_df: pd.DataFrame,
        model_path: Path | str = DEFAULT_TFT_MODEL
        ):
    logger.info(
        f"Columns received by tft_predict(): {pred_df.columns.tolist()}"
    )
    # Step 1: Preprocess raw data
    pred_df = clean_data(pred_df)
    # Step 2: Create time index if not present
    if not "time_idx" in pred_df.columns:
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
        pred_df["time_idx"] = ((pred_df["timestamp"] - pred_df["timestamp"].min()).dt.total_seconds() // (15 * 60)
                          ).astype(int)  
    # Step 3: Feature engineering
    pred_df = feature_engineering(pred_df)
    # Step 4. Fix order here
    pred_df = pred_df.sort_values("time_idx").reset_index(drop=True)
    # Step 5: Add group
    pred_df["series"] = 0
    logger.info(f"Shape: {pred_df.shape}")

    logger.info("NaNs per column:")
    logger.info(pred_df.isna().sum())

    logger.info("time_idx continuous check:")
    logger.info(pred_df["time_idx"].diff().value_counts().head())

    logger.info("Required columns check:")
    logger.info(pred_df.columns.tolist())
    # Step 6: Construct prediction dataset
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        pred_df,
        predict=True,
        stop_randomization=True,
        )
    logger.info(f"Prediction dataset length: {len(prediction_dataset)}")
    # Step 7: Create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    pred_dataloader = prediction_dataset.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=0,
        shuffle=False
        )
    # Step 8: Load tft model
    tft = TemporalFusionTransformer.load_from_checkpoint(
        model_path
        )
    preds = tft.predict(
        pred_dataloader,
        return_x=True,
        return_y=True,
        mode="prediction"
        )
    
    #y_pred = preds.output
    #y_pred = preds.output[0].cpu().numpy()   # shape (96,)
    y_pred = preds.output[0].detach().cpu().numpy().reshape(-1)
    #y_true = preds.y[0]
    y_true = preds.y[0][0].detach().cpu().numpy().reshape(-1)
    decoder_time_idx = preds.x["decoder_time_idx"][0].detach().cpu().numpy().reshape(-1)
    # Step 9: Build output
    output_df = pd.DataFrame({
        "time_idx": decoder_time_idx,
        "predicted_load_demand(kW)": y_pred,
        "actual_load_demand(kW)": y_true,
        })
    return output_df

if __name__ == "__main":
    parser = argparse.ArgumentParser(description="Run inference on new load data (raw)")
    parser.add_argument("--input", type=str, required=True, help="Path to input holdout CSV file")
    parser.add_argument("--tft_output", type=str, default=str(DEFAULT_TFT_OUTPUT), help="Path to save tft predictions")
    parser.add_argument("--gbm_output", type=str, default=str(DEFAULT_GBM_OUTPUT), help="Path to save gbm predictions")
    parser.add_argument("--tft_model", type=str, default=str(DEFAULT_TFT_MODEL), help="Path to trained tft model")
    parser.add_argument("--gbm_model", type=str, default=str(DEFAULT_GBM_MODEL), help="Path to trained gbm model")
    parser.add_argument("--tft_training_dataset", type=str, default=str(DEFAULT_TFT_TRAINING_DATASET), help="Path to tft training dataset")
    
    args = parser.parse_args()

    holdout_df = pd.read_csv(args.input)
    # gbm prediction
    gbm_preds_df = gbm_predict(
        holdout_df,
        model_path=args.gbm_model
        )
    gbm_preds_df.to_csv(args.gbm_output)
    logger.info(
        f"GBM predictions saved to {args.gbm_output}"
    )
    
    # tft prediction
    tft_preds_df = tft_predict(
        holdout_df,
        model_path=args.tft_model
        )
    tft_preds_df.to_csv(args.tft_output)
    logger.info(
        f"TFT predictions saved to {args.tft_output}"
    )