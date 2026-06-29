import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()

# Configuration

bucket = "electrical-load-forecasting-data"
region = "eu-central-1"

# Set preoject root as parent
local_cleaned_data_dir = Path("data/cleaned")
local_raw_data_dir = Path("data/raw")
local_models_dir = Path("artifacts/models")

s3 = boto3.client("s3", region_name = region)

# Helper functions

def s3_file_exists(bucket: str, s3_key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def upload_file(local_path: Path, s3_key: str):
    if not local_path.exists():
        logger.info(f"File not found: {local_path}")
        return
    if s3_file_exists(bucket, s3_key):
        logger.info(f"s3://{bucket}/{s3_key} already exists. Skipping upload.")
        return
    logger.info(
        f"Uploading {local_path} to s3://{bucket}/{s3_key}"
    )
    s3.upload_file(str(local_path), bucket, s3_key)


if __name__ == "__main__":
    # Upload required cleaned datasets
    upload_file(local_cleaned_data_dir/"development.csv", "data/cleaned/development.csv")
    upload_file(local_cleaned_data_dir/"holdout.csv", "data/cleaned/holdout.csv")
    # Upload required raw dataset
    upload_file(local_raw_data_dir/"forecasting_dataset.csv", "data/raw/forecasting_dataset.csv")
    # Upload lightgbm model
    upload_file(local_models_dir/"lightgbm/lightgbm_best_model.pkl", "models/lightgbm/lightgbm_best_model.pkl")
    # Upload tft model
    upload_file(local_models_dir/"tft/best_tft.ckpt", "models/tft/best_tft.ckpt")
    # Upload tft training dataset
    upload_file(local_models_dir/"tft/training_dataset.pkl", "models/tft/training_dataset.pkl")



