import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.etl.clean_data import clean_data
from src.utils.logger import setup_logger

logger = setup_logger()

DATA_DIR = Path('data/cleaned')
RAW_DATA_PATH = Path('data/raw/forecasting_dataset.csv')
HOLDOUT_DAYS = 30
SAMPLES_PER_DAY = 96 # 24 hours


def split_data(df, holdout_days, samples_per_day):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // (15 * 60)
                  ).astype(int)
    holdout_days = 30
    holdout_cutoff = df["time_idx"].max() - holdout_days * samples_per_day
    development_df = df[df.time_idx <= holdout_cutoff]
    holdout_df = df[df.time_idx > holdout_cutoff]
    return development_df, holdout_df

def save_splitted_data(
        raw_path: Path | str,
        output_dir: Path | str,
        holdout_days: int,
        samples_per_day: int
        ):
    """ Load raw dataset and split it into development and holdout,
    and save to output_dir"""

    logger.info(
        "Raw data cleaning and split started"
    )
    
    # load raw dataset
    df = pd.read_csv(raw_path)
    # clean dataset
    df = clean_data(df)
    # splits
    development_df, holdout_df = split_data(df, holdout_days, samples_per_day)
    # save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    development_df.to_csv(output_dir / "development.csv", index=False)
    holdout_df.to_csv(output_dir / "holdout.csv", index=False)
    logger.info(
        "Raw data cleaning and split completed"
    )
    return development_df, holdout_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, clean, and split raw data")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH), help="Path to load raw data")
    parser.add_argument("--output", type=str, default=str(DATA_DIR), help="Path to save splitted cleaned data")
    parser.add_argument("--holdout_days", type=int, default=int(HOLDOUT_DAYS), help="Number of days dedicated for inference")
    parser.add_argument("--samples_per_day", type=int, default=int(SAMPLES_PER_DAY), help="Samples per day")

    args = parser.parse_args()
    save_splitted_data(args.input, args.output, args.holdout_days, args.samples_per_day)



    
