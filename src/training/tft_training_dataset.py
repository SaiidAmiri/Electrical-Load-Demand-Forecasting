from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()
from pytorch_forecasting import TimeSeriesDataSet
from src.etl.feature_engineering import feature_engineering
import pandas as pd
def tft_training_dataset(dev_path: Path | str):
    
    # load development data
    development_df = pd.read_csv(dev_path)
    # create a dummy group
    development_df["series"] = 0
    # Feature engineering
    development_df = feature_engineering(development_df)
    # Dropn NaN 
    logger.info(
        f"Rows before dropna: {len(development_df)}"
    )

    development_df = development_df.dropna()

    logger.info(
        f"Rows after dropna: {len(development_df)}"
    )
    #
    #validation_rows = validation_days * max_prediction_length
    #validation_cutoff = (development_df["time_idx"].max() - validation_rows)
    # split data
    #train_df = development_df[development_df.time_idx <= validation_cutoff]

    time_varying_known_reals = [
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "is_business_hours",
        "is_peak_hour",
        "temperature_(°c)",
        "humidity_(%)",
        "wind_speed_(m/s)",
        "solar_irradiance_(w/m²)",
        "rainfall_(mm)",
        "temp_sq",
        "cooling_degree",
        "temp_humidity",
        "temp_solar",
        ]
    training_dataset = TimeSeriesDataSet(
        development_df,
        #train_df,
        time_idx="time_idx",
        target="load_demand_(kw)",
        group_ids=["series"],

        max_encoder_length=692,  #692, #: 7 days history # 288: Previous 3 days # 692: 7 days history
        min_prediction_length=96,
        max_prediction_length=96,   # forecast 24 hours

        time_varying_known_reals= time_varying_known_reals,

        time_varying_known_categoricals=[
            "season",
            ],

        time_varying_unknown_reals=[
            "load_demand_(kw)"
            ],
        allow_missing_timesteps=True
        )
    training_dataset.save("artifacts/models/tft/training_dataset.pkl")
    return training_dataset

if __name__ == "__main__":

    dev_path = 'data/cleaned/development.csv'
    #validation_days = 180  
    #max_prediction_length = 96  # 24 hours
    tft_training_dataset(dev_path)