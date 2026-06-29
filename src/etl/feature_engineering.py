import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger()

def cyclical_features(df):

    df["hour_sin"] = np.sin(
    2 * np.pi * df["hour_of_day"] / 24
    )
    
    df["hour_cos"] = np.cos(
    2 * np.pi * df["hour_of_day"] / 24
    )
    return df

def calender_features(df):

    df['is_weekend'] = df["day_of_week"].isin([5,6]).astype(int)
    df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
    df["is_peak_hour"] = (
    df["hour_of_day"].between(18, 22)
    ).astype(int)
    return df

def weather_interaction_features(df):
    # Heat index proxy
    df["temp_humidity"] = (
        df["temperature_(°c)"]
        * df["humidity_(%)"]
        )
    # Solar-temperature interaction
    df["temp_solar"] = (
        df["temperature_(°c)"]
        * df["solar_irradiance_(w/m²)"]
        )
    return df

def temperature_demand_interaction_features(df):
    # Temperature squared
    df["temp_sq"] = df["temperature_(°c)"] ** 2
    # Cooling degree
    df["cooling_degree"] = (
        df["temperature_(°c)"] - 24
        ).clip(lower=0)
    return df

def lagged_load_features(df):
    # Previous day
    df["load_lag_96"] = df["load_demand_(kw)"].shift(96)
    # Previous week
    df["load_lag_672"] = df["load_demand_(kw)"].shift(96 * 7)
    return df

def rolling_statistics_features(df):
    # 24-hour moving average
    df["load_mean_24h"] = (
        df["load_demand_(kw)"]
        .shift(1)
        .rolling(96)
        .mean()
        )
    # 24-hour moving standard deviation
    df["load_std_24h"] = (
        df["load_demand_(kw)"]
        .shift(1)
        .rolling(96)
        .std()
        )
    return df


def change_features(df):
    # Temperature change
    df["temp_diff_1h"] = (
        df["temperature_(°c)"]
        - df["temperature_(°c)"].shift(4)
        )
    # Load change
    df["load_diff_1h"] = (
        df["load_demand_(kw)"]
        - df["load_demand_(kw)"].shift(4)
        )
    return df


def feature_engineering(df):
    logger.info(f"Columns entering FE: {list(df.columns)}")

    logger.info(
        "Starting feature engineering pipeline"
    )
    df = df.sort_values("timestamp").set_index("timestamp")
    df["season"] = df["season"].astype("category")

    df = cyclical_features(df)

    df = calender_features(df)

    df = weather_interaction_features(df)

    df = temperature_demand_interaction_features(df)

    df = lagged_load_features(df)

    df = rolling_statistics_features(df)

    df = change_features(df)

    logger.info(
        "Feature engineering complete"
    )

    return df