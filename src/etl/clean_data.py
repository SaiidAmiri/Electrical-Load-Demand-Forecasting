import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger()

def standardize_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    n_removed = before - len(df)
    logger.info(
        f"Removed {n_removed} duplicates"
    )
    return df

def drop_missing(df):
    return df.dropna()

def fill_missing(df):
    df = df.ffill()
    return df

def remove_invalid_values(df):
    df = df[
    (df['temperature_(°c)'].between(-20, 60)) &
    (df['humidity_(%)'].between(0, 100)) &
    (df['wind_speed_(m/s)'].between(0, 100)) &
    (df['rainfall_(mm)'].between(0, 1000)) &
    (df['solar_irradiance_(w/m²)'].between(0, 1400)) &
    (df['gdp_(lkr)'] > 0) &
    (df['per_capita_energy_use_(kwh)'] > 0) &
    (df['electricity_price_(lkr/kwh)'] > 0) &
    (df['day_of_week'].between(0,6)) &
    (df['hour_of_day'].between(0, 23)) &
    (df['month'].between(1, 12)) &
    (df['load_demand_(kw)'] > 0)
    ]
    return df

def clean_data(df):

    logger.info("Starting cleaning pipeline")

    df = standardize_column_names(df)

    #df = remove_duplicates(df)

    #df = fill_missing(df)

    #df = remove_invalid_values(df)

    logger.info("Cleaning complete")

    return df
