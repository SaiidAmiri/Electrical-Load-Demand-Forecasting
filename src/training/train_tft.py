import copy
from src.utils.logger import setup_logger

logger = setup_logger()
from pathlib import Path
import warnings
import tensorflow as tf
import tensorboard as tb
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from pytorch_forecasting.tuning import Tuner
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from pytorch_forecasting import TimeSeriesDataSet
from src.etl.feature_engineering import feature_engineering

pl.seed_everything(42)

def train_tft_model(dev_path: Path | str,
                    validation_days: int,
                    max_prediction_length: int):
    
    # load development data
    development_df = pd.read_csv(dev_path)
    # create a dummy group
    development_df["series"] = 0
    # Feature engineering
    development_df = feature_engineering(development_df)
    # Drop lagged features 
    logger.info(
        f"Columns before deleting lqgged featues: {development_df.columns}"
    )

    lag_cols = [
        "load_lag_96",
        "load_lag_672",
        "load_mean_24h",
        "load_std_24h",
        "temp_diff_1h",
        "load_diff_1h",
        ]
    
    development_df = development_df.drop(columns=lag_cols)

    logger.info(
        f"Rows after dropna: {development_df.columns}"
    )
    #
    validation_rows = validation_days * max_prediction_length
    validation_cutoff = (development_df["time_idx"].max() - validation_rows)
    # split data
    train_df = development_df[development_df.time_idx <= validation_cutoff]
    eval_df = development_df[development_df.time_idx > validation_cutoff]

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
        train_df,
        time_idx="time_idx",
        target="load_demand_(kw)",
        group_ids=["series"],

        max_encoder_length=692, #: 7 days history # 288: Previous 3 days # 692: 7 days history
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
    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, 
        development_df,
        min_prediction_idx=validation_cutoff + 1, 
        stop_randomization=True
        )
    
    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
        )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=0,
        shuffle=False
        )
    # Temporary trainer for LR finder
    lr_trainer = pl.Trainer(
        accelerator="cpu",
        gradient_clip_val=0.1,
        )

    # Temporary model
    lr_tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="ranger",
        )
    
    # Find learning rate
    res = Tuner(lr_trainer).lr_find(
        lr_tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-6,
        max_lr=1e-1,
        )
    
    learning_rate = res.suggestion()
    logger.info(f"Selected learning rate: {learning_rate}")

    # Create callbacks for real training
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        mode="min",
        )

    lr_logger = LearningRateMonitor()

    checkpoint_callback = ModelCheckpoint(
        dirpath="artifacts/models/tft",
        filename="best_tft",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        )
    
    tb_logger = TensorBoardLogger(
        "lightning_logs"
        )
    
    # Create the real trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_logger,
            ],
        logger=tb_logger,
        )

    # Create the real training model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        optimizer="ranger",
        reduce_on_plateau_patience=4,
        )
    logger.info(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )
    
    val_metrics = trainer.validate(tft, val_dataloader)
    
    logger.info(f"Validation metrics: {val_metrics}")

    return val_metrics

if __name__ == "__main__":

    dev_path = 'data/cleaned/development.csv'
    validation_days = 180  
    max_prediction_length = 96  # 24 hours
    # Train and save the TFT-Model
    train_tft_model(dev_path, validation_days, max_prediction_length)

    

    