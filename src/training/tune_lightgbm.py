from src.utils.logger import setup_logger

logger = setup_logger()
#from __future__ import annotations
from src.etl.feature_engineering import feature_engineering
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Optional, Tuple


def load_and_split_data(
        dev_path: Path | str ,
        target_col: str,
        validation_days: int,
        max_prediction_length: int
        ):
    # load development data
    development_df = pd.read_csv(dev_path)
    # Feature engineering
    development_df = feature_engineering(development_df)
    #
    validation_rows = validation_days * max_prediction_length
    validation_cutoff = (development_df["time_idx"].max() - validation_rows)
    # split data
    train_df = development_df[development_df.time_idx <= validation_cutoff]
    eval_df = development_df[development_df.time_idx > validation_cutoff]
    X_train, y_train = train_df.drop(columns = [target_col]), train_df[target_col]
    X_eval, y_eval = eval_df.drop(columns = [target_col]), eval_df[target_col]
    return X_train, y_train, X_eval, y_eval

def tune_model(
        dev_path: Path | str ,
        model_output: Path | str ,
        model_name: str ,
        target_col: str ,
        n_trials: int,
        tracking_uri: Optional[str] ,
        experiment_name: str
        ) -> Tuple[dict, dict]:
    """
    Run optuna tuning, save best model, 
    and return (best_params, best_metrics)
    """
    if tracking_uri:
        tracking_path = Path(tracking_uri)
        tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_path.resolve().as_uri())
    
    mlflow.set_experiment(experiment_name)
    X_train, y_train, X_eval, y_eval = load_and_split_data(dev_path, target_col, validation_days, max_prediction_length)
    
    def objective(trial: optuna.Trial):
        params = {
            # Boosting
            "objective": "regression",
            "metric": "rmse",

            # Core model complexity
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),

            # Tree structure
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

            # Row and feature sampling
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            # LightGBM-specific bagging
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            
            # Regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),

            # General
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            }

        with mlflow.start_run(nested=True):
            if model_name == "lightgbm":
                model = LGBMRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_eval)
                
                mae = float(mean_absolute_error(y_eval, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
                r2 = float(r2_score(y_eval, y_pred))
                mlflow.log_params(params)
                mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
                # Log final model
                # mlflow.lightgbm.log_model(model, name="lightgbm_model")
        return rmse
    
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials= n_trials)
    best_params = study.best_trial.params
    logger.info(
        f"Best parameters from optuna determined: {best_params}"
    )

    # retrain the model
    if model_name == "lightgbm":
        best_model = LGBMRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "tree_method": "hist" })
        best_model.fit(X_train, y_train)
        # metrics
        y_pred = best_model.predict(X_eval)
                
        mae = float(mean_absolute_error(y_eval, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        r2 = float(r2_score(y_eval, y_pred))
        best_metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info(
            f"Best tuning model metrics: mae={mae:.2f}, rmse={rmse:.2f}, r2={r2:.2f}"
            )
        # save best model
        output_path = Path(model_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, output_path)
        logger.info(
            f"Model trained and saved to {output_path}"
            )
        with mlflow.start_run(run_name="lightgbm_best_model"):
            mlflow.log_params(best_params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            # Log final model
            mlflow.lightgbm.log_model(best_model, 
                                     name="lightgbm_best_model",
                                     registered_model_name="lightgbm_best_model"
                                     )
  

    return best_params, best_metrics

if __name__ == "__main__":

    dev_path = 'data/cleaned/development.csv'
    target_col ='load_demand_(kw)'
    validation_days = 180  
    max_prediction_length = 96  # 24 hours
    X_train, y_train, X_eval, y_eval = load_and_split_data(
        dev_path, target_col, validation_days, max_prediction_length
        )
    
    model_name = "lightgbm"
    best_model_path = Path("artifacts/models/lightgbm/lightgbm_best_model.pkl")
    n_trials = 15
    experiment_name = "lightgbm_optuna_wind_forecasting"
    tracking_uri = None

    tune_model(
        dev_path = dev_path,
        model_output = best_model_path,
        model_name = model_name,
        target_col = target_col,
        n_trials = n_trials,
        tracking_uri = tracking_uri,
        experiment_name = experiment_name
        )