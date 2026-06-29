from src.utils.logger import setup_logger

logger = setup_logger()
from src.etl.feature_engineering import feature_engineering
#from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Optional, Tuple
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap

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

def train_and_save_model(model_output, model, X_train, y_train, X_eval, y_eval):

    logger.info("Training model")

    model.fit(X_train, y_train, categorical_feature=["season"])
    y_pred = model.predict(X_eval)
                
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    metrics = {"mae": mae, "rmse": rmse}
    logger.info(
        f"Model metrics: mae={mae:.2f}, rmse={rmse:.2f}"
        )
    # save best model
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(
        f"Model trained and saved to {output_path}"
        )

    return metrics

if __name__ == "__main__":

    dev_path = 'data/cleaned/development.csv'
    target_col ='load_demand_(kw)'
    validation_days = 180  
    max_prediction_length = 96  # 24 hours
    X_train, y_train, X_eval, y_eval = load_and_split_data(
        dev_path, target_col, validation_days, max_prediction_length
        )
    #
    model = LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
        verbosity=-1,
        )
    
    model_output = 'artifacts/models/lightgbm/lightgbm_model.pkl'

    train_and_save_model(model_output, model, X_train, y_train, X_eval, y_eval)
    
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.booster_.feature_importance(
            importance_type="gain"
            )
            })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
        )
    importance_df.to_csv(
        "artifacts/feature_importance/feature_importance.csv",
        index=False
        )
    

    
    lgb.plot_importance(
        model,
        importance_type="gain",
        max_num_features=10
        )
    plt.savefig(
        "artifacts/feature_importance/importance.png"
        )
    #

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    shap.summary_plot(
    shap_values,
    X_eval,
    max_display=20,
    show = False
    )

    plt.tight_layout()
    plt.savefig(
        "artifacts/feature_importance/shap_summary.png",
        dpi=300,
        bbox_inches="tight"
        )
    plt.close()

    