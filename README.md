# ⚡ Production-Ready Electrical Load Demand Forecasting System

An end-to-end machine learning platform for **short-term electrical load forecasting** using **LightGBM** and **Temporal Fusion Transformer (TFT)** models.

The project goes beyond model development by implementing a complete **MLOps workflow** including experiment tracking with **MLflow**, hyperparameter optimization using **Optuna**, Docker containerization, CI/CD automation with **GitHub Actions**, deployment on **AWS ECS**, and a **Streamlit-based forecasting dashboard** for scalable and reproducible inference.

---

## 🚀 Key Highlights

* ⚡ Short-term electrical load forecasting (15-minute intervals)
* 🤖 LightGBM and Temporal Fusion Transformer (TFT)
* 🎯 Hyperparameter optimization with Optuna
* 📊 MLflow experiment tracking and model versioning
* 🐳 Dockerized application stack
* 🔄 Automated GitHub Actions CI/CD pipeline
* ☁️ AWS ECS deployment
* 📈 Interactive Streamlit dashboard with prediction visualization and performance metrics

---

## 🏗️ System Architecture

```text
Raw Data
      │
      ▼
Data Preprocessing & Feature Engineering
      │
      ▼
Optuna Hyperparameter Optimization
      │
      ▼
Model Training (LightGBM / TFT)
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
Best Model Selection
      │
      ▼
Docker Build
      │
      ▼
GitHub Actions CI/CD
      │
      ▼
AWS ECS Deployment
      │
      ▼
Streamlit Forecasting Dashboard
```

---

## 📌 Problem Statement

Accurate electrical load forecasting is essential for modern power systems. Reliable predictions support:

* ⚡ Grid balancing and stability
* 🔋 Battery energy storage optimization
* 📈 Energy trading
* 🌱 Renewable energy integration
* 🏭 Generation scheduling

Traditional forecasting approaches often struggle to capture nonlinear relationships between electricity demand, weather conditions, and temporal patterns. This project combines gradient boosting and deep learning techniques to model these complex dependencies for accurate short-term load forecasting.

---

## 📂 Dataset

**Source:** Kaggle Electrical Load Forecasting Dataset

### Features

**Weather**

* Temperature
* Humidity
* Wind Speed
* Rainfall
* Solar Irradiance

**Economic**

* GDP
* Electricity Price
* Per-Capita Energy Consumption

**Calendar**

* Hour
* Day of Week
* Month
* Season
* Public Event Index

**Target**

* Electrical Load Demand (kW)

---

## ⚙️ Machine Learning Pipeline

### Feature Engineering

* Temporal feature extraction
* Cyclical time encoding
* Lag features
* Rolling statistics
* Weather interaction features
* Peak-hour and business-hour indicators

### Models

* LightGBM
* Temporal Fusion Transformer (TFT)

### Hyperparameter Optimization

Optuna was used to optimize the LightGBM model by searching for the best hyperparameters while minimizing validation RMSE.

---

## 📊 Model Performance

Example forecast for **May 4, 2025**

| Model    |      MAE |     RMSE |
| -------- | -------: | -------: |
| LightGBM | **1.04** | **2.75** |
| TFT      |     4.77 |     5.41 |

The forecasting dashboard provides interactive **Prediction vs Actual** visualization together with real-time MAE and RMSE evaluation.

---

## 🔄 MLOps Workflow

The project implements a production-ready MLOps pipeline including:

### Experiment Tracking

* MLflow experiment tracking
* Parameter logging
* Metric logging
* Model versioning
* Artifact management

### Containerization

* Dockerized training and inference environments
* Reproducible builds
* Environment consistency

### CI/CD

Automated GitHub Actions pipeline for:

* Build
* Docker image creation
* Image publishing
* Deployment to AWS ECS

### Cloud Deployment

AWS ECS provides:

* Scalable model deployment
* Container orchestration
* Production-ready inference service

### Model Serving

The Streamlit application offers:

* Interactive forecasting interface
* Date and model selection
* Prediction vs Actual visualization
* Daily load profile
* Live MAE and RMSE metrics

---

## 🔮 Future Improvements

* Automated model retraining
* Data and concept drift detection
* Prometheus & Grafana monitoring
* Additional forecasting architectures (PatchTST, N-HiTS)
* Automated model registry and deployment
