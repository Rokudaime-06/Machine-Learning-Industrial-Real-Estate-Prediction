# 🏢 Industrial Real Estate Predictive Analytics: Rental Price Modeling in Morocco

This repository contains a localized machine learning pipeline developed to model and predict monthly rental prices for industrial warehouses and logistics zones.

## 📊 Pipeline Architecture & Performance Metrics
- **Algorithm:** RandomForestRegressor (100 estimators)
- **Feature Scaling:** MinMaxScaler (Continuous normalization)
- **Dataset Size:** 15 localized strategic zones
- **Model Evaluation:**
  - **R² Score (Precision):** **82.47 %**
  - **Mean Absolute Error (MAE):** **19,496.67 MAD**

## 💻 Technical Implementation
The pipeline executes a clean 80/20 data split for evaluation. Features (`Superficie_Zone_M2` and `Distance_Port_KM`) are continuous variables normalized using `MinMaxScaler` to prevent distance-bias during training. The final output is plotted using a continuous regression line in `Matplotlib` to inspect data distribution.
