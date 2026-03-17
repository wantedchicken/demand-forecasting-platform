# 📊 Demand Forecasting Platform (Retail Sales)

An end-to-end machine learning system that predicts daily product demand for retail stores using large-scale time series data.

🚀 Built with real-world data (57M+ records) and designed to simulate production-grade data science & data engineering workflows.

---

## 🔥 Live Demo
👉 [Add your Streamlit link here after deployment]

---

## 📌 Problem Statement

Retail businesses need to accurately forecast demand to:

- avoid stockouts  
- reduce excess inventory  
- optimize supply chain decisions  

This project builds a scalable system to answer:

> “How many units of a product will a store sell tomorrow?”

---

## 🧠 Solution Overview

This project implements an end-to-end pipeline:

1. Data ingestion and transformation  
2. Feature engineering (time-series based)  
3. Machine learning modeling (LightGBM)  
4. Rolling backtesting  
5. Batch prediction  
6. Monitoring & evaluation  
7. Interactive dashboard (Streamlit)  

---

## 📂 Dataset

- Source: **M5 Forecasting Dataset (Walmart retail data)**
- Scale: **57M+ rows**
- Hierarchical structure:
  - Product level (`item_id`)
  - Store level (`store_id`)
  - Category & department levels  

---

## ⚙️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **LightGBM**
- **Scikit-learn**
- **Streamlit**
- **Parquet (PyArrow)**
- **Jupyter Notebooks**

---

## 🔧 Key Features

### 📈 Feature Engineering
- Lag features (7, 14, 28 days)
- Rolling statistics (mean, std)
- Calendar-based features (weekday, month)

---

### 🤖 Model
- Global LightGBM model across all products
- Handles large-scale data efficiently
- Learns cross-product demand patterns

---

### 🔁 Rolling Backtesting
- Simulates real-world forecasting
- Multiple time-based validation windows
- Ensures model stability over time

---

### 📊 Evaluation Metrics

| Metric | Meaning |
|------|--------|
MAE | Average error in units sold |
RMSE | Penalizes large errors |
WAPE | Weighted error |
SMAPE | Percentage-based error |

👉 Example:
> MAE ≈ 1 → predictions are off by ~1 unit per day

---

### 📡 Monitoring

- Data quality checks  
- Drift detection  
- Performance tracking across stores and categories  

---

### 🖥️ Interactive Dashboard

Built with Streamlit to visualize:

- Actual vs predicted sales  
- Model performance  
- Store & product-level insights  

💡 Designed for non-technical users to understand predictions easily.

---

## 📦 Project Structure
