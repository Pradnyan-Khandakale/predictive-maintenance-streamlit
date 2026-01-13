Predictive Maintenance for Machine Manufacturing
Overview
This project implements an end-to-end Predictive Maintenance system for manufacturing machines using machine learning and an interactive Streamlit dashboard.
The system analyzes time-series sensor data (temperature, vibration, pressure, RPM) to predict machine failure risk and help prevent unplanned downtime.

Key Features
Multi-machine support (M1, M2, M3)
Predictive failure modeling using:
Random Forest
XGBoost
Time-series feature engineering (rolling statistics, lag features)

Risk classification:
LOW
MEDIUM
HIGH

Interactive Streamlit UI
Deployable on Streamlit Community Cloud
Industry-aligned ML pipeline
Dataset Description

The dataset contains time-series sensor readings for multiple machines.

Columns
Column Name Description
timestamp Date & time of sensor reading
machine_id Machine identifier (M1, M2, M3)
temperature Temperature (°C)
vibration Vibration level
pressure Pressure
rpm Rotations per minute
failure 0 = Normal, 1 = Failure

Risk levels are learned by the model, not manually labeled.

Machine Learning Approach
Problem Type

Binary classification:

0 → No failure

1 → Failure

Feature Engineering

Rolling mean & standard deviation

Lag features

Time-aware preprocessing (no shuffling)

Risk Mapping (UI Logic)
Failure Probability Risk Level
< 0.4 LOW
0.4 – 0.7 MEDIUM

> 0.7 HIGH

Installation & Setup

1. Clone Repository
   git clone <your-repository-url>
   cd Predictive\ Maintainance\ for\ Machine\ Manufacturing

2. Create Virtual Environment (Recommended)
   python -m venv venv
   source venv/bin/activate # macOS / Linux
   venv\Scripts\activate # Windows

3. Install Dependencies
   pip install -r requirements.txt

Model Training

Train the models from the project root directory:

python -m src.train_model

This will create:

models/
├── random_forest.pkl
└── xgboost.pkl

XGBoost training is automatically skipped if the dataset contains only one class.

Run Streamlit Application
streamlit run app.py

Then open the browser at:

http://localhost:8501

Streamlit Dashboard Features

Machine selector (M1 / M2 / M3)

Sensor trend visualization

Failure probability score

Risk indicator (LOW / MEDIUM / HIGH)

Feature importance visualization

Model selection (Random Forest / XGBoost)

Deployment on Streamlit Cloud

Push the project to GitHub

Go to: https://streamlit.io/cloud

Click New App

Select your repository

Set:

Main file: app.py

Deploy

No Docker required.
