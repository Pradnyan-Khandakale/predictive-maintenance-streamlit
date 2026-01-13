import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")

data = load_and_clean_data("data/raw_sensor_data.csv")
machine = st.selectbox("Select Machine", data["machine_id"].unique())

filtered = data[data["machine_id"] == machine]

st.subheader("Sensor Trends")
fig, ax = plt.subplots()
ax.plot(filtered["timestamp"], filtered["temperature"], label="Temperature")
ax.plot(filtered["timestamp"], filtered["vibration"], label="Vibration")
ax.legend()
st.pyplot(fig)

model_choice = st.selectbox("Select Model", ["random_forest", "xgboost"])
model = joblib.load(f"models/{model_choice}.pkl")

latest = create_features(filtered).iloc[-1:]
X_latest = latest.drop(columns=["timestamp", "machine_id", "failure"])

prob = model.predict_proba(X_latest)[0][1]

if prob > 0.7:
    st.error(f"⚠ HIGH FAILURE RISK ({prob:.2f})")
elif prob > 0.4:
    st.warning(f"⚠ MEDIUM FAILURE RISK ({prob:.2f})")
else:
    st.success(f"✅ LOW FAILURE RISK ({prob:.2f})")

st.subheader("Feature Importance")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": X_latest.columns, "Importance": importances})
st.bar_chart(feat_df.set_index("Feature"))
