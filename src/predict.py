import joblib
import pandas as pd

def predict_failure(input_df, model_name="random_forest"):
    model = joblib.load(f"models/{model_name}.pkl")
    probability = model.predict_proba(input_df)[0][1]

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return probability, risk
