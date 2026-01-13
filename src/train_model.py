import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

df = load_and_clean_data("data/raw_sensor_data.csv")
df = create_features(df)

X = df.drop(columns=["timestamp", "machine_id", "failure"])
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# Check label diversity
if y_train.nunique() < 2:
    print("⚠ Only one class present in training data. Skipping XGBoost training.")
else:
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, "models/xgboost.pkl")
    print("XGBoost model trained and saved.")


print("Models trained and saved.")
print("Label distribution:")
print(y.value_counts())

