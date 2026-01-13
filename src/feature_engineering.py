import pandas as pd
def create_features(df, window=3):
    sensors = ["temperature", "vibration", "pressure", "rpm"]

    for sensor in sensors:
        df[f"{sensor}_rolling_mean"] = (
            df.groupby("machine_id")[sensor]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{sensor}_rolling_std"] = (
            df.groupby("machine_id")[sensor]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )

        df[f"{sensor}_lag1"] = (
            df.groupby("machine_id")[sensor]
            .shift(1)
            .fillna(method="bfill")
        )

    df.bfill()
    return df

      
