import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values(["machine_id", "timestamp"], inplace=True)
    df = df.ffill()
    return df
