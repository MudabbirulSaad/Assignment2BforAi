import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- CONFIG ---
RAW_DATA_PATH = "data/raw/data_given.xlsx"
PROCESSED_CSV_PATH = "data/processed/cleaned_data.csv"
SEQUENCE_DATA_PATH = "data/processed/sequence_data.npz"

def reshape_data(filepath):
    try:
        df = pd.read_excel(filepath, header=None, engine='openpyxl')

        # Time headers: row 0, columns 10+ (K onward)
        time_intervals = df.iloc[0, 10:].tolist()

        records = []

        # Iterate from row 2 onward (index 2+)
        for i in range(2, df.shape[0]):
            date_cell = df.iloc[i, 9]  # Column J = index 9
            try:
                date = pd.to_datetime(str(date_cell)).date()
            except:
                print(f"[WARN] Skipping invalid date at row {i}: {date_cell}")
                continue

            for j, time in enumerate(time_intervals):
                try:
                    time_obj = pd.to_datetime(str(time)).time()
                    timestamp = pd.Timestamp.combine(date, time_obj)
                    flow = df.iloc[i, 10 + j]
                    records.append({"Timestamp": timestamp, "Flow": flow})
                except Exception as e:
                    print(f"[WARN] Row {i}, Col {j}: {e}")

        df_long = pd.DataFrame(records).sort_values("Timestamp").set_index("Timestamp")
        print(f"[INFO] Time-series created with {len(df_long)} rows.")
        return df_long

    except Exception as e:
        print("[ERROR] Reshaping failed:", e)
        return None

def clean_and_normalize(df):
    try:
        df = df.resample("15min").mean()
        df = df.fillna(method="ffill")
        df = df[df["Flow"] >= 0]

        scaler = MinMaxScaler()
        df["Flow_norm"] = scaler.fit_transform(df[["Flow"]])
        print("[INFO] Data cleaned and normalized.")
        return df
    except Exception as e:
        print("[ERROR] Cleaning failed:", e)
        return None

def create_sequences(values, window_size):
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])
    return np.array(X), np.array(y)

def split_data(df, window_size=16):
    values = df["Flow_norm"].values
    split = int(len(values) * 0.8)
    X_train, y_train = create_sequences(values[:split], window_size)
    X_test, y_test = create_sequences(values[split:], window_size)

    X_train = X_train.reshape((-1, window_size, 1))
    X_test = X_test.reshape((-1, window_size, 1))
    return X_train, y_train, X_test, y_test

def save_all(df, X_train, y_train, X_test, y_test):
    df.to_csv(PROCESSED_CSV_PATH)
    print(f"[INFO] Saved CSV: {PROCESSED_CSV_PATH}")

    np.savez(SEQUENCE_DATA_PATH,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print(f"[INFO] Saved sequence data: {SEQUENCE_DATA_PATH}")

def main():
    df = reshape_data(RAW_DATA_PATH)
    if df is not None:
        df_clean = clean_and_normalize(df)
        if df_clean is not None:
            X_train, y_train, X_test, y_test = split_data(df_clean)
            save_all(df_clean, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
