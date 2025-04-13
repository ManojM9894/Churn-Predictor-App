import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(df):
    df = df.copy()

    # Drop ID-like columns
    for col in df.columns:
        if "id" in col.lower():
            df.drop(columns=[col], inplace=True)

    df.columns = df.columns.str.strip()
    df.replace(" ", pd.NA, inplace=True)
    df.fillna(method="fill", inplace=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def save_encoders(encoders, path="encoders.pkl"):
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def load_encoders(path="encoders.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_encoders(df, encoders):
    df = df.copy()
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    return df
