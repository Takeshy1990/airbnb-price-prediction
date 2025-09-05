# predict_new.py — Batch predict with quantile intervals
# Διαβάζει predict.csv -> γράφει prediction.csv με pred_price, low_q10, high_q90

import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE = Path(".").resolve()
META_FILE  = BASE/"airbnb_meta.json"
MODEL_CORE = BASE/"airbnb_model_core.pkl"
MODEL_Q10  = BASE/"airbnb_model_q10.pkl"
MODEL_Q90  = BASE/"airbnb_model_q90.pkl"
IN_FILE    = BASE/"predict.csv"
OUT_FILE   = BASE/"prediction.csv"

def baths_to_float(s):
    if pd.isna(s): return np.nan
    m = re.search(r"(\d+(\.\d+)?)", str(s))
    return float(m.group(1)) if m else np.nan

def read_csv_any(p: Path) -> pd.DataFrame:
    for sep in (None, ",", ";", "\t"):
        try:
            return pd.read_csv(p, sep=sep, engine="python")
        except Exception:
            pass
    raise FileNotFoundError(f"Αδυναμία ανάγνωσης: {p}")

def prepare_df(df_raw: pd.DataFrame, feature_columns):
    df = df_raw.copy()

    # αν υπάρχει μόνο bathrooms_text -> δημιουργούμε bathrooms
    if "bathrooms" not in df.columns and "bathrooms_text" in df.columns:
        df["bathrooms"] = df["bathrooms_text"].apply(baths_to_float)

    # extra features όπως στο training
    if "number_of_reviews" in df.columns and "availability_365" in df.columns:
        df["reviews_density"] = df["number_of_reviews"]/(df["availability_365"]+1)
    if "accommodates" in df.columns:
        df["price_per_guest"] = np.nan  # στο predict δεν ξέρουμε τιμή

    # align με τα features του μοντέλου
    return df.reindex(columns=feature_columns, fill_value=np.nan)

def main():
    # meta (για τα feature columns)
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    feature_columns = meta["feature_columns"]

    # input
    df_raw = read_csv_any(IN_FILE)
    X = prepare_df(df_raw, feature_columns)

    # load models
    core = joblib.load(MODEL_CORE)
    q10  = joblib.load(MODEL_Q10)
    q90  = joblib.load(MODEL_Q90)

    # predict log->eur
    y_med = np.expm1(core.predict(X))
    y_lo  = np.expm1(q10.predict(X))
    y_hi  = np.expm1(q90.predict(X))

    # output
    out = df_raw.copy()
    out["pred_price"] = y_med.round(0)
    out["low_q10"]    = y_lo.round(0)
    out["high_q90"]   = y_hi.round(0)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved predictions -> {OUT_FILE}")

if __name__ == "__main__":
    main()