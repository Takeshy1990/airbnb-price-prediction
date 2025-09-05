# airbnb.py — Robust training για listings.csv + Quantile intervals + Feature Importances
# Outputs: airbnb_model_core.pkl, airbnb_model_q10.pkl, airbnb_model_q90.pkl,
#          airbnb_meta.json, feature_importance.csv, prediction_plot.png

import re, math, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ---------- paths ----------
BASE = Path(".").resolve()
DATA_FILE = BASE / "listings.csv"
MODEL_CORE = BASE / "airbnb_model_core.pkl"   # median model
MODEL_Q10  = BASE / "airbnb_model_q10.pkl"
MODEL_Q90  = BASE / "airbnb_model_q90.pkl"
META_FILE  = BASE / "airbnb_meta.json"
PLOT_FILE  = BASE / "prediction_plot.png"
IMP_FILE   = BASE / "feature_importance.csv"

# ---------- helpers ----------
def read_csv_any(p: Path) -> pd.DataFrame:
    """Διαβάζει CSV ακόμη κι αν ο διαχωριστής δεν είναι ','."""
    for sep in (None, ",", ";", "\t"):
        try:
            return pd.read_csv(p, sep=sep, engine="python", on_bad_lines="skip")
        except Exception:
            pass
    raise FileNotFoundError(f"Αδυναμία ανάγνωσης: {p}")

def to_price_num(x):
    """Μετατρέπει τιμές τύπου '$118.00' / '€120,50' / '120' -> float (ή NaN)."""
    if pd.isna(x): return np.nan
    s = str(x)
    s = s.replace("\u20ac","").replace("€","").replace("$","").strip()
    s = re.sub(r"[^0-9,.\-]", "", s)
    if s.count(",") == 1 and s.count(".") == 0:  # 120,50 -> 120.50
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

def baths_to_float(s):
    if pd.isna(s): return np.nan
    m = re.search(r"(\d+(\.\d+)?)", str(s))
    return float(m.group(1)) if m else np.nan

def iqr_clip(s: pd.Series, k=3.0):
    """Κόβει outliers (IQR)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s.clip(lower=lo, upper=hi)

# ---------- load ----------
df = read_csv_any(DATA_FILE)

# Αναμένουμε τις στήλες του listings.csv που μου έδωσες
required = [
    "neighbourhood_cleansed","room_type","accommodates",
    "bedrooms","bathrooms_text","minimum_nights",
    "number_of_reviews","availability_365","review_scores_rating",
    "latitude","longitude","price"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Λείπουν στήλες από το listings.csv: {missing}")

# parsing / cleaning
df["price"] = df["price"].apply(to_price_num)
df["bathrooms"] = df["bathrooms_text"].apply(baths_to_float)

# βασικά φίλτρα & outlier handling
df = df.dropna(subset=["price","neighbourhood_cleansed","room_type","accommodates"])
df = df[(df["price"]>=15) & (df["price"]<=1000)].copy()

for c in ["accommodates","bedrooms","number_of_reviews",
          "availability_365","review_scores_rating","latitude","longitude","minimum_nights"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = iqr_clip(df[c])

# ---------- feature engineering ----------
df["price_per_guest"] = df["price"] / df["accommodates"].replace(0, np.nan)
df["reviews_density"] = df["number_of_reviews"] / (df["availability_365"] + 1)
df["log_price"] = np.log1p(df["price"])

y = df["log_price"].values
X = df.drop(columns=["price","log_price"])

feature_columns = X.columns.tolist()

# ---------- preprocessor ----------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler(with_mean=False))
])
# Συμβατό με όλες τις εκδόσεις scikit-learn
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe)
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], remainder="drop")

# ---------- models ----------
lin = LinearRegression()
gbm = GradientBoostingRegressor(
    random_state=42, n_estimators=800, learning_rate=0.03,
    max_depth=3, subsample=0.9
)

pipe_lin = Pipeline([("prep", pre), ("mdl", lin)])
pipe_gbm = Pipeline([("prep", pre), ("mdl", gbm)])

# ---------- train/test ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def eval_model(pipe, name):
    pipe.fit(Xtr, ytr)
    pred = np.expm1(pipe.predict(Xte))
    r2   = r2_score(np.expm1(yte), pred)
    mae  = mean_absolute_error(np.expm1(yte), pred)
    rmse = math.sqrt(mean_squared_error(np.expm1(yte), pred))
    print(f"{name}: R²={r2:.3f}  MAE={mae:.0f}€  RMSE={rmse:.0f}€")
    return pipe

eval_model(pipe_lin, "Linear")
pipe_gbm = eval_model(pipe_gbm, "GBM")

# ---------- cross-validation ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
yhat_log = cross_val_predict(pipe_gbm, X, y, cv=cv, n_jobs=-1)
yhat     = np.expm1(yhat_log)
R2   = r2_score(np.expm1(y), yhat)
MAE  = mean_absolute_error(np.expm1(y), yhat)
RMSE = math.sqrt(mean_squared_error(np.expm1(y), yhat))
print(f"CV (GBM): R²={R2:.3f}  MAE={MAE:.0f}€  RMSE={RMSE:.0f}€")

# ---------- plot ----------
plt.figure(figsize=(5,5))
plt.scatter(np.expm1(y), yhat, alpha=0.3)
plt.xlabel("Actual (€)"); plt.ylabel("Predicted (€)")
plt.title("Airbnb Actual vs Predicted (CV)")
mn, mx = np.expm1(y).min(), np.expm1(y).max()
plt.plot([mn, mx], [mn, mx])
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.close()

# ---------- quantile models (q10 / q90) ----------
def q_loss(a): return {"loss": "quantile", "alpha": a}

gbm_q10 = GradientBoostingRegressor(
    random_state=42, n_estimators=800, learning_rate=0.03,
    max_depth=3, subsample=0.9, **q_loss(0.10)
)
gbm_q90 = GradientBoostingRegressor(
    random_state=42, n_estimators=800, learning_rate=0.03,
    max_depth=3, subsample=0.9, **q_loss(0.90)
)

pipe_q10 = Pipeline([("prep", pre), ("mdl", gbm_q10)])
pipe_q90 = Pipeline([("prep", pre), ("mdl", gbm_q90)])

pipe_gbm.fit(X, y)
pipe_q10.fit(X, y)
pipe_q90.fit(X, y)

# ---------- feature importances ----------
mdl = pipe_gbm.named_steps["mdl"]
try:
    feat_names = pipe_gbm.named_steps["prep"].get_feature_names_out()
except Exception:
    feat_names = np.array(feature_columns)

if hasattr(mdl, "feature_importances_"):
    imp = pd.DataFrame({"feature": feat_names, "importance": mdl.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(IMP_FILE, index=False, encoding="utf-8")
    print(f"Saved feature_importance -> {IMP_FILE.name}")

# ---------- save all ----------
joblib.dump(pipe_gbm, MODEL_CORE)
joblib.dump(pipe_q10, MODEL_Q10)
joblib.dump(pipe_q90, MODEL_Q90)

meta = {
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "feature_columns": feature_columns,
    "cv": {"R2": float(R2), "MAE": float(MAE), "RMSE": float(RMSE)}
}
META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("Saved:", MODEL_CORE.name, MODEL_Q10.name, MODEL_Q90.name, META_FILE.name, PLOT_FILE.name)