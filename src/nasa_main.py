# -*- coding: utf-8 -*-
"""
Amaç: büyük/sıcak yıldızlarda belirsizliği azaltmak
- Opsiyonel logg filtresi: low-gravity dev yıldızları dışarı at (logg < 3.8)
- Normalized depth: depth_ppm / st_rad^2
- Ek etkileşim/ratio: depth_norm_per_hour, depth_norm_x_logg
- Mevcut feature eng + outlier clipping + calibration + threshold tuning
- Two-stage training (pre->model) + early stopping
"""

import json, joblib, warnings
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# =============================
# Config
# =============================
DATA_PATH = "data/harmonized_exoplanets_clean.csv"
OUT_DIR   = Path("./artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2

# toggles
USE_OUTLIER_CLIP   = True
USE_FEATURE_ENG    = True
USE_LOGG_FILTER    = True   # <— önemli: dev yıldızları ayıkla
LOGG_MIN           = 3.8    # literatürde ana dizi için yaygın eşik
DO_CALIBRATION     = True
DO_THRESHOLD_TUNE  = True

CLIP_BOUNDS = {
    "period_days":   (0.05, 1000.0),
    "duration_hrs":  (0.05, 60.0),
    "depth_ppm":     (0.0,  2e5),
    "radius_rearth": (0.1,  30.0),
    "impact":        (0.0,  1.5),
    "teff":          (2500.0, 8000.0),
    "logg":          (3.3,  5.5),   # filtreden önce geniş tut
    "st_rad":        (0.05,  100.0),
}

# =============================
# Load
# =============================
df = pd.read_csv(DATA_PATH)
assert set(["mission","label"]).issubset(df.columns), "CSV'de mission ve label olmalı"

# =============================
# Optional filter: remove very low-gravity stars (giants)
# =============================
if USE_LOGG_FILTER and "logg" in df.columns:
    before = len(df)
    df = df[df["logg"] >= LOGG_MIN].copy()
    after = len(df)
    print(f"[LOGG FILTER] Kept {after}/{before} rows (logg >= {LOGG_MIN})")

# =============================
# Outlier clipping
# =============================
def clip_outliers(d: pd.DataFrame, bounds: Dict[str, Tuple[float,float]]) -> pd.DataFrame:
    d = d.copy()
    for c,(lo,hi) in bounds.items():
        if c in d.columns:
            d[c] = np.clip(d[c], lo, hi)
    return d

if USE_OUTLIER_CLIP:
    df = clip_outliers(df, CLIP_BOUNDS)

# =============================
# Feature engineering (v3)
# =============================
def add_features_v3(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    eps = 1e-9

    # v2’den gelenler
    if {"depth_ppm","duration_hrs"}.issubset(d.columns):
        d["depth_ratio"] = d["depth_ppm"] / (d["duration_hrs"] + eps)
    if {"radius_rearth","st_rad"}.issubset(d.columns):
        d["radius_ratio"] = d["radius_rearth"] / (d["st_rad"] + eps)
    if "depth_ppm" in d.columns:
        d["flux_drop"] = d["depth_ppm"] / 1e6

    # v3 yeni: normalized depth (yıldız yarıçapına göre)
    if {"depth_ppm","st_rad"}.issubset(d.columns):
        d["depth_norm"] = d["depth_ppm"] / ((d["st_rad"] + eps) ** 2)

    # v3 etkileşim/ratio
    if {"depth_norm","duration_hrs"}.issubset(d.columns):
        d["depth_norm_per_hour"] = d["depth_norm"] / (d["duration_hrs"] + eps)
    if {"depth_norm","logg"}.issubset(d.columns):
        d["depth_norm_x_logg"] = d["depth_norm"] * d["logg"]

    # log1p dönüşümleri
    for col in [
        "depth_ppm","period_days","duration_hrs","radius_rearth",
        "depth_ratio","depth_norm","depth_norm_per_hour"
    ]:
        if col in d.columns:
            d[f"log1p_{col}"] = np.log1p(np.clip(d[col], a_min=0, a_max=None))

    return d

if USE_FEATURE_ENG:
    df = add_features_v3(df)

# =============================
# Split
# =============================
le = LabelEncoder()
y = le.fit_transform(df["label"])
X = df.drop(columns=["label"])

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = ["mission"] if "mission" in X.columns else []

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# =============================
# Two-stage training (pre -> model) + early stopping
# =============================
pre_fitted = clone(pre).fit(X_train, y_train)
Xtr = pre_fitted.transform(X_train)
Xva = pre_fitted.transform(X_test)

algo = "lightgbm"
clf = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight=class_weights,
        learning_rate=0.025,
        n_estimators=6000,
        max_depth=8, num_leaves=79,
        subsample=0.9, colsample_bytree=0.85,
        min_child_samples=40,
        reg_alpha=0.3, reg_lambda=0.3,
        random_state=SEED
)

    
    clf.fit(
        Xtr, y_train,
        eval_set=[(Xva, y_test)],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)]
    )
except Exception:
    try:
        from xgboost import XGBClassifier
        algo = "xgboost"
        clf = XGBClassifier(
            n_estimators=6000, learning_rate=0.025,
            max_depth=8, subsample=0.9, colsample_bytree=0.85,
            min_child_weight=1.0, reg_alpha=0.3, reg_lambda=0.3,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=SEED
        )
        clf.fit(Xtr, y_train, eval_set=[(Xva, y_test)], early_stopping_rounds=200, verbose=False)
    except Exception:
        algo = "random_forest"
        clf = RandomForestClassifier(
            n_estimators=1200, max_depth=None,
            min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
        )
        clf.fit(Xtr, y_train)

# inference pipeline
pipe = Pipeline([("pre", pre_fitted), ("clf", clf)])

# =============================
# Calibration
# =============================
calibrated_pipe = pipe
if DO_CALIBRATION and hasattr(pipe.named_steps["clf"], "predict_proba"):
    calibrated_pipe = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    calibrated_pipe.fit(X_train, y_train)

# =============================
# Eval
# =============================
def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    mf1 = f1_score(y_te, y_pred, average="macro")
    rep = classification_report(y_te, y_pred, target_names=list(le.classes_))
    cm  = confusion_matrix(y_te, y_pred)
    return acc, mf1, rep, cm

acc_raw, mf1_raw, rep_raw, cm_raw = evaluate(pipe, X_test, y_test)
acc_cal, mf1_cal, rep_cal, cm_cal = evaluate(calibrated_pipe, X_test, y_test)

print(f"Algorithm : {algo}")
print(f"[RAW] Accuracy: {acc_raw:.4f} | Macro-F1: {mf1_raw:.4f}")
print(f"[CAL] Accuracy: {acc_cal:.4f} | Macro-F1: {mf1_cal:.4f}")

# =============================
# Threshold tuning for CONFIRMED
# =============================
confirmed_idx = list(le.classes_).index("CONFIRMED")
best_thr = None
thr_note = ""
if DO_THRESHOLD_TUNE and hasattr(calibrated_pipe, "predict_proba"):
    proba = calibrated_pipe.predict_proba(X_test)
    p_confirmed = proba[:, confirmed_idx]
    y_binary = (y_test == confirmed_idx).astype(int)
    prec, rec, thr = precision_recall_curve(y_binary, p_confirmed)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    if len(thr) > 0:
        best_i = int(np.nanargmax(f1s))
        best_thr = float(thr[max(best_i-1,0)])
        thr_note = f"Best threshold for CONFIRMED (by PR-F1): {best_thr:.3f}"
        print(thr_note)

# =============================
# Save
# =============================
(OUT_DIR / "metrics_raw.txt").write_text(
    f"Algorithm: {algo}\nAccuracy: {acc_raw:.4f}\nMacro-F1: {mf1_raw:.4f}\n\n"
    + rep_raw + "\nConfusion Matrix (rows=true, cols=pred):\n" + np.array2string(cm_raw),
    encoding="utf-8"
)
(OUT_DIR / "metrics_calibrated.txt").write_text(
    f"Algorithm: {algo}\nAccuracy: {acc_cal:.4f}\nMacro-F1: {mf1_cal:.4f}\n\n"
    + rep_cal + "\nConfusion Matrix (rows=true, cols=pred):\n" + np.array2string(cm_cal)
    + ("\n" + thr_note if best_thr is not None else ""),
    encoding="utf-8"
)

# feature importance
imp_path = OUT_DIR / "feature_importance.csv"
try:
    model = pipe.named_steps["clf"]
    pre_fit = pipe.named_steps["pre"]
    num_names = [c for c in num_cols]
    cat_names = []
    if "mission" in cat_cols:
        ohe = pre_fit.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(["mission"]).tolist()
    feat_names = num_names + cat_names
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pd.DataFrame({"feature": feat_names, "importance": importances}) \
          .sort_values("importance", ascending=False) \
          .to_csv(imp_path, index=False)
    else:
        raise AttributeError
except Exception:
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1)
    feat_names = num_cols + (["mission"] if len(cat_cols)>0 else [])
    pd.DataFrame({"feature": feat_names, "importance": pi.importances_mean}) \
      .sort_values("importance", ascending=False) \
      .to_csv(imp_path, index=False)

# artifacts
raw_path = OUT_DIR / f"exoplanet_{algo}_raw.pkl"
cal_path = MODEL_DIR / f"exoplanet_{algo}_calibrated.pkl"
lbl_path = OUT_DIR / "label_encoder.pkl"
cfg_path = OUT_DIR / "train_config.json"

joblib.dump(calibrated_pipe, cal_path)
joblib.dump(pipe,            raw_path)
joblib.dump(le,              lbl_path)

cfg = {
    "algo": algo,
    "seed": SEED,
    "test_size": TEST_SIZE,
    "use_outlier_clip": USE_OUTLIER_CLIP,
    "use_feature_eng": USE_FEATURE_ENG,
    "use_logg_filter": USE_LOGG_FILTER,
    "logg_min": LOGG_MIN,
    "do_calibration": DO_CALIBRATION,
    "do_threshold_tune": DO_THRESHOLD_TUNE,
    "clip_bounds": CLIP_BOUNDS,
    "best_threshold_confirmed": best_thr,
}
cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

print("Saved ->", str(cal_path))
print("Saved ->", str(raw_path))
print("Saved ->", str(lbl_path))
print("Saved ->", str(imp_path))
print("Saved ->", str(cfg_path))


unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(le.inverse_transform(unique), counts):
    print(f"{cls:15s}: {count:6d}")

if algo == "lightgbm":
    from collections import Counter
    total = sum(counts)
    weights = {cls: total/c for cls, c in zip(le.inverse_transform(unique), counts)}
    print("Auto LightGBM class weights:", weights)
elif algo == "random_forest":
    print("RandomForest uses class_weight='balanced_subsample'")