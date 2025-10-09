# back.py
# -----------------------------------------------------------
# Exoplanet API (model.pkl)
# - JSON POST /predict ile tahmin döner
# - Girdi: ham alanlar (mission, period_days, duration_hrs, depth_ppm,
#          radius_rearth, st_rad, impact, teff, logg)
# - Sunucu, eğitimdekiyle uyumlu engineered feature'ları üretir:
#   depth_norm, depth_ratio, radius_ratio, flux_drop,
#   depth_norm_per_hour, log1p_depth_norm_per_hour, depth_norm_x_logg,
#   log1p_* türevleri
# -----------------------------------------------------------

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # pip install flask-cors
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =============== App & CORS =================
app = Flask(__name__)
CORS(app)  # Frontend farklı porttaysa gerekli (ör. Live Server 5500)

# =============== Model yükleme ===============
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / f"exoplanet_random_forest_calibrated.pkl"               # pipeline / model
LABEL_PATH = Path("label_encoder.pkl")        # opsiyonel (LabelEncoder)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH.resolve()}")

pipe = joblib.load(MODEL_PATH)

# Sınıf isimleri: çoğu sklearn modelinde classes_ olur; yoksa opsiyonel label_encoder
CLASSES = None
if hasattr(pipe, "classes_"):
    CLASSES = list(pipe.classes_)
elif LABEL_PATH.exists():
    try:
        le = joblib.load(LABEL_PATH)
        if hasattr(le, "classes_"):
            CLASSES = list(le.classes_)
    except Exception:
        CLASSES = None

# Binary eşik kullanacaksan buradan belirleyebilirsin (opsiyonel)
CONF_THR = None  # örn: 0.62

# =============== Feature Engineering =========
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitimde kullanılan feature engineering ile birebir uyumlu olmalı.
    (Eğitimde bu adımlar pipeline içinde yapıldıysa burayı KAPAT.)
    """
    d = d.copy()
    eps = 1e-9

    # Oranlar / temel dönüşümler
    if {"depth_ppm", "duration_hrs"}.issubset(d.columns):
        d["depth_ratio"] = d["depth_ppm"] / (d["duration_hrs"] + eps)

    if {"radius_rearth", "st_rad"}.issubset(d.columns):
        d["radius_ratio"] = d["radius_rearth"] / (d["st_rad"] + eps)

    if "depth_ppm" in d.columns:
        d["flux_drop"] = d["depth_ppm"] / 1e6  # ppm -> oran

    if {"depth_ppm", "st_rad"}.issubset(d.columns):
        d["depth_norm"] = d["depth_ppm"] / ((d["st_rad"] + eps) ** 2)

    # İstenen ekstra engineered kolonlar
    if {"depth_norm", "duration_hrs"}.issubset(d.columns):
        d["depth_norm_per_hour"] = d["depth_norm"] / (d["duration_hrs"] + eps)
        d["log1p_depth_norm_per_hour"] = np.log1p(
            np.clip(d["depth_norm_per_hour"], a_min=0, a_max=None)
        )

    if {"depth_norm", "logg"}.issubset(d.columns):
        d["depth_norm_x_logg"] = d["depth_norm"] * d["logg"]

    # Sık kullanılan log1p kolonları
    for col in [
        "depth_ppm",
        "period_days",
        "duration_hrs",
        "radius_rearth",
        "depth_ratio",
        "depth_norm",
    ]:
        if col in d.columns:
            d[f"log1p_{col}"] = np.log1p(np.clip(d[col], a_min=0, a_max=None))

    return d


def _coerce_payload_to_df(payload: dict) -> pd.DataFrame:
    """
    - String sayıları float'a çevirir (virgül -> nokta).
    - Eksik ham alanlar için eğitimle uyumlu varsayılanlar atar.
    - Ardından engineered feature'ları üretir.
    """
    # Eğitimdeki imputing ile uyumlu basit default'lar
    defaults = {
        "mission": "KEPLER",  # en sık kategori / eğitimdeki default
        "impact": 0.0,
        "teff": 5700.0,
        "logg": 4.4,
    }

    # Beklenen temel ham alanlar (frontend bunları yollamalı)
    expected = [
        "mission",
        "period_days",
        "duration_hrs",
        "depth_ppm",
        "radius_rearth",
        "st_rad",
        "impact",
        "teff",
        "logg",
    ]

    clean = {}
    for key in set(expected) | set(payload.keys()) | set(defaults.keys()):
        v = payload.get(key, defaults.get(key))
        if isinstance(v, str):
            vv = v.replace(",", ".").strip()
            try:
                clean[key] = float(vv)
            except ValueError:
                clean[key] = vv  # mission gibi kategorikler string kalır
        else:
            clean[key] = v

    df = pd.DataFrame([clean])

    # Eğer FE eğitimde pipeline dışında yapıldıysa:
    df = add_features(df)

    return df


# ================== Routes ===================
@app.route("/", methods=["GET"])
def root():
    idx = Path("index.html")
    if idx.exists():
        return send_from_directory(".", "index.html")
    return "Exoplanet API is running. POST /predict"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, dict):
            return jsonify({"error": "JSON body dict olmalı"}), 400

        df = _coerce_payload_to_df(data)

        # Tahmin
        pred = pipe.predict(df)
        out = {}

        # Sınıf adı çözümlenmesi
        label_out = pred[0]
        if CLASSES is not None:
            try:
                # bazı modeller int index dönebilir
                if np.issubdtype(np.asarray(pred).dtype, np.integer):
                    label_out = CLASSES[int(pred[0])]
            except Exception:
                pass

        out["prediction"] = str(label_out)

        # Olasılık / güven
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(df)
            out["confidence"] = float(np.max(proba))

            if (CONF_THR is not None) and (CLASSES is not None):
                try:
                    idx_conf = CLASSES.index("CONFIRMED")
                    p_conf = float(proba[:, idx_conf][0])
                    out["p_confirmed"] = p_conf
                    out["is_confirmed_by_thr"] = p_conf >= CONF_THR
                except Exception:
                    pass

        return jsonify(out)

    except Exception as e:
        # Hata mesajını döndür (geliştirme için)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ağdan da erişilsin diye 0.0.0.0; sadece lokal istersen host="127.0.0.1"
    app.run(host="0.0.0.0", port=5000)
