import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pydeck as pdk  # map
from zoneinfo import ZoneInfo  # Python 3.9+

# Page & theme
st.set_page_config(
    page_title="Aurora Forecast • Rovaniemi",
    page_icon="🌌",
    layout="centered",
)

# Load external CSS
@st.cache_data
def load_css(path: str) -> str:
    """Read a CSS file once and cache the contents for reuse."""
    return Path(path).read_text(encoding="utf-8")

st.markdown(f"<style>{load_css('assets/styles.css')}</style>", unsafe_allow_html=True)

# Mapbox token (optional but recommended for dark basemap)
if os.environ.get("MAPBOX_API_KEY"):
    pdk.settings.mapbox_api_key = os.environ["MAPBOX_API_KEY"]

# Configuration
# Prefer a calibrated model if it exists (better probability calibration)
CALIB_MODEL_PATH = os.environ.get("AURORA_MODEL_CALIBRATED_PATH", "aurora_rf_calibrated.pkl")
MODEL_PATH = CALIB_MODEL_PATH if os.path.exists(CALIB_MODEL_PATH) else os.environ.get("AURORA_MODEL_PATH", "aurora_rf_model.pkl")

LOCAL_FEATURES_FILE = "live_spaceweather.csv"     # timestamp (UTC) + FEATURES
LOCAL_SNAPSHOT_FILE = "latest_spaceweather.json"  # raw snapshot to derive features
CLOUDS_FILE = "cloud_forecast.csv"                # timestamp (UTC), site_id, cloud_cover [0..1]
MODEL_REPORT_FILE = "model_report.json"           # optional metrics

SITES = {
    # name: lat, lon, light emoji, id
    "Arktikum Park 🌆":     {"lat": 66.5075, "lon": 25.7285, "light": "🌆", "site_id": "arktikum"},
    "Ounasvaara Scenic 🌃": {"lat": 66.4980, "lon": 25.7990, "light": "🌃", "site_id": "ounasvaara"},
    "Vikaköngäs Rapids 🌑": {"lat": 66.5610, "lon": 25.7730, "light": "🌑", "site_id": "vikakongas"},
}
SITE_DEFAULT = "Vikaköngäs Rapids 🌑"  # typically darkest of the three

# Data loaders (cached)
@st.cache_resource(show_spinner=False)
# --- UUSI JA PAREMPI KOODI ---
@st.cache_resource(show_spinner=False)
def load_model_and_features(path: str):
    """Load trained model and its required features list (cached)."""
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    try:
        artefact = joblib.load(path)
        model = artefact['model']
        features = artefact['features']
        return model, features
    except Exception as e:
        st.error("Failed to load model artefact.")
        st.exception(e)
        st.stop()
# --- KOODIN LOPPU ---

model, FEATURES = load_model_and_features(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_features_csv():
    """Read space-weather features CSV if present, else None."""
    if not os.path.exists(LOCAL_FEATURES_FILE):
        return None
    df = pd.read_csv(LOCAL_FEATURES_FILE, parse_dates=["timestamp"], comment="#")
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_clouds_csv():
    """Read clouds CSV if present, else None."""
    if not os.path.exists(CLOUDS_FILE):
        return None
    df = pd.read_csv(CLOUDS_FILE, parse_dates=["timestamp"])
    if df.empty:
        return None
    # Validate schema
    need = {"timestamp", "site_id", "cloud_cover"}
    miss = need - set(df.columns)
    if miss:
        st.error(f"`{CLOUDS_FILE}` missing columns: {sorted(miss)}")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # keep only valid site_ids
    df = df[df["site_id"].isin([v["site_id"] for v in SITES.values()])]
    # clamp cloud_cover to [0,1]
    df["cloud_cover"] = df["cloud_cover"].clip(0.0, 1.0)
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_model_report():
    """Optional: metrics JSON produced during training (accuracy, f1, roc_auc, trained_until)."""
    if not os.path.exists(MODEL_REPORT_FILE):
        return None
    try:
        return json.loads(Path(MODEL_REPORT_FILE).read_text(encoding="utf-8"))
    except Exception:
        return None

# Feature engineering
def to_utc_hour(dt_utc: datetime) -> datetime:
    """Round a datetime to the exact UTC hour."""
    return dt_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

def build_engineered_features_from_raw(bz_gsm, by_gsm, flow_speed, proton_density, ey, dst, f107, kp10):
    """
    Build the exact feature vector expected by the model from a single-hour raw snapshot.
    Uses persistence for engineered features (3h aggregates & 1h lags).
    """
    row = {
        "bz_gsm": float(bz_gsm),
        "by_gsm": float(by_gsm),
        "flow_speed": float(flow_speed),
        "proton_density": float(proton_density),
        "ey": float(ey),
        "dst": float(dst),
        "f107": float(f107),
        "kp10_3h_avg": float(kp10),              # persistence
        "bz_gsm_3h_min": float(bz_gsm),          # persistence
        "proton_density_3h_avg": float(proton_density),  # persistence
        "bz_gsm_lag1": float(bz_gsm),            # 1h lag (persistence)
        "kp10_lag1": float(kp10),                # 1h lag (persistence)
    }
    return pd.DataFrame([row], columns=FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)

def pick_nearest_row(df: pd.DataFrame, selected_utc_hour: datetime):
    """Return (row, used_ts str, delta_hours float) nearest to selected_utc_hour."""
    df = df.copy()
    df["diff"] = (df["timestamp"] - selected_utc_hour).abs()
    row = df.iloc[df["diff"].values.argmin()]
    used_ts = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
    delta_hours = float(abs((row["timestamp"] - selected_utc_hour).total_seconds()) / 3600.0)
    return row, used_ts, delta_hours

def get_features(selected_utc_hour: datetime):
    """
    Prefer CSV (nearest hour). Fallback to snapshot JSON. Final fallback to defaults.
    Returns X (DataFrame), meta (dict).
    """
    df_feat = load_features_csv()
    if df_feat is not None:
        missing = sorted(set(FEATURES) - set(df_feat.columns))
        if missing:
            st.error(f"`{LOCAL_FEATURES_FILE}` missing columns: {missing}")
            st.stop()
        row, used_ts, delta = pick_nearest_row(df_feat, selected_utc_hour)
        X = pd.DataFrame([row[FEATURES].to_dict()], columns=FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)
        return X, {"source": "CSV", "timestamp_used": used_ts, "delta_hours": delta}

    # snapshot JSON
    if os.path.exists(LOCAL_SNAPSHOT_FILE):
        data = json.loads(Path(LOCAL_SNAPSHOT_FILE).read_text(encoding="utf-8"))
        required = ["bz_gsm", "by_gsm", "flow_speed", "proton_density", "ey", "dst", "f107", "kp10"]
        missing = sorted(set(required) - set(data))
        if missing:
            st.error(f"`{LOCAL_SNAPSHOT_FILE}` missing keys: {missing}")
            st.stop()
        X = build_engineered_features_from_raw(**{k: data[k] for k in required})
        return X, {"source": "Snapshot"}

    # defaults
    raw = {"bz_gsm": -2.0, "by_gsm": 2.0, "flow_speed": 420.0, "proton_density": 5.0,
           "ey": -1.0, "dst": -15, "f107": 150.0, "kp10": 20}
    X = build_engineered_features_from_raw(**raw)
    return X, {"source": "Demo defaults"}

# Prediction
def predict_proba(model, X: pd.DataFrame) -> float:
    """Return probability from model, robust across estimator types."""
    X = X[FEATURES]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    if hasattr(model, "decision_function"):
        raw = float(model.decision_function(X)[0])
        return float(1.0 / (1.0 + np.exp(-raw)))
    # last resort: interpret predict as probability-like
    return float(np.clip(model.predict(X).astype(float)[0], 0.0, 1.0))

def cloud_at(time_utc: datetime, site_id: str):
    """
    Get cloud fraction [0..1] for nearest record at time_utc & site.
    If file missing, assume 0.30 and say so in meta.
    Returns (cloud_frac, cloud_meta dict).
    """
    df = load_clouds_csv()
    if df is None:
        return 0.30, {"cloud_source": "Assumed (no cloud CSV)", "delta_hours": None, "timestamp_used": None}

    dsite = df[df["site_id"] == site_id]
    if dsite.empty:
        return 0.30, {"cloud_source": f"Assumed (no rows for site_id={site_id})", "delta_hours": None, "timestamp_used": None}

    row, used_ts, delta = pick_nearest_row(dsite, time_utc)
    return float(row["cloud_cover"]), {"cloud_source": "Cloud CSV", "delta_hours": delta, "timestamp_used": used_ts}

def effective_probability(p_space: float, cloud_frac: float) -> float:
    """
    Combine space-weather probability with cloud fraction into an 'effective visibility' probability.
    Heuristic for demo: p_eff = p_space × (1 - cloud).
    """
    return float(np.clip(p_space * (1.0 - cloud_frac), 0.0, 1.0))

def lerp(a, b, t):
    """Linear interpolation helper for colors."""
    return int(a + (b - a) * t)

# Sidebar — controls
st.sidebar.header("Settings")
utc_now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

date_pick = st.sidebar.date_input("📅 Date (UTC)", value=utc_now.date())
hour_pick = st.sidebar.time_input("⏰ Hour (UTC)", value=utc_now.time())
site_name = st.sidebar.selectbox("📍 Location", list(SITES.keys()), index=list(SITES.keys()).index(SITE_DEFAULT))
time_window_h = st.sidebar.slider("⏱️ Available window (hours)", 1, 6, 2, help="Search best hour inside this window.")
threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01,
    help="If effective probability ≥ threshold → GO. Raise to be conservative; lower to be sensitive.")

col_sb1, col_sb2 = st.sidebar.columns(2)
predict_clicked = col_sb1.button("🔮 Calculate")
if col_sb2.button("🔁 Reload data"):
    load_features_csv.clear(); load_clouds_csv.clear()

# Predict on first render
if "pred_once" not in st.session_state:
    st.session_state.pred_once = True

selected_dt_utc = datetime.combine(date_pick, hour_pick).replace(tzinfo=timezone.utc)
selected_utc_hour = to_utc_hour(selected_dt_utc)
selected_local = selected_utc_hour.astimezone(ZoneInfo("Europe/Helsinki"))

# Header
st.markdown('<h1 class="aurora-title">Aurora Forecast • Rovaniemi</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="aurora-caption">'
    'Set date/time and a location around Rovaniemi. We combine space weather '
    'with cloud cover to estimate visibility and show a glowing map.'
    '</div>',
    unsafe_allow_html=True
)

# Core prediction flow
do_predict = predict_clicked or st.session_state.pred_once
if do_predict:
    st.session_state.pred_once = False

    # 1) Base features & model probability (space weather)
    X, meta = get_features(selected_utc_hour)
    # model ja FEATURES already done
    p_space = predict_proba(model, X)

    # 2) Cloud adjustment at selected site/time
    site = SITES[site_name]
    cloud_frac, cloud_meta = cloud_at(selected_utc_hour, site["site_id"])
    p_eff = effective_probability(p_space, cloud_frac)

    # 3) Find best hour inside user's available window using features CSV (if present)
    best_hour, best_p_eff = selected_utc_hour, p_eff
    df_feat = load_features_csv()
    if df_feat is not None and time_window_h > 1:
        for k in range(time_window_h):
            t = to_utc_hour(selected_utc_hour + timedelta(hours=k))
            row, _, _ = pick_nearest_row(df_feat, t)
            Xk = pd.DataFrame([row[FEATURES].to_dict()], columns=FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)
            p_space_k = predict_proba(model, Xk)
            cloud_k, _ = cloud_at(t, site["site_id"])
            p_eff_k = effective_probability(p_space_k, cloud_k)
            if p_eff_k > best_p_eff:
                best_p_eff, best_hour = p_eff_k, t
    elif df_feat is None and time_window_h > 1:
        st.info("Best-hour search limited (no space-weather CSV found).", icon="ℹ️")

    # 4) Warnings for stale data
    if meta.get("delta_hours", 0) > 3:
        st.warning(f"Closest space-weather row is {meta['delta_hours']:.1f} h from the selected time.")
    if cloud_meta.get("delta_hours") and cloud_meta["delta_hours"] > 3:
        st.warning(f"Closest cloud row is {cloud_meta['delta_hours']:.1f} h from the selected time.")

    # 5) GO/NO-GO decision (single, consistent rule)
    go = p_eff >= threshold
    decision = "GO — Aurora likely ✨" if go else "NO-GO — Aurora unlikely"

    # 6) Hero block (color from probability)
    r = lerp(90, 120, p_eff); g = lerp(120, 255, p_eff); b = 180
    sub_bits = []
    if meta.get("source"): sub_bits.append(meta["source"])
    if meta.get("timestamp_used"): sub_bits.append(f"features: {meta['timestamp_used']}")
    if cloud_meta.get("timestamp_used"):
        sub_bits.append(f"clouds: {cloud_meta['timestamp_used']}")
    elif cloud_meta.get("cloud_source"):
        sub_bits.append(cloud_meta["cloud_source"])
    sub_bits.append(f"threshold {threshold:.2f}")
    sub_bits.append(f"Local: {selected_local.strftime('%Y-%m-%d %H:%M')}")
    sub = " • ".join(sub_bits)

    st.markdown(f"""
    <div class="hero" style="box-shadow: 0 0 60px rgba({r},{g},{b},0.18), inset 0 0 120px rgba({r},{g},{b},0.10);">
      <p class="title" style="color: rgba({r},{g},{b},0.95);">
        {decision} <span class="pill">{site_name}</span> <span class="pill">UTC {selected_utc_hour.strftime('%Y-%m-%d %H:%M')}</span>
      </p>
      <p class="percent">{p_eff*100:.1f}%</p>
      <p class="sub">{sub}</p>
    </div>
    """, unsafe_allow_html=True)

    # 7) Map: heatmap around selected site + markers for all sites (color by p_eff per site)
    # Heatmap (synthetic fan around center; intensity = p_eff)
    np.random.seed(42)
    center_lat, center_lon = site["lat"], site["lon"]
    n_points, radius_km = 150, 10
    km2deg_lat = 1.0 / 111.0
    km2deg_lon = 1.0 / (111.0 * np.cos(np.deg2rad(center_lat)))
    rr = (np.random.rand(n_points) ** 0.5) * radius_km * (0.4 + 0.6 * p_eff)
    tt = np.random.rand(n_points) * 2 * np.pi
    lat_pts = center_lat + rr * np.sin(tt) * km2deg_lat
    lon_pts = center_lon + rr * np.cos(tt) * km2deg_lon
    heat_df = pd.DataFrame({"lat": lat_pts, "lon": lon_pts, "w": p_eff})

    # site markers with per-site effective probability (same p_space, different clouds)
    marker_rows = []
    for name, s in SITES.items():
        cfrac, _ = cloud_at(selected_utc_hour, s["site_id"])
        pe = effective_probability(p_space, cfrac)
        marker_rows.append({"name": name, "lat": s["lat"], "lon": s["lon"], "p": pe})
    markers_df = pd.DataFrame(marker_rows)
    markers_df["p_pct"] = (markers_df["p"] * 100).round(1)

    def prob2rgba(p):
        return [lerp(90, 120, p), lerp(120, 255, p), 180, lerp(120, 220, p)]
    markers_df["cr"], markers_df["cg"], markers_df["cb"], markers_df["ca"] = zip(*markers_df["p"].map(prob2rgba))

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_df,
        get_position='[lon, lat]',
        get_weight='w',
        radiusPixels=70,
        aggregation='MEAN',
    )
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=markers_df,
        get_position='[lon, lat]',
        get_radius=300,
        get_fill_color='[cr, cg, cb, ca]',
        pickable=True,
    )
    tooltip = {"text": "{name}\nEffective probability: {p_pct}%"}
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9, bearing=0, pitch=35)
    map_style = "mapbox://styles/mapbox/dark-v9" if os.environ.get("MAPBOX_API_KEY") else None
    deck = pdk.Deck(layers=[heat_layer, marker_layer], initial_view_state=view_state, map_style=map_style, tooltip=tooltip)
    st.pydeck_chart(deck, use_container_width=True)

    # 8) Best hour suggestion inside the window
    if best_hour != selected_utc_hour:
        st.success(
            f"Best hour in your {time_window_h}h window: **{best_hour.strftime('%Y-%m-%d %H:%M UTC')}** "
            f"→ **{best_p_eff*100:.1f}%** effective probability"
        )

    # 9) Inputs & features (for transparency)
    with st.expander("Inputs & features fed to the model"):
        st.write("Feature vector (engineered, single hour):")
        st.dataframe(X[FEATURES], use_container_width=True)
        st.caption(
            "bz_gsm/by_gsm (nT), flow_speed (km/s), proton_density (1/cc), ey (mV/m), dst (nT), "
            "f107 (sfu), kp10_* (Kp×10 aggregates), *_lag1 (1-hour lags, persistence)."
        )

    # 10) Model metrics (optional)
    report = load_model_report()
    with st.expander("Model performance (offline)"):
        if report:
            colA, colB, colC = st.columns(3)
            colA.metric("Accuracy", f"{report.get('accuracy', 0)*100:.1f}%")
            colB.metric("F1-score", f"{report.get('f1', 0):.3f}")
            colC.metric("ROC AUC", f"{report.get('roc_auc', 0):.3f}")
            if report.get("trained_until"):
                st.caption(f"Training data up to: {report['trained_until']}")
        else:
            st.info("Place a `model_report.json` with fields like `accuracy`, `f1`, `roc_auc` (and optional `trained_until`).")
    
    with st.expander("Model limitations & methodology"):
        st.info(
            "**Important Note:** The model's target ('aurora_visibility') is a proxy based on "
            "space weather rules (e.g., `bz_gsm < -5`), not on actual ground-truth camera observations. "
            "Therefore, the model predicts the *potential* for auroras under these conditions."
        )

    # 11) Plan B if NO-GO
    if not go:
        st.markdown("### What to do in Rovaniemi if aurora is unlikely?")
        ideas = [
            "❄️ **Arktikum** — science centre & museum.",
            "🌲 **Ounasvaara** — short evening hike with city views.",
            "🔥 **Sauna + cold plunge** — riverside sauna experiences.",
            "🎅 **Santa Claus Village** — evening stroll & souvenirs.",
            "☕ **Café stop** — pulla + coffee in the city centre.",
        ]
        st.markdown("\n".join(f"- {x}" for x in ideas))

# Footer
if not os.environ.get("MAPBOX_API_KEY"):
    st.caption("Tip: set MAPBOX_API_KEY for a richer basemap.")
st.caption("