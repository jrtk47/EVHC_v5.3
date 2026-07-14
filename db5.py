##
## EV Hosting Capacity Dashboard v5 (Hybrid Ensemble Implementation)
## PEA กฟก.2 Chonburi — Transformer-level EV Integration Tool
##

import io
import json
import zipfile
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════
# LANGUAGE / i18n
# ══════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "hdr_eyebrow":       {"EN": "PEA กฟก.2 · Chonburi Grid", "TH": "PEA กฟก.2 · จังหวัดชลบุรี"},
    "hdr_title":         {"EN": "Static Thermal Rating (STR) based EV Hosting Capacity Predictor"},
    "hdr_sub":           {"EN": "Hybrid Ensemble ML Engine (RidgeCV + XGBoost Residuals) · LOGO-CV Validated",
                           "TH": "โมเดล Hybrid Ensemble (RidgeCV + XGBoost Residuals) · ตรวจสอบยืนยันด้วย LOGO-CV"},
    "tab_fleet":         {"EN": "🗺️  Fleet Overview", "TH": "🗺️  ภาพรวมกลุ่มหม้อแปลง"},
    "tab_analysis":      {"EN": "🔍  Transformer Analysis", "TH": "🔍  วิเคราะห์หม้อแปลง"},
    "show_explanations": {"EN": "Show explanations", "TH": "แสดงคำอธิบาย"},
    "cfg_header":        {"EN": "⚙️ Configuration", "TH": "⚙️ การตั้งค่า"},
    "sb1_title":         {"EN": "1 · Ratings Master File (Tier 1)", "TH": "1 · ไฟล์ Master พิกัดหม้อแปลง (ลำดับ 1)"},
    "sb1_caption":       {"EN": "Upload once per session: a CSV or Excel master file with PEA No / DTMS No, kVA rating, and (optionally) Latitude/Longitude.",
                           "TH": "อัปโหลดครั้งเดียวต่อ session: ไฟล์ CSV หรือ Excel Master ที่มีเลข PEA No / DTMS No, พิกัด kVA และ (ถ้ามี) ละติจูด/ลองจิจูด"},
    "sb1_uploader":      {"EN": "Upload ratings master file (CSV/XLSX)", "TH": "อัปโหลดไฟล์ Master พิกัด (CSV/XLSX)"},
    "ratings_loaded":    {"EN": "Loaded {n} ratings ({c} with coordinates).", "TH": "โหลดพิกัดแล้ว {n} รายการ (มีพิกัด GPS {c} รายการ)"},
    "ratings_error":     {"EN": "Could not read reference file: {e}", "TH": "ไม่สามารถอ่านไฟล์อ้างอิงได้: {e}"},
    "ratings_count":     {"EN": "📋 {n} rating(s) loaded this session.", "TH": "📋 โหลดพิกัดแล้ว {n} รายการใน session นี้"},
    "sb2_title":         {"EN": "2 · Upload Load Curve(s)", "TH": "2 · อัปโหลดกราฟโหลด (Load Curve)"},
    "upload_mode_label": {"EN": "Mode", "TH": "โหมด"},
    "mode_single":       {"EN": "Single transformer", "TH": "หม้อแปลงเดี่ยว"},
    "mode_batch":        {"EN": "Batch (multiple files)", "TH": "หลายไฟล์พร้อมกัน"},
    "shift_utc":         {"EN": "Timestamps are UTC — convert to UTC+7", "TH": "เวลาในไฟล์เป็น UTC — แปลงเป็น UTC+7"},
    "single_uploader":   {"EN": "Upload DTMS transformer CSV", "TH": "อัปโหลดไฟล์ CSV หม้อแปลงจาก DTMS"},
    "batch_uploader":    {"EN": "Upload multiple DTMS CSVs", "TH": "อัปโหลดไฟล์ CSV จาก DTMS หลายไฟล์"},
    "sb3_title":         {"EN": "3 · Transformer Rating — Sidebar Fallback (Tier 2)", "TH": "3 · พิกัดหม้อแปลง — ค่าสำรอง (ลำดับ 2)"},
    "nameplate_label":   {"EN": "Nameplate kVA (used only if no reference match)", "TH": "พิกัด kVA (ใช้เฉพาะเมื่อไม่พบข้อมูลอ้างอิง)"},
    "match_caption":     {"EN": "**{pea_id}**: {val:.0f} kVA (from {matched_by})", "TH": "**{pea_id}**: {val:.0f} kVA (จาก {matched_by})"},
    "nomatch_caption":   {"EN": "**{pea_id}**: no reference match — using {val:.0f} kVA.", "TH": "**{pea_id}**: ไม่พบข้อมูลอ้างอิง — ใช้ {val:.0f} kVA"},
    "sb4_title":         {"EN": "4 · Day Type", "TH": "4 · ประเภทวัน"},
    "profile_eval":      {"EN": "Profile to evaluate", "TH": "รูปแบบวันที่ต้องการประเมิน"},
    "day_weekday":       {"EN": "Weekday", "TH": "วันธรรมดา"},
    "day_weekend":       {"EN": "Weekend / Holiday", "TH": "วันหยุด / วันหยุดนักขัตฤกษ์"},
    "sb5_title":         {"EN": "5 · EV Charging Start Time", "TH": "5 · เวลาเริ่มชาร์จ EV"},
    "peak_hour_label":   {"EN": "Peak charging hour", "TH": "ช่วงเวลาชาร์จสูงสุด"},
    "sb6_title":         {"EN": "6 · Charger Mix (% share per tier)", "TH": "6 · สัดส่วนประเภทเครื่องชาร์จ (%)"},
    "fast_charger_cap":  {"EN": "11 kW fast charger: **{w}%**", "TH": "11 kW fast chargerd: **{w}%**"},
    "weights_error":     {"EN": "Weights must not all be zero.", "TH": "สัดส่วนต้องไม่เป็นศูนย์ทั้งหมด"},
    "inspect_label":     {"EN": "📂 Inspect uploaded transformer", "TH": "📂 ตรวจสอบหม้อแปลงที่อัปโหลด"},
    "missing_cols":      {"EN": "Missing columns: {missing}", "TH": "ไม่พบคอลัมน์: {missing}"},
    "file_read_error":   {"EN": "Could not read file: {e}", "TH": "ไม่สามารถอ่านไฟล์ได้: {e}"},
    "exceeds_rating":    {"EN": "Peak apparent power ({peak:.1f} kVA) already exceeds TR rating ({rated:.0f} kVA). HC = 0 kW.",
                           "TH": "กำลังไฟฟ้าสูงสุด ({peak:.1f} kVA) เกินพิกัดหม้อแปลง ({rated:.0f} kVA) แล้ว — HC = 0 kW"},
    "card_predicted_hc": {"EN": "Predicted HC", "TH": "HC ที่ทำนายได้"},
    "card_hc_sub":       {"EN": "daily, time-distributed (DNOA)", "TH": "รายวัน แบบกระจายตามเวลา (DNOA)"},
    "card_chargers_daily":       {"EN": "Chargers — daily", "TH": "เครื่องชาร์จ — รายวัน"},
    "card_chargers_daily_sub":   {"EN": "spread across charging windows", "TH": "กระจายตามช่วงเวลาชาร์จ"},
    "card_chargers_sim":         {"EN": "Chargers — simultaneous", "TH": "เครื่องชาร์จ — พร้อมกัน"},
    "card_chargers_sim_sub":     {"EN": "drawing power at the same instant", "TH": "ดึงกำลังไฟฟ้าพร้อมกันในขณะเดียวกัน"},
    "card_headroom":     {"EN": "Capacity headroom", "TH": "Capacity Headroom"},
    "card_headroom_sub": {"EN": "TR rating − peak S_TOT", "TH": "พิกัดหม้อแปลง − S_TOT สูงสุด"},
    "section_daily_profile":     {"EN": "Daily load profile", "TH": "กราฟโหลดรายวัน"},
    "section_hc_scenario":       {"EN": "HC by charging scenario", "TH": "HC ตามสถานการณ์การชาร์จ"},
    "section_residential":       {"EN": "Residential load characteristics", "TH": "ลักษณะโหลดที่อยู่อาศัย"},
    "section_fleet_map":         {"EN": "Fleet map", "TH": "แผนที่กลุ่มหม้อแปลง"},
    "section_fleet_table":       {"EN": "📋 Transformer table (edit / add rows)", "TH": "📋 ตารางหม้อแปลง (แก้ไข / เพิ่มแถว)"},
    "mc_card_hc_p5":     {"EN": "Monte Carlo HC (5th pct)", "TH": "Monte Carlo HC (เปอร์เซ็นไทล์ที่ 5)"},
    "mc_card_hybrid_raw":{"EN": "Hybrid Engine (raw)", "TH": "โมเดล Hybrid (ค่าดิบ)"},
    "mc_card_diff":      {"EN": "Difference", "TH": "ผลต่าง"},
    "empty_state":       {"EN": "Upload a transformer load curve to begin", "TH": "อัปโหลดกราฟโหลดหม้อแปลงเพื่อเริ่มต้น"},
    "footer":            {"EN": "PEA กฟก.2 EV Hosting Capacity Tool | Smart Grid Division", "TH": "PEA กฟก.2 EV Hosting Capacity Tool | Smart Grid Division"},
}


def t(key: str, lang: str, **kwargs) -> str:
    """Translate a UI string; falls back to EN if key/lang missing."""
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(lang) or entry.get("EN") or key
    return text.format(**kwargs) if kwargs else text

st.set_page_config(
    page_title="EV Hosting Capacity — PEA กฟก.2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Palette ─────────────────────────────────────────────────────
BLUE     = "#1a56db"
BLUE_LT  = "#e8f0fe"
AMBER    = "#d97706"
AMBER_LT = "#fef3c7"
GREEN    = "#059669"
GREEN_LT = "#f9fff7"
RED      = "#dc2626"
RED_LT   = "#fff3f3"
SLATE    = "#1e293b"
SLATE_MID= "#475569"
BORDER   = "#e2e8f0"
BG_SOFT  = "#f8fafc"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans Thai', sans-serif; color: {SLATE}; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }}
section[data-testid="stMain"] .block-container {{ padding-top: 3.5rem; }}

.hdr {{ background:{SLATE}; border-radius:10px; padding:32px 28px; margin-bottom:20px; }}
.hdr-eyebrow {{ font-size:10px; font-weight:600; letter-spacing:0.15em; text-transform:uppercase; color:#94a3b8; margin-bottom:2px; }}
.hdr-title {{ font-size:22px; font-weight:700; color:#f8fafc; line-height:1.2; }}
.hdr-sub {{ font-size:12px; color:#64748b; margin-top:3px; font-family:'IBM Plex Mono',monospace; }}

.stTabs [data-baseweb="tab-list"] {{ gap:4px; background:{BG_SOFT}; padding:4px; border-radius:10px; border:1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ height:40px; border-radius:8px; padding:0 18px; font-weight:600; font-size:13.5px; color:{SLATE_MID}; }}
.stTabs [aria-selected="true"] {{ background:white; color:{BLUE}; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}

.card {{ background:white; border:1px solid {BORDER}; border-radius:8px; padding:16px 20px; }}
.card-green {{ background:{GREEN_LT}; border:1px solid {GREEN}; border-radius:8px; padding:16px 20px; }}
.card-red   {{ background:{RED_LT};   border:1px solid {RED};   border-radius:8px; padding:16px 20px; }}
.card-label {{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:{SLATE_MID}; margin-bottom:6px; }}
.card-value {{ font-size:28px; font-weight:700; color:{SLATE}; line-height:1; }}
.card-unit {{ font-size:13px; color:{SLATE_MID}; margin-left:3px; font-weight:400; }}
.card-sub {{ font-size:11px; color:{SLATE_MID}; margin-top:4px; }}

.badge {{ display:inline-block; font-size:13px; font-weight:700; border-radius:6px; padding:5px 14px; margin-right:10px; }}
.badge-green  {{ background:{GREEN_LT};  color:{GREEN}; }}
.badge-yellow {{ background:{AMBER_LT};  color:{AMBER}; }}
.badge-red    {{ background:{RED_LT};    color:{RED};   }}

.section-lbl {{ font-size:10px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:{BLUE}; border-bottom:1.5px solid {BLUE_LT}; padding-bottom:4px; margin:20px 0 10px 0; }}
.info-box {{ background:{BLUE_LT}; border-left:3px solid {BLUE}; border-radius:4px; padding:10px 14px; font-size:12px; color:{SLATE}; margin-top:8px; font-family:'IBM Plex Mono',monospace; }}
.empty {{ text-align:center; padding:80px 32px; color:{SLATE_MID}; }}
.empty-icon {{ font-size:40px; margin-bottom:12px; }}
.empty-title {{ font-size:17px; font-weight:600; color:{SLATE}; margin-bottom:6px; }}
.empty-sub {{ font-size:13px; }}
.sb-lbl {{ font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:{SLATE_MID}; margin:18px 0 6px 0; border-bottom:1px solid {BORDER}; padding-bottom:3px; }}

.fleet-stat {{ background:white; border:1px solid {BORDER}; border-radius:8px; padding:14px 18px; text-align:center; }}
.fleet-stat-n {{ font-size:24px; font-weight:700; }}
.fleet-stat-l {{ font-size:11px; color:{SLATE_MID}; text-transform:uppercase; letter-spacing:0.06em; margin-top:2px; }}
</style>
""", unsafe_allow_html=True)

CHARGER_OPTIONS_KW = [3.7, 7.4, 11.0]
BASE_AVG_KW        = 7.55
EV_COUNT_CAP       = 100
SPIKE_CLIP_KW      = 1000.0
EV_CHARGER_PF      = 0.98

PEAK_HOURS = {
    "14:00  Afternoon — off-peak base load":  14.0,
    "17:00  Late afternoon":                  17.0,
    "18:30  Evening arrival (Uncontrolled)":  18.5,
    "20:00  Evening — delayed start":         20.0,
    "22:00  Late night (Smart Charging)":     22.0,
}

EXPECTED_TIMES = [
    pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}:00").time()
    for h in range(24) for m in (0, 15, 30, 45)
]

CHONBURI_LAT, CHONBURI_LON = 0, 0

@st.cache_resource
def load_artifacts():
    # Production upgraded to a Hybrid Model consisting of a Base Estimator (Ridge)
    # plus an explicit Non-Linear Correction Estimator (XGBoost Residual Predictor)
    model = joblib.load(BASE_DIR / 'v5_model.pkl')
    # FIXED: filename now matches what Cell G actually saves (joblib.dump(scaler_final, 'v5_scaler.pkl')).
    # Was previously 'v5_feature_scaler.pkl', which does not exist -> load_artifacts() always failed.
    scaler = joblib.load(BASE_DIR / 'v5_scaler.pkl')
    from xgboost import XGBRegressor
    xgb_residual = XGBRegressor()
    xgb_residual.load_model(BASE_DIR / 'v5_residual.json')

    # Optional: cluster centroids exported by Cell D, used for the residential-
    # shape check. Not a hard dependency — if missing (older training run),
    # the dashboard falls back to a simpler heuristic check further down.
    cluster_artifact = None
    cluster_path = BASE_DIR / 'v5_cluster_centroids.json'
    if cluster_path.exists():
        try:
            with open(cluster_path, 'r') as f:
                cluster_artifact = json.load(f)
        except Exception:
            cluster_artifact = None

    return model, xgb_residual, scaler, cluster_artifact

try:
    model, xgb_residual, scaler, cluster_artifact = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Could not load hybrid engine files: {e}\n\nPlease place all three artifacts inside the folder.")
    st.stop()


def _normalize_dtms_columns(columns) -> dict:
    """
    Maps known alternate column names to the canonical ones the pipeline
    expects, so raw exports in a different shape don't need manual
    pre-cleansing before upload.

    Currently handles: DTMS/InfluxDB (Flux) monthly bulk exports, which use
    '_time' instead of 'timestamp' and include extra metadata columns
    (result, table, _start, _stop, dtmsid, per-phase P_A/P_B/P_C, etc.).
    Those extra columns are NOT dropped here — they simply pass through
    unused, since clean_and_resample() only ever selects P_TOT/S_TOT by name.

    Add more aliases here if PEA exports change format again; this is the
    single place that needs updating, rather than every call site.
    """
    rename_map = {}
    if 'timestamp' not in columns:
        for alt in ('_time', 'Time', 'time', 'Timestamp', 'DATETIME', 'datetime'):
            if alt in columns:
                rename_map[alt] = 'timestamp'
                break
    return rename_map


@st.cache_data(show_spinner=False)
def clean_and_resample(file_bytes: bytes, shift_utc: bool) -> pd.DataFrame:
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df_raw = df_raw.rename(columns=_normalize_dtms_columns(df_raw.columns))
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if shift_utc:
        # .dt.tz_localize(None) strips tz info while keeping the same wall-clock
        # values -- correct here because DTMS/InfluxDB exports are tagged UTC
        # ("+00:00"), so this is equivalent to a UTC->naive conversion, and the
        # +7h below then produces Thailand local time. If a future export uses
        # a non-UTC offset, this step would need tz_convert('UTC') first.
        df['timestamp'] = df['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=7)
    df = df.sort_values('timestamp')
    for col in ['P_TOT', 'S_TOT']:
        df.loc[df[col] <= 0, col] = np.nan
        med   = df[col].rolling(window=7, center=True, min_periods=1).median()
        spike = (df[col] > med * 1.5) & ((df[col] - med) > 20.0)
        df.loc[spike, col] = np.nan
        df[col] = df[col].interpolate(method='linear').bfill().ffill()
        df[col] = df[col].clip(upper=SPIKE_CLIP_KW)
    df_ts = df.set_index('timestamp')[['P_TOT', 'S_TOT']].resample('15min').mean()
    for col in ['P_TOT', 'S_TOT']:
        df_ts.loc[df_ts[col] <= 0, col] = np.nan
        df_ts[col] = df_ts[col].interpolate(method='linear').bfill().ffill()
    return df_ts


def extract_features(df_ts: pd.DataFrame, tr_rated_kva: float, is_weekend: int) -> dict:
    df = df_ts.copy()
    df['time'] = df.index.time
    profile_kva = df.groupby('time')['S_TOT'].median().reindex(EXPECTED_TIMES).interpolate(method='linear').bfill().ffill().values
    profile_kw  = df.groupby('time')['P_TOT'].median().reindex(EXPECTED_TIMES).interpolate(method='linear').bfill().ffill().values
    peak_kva    = float(np.max(profile_kva))
    headroom    = tr_rated_kva - peak_kva
    slot_17     = list(EXPECTED_TIMES).index(datetime.time(17, 0))
    slot_20     = list(EXPECTED_TIMES).index(datetime.time(20, 0))
    evening_ramp = float(profile_kva[slot_20] - profile_kva[slot_17])
    day_load     = float(np.mean(profile_kva[36:64]))
    night_load   = float(np.mean(profile_kva[72:88]))
    nd_ratio     = night_load / day_load if day_load > 0.5 else 0.0
    df['_date'] = df.index.date
    daily_pivot = (
        df.groupby(['_date', 'time'])['S_TOT']
        .mean()
        .unstack(level='time')
        .reindex(columns=EXPECTED_TIMES)
    )
    MIN_VALID_SLOTS = 72
    daily_pivot = daily_pivot[daily_pivot.notna().sum(axis=1) >= MIN_VALID_SLOTS]
    
    if daily_pivot.empty:
        profile_kva_peak_day = profile_kva.copy()
        profile_kva_min_day = profile_kva.copy()
        peak_day_label = 'N/A'
        min_day_label = 'N/A'
    else:
        peak_day_idx = daily_pivot.sum(axis=1).idxmax()
        min_day_idx  = daily_pivot.sum(axis=1).idxmin()
        profile_kva_peak_day = daily_pivot.loc[peak_day_idx].interpolate(method='linear').bfill().ffill().values
        profile_kva_min_day  = daily_pivot.loc[min_day_idx].interpolate(method='linear').bfill().ffill().values
        peak_day_label = str(peak_day_idx)
        min_day_label = str(min_day_idx)

    return {
        'Capacity_Headroom_kVA': round(headroom, 2),
        'Evening_Surge_kVA':      round(evening_ramp, 2),
        'Night_Day_Ratio':       round(nd_ratio, 3),
        'Is_Weekend':            is_weekend,
        '_peak_kva':             round(peak_kva, 2),
        '_profile_kva':          profile_kva,
        '_profile_kw':           profile_kw,
        '_profile_kva_peak_day': profile_kva_peak_day,
        '_profile_kva_min_day':  profile_kva_min_day,
        '_peak_day_label':       peak_day_label,
        '_min_day_label':        min_day_label,
    }


def _compute_normalized_shape(df_ts: pd.DataFrame) -> np.ndarray:
    """
    Reproduces the exact normalization used in Cell D so a single uploaded
    profile can be compared against the exported cluster centroids:
    time-of-day MEAN of P_TOT (not median — matches training), reindexed to
    the 96 15-min slots, then scaled by that profile's own 1st/99th
    percentile and clipped to [0, 1].
    """
    df = df_ts.copy()
    df['time'] = df.index.time
    daily = df.groupby('time')['P_TOT'].mean().reindex(EXPECTED_TIMES).interpolate(method='linear').bfill().ffill()
    mn, mx = daily.quantile(0.01), daily.quantile(0.99)
    if mx - mn < 1e-6:
        return None
    return ((daily - mn) / (mx - mn)).clip(lower=0, upper=1).values


def check_residential_shape(df_ts: pd.DataFrame, feats: dict) -> dict:
    """
    Flags uploads whose load-curve SHAPE doesn't match the residential
    cluster the Hybrid Ensemble was trained on.

    Preferred path (cluster_artifact available): reproduces Cell D's
    normalization on the uploaded profile, computes MSE distance to every
    exported K-Means centroid, and checks (a) whether the nearest centroid is
    the residential cluster, and (b) whether the distance to it is within the
    anomaly threshold used to reject outliers during training. This is a
    direct re-application of the same clustering space, not a re-fit — it
    will not always agree with what a fresh K-Means run would produce if the
    fleet composition has changed a lot since training.

    Fallback path (no cluster_artifact, e.g. older training run without the
    exported centroids): a lightweight heuristic on Night/Day ratio and
    evening surge, the same two signals Cell D used to auto-select the
    residential cluster in the first place, but without a real distance
    measure — directionally consistent, not a guarantee.
    """
    if cluster_artifact is not None:
        shape_vec = _compute_normalized_shape(df_ts)
        if shape_vec is None:
            return {'is_likely_residential': True, 'reasons': [], 'method': 'skipped_flat_profile'}

        centroids = np.array(cluster_artifact['centroids'])
        residential_idx = cluster_artifact['residential_cluster_index']
        threshold = cluster_artifact['anomaly_mse_threshold']

        distances = np.mean((centroids - shape_vec) ** 2, axis=1)
        nearest_idx = int(np.argmin(distances))
        dist_to_residential = float(distances[residential_idx])

        reasons = []
        if nearest_idx != residential_idx:
            reasons.append(
                f"รูปแบบโหลด (normalized shape) ใกล้เคียงกับ Cluster {nearest_idx} มากที่สุด "
                f"ไม่ใช่ Cluster {residential_idx} ซึ่งเป็นกลุ่มที่อยู่อาศัยที่ใช้เทรนโมเดล "
                f"(MSE ถึง cluster ที่อยู่อาศัย = {dist_to_residential:.4f})"
            )
        elif dist_to_residential > threshold:
            reasons.append(
                f"รูปแบบโหลดใกล้เคียง Cluster ที่อยู่อาศัยที่สุดก็จริง แต่ระยะห่าง (MSE = "
                f"{dist_to_residential:.4f}) เกินเกณฑ์ปกติที่ใช้คัดหม้อแปลง anomaly ตอนเทรน "
                f"(threshold = {threshold:.4f}) — รูปร่างโหลดผิดปกติกว่าหม้อแปลงที่อยู่อาศัยทั่วไปในชุดข้อมูล"
            )

        return {
            'is_likely_residential': len(reasons) == 0,
            'reasons': reasons,
            'method': 'centroid_distance',
            'nearest_cluster': nearest_idx,
            'residential_cluster': residential_idx,
            'distance_to_residential': dist_to_residential,
            'anomaly_threshold': threshold,
        }

    # ---- Fallback heuristic (no centroid artifact available) ----
    reasons = []
    nd_ratio = feats['Night_Day_Ratio']
    evening_surge = feats['Evening_Surge_kVA']

    if nd_ratio < 1.0:
        reasons.append(
            f"Night/Day ratio = {nd_ratio:.2f} (< 1.0) — โหลดกลางวัน (09:00–16:00) "
            f"สูงกว่าหรือเท่ากับโหลดกลางคืน (18:00–22:00) ซึ่งตรงข้ามกับหม้อแปลงที่อยู่อาศัย "
            f"ทั้ง 22 ลูกที่ใช้เทรนโมเดล (มี Night/Day ratio > 1.0 ทุกลูก)"
        )
    if evening_surge <= 0:
        reasons.append(
            f"Evening surge (17:00→20:00) = {evening_surge:.1f} kVA (≤ 0) — ไม่มีโหลดไต่ขึ้น "
            f"ช่วงเย็นตามแบบฉบับที่อยู่อาศัย อาจเป็นโหลดเชิงพาณิชย์/อุตสาหกรรมที่คงที่หรือพีคช่วงกลางวัน"
        )

    return {'is_likely_residential': len(reasons) == 0, 'reasons': reasons, 'method': 'heuristic_fallback'}


def predict_and_adjust(features: dict, peak_hour: float, user_weights: list, tr_rated_kva: float, model, xgb_residual, scaler) -> tuple:
    user_avg_kw = sum(c * w for c, w in zip(CHARGER_OPTIONS_KW, user_weights))
    if user_avg_kw <= 0:
        user_avg_kw = BASE_AVG_KW

    peak_x_nightday = peak_hour * features['Night_Day_Ratio']
    peak_x_headroom = peak_hour * features['Capacity_Headroom_kVA']

    feat_vec = pd.DataFrame([[
        features['Capacity_Headroom_kVA'],
        features['Evening_Surge_kVA'],
        features['Night_Day_Ratio'],
        features['Is_Weekend'],
        peak_hour,
        user_avg_kw,
        peak_x_nightday,
        peak_x_headroom,
    ]], columns=['Capacity_Headroom_kVA', 'Evening_Surge_kVA', 'Night_Day_Ratio', 'Is_Weekend',
                 'Peak_Start_Hour', 'Avg_Charger_kW', 'Peak_x_NightDay', 'Peak_x_Headroom'])

    # Hybrid inference pipeline execution
    X_scaled = scaler.transform(feat_vec)
    hc_base = float(model.predict(X_scaled)[0])
    hc_correction = float(xgb_residual.predict(X_scaled)[0])

    # Blended output with structural lower-bound constraint
    hc_raw = max(hc_base + hc_correction, 0.0)

    # NOTE: There is currently no separate post-hoc adjustment stage (e.g. a
    # nameplate/kVA cap) — that logic was intentionally removed (see project
    # notes: "Removing the nameplate cap was necessary because it conflated
    # time-distributed HC with instantaneous headroom"). hc_adj is therefore
    # identical to hc_raw today. The tuple is kept (rather than a single
    # return value) so callers don't need to change if a distinct adjustment
    # step is reintroduced later — but if no such step is planned, prefer
    # collapsing this to a single return value to avoid implying a
    # distinction that doesn't exist.
    hc_adj = hc_raw
    return hc_raw, hc_adj


def run_scenario_monte_carlo(base_profile_kva, tr_rating_kva, weights, peak_hour, n_iter=200):
    base_profile_kva = np.copy(base_profile_kva)
    simulated_hc_kw = []
    traces = []
    ev_load_profiles = []

    for _ in range(n_iter):
        current_ev_load_kva = np.zeros(96)
        current_added_kw = 0.0
        ev_count = 0
        kw_trace = [0.0]
        while True:
            charger_kw  = np.random.choice(CHARGER_OPTIONS_KW, p=weights)
            charger_kva = charger_kw / EV_CHARGER_PF

            start_hour = np.clip(np.random.normal(peak_hour, 2.0), 0, 23.99)
            start_slot = int(start_hour * 4)

            duration_hours = np.clip(np.random.normal(2.0, 0.5), 0.5, 8.0)
            slots_needed   = int(duration_hours * 4)

            new_car_kva = np.zeros(96)
            for j in range(slots_needed):
                idx = (start_slot + j) % 96
                new_car_kva[idx] += charger_kva

            test_ev_load_kva = current_ev_load_kva + new_car_kva
            if np.max(base_profile_kva + test_ev_load_kva) > tr_rating_kva:
                break

            current_ev_load_kva = test_ev_load_kva
            current_added_kw += charger_kw
            ev_count += 1
            kw_trace.append(current_added_kw)

            if ev_count > EV_COUNT_CAP:
                break

        simulated_hc_kw.append(current_added_kw)
        traces.append(kw_trace)
        ev_load_profiles.append(current_ev_load_kva)

    median_hc = float(np.median(simulated_hc_kw))
    rep_idx = int(np.argmin([abs(hc - median_hc) for hc in simulated_hc_kw]))

    return {
        'hc_p5':   round(float(np.percentile(simulated_hc_kw, 5)), 2),
        'hc_mean': round(float(np.mean(simulated_hc_kw)), 2),
        'hc_std':  round(float(np.std(simulated_hc_kw)), 2),
        'samples': simulated_hc_kw,
        'rep_trace':        traces[rep_idx],
        'rep_ev_profile':   ev_load_profiles[rep_idx],
        'rep_hc':           round(simulated_hc_kw[rep_idx], 2),
    }


def classify(hc_kw: float) -> tuple:
    if hc_kw > 22:     return "Ready",    "badge-green"
    elif hc_kw >= 7.4: return "Monitor",  "badge-yellow"
    else:              return "Critical", "badge-red"


def plot_load_curve(profile_kva, profile_kw, tr_rated_kva, hc_adj, peak_hour,
                     profile_kva_peak_day=None, profile_kva_min_day=None,
                     peak_day_label='', min_day_label=''):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(96)
    if profile_kva_peak_day is not None and profile_kva_min_day is not None:
        ax.fill_between(x, profile_kva_min_day, profile_kva_peak_day, alpha=0.07, color=BLUE, zorder=0)
        ax.plot(x, profile_kva_peak_day, color=AMBER, lw=0.9, linestyle='--', alpha=0.5, label=f'Peak day ({peak_day_label})', zorder=2)
        ax.plot(x, profile_kva_min_day,  color=GREEN, lw=0.9, linestyle='--', alpha=0.5, label=f'Min day ({min_day_label})', zorder=2)
    ax.fill_between(x, profile_kva, alpha=0.08, color=BLUE)
    ax.plot(x, profile_kva, color=BLUE, lw=2, label='Median S_TOT (kVA)', zorder=3)
    ax.plot(x, profile_kw,  color='#94a3b8', lw=1.2, linestyle=':', label='Median P_TOT (kW)', zorder=2)
    ax.axhline(tr_rated_kva, color=RED, lw=1.5, linestyle='--', label=f'TR Rating ({tr_rated_kva:.0f} kVA)', zorder=4)
    peak_val   = float(np.max(profile_kva))
    band_color = GREEN if hc_adj > 22 else (AMBER if hc_adj >= 7.4 else RED)
    ax.fill_between(x, [peak_val]*96, [min(peak_val + hc_adj, tr_rated_kva)]*96, alpha=0.25, color=band_color, label=f'HC headroom ({hc_adj:.0f} kW)', zorder=1)
    ph_slot = int(peak_hour * 4)
    ax.axvspan(ph_slot, min(ph_slot + 8, 95), alpha=0.10, color=AMBER, label='Avg EV charging window')
    ax.set_xlim(0, 95)
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96])
    ax.set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=8)
    ax.set_ylabel('Power (kVA / kW)', fontsize=9)
    ax.set_title('Median daily load profile', fontsize=10, fontweight='600')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=1.0)
    return fig


def plot_residential_profile(profile_kva, profile_kw,
                              profile_kva_peak_day=None, profile_kva_min_day=None,
                              peak_day_label='', min_day_label=''):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(96)
    if profile_kva_peak_day is not None and profile_kva_min_day is not None:
        ax.fill_between(x, profile_kva_min_day, profile_kva_peak_day, alpha=0.07, color=BLUE, zorder=0)
        ax.plot(x, profile_kva_peak_day, color=AMBER, lw=0.9, linestyle='--', alpha=0.5, label=f'Peak day ({peak_day_label})', zorder=2)
        ax.plot(x, profile_kva_min_day,  color=GREEN, lw=0.9, linestyle='--', alpha=0.5, label=f'Min day ({min_day_label})', zorder=2)
    ax.fill_between(x, profile_kva, alpha=0.10, color=BLUE, zorder=1)
    ax.plot(x, profile_kva, color=BLUE, lw=2, label='Median S_TOT (kVA)', zorder=3)
    ax.plot(x, profile_kw,  color='#94a3b8', lw=1.2, linestyle=':', label='Median P_TOT (kW)', zorder=2)
    
    max_envelope_val = max(np.nanmax(profile_kva), np.nanmax(profile_kw))
    if profile_kva_peak_day is not None:
        max_envelope_val = max(max_envelope_val, np.nanmax(profile_kva_peak_day))
    ax.set_ylim(0, max_envelope_val * 1.25 if max_envelope_val > 0 else 50)
    
    SLOT_17, SLOT_20 = 68, 80
    ax.axvspan(SLOT_17, SLOT_20, alpha=0.08, color=AMBER, zorder=0)
    ramp_start_kva = float(profile_kva[SLOT_17])
    ramp_end_kva   = float(profile_kva[SLOT_20])
    ramp_kva       = ramp_end_kva - ramp_start_kva
    arrow_x = 74
    ax.annotate('', xy=(arrow_x, ramp_end_kva), xytext=(arrow_x, ramp_start_kva),
                arrowprops=dict(arrowstyle='<->', color=AMBER, lw=1.4), zorder=5)
    ax.text(arrow_x + 1.2, (ramp_start_kva + ramp_end_kva) / 2, f'+{ramp_kva:.1f} kVA',
            fontsize=7.5, color=AMBER, va='center', fontweight='600', zorder=5)
    
    ax.set_xlim(0, 95)
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96])
    ax.set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=8)
    ax.set_ylabel('Power (kVA / kW)', fontsize=9)
    ax.set_title('Residential load characteristics — data-scaled', fontsize=10, fontweight='600')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=1.0)
    return fig


def plot_scenario_bar(features, tr_rated_kva, user_weights, model, xgb_residual, scaler):
    results = []
    for label, ph in PEAK_HOURS.items():
        _, hc = predict_and_adjust(features, ph, user_weights, tr_rated_kva, model, xgb_residual, scaler)
        results.append((label.split('  ')[0], hc))
    hours, hcs = zip(*results)
    colors = [GREEN if v > 22 else (AMBER if v >= 7.4 else RED) for v in hcs]
    fig, ax = plt.subplots(figsize=(6, 3.0))
    bars = ax.bar(hours, hcs, color=colors, width=0.55, edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(22,  color=GREEN, lw=1, linestyle='--', alpha=0.6)
    ax.axhline(7.4, color=RED,   lw=1, linestyle='--', alpha=0.6)
    for bar, val in zip(bars, hcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{val:.0f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='600', color=SLATE)
    ax.set_ylabel('Predicted HC (kW)', fontsize=9)
    ax.set_title('HC across all charging scenarios', fontsize=10, fontweight='600')
    ax.set_ylim(0, max(hcs) * 1.3 if max(hcs) > 0 else 30)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=1.0)
    return fig


def parse_ratings_reference(file_bytes: bytes, filename: str = "") -> tuple:
    """Parse a ratings/coordinates master file (CSV or Excel).

    Recognizes several column layouts:
      - Simple:           Name, Rated_kVA[, Latitude, Longitude]
      - PEA export (EN):  PEA No, TR. Rated[, DTMS No][, Latitude, Longitude]
      - PEA export (EN):  DTMS No, TR. Rated[, PEA No][, Latitude, Longitude]
      - PEA GIS master (TH, e.g. DS_Transformer export):
            PEANO หม้อแปลง, ค่าพิกัด kVA หม้อแปลง[, รหัส TAG][, PEANO (ADS)],
            LATITUDE, LONGITUDE

    Every ID variant found for a row (PEA No, DTMS/TAG No, ADS No, ...) is
    indexed to the same kVA rating and, if present, the same coordinates —
    so a lookup by any of those IDs will match.

    Returns (ratings: dict[str, float], coords: dict[str, tuple[float, float]],
             name_col: str, kva_col: str)
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]

    if "Name" in df.columns and "Rated_kVA" in df.columns:
        name_col, kva_col, other_cols = "Name", "Rated_kVA", []
    elif "PEANO หม้อแปลง" in df.columns and "ค่าพิกัด kVA หม้อแปลง" in df.columns:
        name_col, kva_col = "PEANO หม้อแปลง", "ค่าพิกัด kVA หม้อแปลง"
        other_cols = [c for c in ("รหัส TAG", "PEANO (ADS)") if c in df.columns]
    elif "PEA No" in df.columns and "TR. Rated" in df.columns:
        name_col, kva_col = "PEA No", "TR. Rated"
        other_cols = ["DTMS No"] if "DTMS No" in df.columns else []
    elif "DTMS No" in df.columns and "TR. Rated" in df.columns:
        name_col, kva_col = "DTMS No", "TR. Rated"
        other_cols = ["PEA No"] if "PEA No" in df.columns else []
    else:
        raise ValueError(f"Couldn't find ID/rating columns. Found: {list(df.columns)}")

    lat_col = next((c for c in df.columns if c.strip().upper() == "LATITUDE"), None)
    lon_col = next((c for c in df.columns if c.strip().upper() == "LONGITUDE"), None)

    ratings, coords = {}, {}
    for _, r in df.iterrows():
        try:
            kva_val = float(r[kva_col])
        except (TypeError, ValueError):
            continue

        latlon = None
        if lat_col and lon_col and pd.notna(r.get(lat_col)) and pd.notna(r.get(lon_col)):
            try:
                latlon = (float(r[lat_col]), float(r[lon_col]))
            except (TypeError, ValueError):
                latlon = None

        for col in [name_col] + other_cols:
            id_val = str(r[col]).strip()
            if id_val and id_val.lower() != 'nan':
                ratings[id_val] = kva_val
                if latlon is not None:
                    coords[id_val] = latlon

    return ratings, coords, name_col, kva_col


def lookup_rated_kva(name: str, config_panel_value: float) -> tuple:
    ratings = st.session_state.get("ratings_reference", {})
    if name in ratings:
        return float(ratings[name]), "master file"
    return config_panel_value, None


def lookup_coords(name: str):
    """Returns (lat, lon) tuple from the uploaded master file, or None if unknown."""
    coords = st.session_state.get("coords_reference", {})
    return coords.get(name)


STANDARD_KVA = [30, 50, 100, 160, 250, 315, 400, 500, 630, 800, 1000, 1250]


def set_tr_rated(value: float):
    st.session_state.tr_rated_input = value


def init_fleet():
    if "fleet" not in st.session_state:
        st.session_state.fleet = pd.DataFrame(
            columns=["Name", "Latitude", "Longitude", "Rated_kVA", "HC_kW", "Status", "Rated_Verified"]
        ).astype({
            "Name": "object", "Latitude": "float64", "Longitude": "float64",
            "Rated_kVA": "float64", "HC_kW": "float64", "Status": "object",
            "Rated_Verified": "bool",
        })
    if "jump_to_name" not in st.session_state:
        st.session_state.jump_to_name = None
    if "fleet_files" not in st.session_state:
        st.session_state.fleet_files = {}


def upsert_fleet_result(name, tr_rated, lat, lon, hc_adj, status, verified=False):
    fleet = st.session_state.fleet
    mask = fleet["Name"] == name
    if mask.any():
        fleet.loc[mask, ["Rated_kVA", "HC_kW", "Status", "Rated_Verified"]] = [tr_rated, round(hc_adj, 1), status, bool(verified)]
        if lat is not None:
            fleet.loc[mask, "Latitude"] = lat
        if lon is not None:
            fleet.loc[mask, "Longitude"] = lon
    else:
        new_row = pd.DataFrame([{
            "Name": name, "Latitude": lat if lat is not None else np.nan,
            "Longitude": lon if lon is not None else np.nan,
            "Rated_kVA": tr_rated, "HC_kW": round(hc_adj, 1), "Status": status,
            "Rated_Verified": bool(verified),
        }])
        st.session_state.fleet = pd.concat([fleet, new_row], ignore_index=True)


init_fleet()

if "lang" not in st.session_state:
    st.session_state.lang = "EN"
if "page" not in st.session_state:
    st.session_state.page = "fleet"

lang = st.session_state.lang

with st.sidebar:
    st.markdown(f"""
<div class="hdr">
  <div class="hdr-eyebrow">{t("hdr_eyebrow", lang)}</div>
  <div class="hdr-title">{t("hdr_title", lang)}</div>
  <div class="hdr-sub">{t("hdr_sub", lang)}</div>
</div>
""", unsafe_allow_html=True)

    st.session_state.lang = st.radio(
        "Language / ภาษา", ["EN", "TH"],
        index=["EN", "TH"].index(st.session_state.lang),
        horizontal=True, label_visibility="collapsed",
    )
    st.toggle(t("show_explanations", st.session_state.lang), value=True, key="show_explanations")
    st.markdown("---")
    lang = st.session_state.lang

    # ── 1 · Fleet Overview ──────────────────────────────────────
    if st.button(t("tab_fleet", lang), width='stretch',
                 type=("primary" if st.session_state.page == "fleet" else "secondary")):
        st.session_state.page = "fleet"
        st.rerun()

    export_buf = io.BytesIO()
    with zipfile.ZipFile(export_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("fleet.csv", st.session_state.fleet.to_csv(index=False))
        manifest = {}
        for name, (raw_bytes, file_shift_utc) in st.session_state.fleet_files.items():
            fname = f"raw_csvs/{name}.csv"
            zf.writestr(fname, raw_bytes)
            manifest[name] = {"file": fname, "shift_utc": file_shift_utc}
        zf.writestr("manifest.json", json.dumps(manifest))

    with st.popover("💾 Save / Load Fleet", use_container_width=True):
        st.download_button(
            "⬇ Export fleet (.zip)", data=export_buf.getvalue(),
            file_name="pea_fleet_export.zip", mime="application/zip",
            width='stretch',
        )

        st.markdown("---")
        imported_zip = st.file_uploader("Import fleet (.zip)", type=["zip"], key="_fleet_import")
        if imported_zip is not None and st.session_state.get("_last_imported_name") != imported_zip.name:
            st.session_state._last_imported_name = imported_zip.name
            with zipfile.ZipFile(io.BytesIO(imported_zip.getvalue())) as zf:
                if "fleet.csv" in zf.namelist():
                    st.session_state.fleet = pd.read_csv(io.BytesIO(zf.read("fleet.csv")))
                manifest = {}
                if "manifest.json" in zf.namelist():
                    manifest = json.loads(zf.read("manifest.json"))
                for name, info in manifest.items():
                    raw = zf.read(info["file"])
                    st.session_state.fleet_files[name] = (raw, info.get("shift_utc", True))
            st.success(f"Imported database.")
            st.rerun()

    names_available = [n for n in st.session_state.fleet["Name"].dropna().tolist() if str(n).strip()]
    if names_available:
        with st.popover("📍 Set coordinates", use_container_width=True):
            coord_name = st.selectbox("Transformer", names_available, key="_coord_pick_name")

            if "_coord_lat" not in st.session_state or "_coord_lon" not in st.session_state:
                row = st.session_state.fleet[st.session_state.fleet["Name"] == coord_name].iloc[0]
                st.session_state._coord_lat = float(row["Latitude"]) if pd.notna(row["Latitude"]) else CHONBURI_LAT
                st.session_state._coord_lon = float(row["Longitude"]) if pd.notna(row["Longitude"]) else CHONBURI_LON
                st.session_state._coord_last_pick = coord_name

            if st.session_state.get("_coord_last_pick") != coord_name:
                st.session_state._coord_last_pick = coord_name
                row = st.session_state.fleet[st.session_state.fleet["Name"] == coord_name].iloc[0]
                st.session_state._coord_lat = float(row["Latitude"]) if pd.notna(row["Latitude"]) else CHONBURI_LAT
                st.session_state._coord_lon = float(row["Longitude"]) if pd.notna(row["Longitude"]) else CHONBURI_LON

            with st.form("_coord_form", border=False):
                c1, c2 = st.columns(2)
                with c1:
                    new_lat = st.number_input("Latitude", value=st.session_state._coord_lat, format="%.5f")
                with c2:
                    new_lon = st.number_input("Longitude", value=st.session_state._coord_lon, format="%.5f")

                if st.form_submit_button("Apply", width='stretch'):
                    st.session_state._coord_lat = new_lat
                    st.session_state._coord_lon = new_lon
                    mask = st.session_state.fleet["Name"] == coord_name
                    st.session_state.fleet.loc[mask, "Latitude"] = new_lat
                    st.session_state.fleet.loc[mask, "Longitude"] = new_lon
                    st.success(f"Coordinates updated for {coord_name}.")
                    st.rerun()

    # ── 2 · Transformer Analysis ────────────────────────────────
    if st.button(t("tab_analysis", lang), width='stretch',
                 type=("primary" if st.session_state.page == "analysis" else "secondary")):
        st.session_state.page = "analysis"
        st.rerun()

    st.markdown("---")

prefill = st.session_state.get("jump_to_name")
prefill_row = None
if prefill is not None:
    match = st.session_state.fleet[st.session_state.fleet["Name"] == prefill]
    if not match.empty:
        prefill_row = match.iloc[0]

if "tr_rated_input" not in st.session_state:
    st.session_state.tr_rated_input = float(prefill_row["Rated_kVA"]) if (prefill_row is not None and pd.notna(prefill_row["Rated_kVA"])) else 160.0

# Pre-read the inspect-picker's persisted widget value (if any) *before* the
# sidebar renders the Nameplate kVA selectbox below. Streamlit updates
# st.session_state["_inspect_pick"] as soon as the user picks a new option,
# before the script reruns — so reading it here reflects the just-made
# selection. Applying a verified-match kVA override at this point lands in
# the SAME pass that renders the Nameplate kVA selectbox, avoiding the
# extra st.rerun() that used to be needed when this check ran later.
_early_inspect_pick = st.session_state.get("_inspect_pick")
_cached_names_early = sorted(st.session_state.fleet_files.keys())
if _early_inspect_pick in _cached_names_early and st.session_state.get("_last_inspect_pick") != _early_inspect_pick:
    st.session_state._last_inspect_pick = _early_inspect_pick
    _row = st.session_state.fleet[st.session_state.fleet["Name"] == _early_inspect_pick]
    if not _row.empty and pd.notna(_row.iloc[0]["Rated_kVA"]) and bool(_row.iloc[0].get("Rated_Verified", False)):
        st.session_state.tr_rated_input = float(_row.iloc[0]["Rated_kVA"])

with st.sidebar:
    with st.expander(t("cfg_header", lang), expanded=False):
        st.markdown(f'<div class="sb-lbl">{t("sb1_title", lang)}</div>', unsafe_allow_html=True)
        if st.session_state.show_explanations:
            st.caption(t("sb1_caption", lang))
        ref_upload = st.file_uploader(t("sb1_uploader", lang), type=["csv", "xlsx", "xls"], key="_ratings_ref_upload")
        if ref_upload is not None and st.session_state.get("_last_ref_upload_name") != ref_upload.name:
            st.session_state._last_ref_upload_name = ref_upload.name
            try:
                parsed, parsed_coords, name_col, kva_col = parse_ratings_reference(ref_upload.getvalue(), ref_upload.name)
                st.session_state.setdefault("ratings_reference", {})
                st.session_state.ratings_reference.update(parsed)
                st.session_state.setdefault("coords_reference", {})
                st.session_state.coords_reference.update(parsed_coords)
                st.session_state["_ratings_version"] = st.session_state.get("_ratings_version", 0) + 1
                st.success(t("ratings_loaded", lang, n=len(parsed), c=len(parsed_coords)))
            except Exception as e:
                st.error(t("ratings_error", lang, e=e))
        n_ref = len(st.session_state.get("ratings_reference", {}))
        if n_ref:
            st.caption(t("ratings_count", lang, n=n_ref))

        st.markdown(f'<div class="sb-lbl">{t("sb2_title", lang)}</div>', unsafe_allow_html=True)
        upload_mode = st.radio(
            t("upload_mode_label", lang), ["Single transformer", "Batch (multiple files)"],
            format_func=lambda v: t("mode_single", lang) if v == "Single transformer" else t("mode_batch", lang),
            horizontal=True, label_visibility="collapsed",
        )
        shift_utc = st.checkbox(t("shift_utc", lang), value=True)

        st.session_state.setdefault("_uploader_gen", 0)
        uploaded = None
        batch_files = None
        if upload_mode == "Single transformer":
            uploaded = st.file_uploader(
                t("single_uploader", lang), type=["csv"],
                key=f"_single_uploader_{st.session_state._uploader_gen}",
            )
        else:
            batch_files = st.file_uploader(
                t("batch_uploader", lang), type=["csv"], accept_multiple_files=True,
                key=f"_batch_uploader_{st.session_state._uploader_gen}",
            )

        pending_match_note = None
        if uploaded is not None:
            derived_id = uploaded.name.replace('.csv', '')
            if st.session_state.get("_last_uploaded_name") != uploaded.name:
                st.session_state._last_uploaded_name = uploaded.name
                rated_val, matched_by = lookup_rated_kva(derived_id, st.session_state.tr_rated_input)
                if matched_by:
                    set_tr_rated(rated_val)
                    pending_match_note = ("match", derived_id, rated_val, matched_by)
                else:
                    pending_match_note = ("nomatch", derived_id, st.session_state.tr_rated_input, None)

        st.markdown(f'<div class="sb-lbl">{t("sb3_title", lang)}</div>', unsafe_allow_html=True)
        current_val = st.session_state.get("tr_rated_input", 160.0)
        closest_idx = min(range(len(STANDARD_KVA)), key=lambda i: abs(STANDARD_KVA[i] - current_val))
        
        tr_rated = st.selectbox(
            t("nameplate_label", lang),
            STANDARD_KVA, index=closest_idx,
            format_func=lambda v: f"{v} kVA",
        )
        st.session_state.tr_rated_input = float(tr_rated)
        tr_rated = float(tr_rated)

        if pending_match_note:
            kind, pea_id, val, matched_by = pending_match_note
            if st.session_state.show_explanations:
                if kind == "match":
                    st.caption(t("match_caption", lang, pea_id=pea_id, val=val, matched_by=matched_by))
                else:
                    st.caption(t("nomatch_caption", lang, pea_id=pea_id, val=val))

        st.markdown(f'<div class="sb-lbl">{t("sb4_title", lang)}</div>', unsafe_allow_html=True)
        day_type = st.radio(
            t("profile_eval", lang), ["Weekday", "Weekend / Holiday"],
            format_func=lambda v: t("day_weekday", lang) if v == "Weekday" else t("day_weekend", lang),
            horizontal=True,
        )
        is_weekend = 1 if "Weekend" in day_type else 0

        st.markdown(f'<div class="sb-lbl">{t("sb5_title", lang)}</div>', unsafe_allow_html=True)
        peak_label = st.selectbox(t("peak_hour_label", lang), list(PEAK_HOURS.keys()), index=2)
        peak_hour  = PEAK_HOURS[peak_label]

        st.markdown(f'<div class="sb-lbl">{t("sb6_title", lang)}</div>', unsafe_allow_html=True)
        w_slow = st.slider("3.7 kW slow charger (%)", min_value=0, max_value=100, value=10, step=5)
        w_std  = st.slider("7.4 kW standard charger (%)", min_value=0, max_value=max(0, 100 - w_slow), value=min(70, 100 - w_slow), step=5)
        w_fast = 100 - w_slow - w_std
        st.caption(t("fast_charger_cap", lang, w=w_fast))
        
        total_w = w_slow + w_std + w_fast
        if total_w == 0:
            st.error(t("weights_error", lang))
            st.stop()
        user_weights = [w_slow / total_w, w_std / total_w, w_fast / total_w]
        user_avg_kw  = sum(c * w for c, w in zip(CHARGER_OPTIONS_KW, user_weights))

if uploaded is not None:
    active_bytes = uploaded.getvalue()
    active_id = uploaded.name.replace('.csv', '')
    st.session_state.fleet_files[active_id] = (active_bytes, shift_utc)
    st.session_state.jump_to_name = active_id
    st.session_state._uploader_gen += 1
    st.rerun()

if batch_files:
    batch_signature = tuple(sorted(f.name for f in batch_files)) + (day_type, peak_hour, user_avg_kw, tr_rated, st.session_state.get("_ratings_version", 0))
    if st.session_state.get("_last_batch_sig") != batch_signature:
        st.session_state._last_batch_sig = batch_signature
        batch_summary = []
        prog = st.progress(0.0, text="Processing batch...")
        for i, f in enumerate(batch_files):
            fid = f.name.replace('.csv', '')
            fbytes = f.getvalue()
            try:
                _probe = pd.read_csv(io.BytesIO(fbytes), nrows=0)
                _probe_cols = set(_probe.columns) | set(_normalize_dtms_columns(_probe.columns).values())
                if {'timestamp', 'P_TOT', 'S_TOT'} - _probe_cols:
                    batch_summary.append((fid, None, None, None, "Missing columns", "—"))
                else:
                    dfc = clean_and_resample(fbytes, shift_utc)
                    dfc['_dow'] = dfc.index.dayofweek
                    dday = dfc[dfc['_dow'] >= 5 if is_weekend else dfc['_dow'] < 5].drop(columns='_dow')
                    if dday.empty:
                        dday = dfc.drop(columns='_dow')
                    rated_b, matched_by_b = lookup_rated_kva(fid, tr_rated)
                    feats_b = extract_features(dday, rated_b, is_weekend)
                    shape_check_b = check_residential_shape(dday, feats_b)
                    shape_flag_b = "OK" if shape_check_b['is_likely_residential'] else "⚠️ Non-residential shape?"

                    if user_avg_kw <= 0 or feats_b['Capacity_Headroom_kVA'] <= 0:
                        hc_raw_b, hc_adj_b = 0.0, 0.0
                    else:
                        hc_raw_b, hc_adj_b = predict_and_adjust(feats_b, peak_hour, user_weights, rated_b, model, xgb_residual, scaler)
                        
                    status_b, _ = classify(hc_adj_b)
                    coord_b = lookup_coords(fid)
                    lat_b, lon_b = coord_b if coord_b else (None, None)
                    upsert_fleet_result(fid, rated_b, lat_b, lon_b, hc_adj_b, status_b, verified=bool(matched_by_b))
                    st.session_state.fleet_files[fid] = (fbytes, shift_utc)
                    batch_summary.append((fid, rated_b, matched_by_b or "sidebar fallback", round(hc_adj_b, 1), status_b, shape_flag_b))
            except Exception as e:
                batch_summary.append((fid, None, None, None, f"Error: {e}", "—"))
            prog.progress((i + 1) / len(batch_files), text=f"Processed {i + 1}/{len(batch_files)}")
        prog.empty()
        st.session_state._batch_summary = batch_summary
        st.session_state._uploader_gen += 1
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE — TRANSFORMER ANALYSIS
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "analysis":
    if st.session_state.get("_batch_summary"):
        with st.expander(f"📦 Batch results ({len(st.session_state._batch_summary)} transformers)", expanded=False):
            st.dataframe(
                pd.DataFrame(st.session_state._batch_summary, columns=["Transformer", "Rated (kVA)", "Matched by", "HC (kW)", "Status", "Shape"]),
                hide_index=True, width='stretch'
            )
        st.markdown("---")

    cached_names = sorted(st.session_state.fleet_files.keys())
    inspect_pick = None
    if cached_names:
        default_idx = cached_names.index(prefill) if prefill in cached_names else 0
        inspect_pick = st.selectbox(
            t("inspect_label", lang), cached_names, index=default_idx, key="_inspect_pick"
        )

    active_bytes = None
    active_id = None

    if uploaded is not None:
        active_bytes = uploaded.getvalue()
        active_id = uploaded.name.replace('.csv', '')
    elif inspect_pick is not None and inspect_pick in st.session_state.fleet_files:
        active_bytes, cached_shift_utc = st.session_state.fleet_files[inspect_pick]
        active_id = inspect_pick
        st.info(f"Showing **{active_id}**")

    if active_bytes is None:
        st.markdown(
            f'<div class="empty"><div class="empty-icon">📂</div>'
            f'<div class="empty-title">{t("empty_state", lang)}</div></div>',
            unsafe_allow_html=True
        )
    else:
        missing = None
        try:
            _probe = pd.read_csv(io.BytesIO(active_bytes), nrows=0)
            _probe_cols = set(_probe.columns) | set(_normalize_dtms_columns(_probe.columns).values())
            missing = {'timestamp', 'P_TOT', 'S_TOT'} - _probe_cols
        except Exception as e:
            st.error(t("file_read_error", lang, e=e))

        if missing:
            st.error(t("missing_cols", lang, missing=missing))
        elif missing is not None:
            df_day = clean_and_resample(active_bytes, shift_utc) # Fixed caching mismatch issues[cite: 1]
            df_day['_dow'] = df_day.index.dayofweek
            df_day = df_day[df_day['_dow'] >= 5 if is_weekend else df_day['_dow'] < 5].drop(columns='_dow')
            if df_day.empty:
                df_day = clean_and_resample(active_bytes, shift_utc).drop(columns='_dow')

            feats = extract_features(df_day, tr_rated, is_weekend)

            shape_check = check_residential_shape(df_day, feats)
            if not shape_check['is_likely_residential']:
                method_note = (
                    "(อิงจาก K-Means centroid ที่เทรนไว้จริง)" if shape_check['method'] == 'centroid_distance'
                    else "(heuristic เบื้องต้น — ไม่มีไฟล์ v5_cluster_centroids.json)"
                )
                st.warning(
                    f"⚠️ **รูปแบบโหลดของหม้อแปลงนี้อาจไม่ใช่ประเภทที่อยู่อาศัย (Residential)** {method_note}\n\n"
                    + "\n".join(f"- {r}" for r in shape_check['reasons'])
                    + "\n\nโมเดล Hybrid Ensemble ถูกเทรนด้วยข้อมูลจากหม้อแปลงที่อยู่อาศัย 22 ลูกในเขต "
                      "กฟก. เขต 2 เท่านั้น ผลทำนาย HC ที่แสดงด้านล่างอาจไม่แม่นยำหากนำไปใช้กับหม้อแปลง"
                      "ประเภทอื่น (พาณิชย์ / อุตสาหกรรม / ผสม) — ควรตรวจสอบ feeder type จริงก่อนใช้ผลลัพธ์"
                )

            if feats['Capacity_Headroom_kVA'] <= 0:
                st.warning(t("exceeds_rating", lang, peak=feats['_peak_kva'], rated=tr_rated))
                hc_raw, hc_adj = 0.0, 0.0
            else:
                hc_raw, hc_adj = predict_and_adjust(feats, peak_hour, user_weights, tr_rated, model, xgb_residual, scaler)

            status, badge_cls = classify(hc_adj)
            _effective_avg_kw = user_avg_kw if user_avg_kw > 0 else BASE_AVG_KW
            n_cars_daily = int(hc_adj // _effective_avg_kw)
            n_cars_simultaneous = int(max(feats['Capacity_Headroom_kVA'], 0) // _effective_avg_kw)
            pea_id = active_id
            
            _, matched_by_single = lookup_rated_kva(pea_id, tr_rated)
            coord_single = lookup_coords(pea_id)
            lat_single, lon_single = coord_single if coord_single else (None, None)
            upsert_fleet_result(pea_id, tr_rated, lat_single, lon_single, hc_adj, status, verified=bool(matched_by_single))

            day_type_display = t("day_weekday", lang) if day_type == "Weekday" else t("day_weekend", lang)
            st.markdown(f"#### `{pea_id}` · {tr_rated:.0f} kVA · {day_type_display}")

            ca, cb, cc, cd = st.columns(4)
            CARD_CLASS = {
                "chargers_daily": "card-green",
                "chargers_sim": "card-red",
            }
            for col, card_key, label, val, unit, sub in [
                (ca, "hc", t("card_predicted_hc", lang), f"{hc_adj:.1f}", "kW", t("card_hc_sub", lang)),
                (cb, "chargers_daily", t("card_chargers_daily", lang), f"{n_cars_daily}", "units", t("card_chargers_daily_sub", lang)),
                (cc, "chargers_sim", t("card_chargers_sim", lang), f"{n_cars_simultaneous}", "units", t("card_chargers_sim_sub", lang)),
                (cd, "headroom", t("card_headroom", lang), f"{feats['Capacity_Headroom_kVA']:.1f}", "kVA", t("card_headroom_sub", lang)),
            ]:
                with col:
                    card_cls = CARD_CLASS.get(card_key, "card")
                    sub_html = f'<div class="card-sub">{sub}</div>' if st.session_state.show_explanations else ""
                    st.markdown(
                        f'<div class="{card_cls}"><div class="card-label">{label}</div>'
                        f'<div class="card-value">{val}<span class="card-unit">{unit}</span></div>'
                        f'{sub_html}</div>', unsafe_allow_html=True
                    )

            st.markdown("---")

            p1, p2 = st.columns([1.2, 1])
            with p1:
                st.markdown(f'<div class="section-lbl">{t("section_daily_profile", lang)}</div>', unsafe_allow_html=True)
                fig1 = plot_load_curve(
                    feats['_profile_kva'], feats['_profile_kw'], tr_rated, hc_adj, peak_hour,
                    profile_kva_peak_day=feats['_profile_kva_peak_day'],
                    profile_kva_min_day=feats['_profile_kva_min_day'],
                    peak_day_label=feats['_peak_day_label'],
                    min_day_label=feats['_min_day_label'],
                )
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with p2:
                st.markdown(f'<div class="section-lbl">{t("section_hc_scenario", lang)}</div>', unsafe_allow_html=True)
                fig2 = plot_scenario_bar(feats, tr_rated, user_weights, model, xgb_residual, scaler)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            st.markdown("---")
            st.markdown(f'<div class="section-lbl">{t("section_residential", lang)}</div>', unsafe_allow_html=True)
            fig3 = plot_residential_profile(
                feats['_profile_kva'], feats['_profile_kw'],
                profile_kva_peak_day=feats['_profile_kva_peak_day'],
                profile_kva_min_day=feats['_profile_kva_min_day'],
                peak_day_label=feats['_peak_day_label'],
                min_day_label=feats['_min_day_label'],
            )
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            st.markdown("---")
            with st.expander("Feature values used for prediction", expanded=False):
                user_avg_kw_display = sum(c * w for c, w in zip(CHARGER_OPTIONS_KW, user_weights))
                if user_avg_kw_display <= 0:
                    user_avg_kw_display = BASE_AVG_KW
                peak_x_nightday_display = peak_hour * feats['Night_Day_Ratio']
                peak_x_headroom_display = peak_hour * feats['Capacity_Headroom_kVA']
                feature_data = {
                    "Feature": ["Capacity_Headroom_kVA", "Evening_Surge_kVA", "Night_Day_Ratio", "Is_Weekend",
                                "Peak_Start_Hour", "Avg_Charger_kW", "Peak_x_NightDay", "Peak_x_Headroom"],
                    "Value": [f"{feats['Capacity_Headroom_kVA']:.2f} kVA", f"{feats['Evening_Surge_kVA']:.2f} kVA",
                              f"{feats['Night_Day_Ratio']:.3f}", str(is_weekend), str(peak_hour),
                              f"{user_avg_kw_display:.2f} kW", f"{peak_x_nightday_display:.2f}", f"{peak_x_headroom_display:.2f}"],
                }
                st.dataframe(pd.DataFrame(feature_data), width='stretch', hide_index=True)

            with st.expander("🔧 Debug: Monte Carlo ground truth for this transformer", expanded=False):
                n_iter = st.slider("Monte Carlo iterations", min_value=50, max_value=1000, value=200, step=50)
                run_debug = st.button("▶ Run live Monte Carlo for this transformer")

                if run_debug:
                    with st.spinner(f"Running {n_iter} Monte Carlo iterations..."):
                        np.random.seed(42)
                        mc_result = run_scenario_monte_carlo(feats['_profile_kva'], tr_rated, user_weights, peak_hour, n_iter=n_iter)

                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.markdown(f'<div class="card"><div class="card-label">{t("mc_card_hc_p5", lang)}</div><div class="card-value">{mc_result["hc_p5"]:.1f}<span class="card-unit">kW</span></div></div>', unsafe_allow_html=True)
                    with dc2:
                        st.markdown(f'<div class="card"><div class="card-label">{t("mc_card_hybrid_raw", lang)}</div><div class="card-value">{hc_raw:.1f}<span class="card-unit">kW</span></div></div>', unsafe_allow_html=True)
                    with dc3:
                        delta = hc_raw - mc_result['hc_p5']
                        st.markdown(f'<div class="card"><div class="card-label">{t("mc_card_diff", lang)}</div><div class="card-value">{delta:+.1f}<span class="card-unit">kW</span></div></div>', unsafe_allow_html=True)

                    fig_mc, ax_mc = plt.subplots(figsize=(8, 2.6))
                    ax_mc.hist(mc_result['samples'], bins=30, color=BLUE, alpha=0.7, edgecolor='white')
                    ax_mc.axvline(mc_result['hc_p5'], color=RED, lw=1.5, linestyle='--', label=f'5th percentile = {mc_result["hc_p5"]:.1f} kW')
                    ax_mc.axvline(hc_raw, color=AMBER, lw=1.5, linestyle='--', label=f'Hybrid raw = {hc_raw:.1f} kW')
                    ax_mc.set_xlabel('Simulated HC per Monte Carlo iteration (kW)', fontsize=9)
                    ax_mc.set_ylabel('Count', fontsize=9)
                    ax_mc.legend(fontsize=8, loc='upper right')
                    ax_mc.grid(axis='y', linestyle=':', alpha=0.4)
                    ax_mc.set_facecolor('#fafbfc')
                    fig_mc.patch.set_facecolor('white')
                    st.pyplot(fig_mc, use_container_width=True)
                    plt.close(fig_mc)

                    mc1, mc2 = st.columns([1, 1.1])
                    with mc1:
                        rep_trace = mc_result['rep_trace']
                        ev_index  = np.arange(len(rep_trace))
                        fig_acc, ax_acc = plt.subplots(figsize=(6, 3.0))
                        fig_acc.patch.set_facecolor('white')
                        ax_acc.step(ev_index, rep_trace, where='post', color=BLUE, lw=1.8, zorder=3)
                        ax_acc.fill_between(ev_index, rep_trace, step='post', alpha=0.10, color=BLUE)
                        ax_acc.axhline(mc_result['rep_hc'], color=RED, lw=1.2, linestyle='--', alpha=0.7, label=f'Final HC = {mc_result["rep_hc"]:.1f} kW')
                        ax_acc.set_xlabel('EV charger # added', fontsize=8.5)
                        ax_acc.set_ylabel('Cumulative HC (kW)', fontsize=9)
                        ax_acc.legend(fontsize=8, loc='lower right')
                        ax_acc.grid(True, linestyle=':', alpha=0.4)
                        ax_acc.set_facecolor('#fafbfc')
                        st.pyplot(fig_acc, use_container_width=True)
                        plt.close(fig_acc)

                    with mc2:
                        rep_ev_profile = mc_result['rep_ev_profile']
                        x = np.arange(96)
                        combined = feats['_profile_kva'] + rep_ev_profile
                        fig_load, ax_load = plt.subplots(figsize=(6, 3.0))
                        fig_load.patch.set_facecolor('white')
                        ax_load.fill_between(x, feats['_profile_kva'], color=BLUE, alpha=0.15, zorder=1)
                        ax_load.plot(x, feats['_profile_kva'], color=BLUE, lw=1.5, label='Base load (S_TOT)', zorder=2)
                        ax_load.fill_between(x, feats['_profile_kva'], combined, color=GREEN, alpha=0.30, zorder=1, label='EV load')
                        ax_load.plot(x, combined, color=GREEN, lw=1.5, zorder=3)
                        ax_load.axhline(tr_rated, color=RED, lw=1.5, linestyle='--', label=f'TR Rating ({tr_rated:.0f} kVA)', zorder=4)
                        ax_load.set_xlim(0, 95)
                        ax_load.set_xticks([0, 16, 32, 48, 64, 80, 96])
                        ax_load.set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=8)
                        ax_load.set_ylabel('Power (kVA)', fontsize=9)
                        ax_load.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
                        ax_load.grid(True, linestyle=':', alpha=0.4)
                        ax_load.set_facecolor('#fafbfc')
                        st.pyplot(fig_load, use_container_width=True)
                        plt.close(fig_load)

# ══════════════════════════════════════════════════════════════
# TAB 1 — FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# PAGE — FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "fleet":
    fleet_valid = st.session_state.fleet.dropna(subset=["Latitude", "Longitude"]).copy()

    if not fleet_valid.empty:
        fleet_valid["Status"] = fleet_valid["Status"].fillna("Unassessed")
        fleet_valid["Predicted EV HC"] = fleet_valid["HC_kW"].apply(lambda v: f"{v:.1f} kW" if pd.notna(v) else "not assessed")

        n_ready      = (fleet_valid["Status"] == "Ready").sum()
        n_monitor    = (fleet_valid["Status"] == "Monitor").sum()
        n_critical   = (fleet_valid["Status"] == "Critical").sum()
        n_unassessed = (fleet_valid["Status"] == "Unassessed").sum()

        s1, s2, s3, s4 = st.columns(4)
        for col, label, n, color in [
            (s1, "Ready", n_ready, GREEN), (s2, "Monitor", n_monitor, AMBER),
            (s3, "Critical", n_critical, RED), (s4, "Unassessed", n_unassessed, "#94a3b8"),
        ]:
            with col:
                st.markdown(
                    f'<div class="fleet-stat"><div class="fleet-stat-n" style="color:{color};">{n}</div>'
                    f'<div class="fleet-stat-l">{label}</div></div>', unsafe_allow_html=True
                )

        st.markdown(f'<div class="section-lbl">{t("section_fleet_map", lang)}</div>', unsafe_allow_html=True)

        def compute_center_zoom(lat_tuple, lon_tuple, padding_factor=1.15, zoom_boost=1.3):
            import math
            if not lat_tuple or not lon_tuple:
                return CHONBURI_LAT, CHONBURI_LON, 12.0
            lat_min, lat_max = min(lat_tuple), max(lat_tuple)
            lon_min, lon_max = min(lon_tuple), max(lon_tuple)
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            lat_span = max(lat_max - lat_min, 0.0008) * padding_factor
            lon_span = max(lon_max - lon_min, 0.0008) * padding_factor
            span = max(lat_span, lon_span)
            zoom = math.log2(360.0 / span) - 1.3 + zoom_boost
            zoom = min(max(zoom, 3.0), 18.0)
            return center_lat, center_lon, zoom

        color_map = {"Ready": GREEN, "Monitor": AMBER, "Critical": RED, "Unassessed": "#94a3b8"}
        center_lat, center_lon, auto_zoom = compute_center_zoom(
            tuple(fleet_valid["Latitude"]), tuple(fleet_valid["Longitude"])
        )
        
        fig_map = px.scatter_mapbox(
            fleet_valid, lat="Latitude", lon="Longitude", color="Status",
            color_discrete_map=color_map,
            hover_name="Name",
            hover_data={"Rated_kVA": True, "Predicted EV HC": True, "Latitude": False, "Longitude": False, "Status": False},
            labels={"Rated_kVA": "Rated kVA"},
            height=820,
        )
        fig_map.update_traces(marker=dict(size=15))
        fig_map.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            mapbox_zoom=auto_zoom,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
            uirevision="fleet_map",
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})

    with st.expander(t("section_fleet_table", lang), expanded=False):
        fleet_edit = st.data_editor(
            st.session_state.fleet.copy(),
            num_rows="dynamic",
            width='stretch',
            column_config={
                "Name": st.column_config.TextColumn("Transformer ID", required=True),
                "Latitude": st.column_config.NumberColumn("Latitude", format="%.5f", disabled=True),
                "Longitude": st.column_config.NumberColumn("Longitude", format="%.5f", disabled=True),
                "Rated_kVA": st.column_config.NumberColumn("Rated kVA", format="%.0f", min_value=0.0),
                "HC_kW": st.column_config.NumberColumn("HC (kW)", format="%.1f"),
                "Status": st.column_config.SelectboxColumn("Status", options=["Ready", "Monitor", "Critical", "Unassessed"]),
            },
            key="fleet_editor",
        )

        if not fleet_edit.equals(st.session_state.fleet):
            st.session_state.fleet = fleet_edit
            st.rerun()

st.markdown(
    f'<div style="text-align:center;color:#94a3b8;font-size:11px;margin-top:32px;padding-top:12px;border-top:1px solid {BORDER};">'
    f'{t("footer", st.session_state.lang)}</div>',
    unsafe_allow_html=True
)
