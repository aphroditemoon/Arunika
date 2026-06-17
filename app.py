"""
ARUNIKA — Sistem Prediksi Banjir Kilat Nasional
Backend v3.0  |  Flask + Random Forest

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEBENARAN DATA BMKG (penting dibaca):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API BMKG (prakiraan-cuaca) hanya menyediakan:
  t    → suhu udara (°C)
  hu   → kelembaban relatif udara (%)
  tcc  → tutupan awan (%)
  tp   → total presipitasi prakiraan (mm per 3 JAM, bukan mm/jam)
  weather → kode cuaca (integer WMO)
  weather_desc → deskripsi cuaca (string)
  ws   → kecepatan angin (km/jam)
  wd   → arah angin (string)
  vs   → jarak pandang (meter)

Yang TIDAK tersedia dari BMKG:
  × debit / TMA sungai
  × kelembaban tanah
  × curah hujan historis 24 jam
  × slope / drainase / infrastruktur

Parameter hidrologi yang tidak ada di BMKG
diestimasi dari profil geografis per kota
(elevasi, slope, drainase, urbanisasi, dll.)
berbasis data BPS/BNPB/OpenStreetMap.

Konversi field BMKG yang benar:
  tp (mm/3jam) ÷ 3 = intensitas hujan (mm/jam)
  ws (km/jam)  ÷ 3.6 = kecepatan angin (m/s)
  cuaca nested list: data[0]['cuaca'][hari][slot_3jam]

Strategi prediksi: ambil kondisi TERBURUK
dalam window 6 jam ke depan (2 slot × 3 jam)
→ lebih konservatif dan relevan untuk early warning.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import pickle, os, requests, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model         = None
scaler        = None
feature_names = None
model_metrics = {}

# ══════════════════════════════════════════════════════════════════
# KONFIGURASI KOTA
# ══════════════════════════════════════════════════════════════════

# Kode wilayah adm4 BMKG (Kepmendagri No. 100.1.1-6117/2022)
CITY_CODES = {
    'jakarta':    '31.71.03.1001',
    'jakarta pusat': '31.71.03.1001',
    'bekasi':     '32.16.09.2007',
    'cikarang':   '32.16.09.2007',
    'bandung':    '32.73.03.1001',
    'surabaya':   '35.78.03.1001',
    'semarang':   '33.74.04.1001',
    'yogyakarta': '34.71.02.1001',
    'malang':     '35.73.05.1001',
    'bogor':      '32.01.05.2001',
    'depok':      '32.76.01.1001',
    'tangerang':  '36.71.01.1001',
    'medan':      '12.71.01.1001',
    'palembang':  '16.71.01.1001',
    'makassar':   '73.71.01.1001',
    'balikpapan': '64.72.01.1001',
    'pekanbaru':  '14.71.01.1001',
    'padang':     '13.71.01.1001',
    'manado':     '71.71.01.1001',
    'pontianak':  '61.71.01.1001',
}

# Profil geografis per kota
# Kolom: elev_m, slope_deg, dist_sungai_km, drainase_%, urban_%, deforest_%,
#         luas_km2, kepadatan_jiwa/km2, tipe_wilayah, pesisir (bool)
# Sumber: BPS 2022, BNPB, OpenStreetMap, publikasi hidrologi Indonesia
CITY_GEO = {
    'jakarta':    (8,   1.5, 0.4, 35, 96, 80, 661,  15900, 'Kota Pesisir',        True),
    'bekasi':     (19,  2.0, 0.9, 48, 87, 68, 210,  13200, 'Kota Metropolitan',   False),
    'cikarang':   (32,  3.0, 1.4, 52, 76, 58, 120,  8500,  'Kawasan Industri',    False),
    'bandung':    (768, 8.0, 1.8, 62, 71, 52, 167,  14800, 'Kota Dataran Tinggi', False),
    'surabaya':   (5,   1.2, 0.7, 42, 90, 70, 374,  8700,  'Kota Pesisir',        True),
    'semarang':   (12,  4.5, 1.1, 44, 84, 62, 373,  5700,  'Kota Pesisir',        True),
    'yogyakarta': (114, 3.5, 1.7, 57, 76, 48, 33,   13100, 'Kota Budaya',         False),
    'malang':     (445, 7.0, 2.3, 66, 69, 45, 110,  8400,  'Kota Dataran Tinggi', False),
    'bogor':      (265, 10.0,1.3, 50, 74, 60, 118,  8900,  'Kota Hujan',          False),
    'depok':      (87,  4.0, 1.2, 55, 82, 65, 200,  11000, 'Kota Penyangga',      False),
    'tangerang':  (10,  1.8, 0.6, 40, 88, 72, 164,  14200, 'Kota Industri',       True),
    'medan':      (23,  2.5, 0.8, 38, 80, 75, 265,  8400,  'Kota Pesisir',        True),
    'palembang':  (8,   1.0, 0.5, 32, 77, 78, 400,  5000,  'Kota Sungai',         False),
    'makassar':   (2,   0.8, 0.4, 35, 85, 70, 200,  8500,  'Kota Pesisir',        True),
    'balikpapan': (15,  5.0, 1.5, 55, 65, 55, 500,  3200,  'Kota Industri',       True),
    'pekanbaru':  (30,  2.0, 1.0, 45, 72, 68, 632,  2800,  'Kota Dataran',        False),
    'padang':     (10,  6.0, 0.8, 42, 70, 58, 695,  1700,  'Kota Pesisir',        True),
    'manado':     (5,   8.0, 1.0, 48, 68, 60, 157,  3500,  'Kota Pesisir',        True),
    'pontianak':  (0,   0.5, 0.3, 30, 65, 72, 107,  5800,  'Kota Delta Sungai',   True),
}
DEFAULT_GEO = (30, 3.0, 1.5, 50, 70, 55, 85, 5000, 'Kawasan Umum', False)

# Kenaikan muka laut kota pesisir (cm/tahun, rata-rata BMKG/BIG)
SEA_LEVEL_RISE = {
    'jakarta': 1.0, 'semarang': 1.2, 'surabaya': 0.7,
    'tangerang': 0.6, 'bekasi': 0.4, 'medan': 0.5,
    'palembang': 0.4, 'makassar': 0.4, 'balikpapan': 0.3,
    'padang': 0.4, 'manado': 0.3, 'pontianak': 0.5,
}

# ── Tabel konversi kode cuaca WMO → intensitas hujan (mm/jam) ───
# Ref: WMO No. 306, BMKG SOP Prakiraan Cuaca, literatur hidrologi
WEATHER_CODE_MM_JAM = {
    0:  0.0,   # Cerah
    1:  0.0,   # Cerah berawan
    2:  0.3,   # Berawan
    3:  1.0,   # Berawan tebal
    10: 0.0,   # Berkabut
    45: 0.0,   # Kabut tebal
    60: 2.0,   # Hujan ringan sesekali
    61: 4.0,   # Hujan ringan
    63: 12.0,  # Hujan sedang
    65: 30.0,  # Hujan lebat
    80: 6.0,   # Hujan lokal ringan
    81: 15.0,  # Hujan lokal sedang
    82: 45.0,  # Hujan lokal lebat
    95: 20.0,  # Badai petir
    97: 40.0,  # Badai petir lebat
    99: 60.0,  # Badai petir sangat lebat
}

BMKG_API = "https://api.bmkg.go.id/publik/prakiraan-cuaca"


# ══════════════════════════════════════════════════════════════════
# HELPERS LOOKUP
# ══════════════════════════════════════════════════════════════════

def lookup_city_code(city: str) -> str:
    key = city.lower().strip()
    if key in CITY_CODES:
        return CITY_CODES[key]
    for k, v in CITY_CODES.items():
        if k in key or key in k:
            return v
    return CITY_CODES['cikarang']


def lookup_city_geo(city: str) -> tuple:
    key = city.lower().strip()
    if key in CITY_GEO:
        return CITY_GEO[key]
    for k, v in CITY_GEO.items():
        if k in key or key in k:
            return v
    return DEFAULT_GEO


def lookup_sea_level(city: str) -> float:
    key = city.lower().strip()
    return SEA_LEVEL_RISE.get(key, 0.2)


# ══════════════════════════════════════════════════════════════════
# FETCH & PARSE BMKG
# ══════════════════════════════════════════════════════════════════

def fetch_bmkg(city: str) -> dict:
    """
    Ambil data prakiraan cuaca dari API BMKG.
    Parsing nested list: data[0]['cuaca'] = list of (list of slot 3-jam per hari)
    Strategi: ambil kondisi TERBURUK dalam 6 jam ke depan (2 slot pertama).
    """
    adm4 = lookup_city_code(city)
    url  = f"{BMKG_API}?adm4={adm4}"
    hdrs = {"User-Agent": "ARUNIKA/3.0 Mozilla/5.0", "Accept": "application/json"}

    try:
        log.info(f"BMKG fetch: {city} ({adm4})")
        r = requests.get(url, headers=hdrs, timeout=15)
        r.raise_for_status()
        data   = r.json()
        lokasi = data.get("lokasi", {})

        cuaca_raw = data.get("data", [{}])[0].get("cuaca", [])
        if not cuaca_raw:
            raise ValueError("cuaca kosong")

        # ── Flatten nested list (hari × slot_3jam) ──────────────
        all_slots = []
        for day in cuaca_raw:
            if isinstance(day, list):
                all_slots.extend(day)
            elif isinstance(day, dict):
                all_slots.append(day)   # fallback: flat list

        if not all_slots:
            raise ValueError("Tidak ada slot cuaca")

        # Slot saat ini (index 0) + window 6 jam ke depan (index 0-1)
        current      = all_slots[0]
        window_slots = all_slots[:2]   # 2 slot × 3 jam = 6 jam ke depan

        # ── Ambil nilai terburuk untuk prediksi banjir ──────────
        # (worst-case lebih baik untuk early warning sistem)
        worst = max(window_slots, key=lambda s: s.get("weather", 0))

        # tp (mm per 3 jam) dari slot terburuk → konversi ke mm/jam
        tp_3h   = float(worst.get("tp", 0))
        tp_mmjam = tp_3h / 3.0

        # Kecepatan angin: BMKG dalam km/jam → konversi ke m/s
        ws_kmjam = float(worst.get("ws", 0))
        ws_ms    = ws_kmjam / 3.6

        # Estimasi curah hujan 24 jam sebelumnya:
        # BMKG hanya menyediakan prakiraan ke depan, bukan historis.
        # Estimasi: akumulasi tp dari semua slot hari pertama (8 slot × tp)
        day0_slots = cuaca_raw[0] if isinstance(cuaca_raw[0], list) else all_slots[:8]
        prev_24h_mm = float(sum(s.get("tp", 0) for s in day0_slots))

        log.info(f"BMKG OK: {lokasi.get('kotkab', city)} | "
                 f"cuaca={worst.get('weather_desc')} | "
                 f"tp_max={tp_3h}mm/3h | ws={ws_kmjam}km/h")

        return {
            "source":   "BMKG Real-Time",
            "adm4":     adm4,
            "lokasi":   lokasi,
            "parsed": {
                "temperature":      float(worst.get("t",    28)),
                "humidity_rh":      float(worst.get("hu",   80)),  # kelembaban UDARA (%)
                "cloud_cover_pct":  float(worst.get("tcc",  70)),
                "tp_per_3h_mm":     tp_3h,                          # mm per 3 jam (field asli)
                "rainfall_mmjam":   round(tp_mmjam, 2),             # mm/jam (konversi ÷3)
                "wind_kmjam":       ws_kmjam,                       # km/jam (field asli)
                "wind_ms":          round(ws_ms, 2),                # m/s (konversi ÷3.6)
                "visibility_m":     float(worst.get("vs", 10000)),
                "weather_code":     int(worst.get("weather", 0)),
                "weather_desc":     str(worst.get("weather_desc", "Cerah")),
                "wind_dir":         str(worst.get("wd", "N")),
                "datetime_local":   str(worst.get("local_datetime", "")),
                "prev_24h_mm_est":  prev_24h_mm,    # estimasi dari slot hari ini
                "n_slots_parsed":   len(all_slots),
            }
        }

    except Exception as e:
        log.warning(f"BMKG gagal ({e}) → Demo fallback")
        return {
            "source": "Demo (BMKG tidak tersedia)",
            "adm4":   adm4,
            "lokasi": {"kotkab": city, "kecamatan": city, "desa": city},
            "parsed": {
                "temperature": 29.0, "humidity_rh": 85.0,
                "cloud_cover_pct": 90.0,
                "tp_per_3h_mm": 15.0, "rainfall_mmjam": 5.0,
                "wind_kmjam": 12.0, "wind_ms": 3.3,
                "visibility_m": 3000.0, "weather_code": 63,
                "weather_desc": "Hujan Sedang", "wind_dir": "W",
                "datetime_local": "", "prev_24h_mm_est": 40.0,
                "n_slots_parsed": 0,
            }
        }


# ══════════════════════════════════════════════════════════════════
# KONVERSI BMKG → PARAMETER MODEL
# ══════════════════════════════════════════════════════════════════

def bmkg_to_params(bmkg: dict, city: str) -> dict:
    """
    Konversi field BMKG ke 14 parameter model + 5 engineered features.

    Yang berasal dari BMKG (langsung / konversi unit):
      rainfall_intensity  ← tp÷3  (mm/jam) + koreksi weather_code
      soil_moisture       ← diestimasi dari humidity_rh (empiris)

    Yang diestimasi dari profil kota (tidak ada di BMKG):
      rainfall_duration   ← dari weather_code + wind_speed
      previous_24h_rainfall ← akumulasi tp slot hari ini (estimasi)
      river_water_level   ← fungsi intensitas × durasi × urban_index
      soil_height, terrain_slope, distance_to_river,
      drainage_capability, urbanization_index, deforestation_index,
      area_size, population_density ← dari CITY_GEO

    Keterbatasan ini didokumentasikan di response API.
    """
    p   = bmkg["parsed"]
    geo = lookup_city_geo(city)
    (elev, slope, dist_rv, drain, urban, deforest,
     area, pop_d, area_type, is_coastal) = geo

    # ── 1. Intensitas hujan (mm/jam) ─────────────────────────────
    # Gabungkan: konversi tp + weather_code lookup, pakai yang lebih besar
    rain_from_tp   = p["rainfall_mmjam"]
    rain_from_code = WEATHER_CODE_MM_JAM.get(p["weather_code"], 0.0)
    rainfall_intensity = max(rain_from_tp, rain_from_code)

    # Faktor orografi: kota dataran tinggi → efek hujan lebih besar
    if elev > 300 and "Hujan" in p["weather_desc"]:
        rainfall_intensity *= 1.25

    # Kota dekat laut + angin kencang dari laut → intensitas lebih tinggi
    if is_coastal and p["wind_ms"] > 5:
        rainfall_intensity *= 1.10

    rainfall_intensity = round(float(np.clip(rainfall_intensity, 0, 120)), 2)

    # ── 2. Durasi hujan estimasi (jam) ───────────────────────────
    # Berdasarkan tipe cuaca dan kecepatan angin
    wc = p["weather_code"]
    if   wc >= 95: base_dur = 3.0
    elif wc >= 82: base_dur = 2.0
    elif wc >= 65: base_dur = 4.0
    elif wc >= 63: base_dur = 3.5
    elif wc >= 61: base_dur = 3.0
    elif wc >= 60: base_dur = 2.0
    elif wc >= 80: base_dur = 1.5
    else:          base_dur = 0.5
    # Angin kencang mempersingkat durasi hujan
    wind_factor = max(0.5, 1.0 - p["wind_ms"] / 25)
    rainfall_duration = round(float(np.clip(base_dur * wind_factor, 0.5, 12)), 1)

    # ── 3. Curah hujan 24 jam sebelumnya ─────────────────────────
    # Estimasi terbaik dari data BMKG yang tersedia:
    # akumulasi tp slot hari ini + koreksi cloud_cover
    prev_24h = p["prev_24h_mm_est"]
    # Koreksi: jika cloud cover tinggi tapi tp kecil → kemungkinan ada hujan lebih
    if p["cloud_cover_pct"] > 80 and prev_24h < 10:
        prev_24h = p["cloud_cover_pct"] / 100 * 20
    previous_24h_rainfall = round(float(np.clip(prev_24h, 0, 200)), 1)

    # ── 4. Kelembaban tanah (%) ──────────────────────────────────
    # Estimasi empiris dari kelembaban udara (RH):
    # SM ≈ 0.75 × RH + offset, dikurangi imperviousness kota
    # Ref: Seneviratne et al. (2010), Berg et al. (2017)
    impervious_penalty = (urban - 50) / 100 * 8
    soil_moisture = float(np.clip(
        0.75 * p["humidity_rh"] + 8 - impervious_penalty, 40, 97))

    # ── 5. TMA sungai (m) ────────────────────────────────────────
    # Tidak tersedia dari BMKG → estimasi dari intensitas & durasi & urban
    tma_base  = 0.8 + (rainfall_intensity / 15) * (rainfall_duration / 3)
    tma_urban = tma_base * (1 + urban / 250)
    river_water_level = round(float(np.clip(tma_urban, 0, 9)), 1)

    # ── 6. Sea level rise (cm/tahun) ─────────────────────────────
    sea_lr = lookup_sea_level(city)

    # ── 7. Engineered features ───────────────────────────────────
    runoff_index = (rainfall_intensity * rainfall_duration
                    * (1 - drain / 100) * (urban / 100))

    soil_sat_risk = (soil_moisture / 100) * (previous_24h_rainfall / 100)

    river_prox_risk = river_water_level / (dist_rv + 0.1)

    terrain_ponding = (1.0 if slope < 2 else 0.6 if slope < 5 else 0.3)

    coastal_risk = sea_lr / (elev + 1)

    return {
        # ── 14 fitur original ──
        "soil_height":           float(elev),
        "rainfall_intensity":    rainfall_intensity,
        "rainfall_duration":     rainfall_duration,
        "previous_24h_rainfall": previous_24h_rainfall,
        "river_water_level":     river_water_level,
        "distance_to_river":     float(dist_rv),
        "drainage_capability":   float(drain),
        "urbanization_index":    float(urban),
        "deforestation_index":   float(deforest),
        "sea_level_rise":        sea_lr,
        "soil_moisture":         soil_moisture,
        "terrain_slope":         float(slope),
        "area_size":             float(area),
        "population_density":    float(pop_d),
        # ── 5 engineered features ──
        "runoff_index":          float(runoff_index),
        "soil_sat_risk":         float(soil_sat_risk),
        "river_proximity_risk":  float(river_prox_risk),
        "terrain_ponding":       float(terrain_ponding),
        "coastal_risk":          float(coastal_risk),
        # ── metadata (tidak masuk model) ──
        "_city":         city,
        "_area_type":    area_type,
        "_is_coastal":   is_coastal,
        "_bmkg_temp":    p["temperature"],
        "_bmkg_rh":      p["humidity_rh"],
        "_bmkg_weather": p["weather_desc"],
        "_bmkg_wind_ms": p["wind_ms"],
        "_bmkg_cloud":   p["cloud_cover_pct"],
        "_tp_raw_mm3h":  p["tp_per_3h_mm"],    # field asli BMKG
        "_ri_from_tp":   p["rainfall_mmjam"],   # setelah ÷3
        "_ri_from_code": rain_from_code,         # dari weather_code
        "_data_source":  bmkg["source"],
    }


# ══════════════════════════════════════════════════════════════════
# ENGINEERED FEATURES (manual input)
# ══════════════════════════════════════════════════════════════════

def compute_engineered(p: dict) -> dict:
    p["runoff_index"] = (
        p["rainfall_intensity"] * p["rainfall_duration"]
        * (1 - p["drainage_capability"] / 100)
        * (p["urbanization_index"] / 100))
    p["soil_sat_risk"] = (p["soil_moisture"] / 100) * (p["previous_24h_rainfall"] / 100)
    p["river_proximity_risk"] = p["river_water_level"] / (p["distance_to_river"] + 0.1)
    p["terrain_ponding"] = (1.0 if p["terrain_slope"] < 2
                            else 0.6 if p["terrain_slope"] < 5 else 0.3)
    p["coastal_risk"] = p["sea_level_rise"] / (p["soil_height"] + 1)
    return p


# ══════════════════════════════════════════════════════════════════
# TRAINING DATA & MODEL
# ══════════════════════════════════════════════════════════════════

FEATURES = [
    "soil_height", "rainfall_intensity", "rainfall_duration", "previous_24h_rainfall",
    "river_water_level", "distance_to_river", "drainage_capability", "urbanization_index",
    "deforestation_index", "sea_level_rise", "soil_moisture", "terrain_slope",
    "area_size", "population_density",
    "runoff_index", "soil_sat_risk", "river_proximity_risk", "terrain_ponding", "coastal_risk"
]


def make_physics_label(df: pd.DataFrame) -> np.ndarray:
    """Label banjir berbasis Metode Rasional Hidrologi (SNI 2415:2016)."""
    C = np.clip(
        np.where(df["urbanization_index"] > 70, 0.75,
        np.where(df["urbanization_index"] > 50, 0.60, 0.40))
        + df["deforestation_index"] / 100 * 0.15, 0, 1.0)

    I_eff  = df["rainfall_intensity"] * (1 + df["previous_24h_rainfall"] / 200)
    Q      = C * I_eff * df["area_size"] / 360
    drain  = df["drainage_capability"] * (1 - df["urbanization_index"] / 200)

    sat  = np.where(df["soil_moisture"] > 85, 1.5,
           np.where(df["soil_moisture"] > 70, 1.2, 1.0))
    slp  = np.where(df["terrain_slope"] < 2, 1.4,
           np.where(df["terrain_slope"] < 5, 1.1, 0.8))
    riv  = np.where(df["distance_to_river"] < 0.5, 1.6,
           np.where(df["distance_to_river"] < 1.5, 1.2, 0.9))
    tma  = np.where(df["river_water_level"] > 4, 1.5,
           np.where(df["river_water_level"] > 2.5, 1.2, 1.0))
    sea  = 1 + df["sea_level_rise"] * 0.5
    dur  = 1 + df["rainfall_duration"] / 24

    score = Q * sat * slp * riv * tma * sea * dur / (drain + 1e-6)
    return (score > np.percentile(score, 72)).astype(int)


def generate_training_data(n: int = 15000) -> pd.DataFrame:
    np.random.seed(42)
    ri  = np.concatenate([
          np.random.gamma(3, 8, int(n*.60)),
          np.random.gamma(5, 15, int(n*.25)),
          np.random.gamma(8, 20, int(n*.15))])[:n]
    rd  = np.clip(np.random.gamma(2.5, 2.5, n), 0.5, 18)
    p24 = np.random.gamma(3, 15, n) * (0.6 + 0.4 * np.clip(ri / 80, 0, 1))
    sm  = 55 + np.random.beta(4, 2, n) * 40
    rw  = np.clip(np.random.gamma(2.5, 1.2, n), 0, 9)
    dr  = np.clip(np.random.exponential(2.5, n), 0.1, 20)
    dc  = np.random.beta(2.5, 3, n) * 70 + 20
    ui  = np.random.beta(3, 2, n) * 68 + 30
    di  = np.random.beta(2, 3, n) * 75 + 10
    sl  = np.clip(np.random.gamma(2, .3, n), 0, 1.5)
    sh  = np.clip(np.concatenate([
          np.random.normal(10, 8, int(n*.50)),
          np.random.normal(200, 80, int(n*.25)),
          np.random.normal(450, 150, int(n*.25))])[:n], 0, 900)
    ts  = np.clip(np.random.gamma(2, 4, n), 0.2, 35)
    az  = np.clip(np.random.gamma(3, 25, n), 5, 500)
    pd_ = np.clip(np.concatenate([
          np.random.normal(15000, 4000, int(n*.30)),
          np.random.normal(7000,  2000, int(n*.40)),
          np.random.normal(2000,  800,  int(n*.30))])[:n], 200, 28000)

    df = pd.DataFrame({
        "soil_height": sh, "rainfall_intensity": ri, "rainfall_duration": rd,
        "previous_24h_rainfall": p24, "river_water_level": rw,
        "distance_to_river": dr, "drainage_capability": dc,
        "urbanization_index": ui, "deforestation_index": di,
        "sea_level_rise": sl, "soil_moisture": sm, "terrain_slope": ts,
        "area_size": az, "population_density": pd_,
    })
    df["runoff_index"]         = ri * rd * (1 - dc / 100) * (ui / 100)
    df["soil_sat_risk"]        = (sm / 100) * (p24 / 100)
    df["river_proximity_risk"] = rw / (dr + 0.1)
    df["terrain_ponding"]      = np.where(ts < 2, 1.0, np.where(ts < 5, 0.6, 0.3))
    df["coastal_risk"]         = sl / (sh + 1)
    df["flash_flood"]          = make_physics_label(df)
    return df


def build_model():
    """Build & evaluate model dari data training baru."""
    log.info("Membangun model baru (15.000 sampel, 19 fitur)...")
    df = generate_training_data()
    X, y = df[FEATURES], df["flash_flood"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=18,
        min_samples_split=8, min_samples_leaf=4,
        max_features="sqrt", class_weight={0: cw[0], 1: cw[1]},
        bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)
    rf.fit(Xtr_s, ytr)

    yp  = rf.predict(Xte_s)
    ypr = rf.predict_proba(Xte_s)[:, 1]
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvs = cross_val_score(rf, Xtr_s, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)

    m = {
        "accuracy":         float(rf.score(Xte_s, yte)),
        "oob_accuracy":     float(rf.oob_score_),
        "f1_score":         float(f1_score(yte, yp)),
        "precision":        float(precision_score(yte, yp)),
        "recall":           float(recall_score(yte, yp)),
        "roc_auc":          float(roc_auc_score(yte, ypr)),
        "cv_roc_auc_mean":  float(cvs.mean()),
        "cv_roc_auc_std":   float(cvs.std()),
        "n_estimators":     400,
        "train_samples":    int(len(Xtr)),
        "test_samples":     int(len(Xte)),
        "n_features":       len(FEATURES),
    }
    log.info(f"Model selesai — Test Acc: {m['accuracy']*100:.2f}% | ROC-AUC: {m['roc_auc']:.4f}")
    return rf, sc, m


def load_model():
    global model, scaler, feature_names, model_metrics
    feature_names = FEATURES
    pkl = "flood_model_bmkg.pkl"

    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            model, scaler = d["model"], d["scaler"]
            feature_names = d.get("features", FEATURES)
            model_metrics = d["metrics"]
            log.info(f"Model dimuat dari {pkl} — "
                     f"Acc={model_metrics.get('accuracy',0)*100:.1f}% | "
                     f"AUC={model_metrics.get('roc_auc',0):.4f}")
            return
        except Exception as e:
            log.error(f"Gagal load pkl: {e} — rebuild...")

    model, scaler, model_metrics = build_model()
    with open(pkl, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler,
                     "features": feature_names, "metrics": model_metrics}, f)


load_model()


# ══════════════════════════════════════════════════════════════════
# HELPER RESPONSE
# ══════════════════════════════════════════════════════════════════

def flood_category(prob: float) -> str:
    if prob > 0.75: return "SIAGA 1 — BAHAYA"
    if prob > 0.55: return "SIAGA 2 — WASPADA"
    if prob > 0.35: return "SIAGA 3 — PERHATIAN"
    return "AMAN"


def severity_label(people: int) -> str:
    if people > 100_000: return "Sangat Tinggi"
    if people > 50_000:  return "Tinggi"
    if people > 10_000:  return "Sedang"
    return "Rendah"


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/predict-from-bmkg", methods=["POST"])
def predict_from_bmkg():
    try:
        city  = (request.json or {}).get("city_name", "Cikarang").strip()
        bmkg  = fetch_bmkg(city)
        p     = bmkg_to_params(bmkg, city)

        X  = pd.DataFrame([[p[f] for f in feature_names]], columns=feature_names)
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[0][1])
        pred = int(prob > 0.5)

        area   = p["area_size"]
        pop_d  = p["population_density"]
        aff_km = area * prob * 0.7 if pred else area * prob * 0.3
        people = int(aff_km * pop_d)

        return jsonify({
            "flash_flood":    pred,
            "probability":    round(prob, 4),
            "flood_category": flood_category(prob),
            "location":       p["_city"],

            "area_info": {
                "area_type":          p["_area_type"],
                "area_size_km2":      area,
                "population_density": pop_d,
                "total_population":   int(area * pop_d),
                "is_coastal":         p["_is_coastal"],
            },

            "severity_assessment": {
                "estimated_flood_coverage_km2": round(aff_km, 2),
                "estimated_people_affected":    people,
                "severity_level":               severity_label(people),
            },

            # ── Data BMKG yang benar-benar diambil ──────────────
            "bmkg_data": {
                "source":           bmkg["source"],
                "temperature_c":    p["_bmkg_temp"],
                "humidity_rh_pct":  p["_bmkg_rh"],     # kelembaban UDARA
                "weather":          p["_bmkg_weather"],
                "wind_ms":          p["_bmkg_wind_ms"],
                "cloud_cover_pct":  p["_bmkg_cloud"],
                "tp_raw_mm_per_3h": p["_tp_raw_mm3h"], # field asli BMKG
                "tp_converted_mmjam": p["_ri_from_tp"], # setelah ÷3
            },

            # ── Parameter yang dipakai model ─────────────────────
            "model_input": {
                "rainfall_intensity_mmjam":   p["rainfall_intensity"],
                "rainfall_duration_h":        p["rainfall_duration"],
                "previous_24h_rainfall_mm":   p["previous_24h_rainfall"],
                "soil_moisture_pct":          round(p["soil_moisture"], 1),
                "river_water_level_m":        p["river_water_level"],
                "drainage_capability_pct":    p["drainage_capability"],
                "urbanization_index_pct":     p["urbanization_index"],
                "terrain_slope_deg":          p["terrain_slope"],
                "elevation_m":                p["soil_height"],
                "distance_to_river_km":       p["distance_to_river"],
            },

            # ── Engineered features ──────────────────────────────
            "computed_features": {
                "runoff_index":         round(p["runoff_index"], 2),
                "soil_sat_risk":        round(p["soil_sat_risk"], 3),
                "river_proximity_risk": round(p["river_proximity_risk"], 3),
                "terrain_ponding":      p["terrain_ponding"],
                "coastal_risk":         round(p["coastal_risk"], 4),
            },

            # ── Transparansi sumber data ─────────────────────────
            "data_transparency": {
                "from_bmkg_direct": [
                    "suhu (t)", "kelembaban_udara (hu)",
                    "tutupan_awan (tcc)", "kode_cuaca (weather)",
                    "kecepatan_angin (ws)", "presipitasi_prakiraan (tp mm/3jam)"
                ],
                "estimated_from_city_profile": [
                    "elevasi", "kemiringan", "jarak_sungai",
                    "kapasitas_drainase", "indeks_urbanisasi",
                    "indeks_deforestasi", "luas_wilayah", "kepadatan_penduduk"
                ],
                "estimated_empirically": [
                    "kelembaban_tanah (dari hu)", "TMA_sungai (dari intensitas+durasi)",
                    "curah_hujan_24jam_sebelumnya (dari akumulasi tp hari ini)"
                ],
            },

            "model_info": {
                "version":        "3.0",
                "type":           "Random Forest (Hidrologi Fisika)",
                "n_features":     model_metrics.get("n_features", 19),
                "n_estimators":   model_metrics.get("n_estimators", 400),
                "test_accuracy":  model_metrics.get("accuracy", 0),
                "roc_auc":        model_metrics.get("roc_auc", 0),
                "f1_score":       model_metrics.get("f1_score", 0),
                "cv_roc_auc":     model_metrics.get("cv_roc_auc_mean", 0),
                "training_basis": "Metode Rasional Hidrologi (SNI 2415:2016)",
            }
        })

    except Exception as e:
        log.error(f"predict-bmkg error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_manual():
    try:
        data = request.json or {}
        required_14 = [
            "soil_height", "rainfall_intensity", "rainfall_duration",
            "river_water_level", "drainage_capability", "urbanization_index",
            "deforestation_index", "sea_level_rise", "soil_moisture",
            "terrain_slope", "distance_to_river", "previous_24h_rainfall",
        ]
        p = {k: float(data[k]) for k in required_14}
        p["area_size"]          = float(data.get("area_size", 85))
        p["population_density"] = float(data.get("population_density", 5000))
        p = compute_engineered(p)

        X  = pd.DataFrame([[p[f] for f in feature_names]], columns=feature_names)
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[0][1])
        pred = int(prob > 0.5)

        area   = p["area_size"]
        pop_d  = p["population_density"]
        aff_km = area * prob * 0.7 if pred else area * prob * 0.3
        people = int(aff_km * pop_d)

        return jsonify({
            "flash_flood":    pred,
            "probability":    round(prob, 4),
            "flood_category": flood_category(prob),

            "severity_assessment": {
                "area_size_km2":                area,
                "population_density":           pop_d,
                "estimated_flood_coverage_km2": round(aff_km, 2),
                "estimated_people_affected":    people,
                "severity_level":               severity_label(people),
            },

            "computed_features": {
                "runoff_index":         round(p["runoff_index"], 2),
                "soil_sat_risk":        round(p["soil_sat_risk"], 3),
                "river_proximity_risk": round(p["river_proximity_risk"], 3),
                "terrain_ponding":      p["terrain_ponding"],
                "coastal_risk":         round(p["coastal_risk"], 4),
            },

            "model_info": {
                "version":      "3.0",
                "type":         "Random Forest (Hidrologi Fisika)",
                "data_source":  "Manual Input",
                "test_accuracy": model_metrics.get("accuracy", 0),
                "roc_auc":      model_metrics.get("roc_auc", 0),
                "f1_score":     model_metrics.get("f1_score", 0),
            }
        })

    except KeyError as e:
        return jsonify({"error": f"Parameter wajib tidak ada: {e}"}), 400
    except Exception as e:
        log.error(f"predict-manual error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":   "running",
        "version":  "3.0",
        "model_ok": model is not None,
        "metrics": {
            "test_accuracy": model_metrics.get("accuracy", 0),
            "roc_auc":       model_metrics.get("roc_auc", 0),
            "f1_score":      model_metrics.get("f1_score", 0),
            "cv_roc_auc":    model_metrics.get("cv_roc_auc_mean", 0),
        }
    })


@app.route("/model-info")
def model_info_route():
    return jsonify({
        "version":         "3.0",
        "features":        feature_names,
        "n_features":      len(feature_names),
        "metrics":         model_metrics,
        "cities_supported": sorted(CITY_GEO.keys()),
        "bmkg_fields_used": ["t","hu","tcc","tp","weather","ws","wd","vs"],
        "training_basis":   "SNI 2415:2016 Metode Rasional Hidrologi",
    })


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    print("━" * 55)
    print("🌊  ARUNIKA Flash Flood Prediction  v3.0")
    print("━" * 55)
    print(f"  Features    : {len(feature_names)} (14 original + 5 engineered)")
    print(f"  Test Acc    : {model_metrics.get('accuracy',0)*100:.2f}%")
    print(f"  OOB Acc     : {model_metrics.get('oob_accuracy',0)*100:.2f}%")
    print(f"  ROC-AUC     : {model_metrics.get('roc_auc',0):.4f}")
    print(f"  F1-Score    : {model_metrics.get('f1_score',0):.4f}")
    print(f"  CV AUC      : {model_metrics.get('cv_roc_auc_mean',0):.4f}"
          f" ± {model_metrics.get('cv_roc_auc_std',0):.4f}")
    print("━" * 55)
    app.run(debug=True, port=5000)
