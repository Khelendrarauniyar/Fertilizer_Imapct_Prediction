import csv
import io
import json
import math
import os
import re
import secrets
import sqlite3
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from functools import wraps

import requests
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

# Workaround for protobuf/tensorflow compatibility in dev environments
import os as _os_for_proto
if not _os_for_proto.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
    _os_for_proto.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()

APP_NAME = "AgriNexus"
DATABASE = "fertilizer.db"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CROP_DATASET = os.path.join(DATA_DIR, "crop_recommendation.csv")
FERTILIZER_DATASET = os.path.join(DATA_DIR, "fertilizer_prediction.csv")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(32))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = None
if GEMINI_API_KEY:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    except ImportError:
        GEMINI_MODEL = None

# Optional local ML model integration (AgriGo artifacts)
try:
    import ml_models
    ML_MODELS_AVAILABLE = True
except Exception:
    ML_MODELS_AVAILABLE = False

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SOIL_TYPES = ["Black", "Clayey", "Loamy", "Red", "Sandy", "Silty", "Peaty"]
CROP_TYPES = [
    "Apple",
    "Banana",
    "Barley",
    "Blackgram",
    "Chickpea",
    "Coconut",
    "Coffee",
    "Cotton",
    "Grapes",
    "Ground Nuts",
    "Jute",
    "Kidneybeans",
    "Lentil",
    "Maize",
    "Mango",
    "Millets",
    "Mothbeans",
    "Mungbean",
    "Muskmelon",
    "Oil seeds",
    "Orange",
    "Paddy",
    "Papaya",
    "Pigeonpeas",
    "Pomegranate",
    "Pulses",
    "Rice",
    "Soybean",
    "Sugarcane",
    "Tobacco",
    "Watermelon",
    "Wheat",
]

FERTILIZER_GUIDE = {
    "Urea": {
        "focus": "nitrogen correction",
        "timing": "Split into two to three applications around active vegetative growth.",
    },
    "DAP": {
        "focus": "phosphorus and starter nitrogen",
        "timing": "Place near the root zone before sowing or during early establishment.",
    },
    "10-26-26": {
        "focus": "balanced phosphorus and potassium support",
        "timing": "Use at basal application where P and K are both limiting.",
    },
    "14-35-14": {
        "focus": "high phosphorus with balanced N and K",
        "timing": "Prefer during root development and early crop establishment.",
    },
    "17-17-17": {
        "focus": "balanced NPK maintenance",
        "timing": "Apply when soil nutrients are moderately balanced but total fertility is low.",
    },
    "20-20": {
        "focus": "nitrogen and phosphorus support",
        "timing": "Use for crops needing canopy growth and root reinforcement.",
    },
    "28-28": {
        "focus": "strong nitrogen and phosphorus correction",
        "timing": "Use carefully where both N and P are deficient; avoid overuse in high pH soil.",
    },
    "Organic": {
        "focus": "soil structure and slow nutrient release",
        "timing": "Incorporate before planting or as composted side dressing.",
    },
}

BASE_YIELD_T_HA = {
    "apple": 18.0,
    "banana": 32.0,
    "barley": 3.6,
    "blackgram": 1.1,
    "chickpea": 1.4,
    "coconut": 9.0,
    "coffee": 1.2,
    "cotton": 2.5,
    "grapes": 16.0,
    "ground nuts": 2.2,
    "jute": 2.7,
    "kidneybeans": 1.7,
    "lentil": 1.2,
    "maize": 5.8,
    "mango": 8.0,
    "millets": 2.0,
    "mothbeans": 0.9,
    "mungbean": 1.0,
    "muskmelon": 20.0,
    "oil seeds": 1.6,
    "orange": 15.0,
    "paddy": 4.5,
    "papaya": 38.0,
    "pigeonpeas": 1.3,
    "pomegranate": 12.0,
    "pulses": 1.2,
    "rice": 4.6,
    "soybean": 2.7,
    "sugarcane": 72.0,
    "tobacco": 2.1,
    "watermelon": 28.0,
    "wheat": 3.8,
}


YIELD_RANGE_PATTERNS = [
    re.compile(
        r"(?P<low>\d+(?:\.\d+)?)\s*(?:-|to|and|–|—)\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>kg/ha|kg per hectare|t/ha|tons?/ha|tons? per hectare)",
        re.IGNORECASE,
    ),
    re.compile(
        r"between\s+(?P<low>\d+(?:\.\d+)?)\s*(?:-|to|and|–|—)\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>kg/ha|kg per hectare|t/ha|tons?/ha|tons? per hectare)",
        re.IGNORECASE,
    ),
]


def connect_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                farm_name TEXT,
                location TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                provider TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                soil TEXT,
                weather TEXT,
                fertilizer TEXT,
                amount REAL,
                crop TEXT,
                predicted_yield TEXT,
                description TEXT,
                raw_response TEXT
            )
            """
        )

        required_columns = {
            "prediction_type": "TEXT DEFAULT 'yield'",
            "nitrogen": "REAL",
            "phosphorus": "REAL",
            "potassium": "REAL",
            "ph": "REAL",
            "temperature": "REAL",
            "humidity": "REAL",
            "rainfall": "REAL",
            "moisture": "REAL",
            "risk_level": "TEXT",
            "confidence": "REAL",
            "recommendation": "TEXT",
            "created_at": "TEXT",
            "user_id": "INTEGER",
        }
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for column, definition in required_columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {definition}")
        conn.commit()


def parse_float(value, field, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a valid number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}.")
    return number


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with connect_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Sign in to manage your farm plans.")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


def title_crop(crop):
    return str(crop).replace("_", " ").title()


def summarize_yield_text(predicted_yield, raw_response=None, max_length=72):
    for candidate in (predicted_yield, raw_response):
        if not candidate:
            continue
        text = str(candidate).replace("\r", " ").replace("\n", " ").strip()
        for pattern in YIELD_RANGE_PATTERNS:
            match = pattern.search(text)
            if match:
                low = match.group("low")
                high = match.group("high")
                unit = match.group("unit").lower()
                if "ton" in unit:
                    unit = "t/ha"
                elif "kg" in unit:
                    unit = "kg/ha"
                return f"{low} - {high} {unit}"
        if len(text) <= max_length:
            return text
        return text[: max_length - 1].rstrip() + "…"
    return "-"


def enrich_prediction_row(row):
    item = dict(row)
    item["predicted_yield_display"] = summarize_yield_text(
        item.get("predicted_yield"), item.get("raw_response")
    )
    return item


def load_crop_profiles():
    grouped = defaultdict(list)
    with open(CROP_DATASET, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["label"].strip().lower()].append(
                {feature: float(row[feature]) for feature in FEATURES}
            )

    profiles = {}
    for crop, rows in grouped.items():
        profiles[crop] = {
            feature: sum(row[feature] for row in rows) / len(rows)
            for feature in FEATURES
        }
        profiles[crop]["samples"] = len(rows)
    return profiles


def load_feature_ranges():
    ranges = {feature: [math.inf, -math.inf] for feature in FEATURES}
    with open(CROP_DATASET, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for feature in FEATURES:
                value = float(row[feature])
                ranges[feature][0] = min(ranges[feature][0], value)
                ranges[feature][1] = max(ranges[feature][1], value)
    return {feature: tuple(bounds) for feature, bounds in ranges.items()}


def load_fertilizer_rows():
    rows = []
    with open(FERTILIZER_DATASET, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "temperature": float(row["Temparature"]),
                    "humidity": float(row["Humidity "]),
                    "moisture": float(row["Moisture"]),
                    "soil": row["Soil Type"].strip(),
                    "crop": row["Crop Type"].strip(),
                    "N": float(row["Nitrogen"]),
                    "K": float(row["Potassium"]),
                    "P": float(row["Phosphorous"]),
                    "fertilizer": row["Fertilizer Name"].strip(),
                }
            )
    return rows


CROP_PROFILES = load_crop_profiles()
FEATURE_RANGES = load_feature_ranges()
FERTILIZER_ROWS = load_fertilizer_rows()


def feature_similarity(value, target, feature):
    low, high = FEATURE_RANGES[feature]
    spread = max(high - low, 1)
    distance = abs(value - target) / spread
    return max(0, 1 - distance)


def crop_recommendations(inputs, limit=5):
    weights = {
        "N": 0.11,
        "P": 0.11,
        "K": 0.11,
        "temperature": 0.16,
        "humidity": 0.16,
        "ph": 0.16,
        "rainfall": 0.19,
    }
    scored = []
    for crop, profile in CROP_PROFILES.items():
        score = sum(
            weights[feature] * feature_similarity(inputs[feature], profile[feature], feature)
            for feature in FEATURES
        )
        fit = round(score * 100, 1)
        limiting = sorted(
            FEATURES,
            key=lambda feature: abs(inputs[feature] - profile[feature])
            / max(FEATURE_RANGES[feature][1] - FEATURE_RANGES[feature][0], 1),
            reverse=True,
        )[:2]
        scored.append(
            {
                "crop": title_crop(crop),
                "score": fit,
                "why": (
                    f"Closest match to {title_crop(crop)} profile; watch "
                    f"{', '.join(limiting)}."
                ),
                "profile": {k: round(v, 2) for k, v in profile.items() if k != "samples"},
            }
        )
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


def fertilizer_recommendation(inputs):
    crop = inputs["crop"].lower()
    soil = inputs["soil"].lower()
    candidates = []
    for row in FERTILIZER_ROWS:
        score = 0
        score += 25 if row["soil"].lower() == soil else 0
        score += 25 if row["crop"].lower() == crop else 0
        score += max(0, 15 - abs(inputs["temperature"] - row["temperature"]) * 0.7)
        score += max(0, 15 - abs(inputs["humidity"] - row["humidity"]) * 0.35)
        score += max(0, 10 - abs(inputs["moisture"] - row["moisture"]) * 0.35)
        score += max(0, 10 - abs(inputs["N"] - row["N"]) * 0.25)
        score += max(0, 10 - abs(inputs["P"] - row["P"]) * 0.25)
        score += max(0, 10 - abs(inputs["K"] - row["K"]) * 0.25)
        candidates.append((score, row))

    top_pairs = sorted(candidates, key=lambda item: item[0], reverse=True)[:7]
    top_rows = [row for _, row in top_pairs]
    fertilizer = Counter(row["fertilizer"] for row in top_rows).most_common(1)[0][0]
    guide = FERTILIZER_GUIDE.get(fertilizer, FERTILIZER_GUIDE["Organic"])

    crop_aliases = {"paddy": "rice", "maize": "maize", "ground nuts": "mothbeans", "pulses": "pigeonpeas"}
    matched_crop = crop_aliases.get(crop, crop)
    matched_crop = matched_crop if matched_crop in CROP_PROFILES else "rice"
    crop_profile = CROP_PROFILES.get(matched_crop, CROP_PROFILES["rice"])
    gaps = {
        "Nitrogen": round(crop_profile["N"] - inputs["N"], 1),
        "Phosphorus": round(crop_profile["P"] - inputs["P"], 1),
        "Potassium": round(crop_profile["K"] - inputs["K"], 1),
    }
    limiting = [name for name, gap in gaps.items() if gap > 8]
    if not limiting:
        limiting = ["maintenance nutrition"]

    confidence = round(min(96, max(58, sum(score for score, _ in top_pairs) / max(len(top_pairs), 1))), 1)
    return {
        "fertilizer": fertilizer,
        "confidence": confidence,
        "focus": guide["focus"],
        "timing": guide["timing"],
        "nutrient_gaps": gaps,
        "summary": (
            f"{fertilizer} is recommended for {title_crop(inputs['crop'])} on "
            f"{inputs['soil']} soil, mainly for {', '.join(limiting)}."
        ),
    }


def fertilizer_recommendation_with_ml(inputs):
    # try ML model first if available
    if 'ml_models' in globals() and hasattr(ml_models, 'get_fertilizer_recommendation_ml'):
        try:
            # AgriGo expects numeric features in order: temperature, humidity, moisture, N, P, K
            num_feats = [
                inputs['temperature'],
                inputs['humidity'],
                inputs['moisture'],
                inputs['N'],
                inputs['P'],
                inputs['K'],
            ]
            # map categorical selections to indices used by AgriGo model
            soil_val = inputs.get('soil', '')
            crop_val = inputs.get('crop', '')
            try:
                soil_index = next(i for i, s in enumerate(SOIL_TYPES) if s.lower() == str(soil_val).lower())
            except StopIteration:
                soil_index = 0
            try:
                crop_index = next(i for i, c in enumerate(CROP_TYPES) if c.lower() == str(crop_val).lower())
            except StopIteration:
                crop_index = 0
            cat_feats = [soil_index, crop_index]
            pred = ml_models.get_fertilizer_recommendation_ml(num_feats, cat_feats)
            # map pred index back to class name if possible (use existing FERTILIZER_GUIDE keys or fallback)
            if isinstance(pred, (int, float)):
                classes = list(FERTILIZER_GUIDE.keys())
                idx = int(pred) % len(classes)
                fertilizer = classes[idx]
                guide = FERTILIZER_GUIDE.get(fertilizer, FERTILIZER_GUIDE['Organic'])
                return {
                    'fertilizer': fertilizer,
                    'confidence': 78.0,
                    'focus': guide['focus'],
                    'timing': guide['timing'],
                    'nutrient_gaps': {},
                    'summary': f"{fertilizer} predicted by local ML model",
                }
        except Exception:
            pass
    return fertilizer_recommendation(inputs)


def suitability_for_crop(crop, inputs):
    profile = CROP_PROFILES.get(crop.lower())
    if not profile:
        return 0.72
    scores = [feature_similarity(inputs[feature], profile[feature], feature) for feature in FEATURES]
    return sum(scores) / len(scores)


def local_yield_plan(inputs):
    crop_key = inputs["crop"].lower()
    base_yield = BASE_YIELD_T_HA.get(crop_key, 3.5)
    suitability = suitability_for_crop(crop_key, inputs)
    ph_penalty = 1 - min(abs(inputs["ph"] - 6.6) * 0.055, 0.22)
    moisture_factor = 1 + min(max((inputs["moisture"] - 45) / 180, -0.12), 0.12)
    fertilizer_factor = 1 + min(inputs["amount"] / 900, 0.18)
    estimate = base_yield * (0.68 + suitability * 0.45) * ph_penalty * moisture_factor * fertilizer_factor
    low = max(0.1, estimate * 0.88)
    high = estimate * 1.12

    confidence = round(max(52, min(94, 62 + suitability * 31 - abs(inputs["ph"] - 6.6) * 3)), 1)
    risk_level = "Low"
    if confidence < 68 or inputs["rainfall"] < 45 or inputs["ph"] < 5.4 or inputs["ph"] > 8.2:
        risk_level = "High"
    elif confidence < 78 or inputs["moisture"] < 25:
        risk_level = "Medium"

    actions = []
    if inputs["ph"] < 5.8:
        actions.append("Apply lime in a soil-test-guided dose before the next planting window.")
    elif inputs["ph"] > 7.8:
        actions.append("Use acidifying organic matter or sulfur amendments after local soil testing.")
    if inputs["N"] < 45:
        actions.append("Prioritize split nitrogen feeding to reduce leaching losses.")
    if inputs["P"] < 30:
        actions.append("Place phosphorus close to seed/root zones for early vigor.")
    if inputs["K"] < 30:
        actions.append("Add potassium support to improve stress tolerance and grain or fruit fill.")
    if inputs["rainfall"] < 80:
        actions.append("Plan supplemental irrigation around flowering and yield formation.")
    if not actions:
        actions.append("Maintain current nutrient balance and monitor pests after canopy closure.")

    return {
        "yield_range": f"{low:.2f} - {high:.2f} t/ha",
        "estimate": round(estimate, 2),
        "confidence": confidence,
        "risk_level": risk_level,
        "summary": (
            f"{title_crop(inputs['crop'])} is projected at {low:.2f} - {high:.2f} t/ha "
            f"with {confidence}% confidence under the submitted field conditions."
        ),
        "actions": actions,
    }


def groq_completion(messages, temperature=0.35, max_tokens=700):
    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key is not configured.")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def ai_completion(prompt, system_prompt=None, max_tokens=700):
    if GEMINI_MODEL:
        try:
            gemini_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
            response = GEMINI_MODEL.generate_content(gemini_prompt)
            return response.text.strip(), "Gemini"
        except Exception:
            pass

    if GROQ_API_KEY:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return groq_completion(messages, max_tokens=max_tokens), "Groq"
        except Exception as exc:
            return f"AI enrichment is unavailable. Local scoring was used. Reason: {exc}", "Local"

    return "AI enrichment is not configured; local agronomic scoring was used.", "Local"


def ai_enrichment(inputs, local_plan, fertilizer_plan):
    prompt = f"""
Act as an agronomy advisor. Review this field plan and respond in 5 concise bullets.
Crop: {inputs['crop']}
Soil: {inputs['soil']}
NPK: {inputs['N']}/{inputs['P']}/{inputs['K']}
pH: {inputs['ph']}
Weather: {inputs['temperature']} C, {inputs['humidity']}% humidity, {inputs['rainfall']} mm rainfall
Moisture: {inputs['moisture']}%
Fertilizer: {inputs['fertilizer']} at {inputs['amount']} kg/ha
Local estimate: {local_plan['yield_range']}, risk {local_plan['risk_level']}
Fertilizer recommendation: {fertilizer_plan['fertilizer']} for {fertilizer_plan['focus']}
Give practical actions and caution where uncertainty is high.
""".strip()
    text, provider = ai_completion(
        prompt,
        "You are a precise agricultural decision assistant. Be practical and avoid unsupported guarantees.",
    )
    return f"Provider: {provider}\n\n{text}"


def local_chatbot_response(user_id, message):
    context = build_chat_context(user_id)
    lowered = message.lower()
    tips = []

    if any(word in lowered for word in ["fertil", "urea", "dap", "npk", "nutrient"]):
        tips.append("Focus on matching fertilizer to the crop stage and soil test results rather than using a fixed rate.")
    if any(word in lowered for word in ["yield", "harvest", "production", "output"]):
        tips.append("Yield usually improves most from balanced NPK, stable moisture, and pH close to the crop target.")
    if any(word in lowered for word in ["soil", "ph", "acid", "alkaline"]):
        tips.append("Keep pH near the crop's preferred range; extreme acidity or alkalinity reduces nutrient uptake.")
    if any(word in lowered for word in ["water", "rain", "moisture", "irrig"]):
        tips.append("Water stress is easiest to correct early, especially around flowering and grain or fruit fill.")
    if any(word in lowered for word in ["crop", "plant", "sow", "recommend"]):
        tips.append("Choose the crop with the highest fit score for your soil, rainfall, and temperature window.")

    if not tips:
        tips.append("Share the crop, soil, NPK, pH, rainfall, and moisture values and I can give a practical field plan.")

    if context != "No saved field plans yet.":
        tips.append(f"Latest saved plans: {context}")

    return "\n".join(f"- {tip}" for tip in tips)


def format_chat_message(text):
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[•*]\s+", "- ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_chat_boilerplate(text):
    cleaned = str(text or "")
    patterns = [
        r"\n?-\s*Account-Based Farm Management:.*?(?=\n-\s*\w|\Z)",
        r"\n?Account-Based Farm Management:.*?(?=\n\n|\Z)",
        r"\n?-\s*To access your farm's account, please log in to your AgriNexus account\..*?(?=\n-\s*\w|\Z)",
        r"\n?-\s*You can view your saved decisions, track your crop progress, and access other farm management features\..*?(?=\n-\s*\w|\Z)",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def prediction_payload(form):
    crop = form.get("crop", "").strip()
    soil = form.get("soil", "").strip()
    fertilizer = form.get("fertilizer", "").strip()
    if not crop or not soil or not fertilizer:
        raise ValueError("Crop, soil, and fertilizer are required.")

    return {
        "crop": crop,
        "soil": soil,
        "weather": form.get("weather", "Field observation").strip() or "Field observation",
        "fertilizer": fertilizer,
        "amount": parse_float(form.get("amount"), "Fertilizer amount", 0, 1000),
        "N": parse_float(form.get("nitrogen"), "Nitrogen", 0, 250),
        "P": parse_float(form.get("phosphorus"), "Phosphorus", 0, 250),
        "K": parse_float(form.get("potassium"), "Potassium", 0, 250),
        "ph": parse_float(form.get("ph"), "Soil pH", 3, 10),
        "temperature": parse_float(form.get("temperature"), "Temperature", -5, 60),
        "humidity": parse_float(form.get("humidity"), "Humidity", 0, 100),
        "rainfall": parse_float(form.get("rainfall"), "Rainfall", 0, 500),
        "moisture": parse_float(form.get("moisture"), "Moisture", 0, 100),
    }


def store_prediction(inputs, local_plan, fertilizer_plan, crop_plan, raw_response, user_id):
    recommendation = {
        "yield": local_plan,
        "fertilizer": fertilizer_plan,
        "crop_matches": crop_plan,
    }
    with connect_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                soil, weather, fertilizer, amount, crop, predicted_yield, description,
                raw_response, prediction_type, nitrogen, phosphorus, potassium, ph,
                temperature, humidity, rainfall, moisture, risk_level, confidence,
                recommendation, created_at, user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inputs["soil"],
                inputs["weather"],
                inputs["fertilizer"],
                inputs["amount"],
                inputs["crop"],
                local_plan["yield_range"],
                "AgriNexus hybrid agronomic score",
                raw_response,
                "yield",
                inputs["N"],
                inputs["P"],
                inputs["K"],
                inputs["ph"],
                inputs["temperature"],
                inputs["humidity"],
                inputs["rainfall"],
                inputs["moisture"],
                local_plan["risk_level"],
                local_plan["confidence"],
                json.dumps(recommendation),
                datetime.utcnow().isoformat(timespec="seconds"),
                user_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_all_predictions(user_id=None):
    with connect_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id IS NULL ORDER BY id DESC"
            ).fetchall()
    return [enrich_prediction_row(row) for row in rows]


def get_prediction(prediction_id, user_id=None):
    with connect_db() as conn:
        if user_id:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
                (prediction_id, user_id),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if not row:
        return None
    item = enrich_prediction_row(row)
    try:
        item["recommendation_data"] = json.loads(item.get("recommendation") or "{}")
    except json.JSONDecodeError:
        item["recommendation_data"] = {}
    return item


def dashboard_metrics(user_id=None):
    predictions = get_all_predictions(user_id)
    crop_counts = Counter(item.get("crop") for item in predictions if item.get("crop"))
    risk_counts = Counter(item.get("risk_level") or "Unrated" for item in predictions)
    avg_confidence = 0
    confidences = [item["confidence"] for item in predictions if item.get("confidence") is not None]
    if confidences:
        avg_confidence = round(sum(confidences) / len(confidences), 1)
    return {
        "total": len(predictions),
        "top_crop": crop_counts.most_common(1)[0][0] if crop_counts else "No records",
        "avg_confidence": avg_confidence,
        "risk_counts": dict(risk_counts),
        "recent": predictions[:5],
    }


def chat_history(user_id, limit=20):
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    history = []
    for row in reversed(rows):
        item = dict(row)
        item["display_message"] = format_chat_message(item.get("message"))
        history.append(item)
    return history


def save_chat_message(user_id, role, message, provider=None):
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (user_id, role, message, provider, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, role, message, provider, datetime.utcnow().isoformat(timespec="seconds")),
        )
        conn.commit()


def build_chat_context(user_id):
    recent = get_all_predictions(user_id)[:3]
    if not recent:
        return "No saved field plans yet."
    lines = []
    for item in recent:
        lines.append(
            f"{item['crop']} on {item['soil']} soil: yield {item['predicted_yield']}, "
            f"risk {item.get('risk_level')}, fertilizer {item['fertilizer']}."
        )
    return "\n".join(lines)


def chatbot_reply(user_id, message):
    history = chat_history(user_id, limit=8)
    system_prompt = (
        f"You are {APP_NAME}'s farm operations chatbot. Use only current project capabilities: "
        "yield planning, crop recommendation, fertilizer strategy, and saved field context. "
        "Do not mention login, account management, CSV export, or product help text unless the user explicitly asks. "
        "Give concise, practical agronomy guidance and avoid claiming certainty where soil tests or local extension advice are needed."
    )
    context = build_chat_context(user_id)
    transcript = "\n".join(
        f"{item['role']}: {item['message']}" for item in history[-6:] if item["role"] in {"user", "assistant"}
    )
    prompt = f"""
Recent saved field context:
{context}

Recent conversation:
{transcript or "No previous messages."}

Farmer question:
{message}
""".strip()
    text, provider = ai_completion(prompt, system_prompt, max_tokens=650)
    if provider == "Local":
        text = local_chatbot_response(user_id, message)
    text = strip_chat_boilerplate(text)
    return format_chat_message(text), provider


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        farm_name = request.form.get("farm_name", "").strip()
        location = request.form.get("location", "").strip()
        if not name or not email or len(password) < 6:
            return render_template(
                "auth.html",
                app_name=APP_NAME,
                mode="register",
                error="Name, email, and a 6+ character password are required.",
            )
        try:
            with connect_db() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (name, email, password_hash, farm_name, location, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password),
                        farm_name,
                        location,
                        datetime.utcnow().isoformat(timespec="seconds"),
                    ),
                )
                conn.commit()
                session["user_id"] = cursor.lastrowid
            return redirect(url_for("index"))
        except sqlite3.IntegrityError:
            return render_template(
                "auth.html",
                app_name=APP_NAME,
                mode="register",
                error="An account already exists for that email.",
            )
    return render_template("auth.html", app_name=APP_NAME, mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with connect_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["id"]
            return redirect(request.args.get("next") or url_for("index"))
        return render_template(
            "auth.html",
            app_name=APP_NAME,
            mode="login",
            error="Invalid email or password.",
        )
    return render_template("auth.html", app_name=APP_NAME, mode="login")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/account")
@login_required
def account():
    user = current_user()
    metrics = dashboard_metrics(user["id"])
    with connect_db() as conn:
        chat_count = conn.execute(
            "SELECT COUNT(*) AS total FROM chat_messages WHERE user_id = ?",
            (user["id"],),
        ).fetchone()["total"]
    return render_template(
        "account.html",
        app_name=APP_NAME,
        user=user,
        metrics=metrics,
        chat_count=chat_count,
    )


@app.route("/")
def index():
    user = current_user()
    return render_template(
        "index.html",
        app_name=APP_NAME,
        crops=CROP_TYPES,
        soils=SOIL_TYPES,
        fertilizers=sorted(FERTILIZER_GUIDE),
        metrics=dashboard_metrics(user["id"] if user else None),
        gemini_enabled=bool(GEMINI_MODEL),
        groq_enabled=bool(GROQ_API_KEY),
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))
    try:
        inputs = prediction_payload(request.form)
        local_plan = local_yield_plan(inputs)
        fertilizer_plan = fertilizer_recommendation_with_ml(inputs)
        crop_plan = crop_recommendations(inputs, limit=3)
        raw_response = ai_enrichment(inputs, local_plan, fertilizer_plan)
        prediction_id = store_prediction(
            inputs,
            local_plan,
            fertilizer_plan,
            crop_plan,
            raw_response,
            session["user_id"],
        )
        session["result_id"] = prediction_id
        return redirect(url_for("prediction_detail", prediction_id=prediction_id))
    except ValueError as exc:
        return render_template(
            "index.html",
            app_name=APP_NAME,
            crops=CROP_TYPES,
            soils=SOIL_TYPES,
            fertilizers=sorted(FERTILIZER_GUIDE),
            metrics=dashboard_metrics(session.get("user_id")),
            gemini_enabled=bool(GEMINI_MODEL),
            groq_enabled=bool(GROQ_API_KEY),
            error=str(exc),
        )


@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    result = None
    error = None
    if request.method == "POST":
        try:
            inputs = {
                "N": parse_float(request.form.get("nitrogen"), "Nitrogen", 0, 250),
                "P": parse_float(request.form.get("phosphorus"), "Phosphorus", 0, 250),
                "K": parse_float(request.form.get("potassium"), "Potassium", 0, 250),
                "temperature": parse_float(request.form.get("temperature"), "Temperature", -5, 60),
                "humidity": parse_float(request.form.get("humidity"), "Humidity", 0, 100),
                "ph": parse_float(request.form.get("ph"), "Soil pH", 3, 10),
                "rainfall": parse_float(request.form.get("rainfall"), "Rainfall", 0, 500),
            }
            if ML_MODELS_AVAILABLE:
                try:
                    result = ml_models.get_crop_recommendation_ml(inputs)
                except Exception:
                    result = crop_recommendations(inputs)
            else:
                result = crop_recommendations(inputs)
        except ValueError as exc:
            error = str(exc)
    return render_template("crop_recommend.html", app_name=APP_NAME, result=result, error=error)


@app.route("/fertilizer-recommendation", methods=["GET", "POST"])
def fertilizer_recommend():
    result = None
    error = None
    if request.method == "POST":
        try:
            inputs = {
                "temperature": parse_float(request.form.get("temperature"), "Temperature", -5, 60),
                "humidity": parse_float(request.form.get("humidity"), "Humidity", 0, 100),
                "moisture": parse_float(request.form.get("moisture"), "Moisture", 0, 100),
                "N": parse_float(request.form.get("nitrogen"), "Nitrogen", 0, 250),
                "P": parse_float(request.form.get("phosphorus"), "Phosphorus", 0, 250),
                "K": parse_float(request.form.get("potassium"), "Potassium", 0, 250),
                "soil": request.form.get("soil", "").strip(),
                "crop": request.form.get("crop", "").strip(),
            }
            if not inputs["soil"] or not inputs["crop"]:
                raise ValueError("Soil and crop are required.")
            result = fertilizer_recommendation_with_ml(inputs)
        except ValueError as exc:
            error = str(exc)
    return render_template(
        "fertilizer_recommend.html",
        app_name=APP_NAME,
        soils=SOIL_TYPES,
        crops=CROP_TYPES,
        result=result,
        error=error,
    )


@app.route("/prediction")
@login_required
def prediction():
    prediction_id = session.get("result_id")
    if not prediction_id:
        return redirect(url_for("index"))
    return redirect(url_for("prediction_detail", prediction_id=prediction_id))


@app.route("/prediction/<int:prediction_id>")
@login_required
def prediction_detail(prediction_id):
    prediction_item = get_prediction(prediction_id, session["user_id"])
    if not prediction_item:
        return redirect(url_for("index"))
    return render_template("prediction.html", app_name=APP_NAME, prediction=prediction_item)


@app.route("/all_responses")
@login_required
def all_responses():
    return render_template(
        "all_responses.html",
        app_name=APP_NAME,
        predictions=get_all_predictions(session["user_id"]),
        metrics=dashboard_metrics(session["user_id"]),
    )


@app.route("/export/predictions.csv")
@login_required
def export_predictions():
    rows = get_all_predictions(session["user_id"])
    output = io.StringIO()
    fields = [
        "id",
        "created_at",
        "crop",
        "soil",
        "fertilizer",
        "amount",
        "predicted_yield",
        "risk_level",
        "confidence",
        "nitrogen",
        "phosphorus",
        "potassium",
        "ph",
        "temperature",
        "humidity",
        "rainfall",
        "moisture",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=agrinexus_predictions.csv"},
    )


@app.route("/api/recommendations", methods=["POST"])
def api_recommendations():
    data = request.get_json(silent=True) or {}
    try:
        inputs = {
            "N": parse_float(data.get("nitrogen"), "Nitrogen", 0, 250),
            "P": parse_float(data.get("phosphorus"), "Phosphorus", 0, 250),
            "K": parse_float(data.get("potassium"), "Potassium", 0, 250),
            "temperature": parse_float(data.get("temperature"), "Temperature", -5, 60),
            "humidity": parse_float(data.get("humidity"), "Humidity", 0, 100),
            "ph": parse_float(data.get("ph"), "Soil pH", 3, 10),
            "rainfall": parse_float(data.get("rainfall"), "Rainfall", 0, 500),
            "moisture": parse_float(data.get("moisture", 45), "Moisture", 0, 100),
            "soil": data.get("soil", "Loamy"),
            "crop": data.get("crop", "Rice"),
            "fertilizer": data.get("fertilizer", "Organic"),
            "amount": parse_float(data.get("amount", 80), "Amount", 0, 1000),
            "weather": data.get("weather", "API request"),
        }
        return jsonify(
            {
                "crop_recommendations": crop_recommendations(inputs),
                "fertilizer_recommendation": fertilizer_recommendation(inputs),
                "yield_plan": local_yield_plan(inputs),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    error = None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            save_chat_message(session["user_id"], "user", message)
            reply, provider = chatbot_reply(session["user_id"], message)
            save_chat_message(session["user_id"], "assistant", reply, provider)
        else:
            error = "Type a question before sending."
    return render_template(
        "chat.html",
        app_name=APP_NAME,
        messages=chat_history(session["user_id"]),
        error=error,
        gemini_enabled=bool(GEMINI_MODEL),
        groq_enabled=bool(GROQ_API_KEY),
    )


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400
    save_chat_message(session["user_id"], "user", message)
    reply, provider = chatbot_reply(session["user_id"], message)
    save_chat_message(session["user_id"], "assistant", reply, provider)
    return jsonify({"reply": reply, "provider": provider})


@app.route("/crop-disease", methods=["GET", "POST"])
def crop_disease():
    result = None
    error = None
    image_file = None
    if request.method == "POST":
        crop = request.form.get("crop", "").strip().lower()
        if not crop:
            error = "Please select a crop type."
        elif "file" not in request.files or not request.files["file"].filename:
            error = "Please upload a leaf image."
        else:
            file = request.files["file"]
            try:
                from werkzeug.utils import secure_filename
                upload_dir = os.path.join(app.root_path, "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                prediction_index = ml_models.img_predict(file_path, crop)
                result = ml_models.get_diseases_classes(crop, prediction_index)
                image_file = filename
            except FileNotFoundError as exc:
                error = f"No disease model available for '{crop}'. Supported crops: {', '.join(ml_models.CROP_DISEASE_LIST)}"
            except Exception as exc:
                error = f"Prediction error: {exc}"
    return render_template(
        "crop_disease.html",
        app_name=APP_NAME,
        crops=ml_models.CROP_DISEASE_LIST,
        result=result,
        image_file=image_file,
        error=error,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(app.root_path, "uploads"), filename)


@app.route("/about")
def about():
    return render_template("about.html", app_name=APP_NAME)


@app.route("/developer")
def developer():
    return render_template("developer.html", app_name=APP_NAME)


init_db()

if __name__ == "__main__":
    app.run(debug=True)
