import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "ML_models")

# Feature ordering used by the original AgriGo crop model
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Reuse the same crop ordering as the AgriGo model to map labels -> names
_crops_map = {
    'apple': 1, 'banana': 2, 'blackgram': 3, 'chickpea': 4, 'coconut': 5, 'coffee': 6, 'cotton': 7,
    'grapes': 8, 'jute': 9, 'kidneybeans': 10, 'lentil': 11, 'maize': 12, 'mango': 13, 'mothbeans': 14,
    'mungbean': 15, 'muskmelon': 16, 'orange': 17, 'paddy': 18, 'papaya': 19, 'pigeonpeas': 20,
    'pomegranate': 21, 'rice': 22, 'watermelon': 23
}

CROPS = list(_crops_map.keys())

_crop_scaler = None
_crop_model = None

def _ensure_crop_loaded():
    global _crop_scaler, _crop_model
    if _crop_scaler is not None and _crop_model is not None:
        return
    scaler_path = os.path.join(MODEL_DIR, "crop_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, "crop_model.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Crop model artifacts not found in models/ML_models")
    with open(scaler_path, "rb") as fh:
        _crop_scaler = pickle.load(fh)
    with open(model_path, "rb") as fh:
        _crop_model = pickle.load(fh)

def get_crop_recommendation_ml(inputs, limit=5):
    """Return a list of recommendation dicts compatible with the existing UI.

    inputs: dict with keys matching FEATURES
    """
    _ensure_crop_loaded()
    values = [float(inputs.get(f, 0)) for f in FEATURES]
    arr = np.array(values).reshape(1, -1)
    scaled = _crop_scaler.transform(arr)
    pred = _crop_model.predict(scaled)[0]
    try:
        crop_name = CROPS[int(pred)]
    except Exception:
        crop_name = str(pred)

    return [{
        "crop": crop_name.title(),
        "score": 99.9,
        "why": "Predicted by local ML model",
        "profile": {},
    }]


_fert_scaler = None
_fert_model = None

def _ensure_fert_loaded():
    global _fert_scaler, _fert_model
    if _fert_scaler is not None and _fert_model is not None:
        return
    scaler_path = os.path.join(MODEL_DIR, "fertilizer_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, "fertilizer_model.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Fertilizer model artifacts not found in models/ML_models")
    with open(scaler_path, "rb") as fh:
        _fert_scaler = pickle.load(fh)
    with open(model_path, "rb") as fh:
        _fert_model = pickle.load(fh)

def get_fertilizer_recommendation_ml(num_features, cat_features):
    """Return fertilizer class predicted by AgriGo fertilizer_model.pkl.

    num_features: list-like numeric features
    cat_features: list-like categorical values (assumed already encoded in AgriGo as strings)
    """
    _ensure_fert_loaded()
    num_arr = np.array(num_features).reshape(1, -1)
    scaled = _fert_scaler.transform(num_arr)
    cat_arr = np.array(cat_features).reshape(1, -1)
    item = np.concatenate([scaled, cat_arr], axis=1)
    pred = _fert_model.predict(item)[0]
    try:
        return int(pred)
    except Exception:
        return pred


CROP_DISEASE_CLASSES = {
    'strawberry': [(0, 'Leaf_scorch'), (1, 'healthy')],
    'patato': [(0, 'Early_blight'), (1, 'Late_blight'), (2, 'healthy')],
    'corn': [
        (0, 'Cercospora_leaf_spot Gray_leaf_spot'),
        (1, 'Common_rust_'),
        (2, 'Northern_Leaf_Blight'),
        (3, 'healthy'),
    ],
    'apple': [(0, 'Apple_scab'), (1, 'Black_rot'), (2, 'Cedar_apple_rust'), (3, 'healthy')],
    'cherry': [(0, 'Powdery_mildew'), (1, 'healthy')],
    'grape': [
        (0, 'Black_rot'),
        (1, 'Esca_(Black_Measles)'),
        (2, 'Leaf_blight_(Isariopsis_Leaf_Spot)'),
        (3, 'healthy'),
    ],
    'peach': [(0, 'Bacterial_spot'), (1, 'healthy')],
    'pepper': [(0, 'Bacterial_spot'), (1, 'healthy')],
    'tomato': [
        (0, 'Bacterial_spot'),
        (1, 'Early_blight'),
        (2, 'Late_blight'),
        (3, 'Leaf_Mold'),
        (4, 'Septoria_leaf_spot'),
        (5, 'Spider_mites Two-spotted_spider_mite'),
        (6, 'Target_Spot'),
        (7, 'Tomato_Yellow_Leaf_Curl_Virus'),
        (8, 'Tomato_mosaic_virus'),
        (9, 'healthy'),
    ],
}

CROP_DISEASE_LIST = list(CROP_DISEASE_CLASSES.keys())


def get_diseases_classes(crop, prediction):
    """Return the human-readable disease name for a crop and prediction index."""
    classes = CROP_DISEASE_CLASSES.get(crop)
    if not classes:
        raise ValueError(f"No disease classes defined for crop '{crop}'.")
    for idx, name in classes:
        if idx == prediction:
            return name.replace("_", " ")
    return "Unknown"


def img_predict(path, crop):
    model_path = os.path.join(BASE_DIR, 'models', 'DL_models', f'{crop}_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No DL model for crop '{crop}'")
    model = load_model(model_path, compile=False)
    data = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(data).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255
    if hasattr(model, 'predict'):
        p = model.predict(data)[0]
        if p.shape and p.shape[0] > 1:
            return int(np.argmax(p))
        return int(np.round(p)[0])
    return 0
