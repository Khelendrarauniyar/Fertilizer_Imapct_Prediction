# AgriNexus

AgriNexus is an account-based farm decision platform for yield prediction, crop suitability, fertilizer strategy, crop disease detection, decision history, CSV export, and AI chatbot support.

## Features

- **Secure user registration and login** with password hashing and session management.
- **Personal dashboard** for saved field plans and account metrics.
- **Yield planner** using crop, soil, NPK, pH, rainfall, moisture, weather, and fertilizer rate — with confidence scoring and risk assessment.
- **Crop advisor** that ranks suitable crops for submitted field conditions using weighted feature similarity.
- **Fertilizer advisor** with nutrient gap analysis, timing guidance, and confidence scoring — backed by ML models.
- **Crop disease detection** — upload leaf images and diagnose diseases using trained deep learning models (9 crops, 37 disease classes).
- **Gemini-first AI agronomy enrichment** with Groq fallback for practical field advice.
- **Chatbot** grounded in each user's recent saved field plans, with local fallback advice.
- **SQLite storage** for users, predictions, and chatbot messages.
- **CSV export** and **JSON recommendation API**.
- **Docker support** for easy deployment.

## Setup

```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Environment

Create a `.env` file in the project root:

```text
SECRET_KEY=replace_with_a_stable_secret
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

Gemini is used first for AI responses. If Gemini is unavailable and `GROQ_API_KEY` is configured, AgriNexus automatically falls back to Groq. If neither key is configured, local scoring still works for all features.

## Routes

| Route | Description |
|-------|-------------|
| `/` | Home / Yield Planner |
| `/register` | User registration |
| `/login` | User login |
| `/logout` | Logout |
| `/account` | Account dashboard with metrics |
| `/predict` | Submit yield prediction (POST) |
| `/prediction/<id>` | View a saved prediction detail |
| `/all_responses` | View all saved predictions |
| `/crop-recommendation` | Crop suitability advisor |
| `/fertilizer-recommendation` | Fertilizer recommendation |
| `/crop-disease` | Leaf image disease detection |
| `/chat` | AI-powered farm chatbot |
| `/about` | About page |
| `/developer` | Developer information |
| `/export/predictions.csv` | Export predictions as CSV |

## API

### POST /api/recommendations

Full field analysis returning crop recommendations, fertilizer advice, and yield plan.

```json
{
  "crop": "Rice",
  "soil": "Loamy",
  "nitrogen": 60,
  "phosphorus": 45,
  "potassium": 40,
  "temperature": 24,
  "humidity": 70,
  "ph": 6.6,
  "rainfall": 180,
  "moisture": 45,
  "fertilizer": "Organic",
  "amount": 80
}
```

### POST /api/chat

Send a message to the AI chatbot (requires authentication).

```json
{
  "message": "What fertilizer should I use for rice on loamy soil?"
}
```

## Project Structure

```
Fertilizer_Impact_And_Yeild_Prediction/
├── app.py                 # Main Flask application (1232 lines)
├── ml_models.py           # ML model loading, inference, disease classes
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build
├── .env                   # Environment variables (not committed)
├── .gitignore
├── data/
│   ├── crop_recommendation.csv    # 2200+ rows, 22 crop types
│   └── fertilizer_prediction.csv  # 100 rows, 7 fertilizer types
├── models/
│   ├── DL_models/         # Keras H5 models for crop disease (9 crops)
│   └── ML_models/         # Pickle models + scalers (crop & fertilizer)
├── static/
│   ├── style.css          # Main responsive stylesheet
│   ├── disease_style.css  # Disease detection page styles
│   └── images/            # Hero and feature images
├── templates/
│   ├── base.html          # Base layout with navigation
│   ├── index.html         # Home / yield planner
│   ├── auth.html          # Login / register
│   ├── account.html       # User dashboard
│   ├── prediction.html    # Prediction detail
│   ├── all_responses.html # Prediction history
│   ├── crop_recommend.html
│   ├── fertilizer_recommend.html
│   ├── crop_disease.html  # Disease detection UI
│   ├── chat.html          # AI chatbot
│   ├── about.html
│   └── developer.html
└── external/
    └── AgriGo/            # External AgriGo reference project
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python Flask |
| Database | SQLite |
| ML Models | scikit-learn, TensorFlow 2.8 (CPU), Keras |
| AI APIs | Google Gemini (primary), Groq (fallback) |
| Frontend | Jinja2 templates, Vanilla CSS |
| Containerization | Docker (Python 3.11-slim) |
| Image Processing | Pillow |

## Docker

```bash
docker build -t agrinexus -f docker .
docker run -p 5000:5000 agrinexus
```

## License

MIT License — see [LICENSE](LICENSE).