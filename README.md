# AgriNexus

AgriNexus is an account-based farm decision platform for yield prediction, crop suitability, fertilizer strategy, decision history, CSV export, and AI chatbot support.

## Features

- Secure user registration and login with password hashing.
- Personal dashboard for saved field plans and account metrics.
- Yield planner using crop, soil, NPK, pH, rainfall, moisture, weather, and fertilizer rate.
- Crop advisor that ranks suitable crops for submitted field conditions.
- Fertilizer advisor with nutrient gaps, timing guidance, and confidence scoring.
- Gemini-first AI agronomy enrichment with Groq fallback.
- Chatbot grounded in each user's recent saved field plans.
- SQLite storage for users, predictions, and chatbot messages.
- CSV export and JSON recommendation API.

## Setup

```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Environment

```text
SECRET_KEY=replace_with_a_stable_secret
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

Gemini is used first for AI responses. If Gemini is unavailable and `GROQ_API_KEY` is configured, AgriNexus automatically falls back to Groq. If neither key is configured, local scoring still works.

## API

`POST /api/recommendations`

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
