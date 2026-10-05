"""
Flask Backend API - IMDB Sentiment Analysis
Endpoints:
  POST /predict   -> { "review": "..." }  -> { "sentiment": "positive/negative", "confidence": 0.xx }
  GET  /metrics   -> returns evaluation metrics JSON
  GET  /health    -> health check
"""

import os
import json
import re
import joblib
import nltk
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model      = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

with open(os.path.join(BASE_DIR, "static", "metrics.json")) as f:
    METRICS = json.load(f)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


# ── Text cleaning (mirrors training pipeline) ─────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "LogisticRegression", "vectorizer": "TF-IDF"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Empty review text provided."}), 400

    cleaned   = clean_text(review)
    features  = vectorizer.transform([cleaned])
    label     = int(model.predict(features)[0])
    proba     = model.predict_proba(features)[0]
    confidence = float(proba[label])
    sentiment  = "positive" if label == 1 else "negative"

    return jsonify({
        "sentiment":   sentiment,
        "confidence":  round(confidence, 4),
        "label":       label,
        "review_preview": review[:200]
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(METRICS)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/report", methods=["GET"])
def report():
    return render_template("report.html")


if __name__ == "__main__":
    print("Starting Flask API on http://127.0.0.1:5000")
    app.run(debug=False, port=5000, use_reloader=False)
